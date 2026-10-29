"""Offline tests: no API key, GPU, or network needed (LLM_PROVIDER=mock, RETRIEVER=tfidf, logreg risk model)."""
from __future__ import annotations

import sqlite3

import pytest

from fraud_agent import data_gen, risk_model
from fraud_agent.config import Settings
from fraud_agent.db import QueryRejected, get_alert, get_ground_truth, safe_select
from fraud_agent.llm import MockLLM, StructuredOutputError, Usage
from fraud_agent.nodes import Nodes, check_citations, run_without_langgraph
from fraud_agent.pii import mask
from fraud_agent.schemas import CaseSummary, Claim, ToolCall
from fraud_agent.tools import Toolbox


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    d = tmp_path_factory.mktemp("data")
    s = Settings(data_dir=d, results_dir=d / "results", llm_provider="mock", retriever="tfidf",
                 risk_model_type="logreg", tool_timeout_s=10)
    rows = data_gen.Generator(n_customers=300, alerts_per_type=4, benign_alerts=10, seed=3).run()
    data_gen.write_db(rows, s.db_path)
    risk_model.train("logreg", out_path=s.model_path)
    return s


@pytest.fixture(scope="session")
def nodes(settings) -> Nodes:
    return Nodes(settings, MockLLM(), Toolbox(settings))


def test_ground_truth_is_balanced(settings):
    gt = get_ground_truth(settings.db_path)
    typ = {g["typology"] for g in gt.values()}
    assert typ == {"account_takeover", "card_testing", "money_mule", "structuring", "benign"}


def test_sql_rejects_non_whitelisted_table(settings):
    with pytest.raises(QueryRejected):
        safe_select(settings.db_path, "alerts", "AC00001", "2026-03-01T00:00:00")


def test_sql_never_exposes_labels(settings):
    rows = safe_select(settings.db_path, "transactions", "AC00001", "2026-04-01T00:00:00")
    assert rows and "is_fraud" not in rows[0] and "typology" not in rows[0]


def test_connection_is_read_only(settings):
    from fraud_agent.db import connect_ro

    with connect_ro(settings.db_path) as con, pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM transactions")


def test_sql_injection_is_inert(settings):
    rows = safe_select(settings.db_path, "transactions", "AC00001' OR '1'='1", "2026-04-01T00:00:00")
    assert rows == []


def test_pii_masking():
    out = mask({"full_name": "Ava Patel", "email": "a@b.com", "payee_account": "1234567890",
                "note": "call +1-555-123-4567 or mail x.y@example.com"})
    assert out["full_name"] == "[REDACTED]" and out["payee_account"] == "****7890"
    assert "[PHONE]" in out["note"] and "[EMAIL]" in out["note"]


def test_tool_failure_degrades_gracefully(settings):
    tb = Toolbox(settings)
    alert = get_alert(settings.db_path, "A000001")
    ev = tb.execute(ToolCall(tool="typology_classifier", text="x", reason="t"), alert, "E1")
    assert ev.status == "error" and "unavailable" in ev.error


def test_bad_args_are_not_retried(settings):
    tb = Toolbox(settings)
    alert = get_alert(settings.db_path, "A000001")
    ev = tb.execute(ToolCall(tool="sql_lookup", reason="no table"), alert, "E1")
    assert ev.status == "error" and ev.attempts == 1


def test_citation_check_flags_unknown_evidence():
    summary = CaseSummary(alert_id="A1", disposition="close", typology="benign", confidence="low", summary="s",
                          claims=[Claim(statement="ok", evidence_ids=["E1"]),
                                  Claim(statement="made up", evidence_ids=["E9"])])
    ev = [{"evidence_id": "E1", "status": "ok", "data": {"x": 1}}]
    check = check_citations(summary, ev)
    assert not check["passed"] and check["unsupported"][0]["invalid_ids"] == ["E9"]


def test_claims_without_evidence_fail_validation():
    with pytest.raises(ValueError):
        CaseSummary(alert_id="A1", disposition="close", typology="benign", confidence="low", summary="s",
                    claims=[Claim(statement="no evidence", evidence_ids=[])])


@pytest.mark.parametrize("typology", ["account_takeover", "card_testing", "money_mule", "structuring", "benign"])
def test_end_to_end_offline(settings, nodes, typology):
    gt = get_ground_truth(settings.db_path)
    alert_id = next(a for a, g in gt.items() if g["typology"] == typology)
    state = run_without_langgraph(nodes, alert_id)
    assert state["summary"]["disposition"] == gt[alert_id]["disposition"]
    assert state["citation_check"]["passed"]
    assert set(gt[alert_id]["expected_tools"]) <= {e["tool"] for e in state["evidence"]}
    assert state["steps"] <= settings.max_steps


def test_planner_never_repeats_calls(settings, nodes):
    state = {"alert_id": "A000001"}
    state.update(nodes.load_alert(state))
    state.update(nodes.planner(state))
    keys = [ToolCall(**c).key() for c in state["pending_calls"]]
    assert len(keys) == len(set(keys))


def test_writer_repairs_then_falls_back(settings):
    class BrokenLLM:
        model_name = "broken"

        def structured(self, *a, **k):
            raise StructuredOutputError("bad json", Usage(10, 5, 1))

    n = Nodes(settings, BrokenLLM(), Toolbox(settings))
    state = run_without_langgraph(n, "A000001")
    assert any("fallback" in d for d in state["degraded"])
    assert state["summary"]["confidence"] == "low"


def test_sar_requires_review(settings, nodes):
    gt = get_ground_truth(settings.db_path)
    alert_id = next(a for a, g in gt.items() if g["disposition"] == "file_sar")
    state = run_without_langgraph(nodes, alert_id, auto_approve=False)
    assert state["review"]["approved"] is False and state["summary"]["disposition"] == "escalate"


def test_langgraph_wiring(settings, nodes):
    pytest.importorskip("langgraph")
    from fraud_agent.graph import build_graph, resume_case, run_case

    graph = build_graph(nodes)
    gt = get_ground_truth(settings.db_path)
    sar = next(a for a, g in gt.items() if g["disposition"] == "file_sar")
    out = run_case(graph, settings, sar, auto_approve=False, thread_id="t1")
    assert "__interrupt__" in out
    out = resume_case(graph, "t1", approved=True)
    assert out["review"]["approved"] is True and out["summary"]["disposition"] == "file_sar"
