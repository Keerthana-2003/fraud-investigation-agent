"""Graph node logic as plain functions, so it can be unit-tested without LangGraph.

State keys (all JSON-serialisable):
    alert_id, alert, evidence (list[dict]), call_keys, pending_calls, steps, done,
    summary, citation_check, review, trace, usage, degraded
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from pydantic import ValidationError

from . import rules
from .config import Settings
from .db import get_alert
from .llm import StructuredOutputError, Usage
from .pii import mask
from .schemas import CaseSummary, Evidence, PlannerDecision, ToolCall
from .tools import TOOL_DESCRIPTIONS, Toolbox

PLANNER_SYSTEM = """You are a fraud investigation planner at a bank. Decide which evidence to gather next for the
alert. Available tools:
{tools}

Rules:
- Request only evidence that is not already collected. Never repeat a call.
- Request at most {max_calls} tool calls per step.
- Always consult the policy that applies to this alert type before finishing.
- Set done=true when the evidence is enough to apply a policy rule or rule one out.
- Customer PII has been masked. Do not ask for it."""

WRITER_SYSTEM = """You are a senior fraud investigator writing the case file for an alert.

Dispositions:
- close: activity is explained by legitimate behavior.
- escalate: likely fraud against the customer (e.g. account takeover, card testing).
- file_sar: suspicious activity by or through the customer (e.g. money mule, structuring).

Requirements:
- Apply the retrieved policy rules exactly as written. Name the rules you applied (e.g. ATO-1).
- Every claim must cite the evidence IDs that support it (e.g. ["E2", "E5"]); the alert itself is "ALERT".
- Cite only evidence IDs that appear below with status "ok". Do not state numbers that are not in the cited evidence.
- If a tool failed, say so and lower your confidence; do not guess what it would have returned.
- Customer PII has been masked; never try to infer it."""


# --------------------------------------------------------------------------- helpers
def _compact(e: dict[str, Any], max_rows: int = 60, max_chars: int = 9000) -> dict[str, Any]:
    data = e.get("data")
    if isinstance(data, dict) and "rows" in data and len(data["rows"]) > max_rows:
        data = {**data, "rows": data["rows"][:max_rows], "rows_truncated": True}
    out = {"evidence_id": e["evidence_id"], "tool": e["tool"], "args": e["args"], "status": e["status"]}
    if e["status"] == "ok":
        text = json.dumps(mask(data), default=str)
        out["data"] = json.loads(text) if len(text) <= max_chars else text[:max_chars] + "...(truncated)"
    else:
        out["error"] = e.get("error")
    return out


def _alert_view(alert: dict[str, Any]) -> dict[str, Any]:
    return mask(alert)


def _add_usage(state_usage: dict[str, int] | None, u: Usage) -> dict[str, int]:
    base = dict(state_usage or {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0})
    base["input_tokens"] += u.input_tokens
    base["output_tokens"] += u.output_tokens
    base["llm_calls"] += u.calls
    return base


def _trace(state: dict[str, Any], node: str, detail: Any) -> list[dict[str, Any]]:
    return list(state.get("trace", [])) + [{"node": node, "t": round(time.time(), 3), "detail": detail}]


NUM_RE = re.compile(r"\d[\d,]*\.\d+")


def check_citations(summary: CaseSummary, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Every claim must cite at least one existing, successful evidence item.

    Also flags decimal numbers (amounts, scores) that do not appear in any cited evidence item. That second
    check is a heuristic warning, reported separately from unsupported claims.
    """
    ok = {e["evidence_id"]: e for e in evidence if e["status"] == "ok"}
    valid = set(ok) | {"ALERT"}
    unsupported, numeric_warnings = [], []
    for i, c in enumerate(summary.claims):
        bad = [x for x in c.evidence_ids if x not in valid]
        if not c.evidence_ids or bad:
            unsupported.append({"claim_index": i, "statement": c.statement, "invalid_ids": bad})
            continue
        cited_text = " ".join(json.dumps(ok[x]["data"], default=str) for x in c.evidence_ids if x in ok)
        cited_text += " " + json.dumps(summary.alert_id)
        for num in NUM_RE.findall(c.statement):
            plain = num.replace(",", "").rstrip("0").rstrip(".")
            if plain and plain not in cited_text.replace(",", ""):
                numeric_warnings.append({"claim_index": i, "number": num})
    return {"passed": not unsupported, "total_claims": len(summary.claims), "unsupported": unsupported,
            "numeric_warnings": numeric_warnings}


# --------------------------------------------------------------------------- nodes
class Nodes:
    def __init__(self, settings: Settings, llm: Any, toolbox: Toolbox):
        self.s, self.llm, self.tools = settings, llm, toolbox

    def load_alert(self, state: dict[str, Any]) -> dict[str, Any]:
        alert = get_alert(self.s.db_path, state["alert_id"])
        return {"alert": alert, "evidence": [], "call_keys": [], "pending_calls": [], "steps": 0, "done": False,
                "usage": {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}, "degraded": [],
                "trace": _trace(state, "load_alert", _alert_view(alert))}

    def planner(self, state: dict[str, Any]) -> dict[str, Any]:
        if state["steps"] >= self.s.max_steps:
            return {"done": True, "pending_calls": [],
                    "trace": _trace(state, "planner", "max steps reached; moving to writer")}
        system = PLANNER_SYSTEM.format(tools="\n".join(f"- {k}: {v}" for k, v in TOOL_DESCRIPTIONS.items()),
                                       max_calls=self.s.max_calls_per_step)
        user = json.dumps({"alert": _alert_view(state["alert"]),
                           "evidence_collected": [_compact(e, max_rows=15, max_chars=2500) for e in state["evidence"]],
                           "step": state["steps"] + 1, "max_steps": self.s.max_steps}, default=str)
        usage = dict(state["usage"])
        degraded = list(state.get("degraded", []))
        try:
            decision, u = self.llm.structured(system, user, PlannerDecision,
                                              context={"alert": state["alert"], "evidence": state["evidence"]})
            usage = _add_usage(usage, u)
        except (StructuredOutputError, ValidationError, Exception) as e:  # degrade: fall back to a default plan
            if isinstance(e, StructuredOutputError):
                usage = _add_usage(usage, e.usage)
            from .llm import MockLLM

            decision, _ = MockLLM().structured("", "", PlannerDecision,
                                               context={"alert": state["alert"], "evidence": state["evidence"]})
            degraded.append(f"planner fallback: {type(e).__name__}")
        seen = set(state["call_keys"])
        new_calls = []
        for c in decision.calls:
            if c.key() not in seen:
                seen.add(c.key())
                new_calls.append(c.model_dump())
        new_calls = new_calls[: self.s.max_calls_per_step]
        done = decision.done or not new_calls
        return {"pending_calls": [] if done else new_calls, "done": done, "usage": usage, "degraded": degraded,
                "call_keys": sorted(seen),
                "trace": _trace(state, "planner", {"rationale": decision.rationale, "done": done,
                                                   "calls": new_calls})}

    def executor(self, state: dict[str, Any]) -> dict[str, Any]:
        evidence = list(state["evidence"])
        results = []
        for raw in state["pending_calls"]:
            call = ToolCall(**raw)
            ev = self.tools.execute(call, state["alert"], f"E{len(evidence) + 1}")
            evidence.append(ev.model_dump())
            results.append({"evidence_id": ev.evidence_id, "tool": ev.tool, "status": ev.status,
                            "error": ev.error, "latency_ms": ev.latency_ms})
        return {"evidence": evidence, "pending_calls": [], "steps": state["steps"] + 1,
                "trace": _trace(state, "executor", results)}

    def writer(self, state: dict[str, Any]) -> dict[str, Any]:
        evidence = state["evidence"]
        failed = [e["tool"] for e in evidence if e["status"] == "error"]
        base_user = json.dumps({"alert": _alert_view(state["alert"]), "failed_tools": failed,
                                "evidence": [_compact(e) for e in evidence]}, default=str)
        user, usage, degraded = base_user, dict(state["usage"]), list(state.get("degraded", []))
        summary, check, attempts = None, None, 0
        for attempt in range(self.s.max_output_repairs + 1):
            attempts = attempt + 1
            try:
                summary, u = self.llm.structured(
                    WRITER_SYSTEM, user, CaseSummary,
                    context={"alert": state["alert"], "evidence": [Evidence(**e) for e in evidence]})
                usage = _add_usage(usage, u)
            except (StructuredOutputError, ValidationError) as e:
                if isinstance(e, StructuredOutputError):
                    usage = _add_usage(usage, e.usage)
                user = base_user + f"\n\nYour previous output was invalid: {str(e)[:800]}. Return a valid CaseSummary."
                continue
            summary.alert_id = state["alert"]["alert_id"]
            check = check_citations(summary, evidence)
            if check["passed"]:
                break
            user = base_user + ("\n\nThese claims cite evidence IDs that do not exist or failed: "
                                + json.dumps(check["unsupported"]) + ". Fix the citations or remove the claims.")
        if summary is None:  # LLM never produced valid output: degrade to the rules engine
            r = rules.evaluate(state["alert"], [Evidence(**e) for e in evidence])
            summary = CaseSummary(alert_id=state["alert"]["alert_id"], disposition=r["disposition"],
                                  typology=r["typology"], confidence="low",
                                  summary="LLM output unavailable; decision produced by the rules engine.",
                                  claims=r["claims"], policy_rules_applied=r["rules"],
                                  recommended_actions=r["actions"])
            check = check_citations(summary, evidence)
            degraded.append("writer fallback: rules engine")
        return {"summary": summary.model_dump(), "citation_check": {**check, "attempts": attempts},
                "usage": usage, "degraded": degraded,
                "trace": _trace(state, "writer", {"disposition": summary.disposition, "typology": summary.typology,
                                                  "citations_passed": check["passed"], "attempts": attempts})}

    @staticmethod
    def needs_review(state: dict[str, Any]) -> bool:
        return state["summary"]["disposition"] == "file_sar"

    @staticmethod
    def apply_review(state: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        summary = dict(state["summary"])
        approved = bool(decision.get("approved"))
        if not approved:
            summary["disposition"] = "escalate"
            summary["recommended_actions"] = ["SAR recommendation rejected by reviewer: manual investigation"] + \
                list(summary.get("recommended_actions", []))
        review = {"approved": approved, "reviewer": decision.get("reviewer", "unknown"),
                  "note": decision.get("note", "")}
        return {"summary": summary, "review": review, "trace": _trace(state, "human_review", review)}


def run_without_langgraph(nodes: Nodes, alert_id: str, auto_approve: bool = True) -> dict[str, Any]:
    """Same control flow as graph.py, in plain Python. Used by tests and as a fallback runner."""
    state: dict[str, Any] = {"alert_id": alert_id}
    state.update(nodes.load_alert(state))
    while True:
        state.update(nodes.planner(state))
        if state["done"]:
            break
        state.update(nodes.executor(state))
    state.update(nodes.writer(state))
    if nodes.needs_review(state):
        state.update(nodes.apply_review(state, {"approved": auto_approve, "reviewer": "auto"}))
    return state
