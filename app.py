"""Streamlit demo: pick an alert, watch the agent work step by step, approve or reject SAR recommendations.

    streamlit run app.py
"""
from __future__ import annotations

import json
import sqlite3
import uuid

import pandas as pd
import streamlit as st
from langgraph.types import Command

from fraud_agent.config import get_settings
from fraud_agent.graph import make_agent

st.set_page_config(page_title="Fraud Case Investigation Agent", layout="wide")


@st.cache_resource
def load_agent():
    return make_agent()


@st.cache_data
def load_alerts(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    df = pd.read_sql_query("SELECT alert_id, alert_type, account_id, created_ts, rule_triggered, true_disposition "
                           "FROM alerts ORDER BY alert_id", con)
    con.close()
    return df


graph, settings = load_agent()
alerts = load_alerts(str(settings.db_path))
ss = st.session_state

st.title("Fraud Case Investigation Agent")
st.caption(f"LangGraph planner -> tools -> writer -> human review  |  model: {settings.llm_provider}/{settings.llm_model}")

with st.sidebar:
    st.header("Alert queue")
    alert_type = st.selectbox("Alert type", ["all"] + sorted(alerts["alert_type"].unique()))
    view = alerts if alert_type == "all" else alerts[alerts["alert_type"] == alert_type]
    alert_id = st.selectbox("Alert", view["alert_id"])
    row = alerts.set_index("alert_id").loc[alert_id]
    st.write(f"**{row['alert_type']}** on {row['account_id']}")
    st.write(f"Rule: {row['rule_triggered']}")
    with st.expander("Ground truth (synthetic data, hidden from the agent)"):
        st.write(row["true_disposition"])
    if st.button("Investigate", type="primary"):
        ss.thread_id, ss.alert_id, ss.events = str(uuid.uuid4()), alert_id, []
        ss.run_input = {"alert_id": alert_id, "auto_approve": False}


def config() -> dict:
    return {"configurable": {"thread_id": ss.thread_id}, "recursion_limit": 2 * settings.max_steps + 10}


def stream(payload) -> None:
    for update in graph.stream(payload, config(), stream_mode="updates"):
        for node, data in update.items():
            if node == "__interrupt__":
                continue
            detail = (data or {}).get("trace", [{}])[-1].get("detail") if data else None
            ss.events.append({"node": node, "detail": detail})
            with st.status(f"{node}", state="complete"):
                st.json(detail if isinstance(detail, (dict, list)) else {"info": detail}, expanded=False)


if ss.get("run_input"):
    payload, ss.run_input = ss.run_input, None
    stream(payload)
elif ss.get("resume"):
    payload, ss.resume = ss.resume, None
    stream(Command(resume=payload))
else:
    for ev in ss.get("events", []):
        with st.status(ev["node"], state="complete"):
            st.json(ev["detail"] if isinstance(ev["detail"], (dict, list)) else {"info": ev["detail"]},
                    expanded=False)

if ss.get("thread_id"):
    snapshot = graph.get_state(config())
    state = snapshot.values
    if summary := state.get("summary"):
        st.subheader("Case file")
        c1, c2, c3 = st.columns(3)
        c1.metric("Disposition", summary["disposition"])
        c2.metric("Typology", summary["typology"])
        c3.metric("Confidence", summary["confidence"])
        st.write(summary["summary"])
        st.markdown("**Claims and evidence**")
        for c in summary["claims"]:
            st.markdown(f"- {c['statement']}  `{', '.join(c['evidence_ids'])}`")
        st.markdown("**Recommended actions:** " + "; ".join(summary.get("recommended_actions", [])))
        check = state.get("citation_check", {})
        st.caption(f"Citation check: {'passed' if check.get('passed') else 'FAILED'} | "
                   f"policy rules: {', '.join(summary.get('policy_rules_applied', [])) or 'none'} | "
                   f"tokens: {state.get('usage')} | degraded: {state.get('degraded') or 'no'}")
        with st.expander("Evidence"):
            for e in state.get("evidence", []):
                st.markdown(f"**{e['evidence_id']}** `{e['tool']}` {json.dumps(e['args'])} - {e['status']}")
    if snapshot.next:  # paused at human_review
        st.warning("SAR recommendation requires human approval (FIN-POL-050).")
        note = st.text_input("Reviewer note")
        a, b = st.columns(2)
        if a.button("Approve SAR"):
            ss.resume = {"approved": True, "reviewer": "analyst", "note": note}
            st.rerun()
        if b.button("Reject"):
            ss.resume = {"approved": False, "reviewer": "analyst", "note": note}
            st.rerun()
    elif state.get("review"):
        st.info(f"Review: {state['review']}")
