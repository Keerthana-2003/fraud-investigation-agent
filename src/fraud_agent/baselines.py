"""Baselines the LangGraph planner is compared against.

single_prompt: one LLM call over a fixed context (recent transactions, logins, payees, devices).
               No tools, no policy retrieval, no network check.
react:         LangGraph's prebuilt ReAct agent with the same tools and guardrails but no planner,
               no evidence state, and no citation repair loop.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .config import Settings
from .db import get_alert
from .llm import StructuredOutputError
from .nodes import WRITER_SYSTEM, _add_usage, _alert_view, _compact, check_citations
from .schemas import CaseSummary, Evidence, ToolCall
from .tools import TOOL_DESCRIPTIONS, Toolbox

SINGLE_PROMPT_CONTEXT = [("transactions", 24 * 7), ("logins", 48), ("payees", None), ("devices", None)]

REACT_SYSTEM = (
    "You are a fraud investigator. Use the tools to gather evidence about the alert, then produce a case "
    "summary with a disposition (close, escalate, file_sar). Every claim must cite evidence IDs returned by the "
    "tools (e.g. E3) or 'ALERT'. Apply the retrieved policy rules exactly as written."
)


def run_single_prompt(settings: Settings, llm: Any, toolbox: Toolbox, alert_id: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    alert = get_alert(settings.db_path, alert_id)
    evidence = [toolbox.execute(ToolCall(tool="sql_lookup", table=t, since_hours=h, reason="fixed context"),
                                alert, f"E{i + 1}").model_dump()
                for i, (t, h) in enumerate(SINGLE_PROMPT_CONTEXT)]
    user = json.dumps({"alert": _alert_view(alert), "evidence": [_compact(e) for e in evidence]}, default=str)
    usage: dict[str, int] = {}
    try:
        summary, u = llm.structured(WRITER_SYSTEM, user, CaseSummary,
                                    context={"alert": alert, "evidence": [Evidence(**e) for e in evidence]})
        usage = _add_usage(usage, u)
        summary.alert_id = alert_id
        result = {"summary": summary.model_dump(), "citation_check": check_citations(summary, evidence)}
    except (StructuredOutputError, ValueError) as e:
        result = {"summary": None, "citation_check": None, "error": str(e)[:300]}
    return {**result, "evidence": evidence, "usage": usage, "steps": 1, "degraded": [],
            "latency_s": time.perf_counter() - t0}


def run_react(settings: Settings, llm: Any, toolbox: Toolbox, alert_id: str) -> dict[str, Any]:
    """Requires a real LangChain chat model (llm.chat)."""
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import create_react_agent

    t0 = time.perf_counter()
    alert = get_alert(settings.db_path, alert_id)
    evidence: list[dict[str, Any]] = []

    def _run(call: ToolCall) -> str:
        ev = toolbox.execute(call, alert, f"E{len(evidence) + 1}")
        evidence.append(ev.model_dump())
        return json.dumps(_compact(ev.model_dump()), default=str)

    def sql_lookup(table: str, since_hours: int | None = None) -> str:
        return _run(ToolCall(tool="sql_lookup", table=table, since_hours=since_hours, reason="react"))

    def risk_score() -> str:
        return _run(ToolCall(tool="risk_score", reason="react"))

    def policy_search(query: str) -> str:
        return _run(ToolCall(tool="policy_search", query=query, reason="react"))

    def network_check() -> str:
        return _run(ToolCall(tool="network_check", reason="react"))

    tools = [StructuredTool.from_function(func=f, name=f.__name__, description=TOOL_DESCRIPTIONS[f.__name__])
             for f in (sql_lookup, risk_score, policy_search, network_check)]
    agent = create_react_agent(llm.chat, tools, prompt=REACT_SYSTEM, response_format=CaseSummary)
    try:
        out = agent.invoke({"messages": [("user", json.dumps({"alert": _alert_view(alert)}, default=str))]},
                           {"recursion_limit": 4 * settings.max_steps + 10})
        messages = out["messages"]
        summary = out.get("structured_response")
        usage: dict[str, int] = {}
        steps = 0
        for m in messages:
            meta = getattr(m, "usage_metadata", None)
            if meta:
                usage["input_tokens"] = usage.get("input_tokens", 0) + meta.get("input_tokens", 0)
                usage["output_tokens"] = usage.get("output_tokens", 0) + meta.get("output_tokens", 0)
                usage["llm_calls"] = usage.get("llm_calls", 0) + 1
            if getattr(m, "tool_calls", None):
                steps += 1
        if summary is None:
            raise ValueError("ReAct agent returned no structured response")
        if isinstance(summary, dict):
            summary = CaseSummary(**summary)
        summary.alert_id = alert_id
        result = {"summary": summary.model_dump(), "citation_check": check_citations(summary, evidence)}
    except Exception as e:  # recursion limit, invalid output, etc. count as task failures
        result, usage, steps = {"summary": None, "citation_check": None, "error": f"{type(e).__name__}: {e}"[:300]}, \
            {}, 0
    return {**result, "evidence": evidence, "usage": usage, "steps": steps, "degraded": [],
            "latency_s": time.perf_counter() - t0}
