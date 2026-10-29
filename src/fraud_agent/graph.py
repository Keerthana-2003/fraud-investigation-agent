"""LangGraph wiring for the investigation agent.

    load_alert -> planner -> (executor -> planner)* -> writer -> human_review -> END

The planner/executor loop is bounded by AGENT_MAX_STEPS. human_review pauses the graph with `interrupt()`
before any file_sar recommendation is finalized; resume it with `Command(resume={"approved": True, ...})`.
"""
from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .config import Settings, get_settings
from .llm import build_llm
from .nodes import Nodes
from .tools import Toolbox


class AgentState(TypedDict, total=False):
    alert_id: str
    auto_approve: bool
    alert: dict
    evidence: list
    call_keys: list
    pending_calls: list
    steps: int
    done: bool
    summary: dict
    citation_check: dict
    review: dict
    trace: list
    usage: dict
    degraded: list


def build_graph(nodes: Nodes, checkpointer: Any | None = None):
    def human_review(state: AgentState) -> dict:
        if not Nodes.needs_review(state):
            return {"review": {"approved": None, "note": "no SAR recommendation; review not required"}}
        if state.get("auto_approve"):
            decision = {"approved": True, "reviewer": "auto (evaluation mode)"}
        else:
            decision = interrupt({"question": "Approve this SAR recommendation?", "summary": state["summary"]})
        return Nodes.apply_review(state, decision)

    g = StateGraph(AgentState)
    g.add_node("load_alert", nodes.load_alert)
    g.add_node("planner", nodes.planner)
    g.add_node("executor", nodes.executor)
    g.add_node("writer", nodes.writer)
    g.add_node("human_review", human_review)
    g.add_edge(START, "load_alert")
    g.add_edge("load_alert", "planner")
    g.add_conditional_edges("planner", lambda s: "writer" if s["done"] else "executor",
                            {"writer": "writer", "executor": "executor"})
    g.add_edge("executor", "planner")
    g.add_edge("writer", "human_review")
    g.add_edge("human_review", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def make_agent(settings: Settings | None = None):
    settings = settings or get_settings()
    nodes = Nodes(settings, build_llm(settings), Toolbox(settings))
    return build_graph(nodes), settings


def run_case(graph, settings: Settings, alert_id: str, auto_approve: bool = True,
             thread_id: str | None = None) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())},
              "recursion_limit": 2 * settings.max_steps + 10}
    return graph.invoke({"alert_id": alert_id, "auto_approve": auto_approve}, config)


def resume_case(graph, thread_id: str, approved: bool, reviewer: str = "analyst", note: str = "") -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(Command(resume={"approved": approved, "reviewer": reviewer, "note": note}), config)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Investigate one alert")
    ap.add_argument("alert_id")
    ap.add_argument("--interactive", action="store_true", help="pause for human approval of SAR recommendations")
    a = ap.parse_args()
    graph, s = make_agent()
    tid = str(uuid.uuid4())
    out = run_case(graph, s, a.alert_id, auto_approve=not a.interactive, thread_id=tid)
    if "__interrupt__" in out:
        print(json.dumps(out["summary"], indent=2))
        ok = input("Approve SAR recommendation? [y/N] ").strip().lower() == "y"
        out = resume_case(graph, tid, ok, note=input("Note: "))
    print(json.dumps({k: out.get(k) for k in ["summary", "citation_check", "review", "usage", "degraded"]},
                     indent=2, default=str))
