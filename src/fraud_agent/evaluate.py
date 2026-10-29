"""Evaluation harness: compare the LangGraph planner with single-prompt and ReAct baselines.

    python -m fraud_agent.evaluate --systems langgraph,single_prompt,react
    python -m fraud_agent.evaluate --systems langgraph --limit 20          # quick run

Writes results/results.md (headline table), results/summary.json and results/cases.csv.
Metrics:
    disposition_accuracy   share of alerts with the correct close / escalate / file_sar
    typology_accuracy      share with the correct fraud typology
    task_completion        valid case file produced, citations passed, no fallback used
    tool_recall            share of the expected tools (per alert type) that were called
    avg_tool_calls         tool calls per case
    avg_steps              planner/executor iterations (ReAct: tool-calling turns)
    hallucination_rate     claims citing evidence that does not exist or failed / all claims
    numeric_warning_rate   claims containing a decimal number not found in the cited evidence / all claims
    latency (mean, p95), tokens and cost per case
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import cost_usd, get_settings
from .db import get_ground_truth
from .llm import build_llm
from .tools import Toolbox


def _langgraph_runner(settings, llm, toolbox):
    try:
        from .graph import build_graph, run_case
        from .nodes import Nodes

        graph = build_graph(Nodes(settings, llm, toolbox))

        def run(alert_id: str) -> dict[str, Any]:
            t0 = time.perf_counter()
            out = run_case(graph, settings, alert_id, auto_approve=True)
            return {**out, "latency_s": time.perf_counter() - t0}
    except ImportError:  # LangGraph not installed: identical control flow in plain Python
        from .nodes import Nodes, run_without_langgraph

        nodes = Nodes(settings, llm, toolbox)

        def run(alert_id: str) -> dict[str, Any]:
            t0 = time.perf_counter()
            out = run_without_langgraph(nodes, alert_id)
            return {**out, "latency_s": time.perf_counter() - t0}
    return run


def score_case(system: str, alert_id: str, truth: dict[str, Any], out: dict[str, Any], model: str) -> dict[str, Any]:
    summary, check = out.get("summary"), out.get("citation_check")
    called = {e["tool"] for e in out.get("evidence", [])}
    expected = set(truth["expected_tools"])
    usage = out.get("usage") or {}
    tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return {
        "system": system, "alert_id": alert_id, "alert_type": truth["alert_type"],
        "true_typology": truth["typology"], "true_disposition": truth["disposition"],
        "pred_typology": summary["typology"] if summary else None,
        "pred_disposition": summary["disposition"] if summary else None,
        "completed": bool(summary and check and check["passed"] and not out.get("degraded")),
        "tool_recall": len(called & expected) / len(expected),
        "tool_calls": len(out.get("evidence", [])),
        "tools_called": ",".join(sorted(called)),
        "steps": out.get("steps", 0),
        "claims": check["total_claims"] if check else 0,
        "unsupported_claims": len(check["unsupported"]) if check else 0,
        "numeric_warnings": len(check["numeric_warnings"]) if check else 0,
        "latency_s": round(out.get("latency_s", 0.0), 3),
        "input_tokens": tin, "output_tokens": tout, "cost_usd": cost_usd(model, tin, tout),
        "degraded": ";".join(out.get("degraded", [])), "error": out.get("error", ""),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    lat = np.array([r["latency_s"] for r in rows])
    claims = sum(r["claims"] for r in rows) or 1
    return {
        "cases": n,
        "disposition_accuracy": sum(r["pred_disposition"] == r["true_disposition"] for r in rows) / n,
        "typology_accuracy": sum(r["pred_typology"] == r["true_typology"] for r in rows) / n,
        "task_completion": sum(r["completed"] for r in rows) / n,
        "tool_recall": float(np.mean([r["tool_recall"] for r in rows])),
        "avg_tool_calls": float(np.mean([r["tool_calls"] for r in rows])),
        "avg_steps": float(np.mean([r["steps"] for r in rows])),
        "hallucination_rate": sum(r["unsupported_claims"] for r in rows) / claims,
        "numeric_warning_rate": sum(r["numeric_warnings"] for r in rows) / claims,
        "latency_mean_s": float(lat.mean()), "latency_p95_s": float(np.percentile(lat, 95)),
        "avg_tokens": float(np.mean([r["input_tokens"] + r["output_tokens"] for r in rows])),
        "cost_per_case_usd": float(np.mean([r["cost_usd"] for r in rows])),
        "disposition_confusion": {f"{t}->{p}": c for (t, p), c in
                                  Counter((r["true_disposition"], r["pred_disposition"]) for r in rows).items()},
    }


def to_markdown(results: dict[str, dict[str, Any]], meta: dict[str, Any]) -> str:
    cols = [("disposition_accuracy", "Disposition acc.", "{:.1%}"), ("typology_accuracy", "Typology acc.", "{:.1%}"),
            ("task_completion", "Task completion", "{:.1%}"), ("tool_recall", "Tool recall", "{:.1%}"),
            ("avg_tool_calls", "Avg tool calls", "{:.1f}"), ("avg_steps", "Avg steps", "{:.1f}"),
            ("hallucination_rate", "Hallucination rate", "{:.1%}"),
            ("latency_mean_s", "Latency mean (s)", "{:.1f}"), ("latency_p95_s", "Latency p95 (s)", "{:.1f}"),
            ("cost_per_case_usd", "Cost / case", "${:.4f}")]
    lines = [f"# Evaluation results\n",
             f"Model: `{meta['model']}` ({meta['provider']}) | alerts: {meta['cases']} | date: {meta['date']}\n",
             "| System | " + " | ".join(c[1] for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for system, agg in results.items():
        lines.append(f"| {system} | " + " | ".join(f.format(agg[k]) for k, _, f in cols) + " |")
    if meta["provider"] == "mock":
        lines.append("\n> **Offline mock run.** These numbers come from the rules-based MockLLM, not a language model. "
                     "They only verify that the pipeline works. Do not report them.")
    return "\n".join(lines) + "\n"


def main(systems: list[str], limit: int | None = None, seed: int = 0) -> dict[str, Any]:
    settings = get_settings()
    truth = get_ground_truth(settings.db_path)
    ids = sorted(truth)
    if limit:
        rng = np.random.default_rng(seed)
        ids = sorted(rng.choice(ids, size=min(limit, len(ids)), replace=False).tolist())
    llm, toolbox = build_llm(settings), Toolbox(settings)
    from .baselines import run_react, run_single_prompt

    runners = {
        "langgraph": _langgraph_runner(settings, llm, toolbox),
        "single_prompt": lambda a: run_single_prompt(settings, llm, toolbox, a),
        "react": lambda a: run_react(settings, llm, toolbox, a),
    }
    if settings.llm_provider == "mock" and "react" in systems:
        print("Skipping react: it needs a real chat model (set LLM_PROVIDER=openai or anthropic).")
        systems = [s for s in systems if s != "react"]

    all_rows, results = [], {}
    for system in systems:
        rows = []
        for i, alert_id in enumerate(ids, 1):
            out = runners[system](alert_id)
            rows.append(score_case(system, alert_id, truth[alert_id], out, settings.llm_model
                                   if settings.llm_provider != "mock" else "mock"))
            if i % 10 == 0 or i == len(ids):
                acc = sum(r["pred_disposition"] == r["true_disposition"] for r in rows) / len(rows)
                print(f"[{system}] {i}/{len(ids)}  running disposition accuracy {acc:.1%}")
        results[system] = aggregate(rows)
        all_rows += rows

    out_dir = settings.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": settings.llm_model if settings.llm_provider != "mock" else "mock",
            "provider": settings.llm_provider, "cases": len(ids), "date": time.strftime("%Y-%m-%d")}
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    with open(out_dir / "cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    md = to_markdown(results, meta)
    (out_dir / "results.md").write_text(md)
    print(md)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="langgraph,single_prompt,react")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a random subset of alerts")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main([s.strip() for s in a.systems.split(",") if s.strip()], a.limit, a.seed)
