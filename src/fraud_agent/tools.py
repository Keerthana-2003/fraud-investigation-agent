"""Agent tools with guardrails: whitelisting, timeouts, retries, and graceful degradation.

Every call returns an Evidence record (never raises), so a failing tool degrades the investigation
instead of crashing it. The writer is told which tools failed and must say so in the case file.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable

from .config import Settings
from .db import QueryRejected, network_neighbors, safe_select
from .retriever import build_retriever
from .schemas import Evidence, ToolCall

NON_RETRYABLE = (QueryRejected, KeyError, ValueError)

TOOL_DESCRIPTIONS = {
    "sql_lookup": "Read-only records for the alerted account. Args: table (transactions, logins, payees, devices, "
                  "accounts, customers, prior_alerts) and optional since_hours look-back window before the alert.",
    "risk_score": "Transaction risk model score (0-1) for the alerted transaction plus top SHAP drivers. No args.",
    "policy_search": "Search internal fraud and AML policies. Args: query.",
    "network_check": "Accounts sharing a device, IP, or payee with the alerted account, plus 24h inflow, outflow, "
                     "and pass-through ratio. No args.",
    "typology_classifier": "Fine-tuned model that classifies a free-text case narrative into a fraud typology. "
                           "Args: text. May be unavailable.",
}


class Toolbox:
    def __init__(self, settings: Settings, scorer: Any | None = None, retriever: Any | None = None):
        self.settings = settings
        self._scorer = scorer
        self._retriever = retriever
        self._pool = ThreadPoolExecutor(max_workers=4)

    # lazy so the app starts even if the model file or embeddings are missing
    @property
    def scorer(self):
        if self._scorer is None:
            from .risk_model import RiskScorer

            self._scorer = RiskScorer(self.settings.db_path, self.settings.model_path)
        return self._scorer

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = build_retriever(self.settings.retriever, self.settings.policies_dir,
                                              self.settings.chroma_dir, self.settings.embedding_model)
        return self._retriever

    # ------------------------------------------------------------------ tool implementations
    def sql_lookup(self, alert: dict, table: str | None, since_hours: int | None) -> Any:
        if not table:
            raise QueryRejected("sql_lookup requires a table")
        rows = safe_select(self.settings.db_path, table, alert["account_id"], alert["created_ts"], since_hours)
        return {"table": table, "since_hours": since_hours, "row_count": len(rows), "rows": rows}

    def risk_score(self, alert: dict) -> Any:
        return self.scorer.score(alert["txn_id"])

    def policy_search(self, query: str | None) -> Any:
        if not query:
            raise ValueError("policy_search requires a query")
        return {"query": query, "results": self.retriever.search(query, k=3)}

    def network_check(self, alert: dict) -> Any:
        return network_neighbors(self.settings.db_path, alert["account_id"], alert["created_ts"])

    def typology_classifier(self, text: str | None) -> Any:
        if not self.settings.typology_api_url:
            raise RuntimeError("typology_classifier unavailable: TYPOLOGY_API_URL is not set")
        if not text:
            raise ValueError("typology_classifier requires text")
        import requests

        r = requests.post(self.settings.typology_api_url.rstrip("/") + "/extract", json={"text": text[:4000]},
                          timeout=self.settings.tool_timeout_s)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ execution wrapper
    def _dispatch(self, call: ToolCall, alert: dict) -> Callable[[], Any]:
        return {
            "sql_lookup": lambda: self.sql_lookup(alert, call.table, call.since_hours),
            "risk_score": lambda: self.risk_score(alert),
            "policy_search": lambda: self.policy_search(call.query),
            "network_check": lambda: self.network_check(alert),
            "typology_classifier": lambda: self.typology_classifier(call.text),
        }[call.tool]

    def execute(self, call: ToolCall, alert: dict, evidence_id: str) -> Evidence:
        args = call.model_dump(exclude_none=True, exclude={"reason"})
        fn = self._dispatch(call, alert)
        start, attempts, error = time.perf_counter(), 0, None
        for attempt in range(self.settings.tool_retries + 1):
            attempts = attempt + 1
            future = self._pool.submit(fn)
            try:
                data = future.result(timeout=self.settings.tool_timeout_s)
                return Evidence(evidence_id=evidence_id, tool=call.tool, args=args, status="ok", data=data,
                                latency_ms=round((time.perf_counter() - start) * 1000, 1), attempts=attempts)
            except FutureTimeout:
                future.cancel()
                error = f"timed out after {self.settings.tool_timeout_s}s"
            except NON_RETRYABLE as e:
                error = f"{type(e).__name__}: {e}"
                break
            except Exception as e:  # transient: network, model loading, etc.
                error = f"{type(e).__name__}: {e}"
            time.sleep(min(0.25 * 2 ** attempt, 2.0))
        return Evidence(evidence_id=evidence_id, tool=call.tool, args=args, status="error", error=error,
                        latency_ms=round((time.perf_counter() - start) * 1000, 1), attempts=attempts)
