"""LLM adapters returning validated Pydantic objects plus token usage.

* LangChainLLM: OpenAI or Anthropic chat models through LangChain's `with_structured_output`.
* MockLLM: offline, deterministic stand-in for tests and demos (planner heuristics + rules engine writer).
  Mock results are NOT model results and must never be reported as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from .config import Settings

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


class StructuredOutputError(Exception):
    def __init__(self, message: str, usage: Usage | None = None):
        super().__init__(message)
        self.usage = usage or Usage(calls=1)


class LangChainLLM:
    def __init__(self, settings: Settings):
        self.model_name = settings.llm_model
        self.provider = settings.llm_provider
        if settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            self.chat = ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature, timeout=60,
                                   max_retries=2)
        elif settings.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            self.chat = ChatAnthropic(model=settings.llm_model, temperature=settings.llm_temperature,
                                      timeout=60, max_retries=2)
        else:
            raise ValueError(f"unsupported provider {settings.llm_provider}")

    def structured(self, system: str, user: str, schema: Type[T], context: dict[str, Any] | None = None
                   ) -> tuple[T, Usage]:
        from langchain_core.messages import HumanMessage, SystemMessage

        kwargs = {"method": "function_calling"} if self.provider == "openai" else {}
        runnable = self.chat.with_structured_output(schema, include_raw=True, **kwargs)
        out = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        meta = getattr(out["raw"], "usage_metadata", None) or {}
        usage = Usage(meta.get("input_tokens", 0), meta.get("output_tokens", 0), 1)
        if out.get("parsing_error") is not None or out.get("parsed") is None:
            raise StructuredOutputError(str(out.get("parsing_error") or "no structured output returned"), usage)
        return out["parsed"], usage


@dataclass
class MockLLM:
    """Deterministic offline stand-in. Uses `context` (alert + evidence) instead of reading the prompt."""
    model_name: str = "mock"
    calls: list[str] = field(default_factory=list)

    def structured(self, system: str, user: str, schema: Type[T], context: dict[str, Any] | None = None
                   ) -> tuple[T, Usage]:
        from . import rules
        from .schemas import CaseSummary, PlannerDecision, ToolCall

        context = context or {}
        self.calls.append(schema.__name__)
        usage = Usage(len(system + user) // 4, 200, 1)
        if schema is PlannerDecision:
            alert, evidence = context["alert"], context["evidence"]
            if evidence:
                called = {e.tool if hasattr(e, "tool") else e["tool"] for e in evidence}
                if alert["alert_type"] == "rapid_movement" and "network_check" not in called:
                    return PlannerDecision(rationale="Check for shared devices and pass-through.", done=False,
                                           calls=[ToolCall(tool="network_check", reason="mule indicators")]), usage
                return PlannerDecision(rationale="Enough evidence collected.", done=True, calls=[]), usage
            policy_q = {"large_transfer": "account takeover new payee new device password reset",
                        "card_burst": "card testing small transactions merchants",
                        "rapid_movement": "money mule pass-through shared device",
                        "cash_threshold": "structuring cash deposits under $10,000",
                        "device_anomaly": "travel notice new device login"}[alert["alert_type"]]
            calls = [ToolCall(tool="sql_lookup", table="transactions", since_hours=24 * 7, reason="recent activity"),
                     ToolCall(tool="sql_lookup", table="logins", since_hours=48, reason="credential changes"),
                     ToolCall(tool="sql_lookup", table="payees", reason="payee age"),
                     ToolCall(tool="sql_lookup", table="devices", reason="device age"),
                     ToolCall(tool="risk_score", reason="model score"),
                     ToolCall(tool="policy_search", query=policy_q, reason="applicable policy")]
            return PlannerDecision(rationale="Gather account records, model score, and policy.", done=False,
                                   calls=calls), usage
        if schema is CaseSummary:
            alert = context["alert"]
            r = rules.evaluate(alert, context["evidence"])
            return CaseSummary(alert_id=alert["alert_id"], disposition=r["disposition"], typology=r["typology"],
                               confidence=r["confidence"],
                               summary=f"Rules engine decision: {r['typology']} -> {r['disposition']}.",
                               claims=r["claims"], policy_rules_applied=r["rules"],
                               recommended_actions=r["actions"]), usage
        raise StructuredOutputError(f"MockLLM cannot produce {schema.__name__}", usage)


def build_llm(settings: Settings):
    if settings.llm_provider == "mock":
        return MockLLM()
    return LangChainLLM(settings)
