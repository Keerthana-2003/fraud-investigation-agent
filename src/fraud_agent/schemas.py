"""Structured output schemas. Every LLM output is validated against these with Pydantic."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ToolName = Literal["sql_lookup", "risk_score", "policy_search", "network_check", "typology_classifier"]
Table = Literal["transactions", "logins", "payees", "devices", "accounts", "customers", "prior_alerts"]
Disposition = Literal["close", "escalate", "file_sar"]
Typology = Literal["account_takeover", "card_testing", "money_mule", "structuring", "benign", "other"]


class ToolCall(BaseModel):
    """A single tool request. Flat optional fields keep the schema portable across LLM providers."""
    tool: ToolName
    reason: str = Field(description="Why this evidence is needed for the alert")
    table: Optional[Table] = Field(None, description="sql_lookup only")
    since_hours: Optional[int] = Field(None, description="sql_lookup only: look-back window before the alert")
    query: Optional[str] = Field(None, description="policy_search only: what policy to look for")
    text: Optional[str] = Field(None, description="typology_classifier only: narrative text to classify")

    def key(self) -> str:
        return f"{self.tool}|{self.table}|{self.since_hours}|{(self.query or '').lower().strip()}|{self.text}"


class PlannerDecision(BaseModel):
    rationale: str = Field(description="One or two sentences on what evidence is still missing")
    done: bool = Field(description="True when enough evidence has been gathered to decide")
    calls: list[ToolCall] = Field(default_factory=list, description="Tools to call next (empty if done)")


class Claim(BaseModel):
    statement: str = Field(description="One factual statement about the case")
    evidence_ids: list[str] = Field(description="IDs of the evidence items (e.g. E3) that support the statement")


class CaseSummary(BaseModel):
    alert_id: str
    disposition: Disposition
    typology: Typology
    confidence: Literal["low", "medium", "high"]
    summary: str = Field(description="2-4 sentence analyst-ready summary")
    claims: list[Claim] = Field(min_length=1)
    policy_rules_applied: list[str] = Field(default_factory=list, description="e.g. ATO-1, ST-1")
    recommended_actions: list[str] = Field(default_factory=list)

    @field_validator("claims")
    @classmethod
    def claims_need_evidence(cls, v: list[Claim]) -> list[Claim]:
        for c in v:
            if not c.evidence_ids:
                raise ValueError(f"claim has no evidence_ids: {c.statement!r}")
        return v


class Evidence(BaseModel):
    evidence_id: str
    tool: str
    args: dict[str, Any]
    status: Literal["ok", "error"]
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    attempts: int = 1
