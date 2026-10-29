"""Central configuration. Everything can be overridden with environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    results_dir: Path = field(default_factory=lambda: Path(_env("RESULTS_DIR", str(ROOT / "results"))))

    # LLM provider: "openai", "anthropic", or "mock" (offline, rule-based; for tests and demos only)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0")))

    # Retrieval backend for policy_search: "chroma" (embeddings) or "tfidf" (offline)
    retriever: str = field(default_factory=lambda: _env("RETRIEVER", "chroma"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small"))

    # Risk model: "xgboost" (default) or "logreg" (lightweight fallback)
    risk_model_type: str = field(default_factory=lambda: _env("RISK_MODEL_TYPE", "xgboost"))

    # Optional endpoint from the fraud-typology-extractor project
    typology_api_url: str = field(default_factory=lambda: _env("TYPOLOGY_API_URL", ""))

    # Agent limits
    max_steps: int = field(default_factory=lambda: int(_env("AGENT_MAX_STEPS", "4")))
    max_calls_per_step: int = field(default_factory=lambda: int(_env("AGENT_MAX_CALLS_PER_STEP", "6")))
    tool_timeout_s: float = field(default_factory=lambda: float(_env("TOOL_TIMEOUT_S", "15")))
    tool_retries: int = field(default_factory=lambda: int(_env("TOOL_RETRIES", "2")))
    max_output_repairs: int = field(default_factory=lambda: int(_env("MAX_OUTPUT_REPAIRS", "2")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "fraud.db"

    @property
    def policies_dir(self) -> Path:
        return ROOT / "data" / "policies"

    @property
    def model_path(self) -> Path:
        return self.data_dir / "risk_model.joblib"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


# USD per 1M tokens (input, output). Check your provider's current pricing and update before reporting costs.
PRICES_PER_1M = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "mock": (0.0, 0.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES_PER_1M.get(model, (0.0, 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


def get_settings() -> Settings:
    return Settings()
