"""PII masking applied to every tool result before it is placed in an LLM prompt."""
from __future__ import annotations

import re
from typing import Any

DROP_KEYS = {"full_name", "email", "phone", "dob"}
LAST4_KEYS = {"payee_account"}
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s().]{8,}\d")


def mask_value(key: str, value: Any) -> Any:
    if key in DROP_KEYS:
        return "[REDACTED]"
    if key in LAST4_KEYS and isinstance(value, str) and len(value) > 4:
        return "****" + value[-4:]
    if isinstance(value, str):
        value = EMAIL_RE.sub("[EMAIL]", value)
        value = PHONE_RE.sub("[PHONE]", value)
    return value


def mask(obj: Any, key: str = "") -> Any:
    if isinstance(obj, dict):
        return {k: mask(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask(v, key) for v in obj]
    return mask_value(key, obj)
