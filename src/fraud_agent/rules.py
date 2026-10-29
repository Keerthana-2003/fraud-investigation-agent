"""Deterministic policy rules (ATO-1, CT-1, MM-1, ST-1) evaluated over collected evidence.

Used in three places:
  * the offline MockLLM (so the whole graph runs without an API key),
  * a "rules-only" fallback when the LLM is unavailable,
  * a sanity check shown next to the LLM's decision in the demo app.
It is NOT the system being evaluated; the LLM agent is.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .schemas import Evidence


def _ok(evidence: list[Evidence], tool: str, table: str | None = None) -> list[Evidence]:
    return [e for e in evidence if e.status == "ok" and e.tool == tool
            and (table is None or (isinstance(e.data, dict) and e.data.get("table") == table))]


def _rows(evidence: list[Evidence], table: str) -> tuple[list[dict], list[str]]:
    seen, rows, ids = set(), [], []
    for e in _ok(evidence, "sql_lookup", table):
        ids.append(e.evidence_id)
        for r in e.data["rows"]:
            key = tuple(sorted(r.items()))
            if key not in seen:
                seen.add(key)
                rows.append(r)
    return rows, ids


def _policy_ids(evidence: list[Evidence], needle: str) -> list[str]:
    return [e.evidence_id for e in _ok(evidence, "policy_search")
            if any(needle in r["text"] for r in e.data["results"])]


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def evaluate(alert: dict[str, Any], evidence: list[Evidence]) -> dict[str, Any]:
    txns, txn_ids = _rows(evidence, "transactions")
    logins, login_ids = _rows(evidence, "logins")
    payees, payee_ids = _rows(evidence, "payees")
    devices, device_ids = _rows(evidence, "devices")
    net = _ok(evidence, "network_check")
    risk = _ok(evidence, "risk_score")
    as_of, txn_ts = _dt(alert["created_ts"]), _dt(alert["txn_ts"])
    claims: list[tuple[str, list[str]]] = [(f"Alert {alert['alert_id']} ({alert['alert_type']}) was raised on account {alert['account_id']} "
               f"for transaction {alert['txn_id']}.", ["ALERT"])]
    if risk:
        r = risk[0].data
        claims.append((f"The risk model scored transaction {r['txn_id']} at {r['risk_score']} ({r['risk_band']}); "
                       f"top driver: {r['top_drivers'][0]['feature']}.", [risk[0].evidence_id]))

    # ST-1 structuring
    near = [t for t in txns if t["channel"] == "cash_deposit" and 8000 <= t["amount"] < 10000
            and _dt(t["ts"]) >= as_of - timedelta(days=7)]
    if len(near) >= 3:
        pol = _policy_ids(evidence, "ST-1")
        claims.append((f"{len(near)} cash deposits between $8,000 and $9,999.99 within 7 days, e.g. "
                       f"{near[0]['txn_id']} for {near[0]['amount']}.", txn_ids + pol))
        return _result("structuring", "file_sar", "high", ["ST-1"], claims,
                       ["Refer to BSA/AML for SAR preparation", "Do not notify the customer (no tipping-off)"])

    # CT-1 card testing
    small = [t for t in txns if t["channel"] == "card" and t["amount"] < 5
             and txn_ts - timedelta(minutes=60) <= _dt(t["ts"]) <= txn_ts]
    merchants = {t["merchant"] for t in small}
    if len(small) >= 10 and len(merchants) >= 3:
        pol = _policy_ids(evidence, "CT-1")
        claims.append((f"{len(small)} card transactions under $5 across {len(merchants)} merchants within 60 "
                       f"minutes, including {small[0]['txn_id']} at {small[0]['merchant']}.", txn_ids + pol))
        return _result("card_testing", "escalate", "high", ["CT-1"], claims,
                       ["Block and reissue the card", "Decline pending authorizations", "Notify the customer"])

    # MM-1 money mule
    if net:
        n = net[0].data
        linked = len(set(n["shared_device_accounts"]) | set(n["shared_ip_accounts"]))
        if linked >= 2 and n["pass_through_ratio_24h"] >= 0.8:
            pol = _policy_ids(evidence, "MM-1")
            claims.append((f"Account shares a device or IP with {linked} other accounts "
                           f"({', '.join(n['shared_device_accounts'][:3])}) and moved out "
                           f"{n['pass_through_ratio_24h']} of 24h inflow {n['inflow_24h']}.",
                           [net[0].evidence_id] + pol))
            return _result("money_mule", "file_sar", "high", ["MM-1"], claims,
                           ["Refer to BSA/AML for SAR preparation", "Review all linked accounts"])

    # ATO-1 account takeover
    alert_txn = next((t for t in txns if t["txn_id"] == alert["txn_id"]), None)
    if alert_txn and alert_txn.get("payee_id") and alert["amount"] > 2000:
        payee = next((p for p in payees if p["payee_id"] == alert_txn["payee_id"]), None)
        new_payee = payee and txn_ts - _dt(payee["added_ts"]) < timedelta(hours=24)
        creds = [lg for lg in logins if lg["event"] in ("password_reset", "mfa_change")
                 and txn_ts - timedelta(hours=24) <= _dt(lg["ts"]) <= txn_ts]
        dev = next((d for d in devices if d["device_id"] == alert_txn.get("device_id")), None)
        new_device = dev and txn_ts - _dt(dev["first_seen"]) < timedelta(hours=24)
        if new_payee and (creds or new_device):
            pol = _policy_ids(evidence, "ATO-1")
            claims.append((f"Payee {payee['payee_id']} was added at {payee['added_ts']}, less than 24 hours "
                           f"before the transfer of {alert['amount']}.", payee_ids + txn_ids))
            if creds:
                claims.append((f"A {creds[0]['event']} occurred at {creds[0]['ts']} from device "
                               f"{creds[0]['device_id']}.", login_ids))
            if new_device:
                claims.append((f"Device {dev['device_id']} was first seen at {dev['first_seen']}.", device_ids))
            if pol:
                claims.append(("Policy ATO-1 requires escalation when a new payee coincides with a new device or "
                               "credential change and the transfer exceeds $2,000.", pol))
            return _result("account_takeover", "escalate", "high", ["ATO-1"], claims,
                           ["Freeze outbound transfers", "Revoke sessions on the new device",
                            "Contact the customer through a verified channel"])

    claims.append(("No fraud rule (ATO-1, CT-1, MM-1, ST-1) is met by the collected records.",
                   [e.evidence_id for e in evidence if e.status == "ok"][:4] or ["ALERT"]))
    return _result("benign", "close", "medium", [], claims, ["Close as explained activity"])


def _result(typology: str, disposition: str, confidence: str, rules: list[str],
            claims: list[tuple[str, list[str]]], actions: list[str]) -> dict[str, Any]:
    return {"typology": typology, "disposition": disposition, "confidence": confidence, "rules": rules,
            "claims": [{"statement": s, "evidence_ids": sorted(set(ids))} for s, ids in claims],
            "actions": actions}
