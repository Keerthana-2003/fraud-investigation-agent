"""Read-only, whitelisted, parameterized database access.

The agent never writes SQL text. It chooses a whitelisted table and structured filters, and this module
builds a parameterized SELECT. The connection itself is opened in SQLite read-only mode, so even a bug
here cannot modify data.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

# table -> (allowed columns, timestamp column used for time windows or None)
WHITELIST: dict[str, tuple[list[str], str | None]] = {
    "transactions": (["txn_id", "account_id", "ts", "amount", "channel", "direction", "counterparty",
                      "payee_id", "merchant", "mcc", "device_id"], "ts"),
    "logins": (["login_id", "account_id", "device_id", "ip_address", "ts", "event", "geo"], "ts"),
    "payees": (["payee_id", "account_id", "payee_name", "payee_account", "added_ts"], "added_ts"),
    "devices": (["device_id", "account_id", "first_seen", "last_seen", "ip_address", "device_type"], None),
    "accounts": (["account_id", "customer_id", "account_type", "opened_date", "status"], None),
    "customers": (["customer_id", "full_name", "email", "phone", "dob", "city", "signup_date", "kyc_level",
                   "travel_notice_until"], None),
    "prior_alerts": (["prior_alert_id", "account_id", "ts", "alert_type", "outcome"], "ts"),
}
# Ground-truth columns (is_fraud, typology, true_*) are deliberately never exposed to the agent.

MAX_ROWS = 200
MAX_WINDOW_HOURS = 24 * 30


class QueryRejected(ValueError):
    """Raised when a request falls outside the whitelist or limits."""


@contextmanager
def connect_ro(db_path: Path) -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def safe_select(db_path: Path, table: str, account_id: str, as_of: str, since_hours: int | None = None,
                limit: int = 100) -> list[dict[str, Any]]:
    """SELECT whitelisted columns for one account, optionally within [as_of - since_hours, as_of]."""
    if table not in WHITELIST:
        raise QueryRejected(f"table '{table}' is not allowed; allowed: {sorted(WHITELIST)}")
    if not account_id or not isinstance(account_id, str):
        raise QueryRejected("account_id is required")
    columns, ts_col = WHITELIST[table]
    limit = max(1, min(int(limit or 100), MAX_ROWS))
    params: list[Any] = []

    if table == "customers":
        sql = (f"SELECT {', '.join('c.' + c for c in columns)} FROM customers c "
               "JOIN accounts a ON a.customer_id = c.customer_id WHERE a.account_id = ?")
        params.append(account_id)
    else:
        sql = f"SELECT {', '.join(columns)} FROM {table} WHERE account_id = ?"
        params.append(account_id)
        if ts_col:
            sql += f" AND {ts_col} <= ?"
            params.append(as_of)
            if since_hours:
                hours = max(1, min(int(since_hours), MAX_WINDOW_HOURS))
                sql += f" AND {ts_col} >= ?"
                params.append((_parse(as_of) - timedelta(hours=hours)).isoformat(timespec="seconds"))
            sql += f" ORDER BY {ts_col} DESC"
    sql += " LIMIT ?"
    params.append(limit)

    with connect_ro(db_path) as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def get_alert(db_path: Path, alert_id: str) -> dict[str, Any]:
    with connect_ro(db_path) as con:
        row = con.execute(
            "SELECT a.alert_id, a.account_id, a.txn_id, a.alert_type, a.created_ts, a.rule_triggered, "
            "t.ts AS txn_ts, t.amount, t.channel, t.direction, t.merchant, t.payee_id "
            "FROM alerts a JOIN transactions t ON t.txn_id = a.txn_id WHERE a.alert_id = ?", (alert_id,)).fetchone()
    if row is None:
        raise KeyError(f"alert {alert_id} not found")
    return dict(row)


def get_ground_truth(db_path: Path) -> dict[str, dict[str, Any]]:
    """Evaluation only. Never passed to the agent."""
    import json

    with connect_ro(db_path) as con:
        rows = con.execute("SELECT alert_id, alert_type, true_typology, true_disposition, expected_tools "
                           "FROM alerts ORDER BY alert_id").fetchall()
    return {r["alert_id"]: {"alert_type": r["alert_type"], "typology": r["true_typology"],
                            "disposition": r["true_disposition"], "expected_tools": json.loads(r["expected_tools"])}
            for r in rows}


def network_neighbors(db_path: Path, account_id: str, as_of: str) -> dict[str, Any]:
    """Accounts linked through a shared device or IP, or a shared payee name added in the last 30 days."""
    with connect_ro(db_path) as con:
        dev = con.execute(
            "SELECT DISTINCT d2.account_id, d1.device_id FROM devices d1 JOIN devices d2 "
            "ON d1.device_id = d2.device_id AND d2.account_id != d1.account_id "
            "WHERE d1.account_id = ? AND d2.first_seen <= ?", (account_id, as_of)).fetchall()
        ip = con.execute(
            "SELECT DISTINCT d2.account_id, d1.ip_address FROM devices d1 JOIN devices d2 "
            "ON d1.ip_address = d2.ip_address AND d2.account_id != d1.account_id "
            "WHERE d1.account_id = ? AND d2.first_seen <= ?", (account_id, as_of)).fetchall()
        payee = con.execute(
            "SELECT DISTINCT p2.account_id, p1.payee_name FROM payees p1 JOIN payees p2 "
            "ON p1.payee_name = p2.payee_name AND p2.account_id != p1.account_id "
            "WHERE p1.account_id = ? AND p2.added_ts <= ? AND p1.added_ts >= ?",
            (account_id, as_of, (_parse(as_of) - timedelta(days=30)).isoformat(timespec="seconds"))).fetchall()
        start = (_parse(as_of) - timedelta(hours=24)).isoformat(timespec="seconds")
        flows = con.execute(
            "SELECT direction, SUM(amount) FROM transactions WHERE account_id = ? AND channel = 'transfer' "
            "AND ts BETWEEN ? AND ? GROUP BY direction", (account_id, start, as_of)).fetchall()
    inflow = next((f[1] for f in flows if f[0] == "credit"), 0.0) or 0.0
    outflow = next((f[1] for f in flows if f[0] == "debit"), 0.0) or 0.0
    linked = {r[0] for r in dev} | {r[0] for r in ip}
    return {
        "account_id": account_id,
        "shared_device_accounts": sorted({r[0] for r in dev}),
        "shared_devices": sorted({r[1] for r in dev}),
        "shared_ip_accounts": sorted({r[0] for r in ip}),
        "shared_payee_accounts": sorted({r[0] for r in payee}),
        "linked_account_count": len(linked),
        "inflow_24h": round(inflow, 2),
        "outflow_24h": round(outflow, 2),
        "pass_through_ratio_24h": round(outflow / inflow, 3) if inflow else 0.0,
    }
