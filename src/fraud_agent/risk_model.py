"""Transaction risk model (XGBoost) with SHAP explanations.

To avoid leakage, the model is trained on a *separate* synthetic dataset (a different random seed) from
the one the agent is evaluated on. Explanations use XGBoost's built-in TreeSHAP (pred_contribs=True), which
returns exact SHAP values for tree ensembles without needing the `shap` package at inference time.

    python -m fraud_agent.risk_model          # train and save data/risk_model.joblib
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import get_settings

FEATURES = [
    "amount", "log_amount", "hour", "is_card", "is_transfer", "is_cash", "is_debit", "is_business",
    "payee_age_hours", "new_payee_24h", "device_age_hours", "new_device_24h", "credential_changes_24h",
    "small_card_txn_1h", "near_threshold_cash_7d", "inflow_24h", "pass_through_ratio",
    "shared_device_accounts",
]
NO_HISTORY_HOURS = 24 * 365.0


def _counts_in_window(event_times: np.ndarray, query_times: np.ndarray, hours: float) -> np.ndarray:
    """For each query time q, count events e with q - hours < e <= q."""
    event_times = np.sort(event_times)
    hi = np.searchsorted(event_times, query_times, side="right")
    lo = np.searchsorted(event_times, query_times - np.timedelta64(int(hours * 3600), "s"), side="right")
    return hi - lo


def compute_features(transactions: pd.DataFrame, logins: pd.DataFrame, payees: pd.DataFrame,
                     devices: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features: each row only uses information available at its timestamp."""
    t = transactions.copy()
    t["ts"] = pd.to_datetime(t["ts"])
    t = t.sort_values(["account_id", "ts"]).reset_index(drop=True)
    t["log_amount"] = np.log1p(t["amount"])
    t["hour"] = t["ts"].dt.hour
    t["is_card"] = (t["channel"] == "card").astype(int)
    t["is_transfer"] = (t["channel"] == "transfer").astype(int)
    t["is_cash"] = (t["channel"] == "cash_deposit").astype(int)
    t["is_debit"] = (t["direction"] == "debit").astype(int)
    acct_type = accounts.set_index("account_id")["account_type"]
    t["is_business"] = (t["account_id"].map(acct_type) == "business").astype(int)

    p = payees[["payee_id", "added_ts"]].copy()
    p["added_ts"] = pd.to_datetime(p["added_ts"])
    t = t.merge(p, on="payee_id", how="left")
    t["payee_age_hours"] = ((t["ts"] - t["added_ts"]).dt.total_seconds() / 3600).fillna(NO_HISTORY_HOURS)
    t["new_payee_24h"] = ((t["payee_age_hours"] >= 0) & (t["payee_age_hours"] < 24)).astype(int)

    d = devices[["device_id", "account_id", "first_seen"]].copy()
    d["first_seen"] = pd.to_datetime(d["first_seen"])
    t = t.merge(d.rename(columns={"first_seen": "dev_first_seen"}), on=["device_id", "account_id"], how="left")
    t["device_age_hours"] = ((t["ts"] - t["dev_first_seen"]).dt.total_seconds() / 3600).fillna(NO_HISTORY_HOURS)
    t["new_device_24h"] = ((t["device_age_hours"] >= 0) & (t["device_age_hours"] < 24)).astype(int)

    shared = d.groupby("device_id")["account_id"].nunique()
    d["others"] = d["device_id"].map(shared) - 1
    t["shared_device_accounts"] = t["account_id"].map(d.groupby("account_id")["others"].max()).fillna(0)

    lg = logins[logins["event"].isin(["password_reset", "mfa_change"])].copy()
    lg["ts"] = pd.to_datetime(lg["ts"])
    cred_by_acct = {a: g["ts"].values for a, g in lg.groupby("account_id")}

    cols = {"credential_changes_24h": [], "small_card_txn_1h": [], "near_threshold_cash_7d": [],
            "inflow_24h": []}
    for acct, g in t.groupby("account_id", sort=False):
        q = g["ts"].values
        ev = cred_by_acct.get(acct, np.array([], dtype="datetime64[ns]"))
        cols["credential_changes_24h"].append(pd.Series(_counts_in_window(ev, q, 24), index=g.index))
        small = g.loc[(g["channel"] == "card") & (g["amount"] < 5), "ts"].values
        cols["small_card_txn_1h"].append(pd.Series(_counts_in_window(small, q, 1), index=g.index))
        near = g.loc[(g["channel"] == "cash_deposit") & g["amount"].between(8000, 9999.99), "ts"].values
        cols["near_threshold_cash_7d"].append(pd.Series(_counts_in_window(near, q, 24 * 7), index=g.index))
        credits = g.loc[(g["channel"] == "transfer") & (g["direction"] == "credit")]
        ct, ca = credits["ts"].values, credits["amount"].values
        order = np.argsort(ct)
        ct, cum = ct[order], np.concatenate([[0.0], np.cumsum(ca[order])])
        hi = np.searchsorted(ct, q, side="right")
        lo = np.searchsorted(ct, q - np.timedelta64(24 * 3600, "s"), side="right")
        cols["inflow_24h"].append(pd.Series(cum[hi] - cum[lo], index=g.index))
    for name, parts in cols.items():
        t[name] = pd.concat(parts).sort_index() if parts else 0
    t["pass_through_ratio"] = np.where((t["is_transfer"] == 1) & (t["is_debit"] == 1) & (t["inflow_24h"] > 0),
                                       t["amount"] / t["inflow_24h"].replace(0, np.nan), 0.0)
    t["pass_through_ratio"] = t["pass_through_ratio"].fillna(0).clip(0, 5)
    return t


def load_tables(db_path: Path, account_ids: list[str] | None = None) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = {}
        for table in ["transactions", "logins", "payees", "devices", "accounts"]:
            if account_ids is not None and table != "devices":
                marks = ",".join("?" * len(account_ids))
                out[table] = pd.read_sql_query(f"SELECT * FROM {table} WHERE account_id IN ({marks})", con,
                                               params=account_ids)
            else:
                out[table] = pd.read_sql_query(f"SELECT * FROM {table}", con)  # devices: needed for sharing
        return out
    finally:
        con.close()


def _tables_from_rows(rows: dict[str, list[tuple]]) -> dict[str, pd.DataFrame]:
    cols = {
        "transactions": ["txn_id", "account_id", "ts", "amount", "channel", "direction", "counterparty",
                         "payee_id", "merchant", "mcc", "device_id", "is_fraud", "typology"],
        "logins": ["login_id", "account_id", "device_id", "ip_address", "ts", "event", "geo"],
        "payees": ["payee_id", "account_id", "payee_name", "payee_account", "added_ts"],
        "devices": ["device_id", "account_id", "first_seen", "last_seen", "ip_address", "device_type"],
        "accounts": ["account_id", "customer_id", "account_type", "opened_date", "status"],
    }
    return {k: pd.DataFrame(rows[k], columns=v) for k, v in cols.items()}


def train(model_type: str | None = None, seed: int = 1007, out_path: Path | None = None) -> dict[str, Any]:
    """Train on an independent synthetic dataset and save the model. Returns validation metrics."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    from .data_gen import Generator

    settings = get_settings()
    model_type = model_type or settings.risk_model_type
    out_path = out_path or settings.model_path
    tables = _tables_from_rows(Generator(n_customers=1500, alerts_per_type=40, benign_alerts=60, seed=seed).run())
    feats = compute_features(**tables)
    # time-based split: first 70 days train, last 20 validate
    cutoff = feats["ts"].min() + pd.Timedelta(days=70)
    tr, va = feats[feats["ts"] < cutoff], feats[feats["ts"] >= cutoff]
    X_tr, y_tr, X_va, y_va = tr[FEATURES].values, tr["is_fraud"].values, va[FEATURES].values, va["is_fraud"].values

    bundle: dict[str, Any] = {"model_type": model_type, "features": FEATURES}
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
                              colsample_bytree=0.9, scale_pos_weight=pos_weight, eval_metric="aucpr",
                              random_state=seed)
        model.fit(X_tr, y_tr)
        bundle["model"] = model
    elif model_type == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(X_tr)
        model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_tr), y_tr)
        bundle.update(model=model, scaler=scaler)
    else:
        raise ValueError(f"unknown model_type {model_type}")

    scores = _predict(bundle, X_va)
    metrics = {"model_type": model_type, "val_pr_auc": round(float(average_precision_score(y_va, scores)), 4),
               "val_roc_auc": round(float(roc_auc_score(y_va, scores)), 4), "train_rows": int(len(tr)),
               "val_rows": int(len(va)), "val_positives": int(y_va.sum())}
    bundle["metrics"] = metrics
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return metrics


def _predict(bundle: dict[str, Any], X: np.ndarray) -> np.ndarray:
    if bundle["model_type"] == "xgboost":
        return bundle["model"].predict_proba(X)[:, 1]
    return bundle["model"].predict_proba(bundle["scaler"].transform(X))[:, 1]


def _contributions(bundle: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """SHAP values (xgboost TreeSHAP) or coefficient * standardized value (logreg)."""
    if bundle["model_type"] == "xgboost":
        import xgboost as xgb

        contribs = bundle["model"].get_booster().predict(xgb.DMatrix(X, feature_names=None), pred_contribs=True)
        return contribs[:, :-1]  # last column is the bias term
    return bundle["model"].coef_[0] * bundle["scaler"].transform(X)


class RiskScorer:
    def __init__(self, db_path: Path, model_path: Path):
        self.db_path = db_path
        self.bundle = joblib.load(model_path)

    def score(self, txn_id: str, top_k: int = 5) -> dict[str, Any]:
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT account_id FROM transactions WHERE txn_id = ?", (txn_id,)).fetchone()
        finally:
            con.close()
        if row is None:
            raise KeyError(f"transaction {txn_id} not found")
        feats = compute_features(**load_tables(self.db_path, [row[0]]))
        x = feats.loc[feats["txn_id"] == txn_id, FEATURES]
        X = x.values.astype(float)
        score = float(_predict(self.bundle, X)[0])
        contrib = _contributions(self.bundle, X)[0]
        order = np.argsort(-np.abs(contrib))[:top_k]
        band = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
        return {
            "txn_id": txn_id,
            "risk_score": round(score, 4),
            "risk_band": band,
            "top_drivers": [{"feature": FEATURES[i], "value": round(float(X[0, i]), 3),
                             "shap_contribution": round(float(contrib[i]), 4)} for i in order],
            "model": self.bundle["model_type"],
        }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", default=None, choices=["xgboost", "logreg"])
    print(json.dumps(train(ap.parse_args().model_type), indent=2))
