"""Synthetic data generator.

Builds a realistic-looking but fully synthetic banking dataset in SQLite, then plants known
fraud patterns so every alert has a ground-truth typology and disposition:

    account_takeover  -> escalate      (new device + password reset + transfer to brand-new payee)
    card_testing      -> escalate      (burst of tiny card charges across merchants)
    money_mule        -> file_sar      (accounts sharing a device, money in and straight back out)
    structuring       -> file_sar      (repeated cash deposits just under the $10,000 CTR threshold)
    benign            -> close         (look-alike legitimate activity: travel, known payees, transit taps,
                                        cash-heavy businesses, salary-then-rent)

No real customer or employer data is used. Run:  python -m fraud_agent.data_gen
"""
from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .config import get_settings

START = datetime(2026, 1, 1)
DAYS = 90

FIRST = ["Ava", "Liam", "Maya", "Noah", "Priya", "Ethan", "Sofia", "Arjun", "Chloe", "Mateo", "Zara", "Lucas",
         "Aisha", "Ryan", "Mei", "Omar", "Grace", "Diego", "Nina", "Kai", "Leah", "Ravi", "Emma", "Jonah"]
LAST = ["Patel", "Garcia", "Kim", "Johnson", "Nguyen", "Smith", "Reddy", "Lopez", "Chen", "Brown", "Singh",
        "Davis", "Martin", "Ali", "Wilson", "Khan", "Moore", "Rao", "Clark", "Lee"]
CITIES = ["Newark", "Jersey City", "New York", "Boston", "Chicago", "Austin", "Seattle", "Atlanta", "Denver"]
MERCHANTS = [("Grocery Mart", "5411"), ("FuelUp", "5541"), ("Cafe Bloom", "5814"), ("StreamFlix", "4899"),
             ("ShopZone Online", "5999"), ("PharmaPlus", "5912"), ("CityRide Transit", "4111"),
             ("HomeGoods Depot", "5200"), ("TechHub", "5732"), ("BookNook", "5942")]
TEST_MERCHANTS = [("QuickDonate.org", "8398"), ("eGift Cards Co", "5999"), ("AppStore Credit", "5818"),
                  ("WebTrial Pro", "5968"), ("PayLink Tips", "7299"), ("MiniCharge LLC", "5999")]

SCHEMA = """
CREATE TABLE customers (customer_id TEXT PRIMARY KEY, full_name TEXT, email TEXT, phone TEXT, dob TEXT,
    city TEXT, signup_date TEXT, kyc_level TEXT, travel_notice_until TEXT);
CREATE TABLE accounts (account_id TEXT PRIMARY KEY, customer_id TEXT, account_type TEXT, opened_date TEXT,
    status TEXT);
CREATE TABLE devices (device_id TEXT, account_id TEXT, first_seen TEXT, last_seen TEXT, ip_address TEXT,
    device_type TEXT);
CREATE TABLE logins (login_id TEXT PRIMARY KEY, account_id TEXT, device_id TEXT, ip_address TEXT, ts TEXT,
    event TEXT, geo TEXT);
CREATE TABLE payees (payee_id TEXT PRIMARY KEY, account_id TEXT, payee_name TEXT, payee_account TEXT,
    added_ts TEXT);
CREATE TABLE transactions (txn_id TEXT PRIMARY KEY, account_id TEXT, ts TEXT, amount REAL, channel TEXT,
    direction TEXT, counterparty TEXT, payee_id TEXT, merchant TEXT, mcc TEXT, device_id TEXT,
    is_fraud INTEGER DEFAULT 0, typology TEXT);
CREATE TABLE alerts (alert_id TEXT PRIMARY KEY, account_id TEXT, txn_id TEXT, alert_type TEXT,
    created_ts TEXT, rule_triggered TEXT, true_typology TEXT, true_disposition TEXT, expected_tools TEXT);
CREATE TABLE prior_alerts (prior_alert_id TEXT PRIMARY KEY, account_id TEXT, ts TEXT, alert_type TEXT,
    outcome TEXT);
CREATE INDEX idx_txn_acct_ts ON transactions(account_id, ts);
CREATE INDEX idx_login_acct_ts ON logins(account_id, ts);
CREATE INDEX idx_dev_dev ON devices(device_id);
CREATE INDEX idx_dev_acct ON devices(account_id);
CREATE INDEX idx_payee_acct ON payees(account_id);
"""

DISPOSITION = {"account_takeover": "escalate", "card_testing": "escalate", "money_mule": "file_sar",
               "structuring": "file_sar", "benign": "close"}

BASE_TOOLS = ["sql_lookup", "risk_score", "policy_search"]


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class Counters:
    txn: int = 0
    login: int = 0
    payee: int = 0
    alert: int = 0
    device: int = 0

    def next(self, kind: str, prefix: str) -> str:
        value = getattr(self, kind) + 1
        setattr(self, kind, value)
        return f"{prefix}{value:06d}"


class Generator:
    def __init__(self, n_customers: int = 1500, alerts_per_type: int = 25, benign_alerts: int = 40, seed: int = 7):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.n_customers = n_customers
        self.alerts_per_type = alerts_per_type
        self.benign_alerts = benign_alerts
        self.c = Counters()
        self.rows: dict[str, list[tuple]] = {k: [] for k in
                                             ["customers", "accounts", "devices", "logins", "payees",
                                              "transactions", "alerts", "prior_alerts"]}
        self.account_devices: dict[str, list[str]] = {}
        self.account_payees: dict[str, list[str]] = {}
        self.account_type: dict[str, str] = {}
        self.used_for_alert: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def rand_ts(self, day_lo: int = 0, day_hi: int = DAYS) -> datetime:
        return START + timedelta(days=self.rng.uniform(day_lo, day_hi), hours=0)

    def ip(self) -> str:
        return f"{self.rng.randint(11, 223)}.{self.rng.randint(0, 255)}.{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}"

    def add_txn(self, account_id: str, ts: datetime, amount: float, channel: str, direction: str,
                counterparty: str = "", payee_id: str = "", merchant: str = "", mcc: str = "",
                device_id: str = "", is_fraud: int = 0, typology: str = "") -> str:
        txn_id = self.c.next("txn", "T")
        self.rows["transactions"].append((txn_id, account_id, iso(ts), round(amount, 2), channel, direction,
                                          counterparty, payee_id, merchant, mcc, device_id, is_fraud, typology))
        return txn_id

    def add_login(self, account_id: str, device_id: str, ip: str, ts: datetime, event: str = "login",
                  geo: str = "US") -> None:
        self.rows["logins"].append((self.c.next("login", "L"), account_id, device_id, ip, iso(ts), event, geo))

    def add_device(self, account_id: str, first_seen: datetime, device_id: str | None = None,
                   ip: str | None = None, device_type: str | None = None) -> str:
        device_id = device_id or self.c.next("device", "D")
        self.rows["devices"].append((device_id, account_id, iso(first_seen), iso(START + timedelta(days=DAYS)),
                                     ip or self.ip(), device_type or self.rng.choice(["ios", "android", "web"])))
        self.account_devices.setdefault(account_id, []).append(device_id)
        return device_id

    def add_payee(self, account_id: str, added: datetime, name: str | None = None) -> str:
        payee_id = self.c.next("payee", "P")
        name = name or f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"
        self.rows["payees"].append((payee_id, account_id, name, f"{self.rng.randint(10**9, 10**10 - 1)}",
                                    iso(added)))
        self.account_payees.setdefault(account_id, []).append(payee_id)
        return payee_id

    def add_alert(self, account_id: str, txn_id: str, alert_type: str, ts: datetime, rule: str,
                  typology: str, extra_tools: list[str] | None = None) -> None:
        tools = BASE_TOOLS + (extra_tools or [])
        self.rows["alerts"].append((self.c.next("alert", "A"), account_id, txn_id, alert_type,
                                    iso(ts + timedelta(minutes=5)), rule, typology, DISPOSITION[typology],
                                    json.dumps(sorted(set(tools)))))
        self.used_for_alert.add(account_id)

    def pick_account(self, account_type: str | None = None) -> str:
        candidates = [a for a, t in self.account_type.items()
                      if a not in self.used_for_alert and (account_type is None or t == account_type)]
        return self.rng.choice(candidates)

    # ------------------------------------------------------------------ population
    def population(self) -> None:
        for i in range(1, self.n_customers + 1):
            cid = f"C{i:05d}"
            first, last = self.rng.choice(FIRST), self.rng.choice(LAST)
            signup = START - timedelta(days=self.rng.randint(60, 2500))
            self.rows["customers"].append((cid, f"{first} {last}", f"{first.lower()}.{last.lower()}{i}@example.com",
                                           f"+1-555-{self.rng.randint(100, 999)}-{self.rng.randint(1000, 9999)}",
                                           f"{self.rng.randint(1950, 2004)}-{self.rng.randint(1, 12):02d}-15",
                                           self.rng.choice(CITIES), signup.date().isoformat(),
                                           self.rng.choice(["standard", "standard", "enhanced"]), ""))
            acct_type = "business" if self.rng.random() < 0.08 else "checking"
            aid = f"AC{i:05d}"
            self.rows["accounts"].append((aid, cid, acct_type, signup.date().isoformat(), "active"))
            self.account_type[aid] = acct_type
            for _ in range(self.rng.choice([1, 1, 2])):
                self.add_device(aid, signup + timedelta(days=self.rng.randint(0, 50)))
            for _ in range(self.rng.randint(1, 3)):
                self.add_payee(aid, START - timedelta(days=self.rng.randint(30, 400)))
            self.normal_activity(aid, acct_type)

    def normal_activity(self, aid: str, acct_type: str) -> None:
        devices = self.account_devices[aid]
        # salary / revenue
        for d in range(0, DAYS, 14):
            amt = self.rng.uniform(1800, 4200) if acct_type == "checking" else self.rng.uniform(4000, 15000)
            self.add_txn(aid, START + timedelta(days=d + self.rng.uniform(0, 1)), amt, "transfer", "credit",
                         counterparty="Employer Payroll" if acct_type == "checking" else "Customer Payments")
        # card spend
        for _ in range(self.rng.randint(15, 40)):
            m, mcc = self.rng.choice(MERCHANTS)
            self.add_txn(aid, self.rand_ts(), float(self.np_rng.lognormal(3.4, 0.8)), "card", "debit",
                         merchant=m, mcc=mcc, device_id=self.rng.choice(devices))
        # payee transfers
        for pid in self.account_payees[aid]:
            for _ in range(self.rng.randint(0, 3)):
                self.add_txn(aid, self.rand_ts(), self.rng.uniform(50, 1500), "transfer", "debit", payee_id=pid,
                             device_id=self.rng.choice(devices))
        # occasional cash deposits (business accounts deposit more often)
        n_cash = self.rng.randint(4, 10) if acct_type == "business" else self.rng.randint(0, 2)
        for _ in range(n_cash):
            self.add_txn(aid, self.rand_ts(), self.rng.uniform(200, 6000), "cash_deposit", "credit",
                         counterparty="Branch")
        # logins
        for _ in range(self.rng.randint(10, 30)):
            dev = self.rng.choice(devices)
            self.add_login(aid, dev, self.ip(), self.rand_ts())
        # past alert history
        if self.rng.random() < 0.05:
            self.rows["prior_alerts"].append((f"PA{len(self.rows['prior_alerts']) + 1:05d}", aid,
                                              iso(START - timedelta(days=self.rng.randint(10, 300))),
                                              self.rng.choice(["large_transfer", "device_anomaly"]),
                                              "closed_false_positive"))

    # ------------------------------------------------------------------ fraud patterns
    def plant_account_takeover(self) -> None:
        aid = self.pick_account("checking")
        t0 = self.rand_ts(30, DAYS - 2)
        new_dev = self.add_device(aid, t0 - timedelta(hours=self.rng.uniform(1, 5)))
        bad_ip = self.ip()
        self.add_login(aid, new_dev, bad_ip, t0 - timedelta(hours=self.rng.uniform(0.8, 3)), "login",
                       geo=self.rng.choice(["US", "RO", "NG", "VN"]))
        self.add_login(aid, new_dev, bad_ip, t0 - timedelta(minutes=self.rng.uniform(20, 50)), "password_reset")
        if self.rng.random() < 0.5:
            self.add_login(aid, new_dev, bad_ip, t0 - timedelta(minutes=self.rng.uniform(10, 19)), "mfa_change")
        payee = self.add_payee(aid, t0 - timedelta(minutes=self.rng.uniform(3, 9)))
        txn = self.add_txn(aid, t0, self.rng.uniform(2500, 9500), "transfer", "debit", payee_id=payee,
                           device_id=new_dev, is_fraud=1, typology="account_takeover")
        self.add_alert(aid, txn, "large_transfer", t0, "transfer > $2,000 to payee", "account_takeover")

    def plant_card_testing(self) -> None:
        aid = self.pick_account("checking")
        t0 = self.rand_ts(20, DAYS - 1)
        n = self.rng.randint(11, 24)
        last = None
        for k in range(n):
            m, mcc = self.rng.choice(TEST_MERCHANTS)
            last = self.add_txn(aid, t0 + timedelta(minutes=k * self.rng.uniform(1, 3.2)),
                                self.rng.uniform(0.5, 4.99), "card", "debit", merchant=m, mcc=mcc,
                                is_fraud=1, typology="card_testing")
        self.add_alert(aid, last, "card_burst", t0 + timedelta(minutes=n * 2.1),
                       "> 5 card transactions under $5 in 60 minutes", "card_testing")

    def plant_mule_ring(self) -> None:
        ring = [self.pick_account("checking") for _ in range(self.rng.randint(3, 4))]
        for a in ring:
            self.used_for_alert.add(a)
        shared_dev = self.c.next("device", "D")
        t0 = self.rand_ts(25, DAYS - 3)
        collector = f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)} Holdings"
        ring_ip = self.ip()
        alert_txn = None
        for a in ring:
            self.add_device(a, t0 - timedelta(days=self.rng.uniform(2, 10)), device_id=shared_dev,
                            ip=ring_ip, device_type="android")
            payee = self.add_payee(a, t0 - timedelta(days=self.rng.uniform(1, 6)), name=collector)
            total_in = 0.0
            for _ in range(self.rng.randint(2, 4)):
                amt = self.rng.uniform(900, 3800)
                total_in += amt
                self.add_txn(a, t0 + timedelta(hours=self.rng.uniform(0, 8)), amt, "transfer", "credit",
                             counterparty=f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}", is_fraud=1,
                             typology="money_mule")
            out = self.add_txn(a, t0 + timedelta(hours=self.rng.uniform(9, 20)),
                               total_in * self.rng.uniform(0.86, 0.98), "transfer", "debit", payee_id=payee,
                               device_id=shared_dev, is_fraud=1, typology="money_mule")
            self.add_login(a, shared_dev, ring_ip, t0 + timedelta(hours=8.5))
            if alert_txn is None:
                alert_txn = (a, out, t0 + timedelta(hours=20))
        a, out, ts = alert_txn
        self.add_alert(a, out, "rapid_movement", ts, "> 80% of inbound funds moved out within 24h",
                       "money_mule", ["network_check"])

    def plant_structuring(self) -> None:
        aid = self.pick_account()
        t0 = self.rand_ts(20, DAYS - 7)
        last = None
        for k in range(self.rng.randint(3, 5)):
            last = self.add_txn(aid, t0 + timedelta(days=k * self.rng.uniform(0.8, 1.6)),
                                self.rng.uniform(8200, 9900), "cash_deposit", "credit", counterparty="Branch",
                                is_fraud=1, typology="structuring")
        self.add_alert(aid, last, "cash_threshold", t0 + timedelta(days=6), "cumulative cash > $20,000 in 7 days",
                       "structuring")

    # ------------------------------------------------------------------ benign look-alikes
    def plant_benign(self, variant: str) -> None:
        if variant == "travel":
            aid = self.pick_account("checking")
            t0 = self.rand_ts(20, DAYS - 2)
            cid = "C" + aid[2:]
            for i, row in enumerate(self.rows["customers"]):
                if row[0] == cid:
                    self.rows["customers"][i] = row[:-1] + (iso(t0 + timedelta(days=10)),)
            dev = self.add_device(aid, t0 - timedelta(hours=2))
            self.add_login(aid, dev, self.ip(), t0 - timedelta(hours=1), "login", geo=self.rng.choice(["FR", "MX", "JP"]))
            m, mcc = self.rng.choice(MERCHANTS)
            txn = self.add_txn(aid, t0, self.rng.uniform(80, 900), "card", "debit", merchant=m, mcc=mcc, device_id=dev)
            self.add_alert(aid, txn, "device_anomaly", t0, "login from new device in new country", "benign")
        elif variant == "known_payee":
            aid = self.pick_account("checking")
            t0 = self.rand_ts(30, DAYS - 1)
            payee = self.account_payees[aid][0]
            self.add_txn(aid, t0 - timedelta(days=30), self.rng.uniform(2000, 6000), "transfer", "debit",
                         payee_id=payee, device_id=self.account_devices[aid][0])
            txn = self.add_txn(aid, t0, self.rng.uniform(2500, 9000), "transfer", "debit", payee_id=payee,
                               device_id=self.account_devices[aid][0])
            self.add_alert(aid, txn, "large_transfer", t0, "transfer > $2,000 to payee", "benign")
        elif variant == "transit":
            aid = self.pick_account("checking")
            t0 = self.rand_ts(10, DAYS - 1)
            last = None
            for k in range(self.rng.randint(6, 9)):
                last = self.add_txn(aid, t0 + timedelta(minutes=k * self.rng.uniform(5, 8)), self.rng.uniform(1.5, 3.0),
                                    "card", "debit", merchant="CityRide Transit", mcc="4111")
            self.add_alert(aid, last, "card_burst", t0 + timedelta(minutes=50),
                           "> 5 card transactions under $5 in 60 minutes", "benign")
        elif variant == "cash_business":
            aid = self.pick_account("business")
            t0 = self.rand_ts(20, DAYS - 7)
            last = None
            for k in range(self.rng.randint(3, 5)):
                last = self.add_txn(aid, t0 + timedelta(days=k * 1.3), self.rng.choice(
                    [self.rng.uniform(3000, 7500), self.rng.uniform(10500, 16000)]), "cash_deposit", "credit",
                    counterparty="Branch")
            self.add_alert(aid, last, "cash_threshold", t0 + timedelta(days=6), "cumulative cash > $20,000 in 7 days",
                           "benign")
        elif variant == "salary_rent":
            aid = self.pick_account("checking")
            t0 = self.rand_ts(20, DAYS - 2)
            sal = self.rng.uniform(3000, 5000)
            self.add_txn(aid, t0, sal, "transfer", "credit", counterparty="Employer Payroll")
            payee = self.account_payees[aid][0]
            txn = self.add_txn(aid, t0 + timedelta(hours=self.rng.uniform(3, 15)), sal * self.rng.uniform(0.82, 0.9),
                               "transfer", "debit", payee_id=payee, device_id=self.account_devices[aid][0])
            self.add_alert(aid, txn, "rapid_movement", t0 + timedelta(hours=16),
                           "> 80% of inbound funds moved out within 24h", "benign", ["network_check"])

    # ------------------------------------------------------------------ run
    def run(self) -> dict[str, list[tuple]]:
        self.population()
        for _ in range(self.alerts_per_type):
            self.plant_account_takeover()
            self.plant_card_testing()
            self.plant_mule_ring()
            self.plant_structuring()
        variants = ["travel", "known_payee", "transit", "cash_business", "salary_rent"]
        for i in range(self.benign_alerts):
            self.plant_benign(variants[i % len(variants)])
        return self.rows


def write_db(rows: dict[str, list[tuple]], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    for table, data in rows.items():
        if data:
            placeholders = ",".join("?" * len(data[0]))
            con.executemany(f"INSERT INTO {table} VALUES ({placeholders})", data)
    con.commit()
    con.close()


def main(n_customers: int = 1500, alerts_per_type: int = 25, benign_alerts: int = 40, seed: int = 7) -> Path:
    settings = get_settings()
    rows = Generator(n_customers, alerts_per_type, benign_alerts, seed).run()
    write_db(rows, settings.db_path)
    print(f"Wrote {settings.db_path}: " + ", ".join(f"{k}={len(v)}" for k, v in rows.items()))
    return settings.db_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--customers", type=int, default=1500)
    p.add_argument("--alerts-per-type", type=int, default=25)
    p.add_argument("--benign-alerts", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    main(a.customers, a.alerts_per_type, a.benign_alerts, a.seed)
