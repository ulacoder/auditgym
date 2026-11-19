"""Базовые политики. Все, кроме oracle, видят только наблюдения среды.

* random_explorer   — случайный исследователь с фиксированным seed;
* naive_reconciler  — простой алгоритм сверки (наивные даты, дубли по сумме);
* flag_all          — атака «пометить всё»;
* oracle            — только для оценщика: отправляет эталон с доказательствами;
* scripted_auditor  — скриптовая политика команды для демо (читает всё и сверяет по политике).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

from ..analysis import audit

TABLE_ORDER = ("invoices", "payments", "fx", "ledger")


class Policy:
    name = "policy"

    def reset(self, obs: dict, info: dict) -> None:
        self.obs0, self.info = obs, info
        self.episode_id = obs["manifest"]["episode_id"]

    def act(self, obs: dict) -> dict:
        raise NotImplementedError


class _Reader(Policy):
    """Читает все таблицы постранично, затем вызывает decide()."""

    def reset(self, obs, info):
        super().reset(obs, info)
        self.tables = {t: [] for t in TABLE_ORDER}
        self.queue = list(TABLE_ORDER)
        self.cursor = None
        self.pending = None

    def act(self, obs):
        if self.pending and obs.get("result") and obs["result"].get("table") == self.pending:
            self.tables[self.pending].extend(obs["result"]["records"])
            self.cursor = obs["result"]["next_cursor"]
            if self.cursor is None:
                self.queue.pop(0)
        if self.queue and obs["steps_remaining"] > 1:
            self.pending = self.queue[0]
            return {"tool": "list_records", "args": {"table": self.pending, "cursor": self.cursor, "limit": 50}}
        return {"tool": "submit", "args": {"findings": {"episode_id": self.episode_id,
                                                        "findings": self.decide()[:30]}}}

    def decide(self) -> list[dict]:
        raise NotImplementedError


class ScriptedAuditor(_Reader):
    name = "scripted_auditor"

    def decide(self):
        return audit(self.tables, self.obs0["manifest"])


class NaiveReconciler(_Reader):
    """Типичная «простая сверка»: даты берутся из строки как есть (без часового пояса
    компании и без правила «последний курс не позже»), дубли ищутся по сумме, любой
    unapplied_cash считается ошибкой распределения."""
    name = "naive_reconciler"

    def decide(self):
        m = self.obs0["manifest"]
        base, exps = m["base_currency"], m["currency_exponents"]
        p0, p1 = m["period_start"][:10], m["period_end"][:10]
        rates = {(r["date"], r["source_currency"]): Decimal(r["rate"]) for r in self.tables["fx"]}
        led = self.tables["ledger"]
        inv_ar = {r["invoice_id"]: r for r in led if r["invoice_id"] and not r["payment_id"]
                  and r["account"] == "accounts_receivable"}
        out = []

        def conv(amount, cur, d):
            if cur == base:
                return amount
            rt = rates.get((d, cur))
            if rt is None:
                return None
            v = Decimal(amount) * rt * Decimal(10) ** (exps[base] - exps[cur])
            return int(v.quantize(Decimal(1), rounding=ROUND_HALF_UP))

        seen = {}
        for inv in sorted(self.tables["invoices"], key=lambda i: i["issued_at"]):
            ar = inv_ar.get(inv["invoice_id"])
            if not ar:
                continue
            d = inv["issued_at"][:10]
            if not (p0 <= d < p1):
                out.append(_mk("period_cutoff", [inv["invoice_id"]], ar["debit_minor"], base,
                               [inv["doc_id"], ar["row_id"]], "date outside period"))
            exp = conv(inv["total_minor"], inv["currency"], d)
            if exp is not None and exp != ar["debit_minor"]:
                out.append(_mk("fx_conversion", [inv["invoice_id"]], abs(exp - ar["debit_minor"]), base,
                               [inv["doc_id"], ar["row_id"]], "rate mismatch"))
            k = (inv["entity_id"], inv["currency"], inv["total_minor"])
            if k in seen:
                out.append(_mk("duplicate_invoice", [inv["invoice_id"]], ar["debit_minor"], base,
                               [seen[k], inv["doc_id"]], "same amount"))
            else:
                seen[k] = inv["doc_id"]
        pays = {p["payment_id"]: p for p in self.tables["payments"]}
        for r in led:
            if r["account"] == "unapplied_cash" and r["credit_minor"] > 0 and r["payment_id"] in pays:
                p = pays[r["payment_id"]]
                aff = [p["payment_id"]] + ([p["invoice_refs"][0]["invoice_id"]] if p["invoice_refs"] else [])
                out.append(_mk("payment_allocation", aff, r["credit_minor"], base, [p["doc_id"], r["row_id"]],
                               "unapplied cash"))
        return out


class FlagAll(Policy):
    """Атака: прочитать первые страницы документов и пометить всё подряд всеми типами."""
    name = "flag_all"

    def reset(self, obs, info):
        super().reset(obs, info)
        self.stage = 0
        self.inv, self.pay = [], []

    def act(self, obs):
        if obs.get("result") and obs["result"].get("table") == "invoices":
            self.inv = obs["result"]["records"]
        if obs.get("result") and obs["result"].get("table") == "payments":
            self.pay = obs["result"]["records"]
        self.stage += 1
        if self.stage == 1:
            return {"tool": "list_records", "args": {"table": "invoices"}}
        if self.stage == 2:
            return {"tool": "list_records", "args": {"table": "payments"}}
        base = self.obs0["manifest"]["base_currency"]
        fs = []
        all_docs = [i["doc_id"] for i in self.inv[:5]] + [p["doc_id"] for p in self.pay[:5]]
        types = ["duplicate_invoice", "fx_conversion", "period_cutoff"]
        for k, inv in enumerate(self.inv):
            fs.append(_mk(types[k % 3], [inv["invoice_id"]], inv["total_minor"], base,
                          [inv["doc_id"]] + all_docs[:9], "flag everything"))
        for p in self.pay:
            if p["invoice_refs"]:
                fs.append(_mk("payment_allocation", [p["payment_id"], p["invoice_refs"][0]["invoice_id"]],
                              p["amount_minor"], base, [p["doc_id"]] + all_docs[:9], "flag everything"))
        fs = fs[:20] + [f for f in fs if f["type"] == "payment_allocation"][:10]
        return {"tool": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": fs[:30]}}}


class RandomExplorer(Policy):
    name = "random_explorer"

    def __init__(self, seed: int = 12345):
        self.seed = seed

    def reset(self, obs, info):
        super().reset(obs, info)
        self.rng = random.Random(f"{self.seed}|{self.episode_id}")
        self.budget = self.rng.randint(5, 40)
        self.ids = defaultdict(list)
        self.docs = []

    def act(self, obs):
        res = obs.get("result") or {}
        for r in res.get("records", []) + ([res["record"]] if "record" in res else []):
            for k in ("row_id", "invoice_id", "payment_id"):
                if r.get(k):
                    self.ids[k].append(r[k])
            if r.get("doc_id"):
                self.docs.append(r["doc_id"])
        self.budget -= 1
        R = self.rng
        if self.budget > 0 and obs["steps_remaining"] > 1:
            c = R.random()
            if c < 0.45 or not self.docs:
                t = R.choice(["ledger", "invoices", "payments", "fx"])
                return {"tool": "list_records", "args": {"table": t, "cursor": f"{t}@{R.randint(0, 3) * 50}",
                                                          "limit": R.randint(1, 50)}}
            if c < 0.65:
                return {"tool": "get_document", "args": {"doc_id": R.choice(self.docs)}}
            if c < 0.8 and self.ids["row_id"]:
                return {"tool": "get_record", "args": {"table": "ledger", "id": R.choice(self.ids["row_id"])}}
            if c < 0.9:
                return {"tool": "calculator", "args": {"expression": f"{R.randint(1, 999)} * {R.random():.4f}"}}
            return {"tool": "get_record", "args": {"table": "invoices", "id": "INV-DOESNOTEXIST"}}
        fs = []
        base = self.obs0["manifest"]["base_currency"]
        for _ in range(R.randint(0, 3)):
            t = R.choice(["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"])
            ids = self.ids["invoice_id"] or ["INV-UNKNOWN"]
            fs.append(_mk(t, [R.choice(ids)], R.randint(1, 10 ** 6), base, R.sample(self.docs, min(2, len(self.docs))),
                          "random guess"))
        return {"tool": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": fs}}}


class Oracle(Policy):
    """Только для оценщика: получает эталон от харнесса, не из наблюдений."""
    name = "oracle"

    def __init__(self, truth_provider):
        self.truth_provider = truth_provider

    def act(self, obs):
        truth = self.truth_provider(self.episode_id)
        fs = [_mk(t["type"], t["affected_ids"], t["expected_amount_minor"], t["currency"], t["evidence_sets"][0],
                  "oracle") for t in truth["issues"]]
        return {"tool": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": fs}}}


def _mk(t, aff, amt, cur, ev, ex):
    return {"type": t, "affected_ids": list(aff), "claimed_amount_minor": int(amt), "currency": cur,
            "evidence_ids": list(ev)[:10], "explanation": ex}


BASELINES = ("random_explorer", "naive_reconciler", "flag_all", "oracle")


def make_policy(name: str, truth_provider=None) -> Policy:
    if name == "random_explorer":
        return RandomExplorer()
    if name == "naive_reconciler":
        return NaiveReconciler()
    if name == "flag_all":
        return FlagAll()
    if name == "oracle":
        return Oracle(truth_provider)
    if name == "scripted_auditor":
        return ScriptedAuditor()
    raise KeyError(name)
