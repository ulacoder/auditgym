#!/usr/bin/env python3
"""Independent verifier for exported AuditGym episodes.

Written from the case text (docs/CASE_FINTECH_RU.md, sections 03/05/10) and the
files on disk only.  It does NOT import or read anything from the `auditgym`
package.  Stdlib only, Python 3.11+.

For every exported episode (episodes/<split>/<name>, any split; --root DIR for another tree) it:
  1. parses the public files strictly (schema check);
  2. checks accounting equalities (global, per journal, invoice lines, payment refs,
     referential integrity);
  3. recomputes every truth issue from public records (type, affected ids,
     expected amount, evidence sufficiency);
  4. runs its own full audit and diffs it against truth (undeclared / unreproducible);
  5. reconstructs the pre-error ledger from provenance.json and checks that it is
     balanced and clean (accounting equalities *before* the injected errors);
  6. runs realism sanity checks.

Usage (from repo root):  python3 independent_check/verify.py [-v] [episode_name ...]
Exit status is non-zero if any episode FAILs.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, getcontext

getcontext().prec = 60

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "--root" in sys.argv:  # alternate tree with episodes/ and grader_data/ (used by self-test)
    _i = sys.argv.index("--root")
    REPO = os.path.abspath(sys.argv[_i + 1])
    del sys.argv[_i:_i + 2]

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?[+-]\d{2}:\d{2}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
INT_RE = re.compile(r"^(0|[1-9]\d*)$")
SINT_RE = re.compile(r"^-?(0|[1-9]\d*)$")
RATE_RE = re.compile(r"^\d+(\.\d+)?$")
CUR_RE = re.compile(r"^[A-Z]{3}$")
TZ_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")

LEDGER_HEADER = ["row_id", "timestamp", "account", "debit_minor", "credit_minor",
                 "currency", "entity_id", "invoice_id", "payment_id"]
FX_HEADER = ["date", "source_currency", "target_currency", "rate"]
INVOICE_KEYS = {"doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency",
                "total_minor", "line_items"}
PAYMENT_KEYS = {"doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"}
REF_KEYS = {"invoice_id", "amount_minor"}
MANIFEST_REQUIRED = {"schema_version", "episode_id", "base_currency", "period_start",
                     "period_end", "timezone", "counts", "rounding", "currency_exponents"}
PUBLIC_FILES = {"manifest.json", "policy.json", "ledger.csv", "invoices.jsonl",
                "payments.jsonl", "fx.csv"}
ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")
EXPENSE_ACCOUNTS_SUFFIX = "_expense"
LEAK_WORDS = re.compile(r"(ISS-\d|duplicat|backdat|wrong|inject|truth|oracle|mistake|error|copy of)",
                        re.I)

# Rough USD value of one unit of each currency (2025-2027 plausible band centre).
USD_VALUE = {"USD": 1.0, "EUR": 1.12, "GBP": 1.30, "CHF": 1.18, "JPY": 0.0067,
             "CNY": 0.139, "KWD": 3.26, "KZT": 0.0019}


# ----------------------------------------------------------------------------- helpers
class Report:
    def __init__(self, name):
        self.name = name
        self.msgs = defaultdict(list)  # section -> [(level, text)]

    def add(self, section, level, text):
        self.msgs[section].append((level, text))

    def fail(self, section, text):
        self.add(section, "FAIL", text)

    def warn(self, section, text):
        self.add(section, "WARN", text)

    def info(self, section, text):
        self.add(section, "INFO", text)

    def status(self, section):
        lv = [l for l, _ in self.msgs.get(section, [])]
        if "FAIL" in lv:
            return "FAIL"
        if "WARN" in lv:
            return "warn"
        return "ok"

    @property
    def failed(self):
        return any(l == "FAIL" for v in self.msgs.values() for l, _ in v)


def parse_iso(s):
    if not isinstance(s, str) or not ISO_RE.match(s):
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def parse_tz(s):
    m = TZ_RE.match(s or "")
    if not m:
        return None
    sign = 1 if m.group(1) == "+" else -1
    return timezone(sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3))))


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def half_up(d: Decimal) -> int:
    return int(d.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def fmt(n):
    return f"{n:,}".replace(",", "_") if isinstance(n, int) else str(n)


# ----------------------------------------------------------------------------- episode
class Episode:
    def __init__(self, split, name):
        self.split, self.name = split, name
        self.pub = os.path.join(REPO, "episodes", split, name)
        self.hid = os.path.join(REPO, "grader_data", split, name)
        self.r = Report(f"{split}/{name}")
        self.ok_parse = True

    # ------------------------------------------------------------------ 1. schema
    def load(self):
        S = "schema"
        r = self.r
        files = set(os.listdir(self.pub))
        if files != PUBLIC_FILES:
            r.fail(S, f"public dir files {sorted(files)} != expected {sorted(PUBLIC_FILES)}")
        with open(os.path.join(self.pub, "manifest.json"), encoding="utf-8") as f:
            self.manifest = m = json.load(f)
        with open(os.path.join(self.pub, "policy.json"), encoding="utf-8") as f:
            self.policy = json.load(f)
        miss = MANIFEST_REQUIRED - set(m)
        if miss:
            r.fail(S, f"manifest missing keys {sorted(miss)}")
        self.base = m.get("base_currency")
        self.exps = m.get("currency_exponents", {})
        if self.base not in self.exps:
            r.fail(S, f"base currency {self.base} has no exponent")
        for c, e in self.exps.items():
            if not CUR_RE.match(c) or not is_int(e) or not 0 <= e <= 4:
                r.fail(S, f"bad currency exponent {c}={e}")
        self.tz = parse_tz(m.get("timezone"))
        if self.tz is None:
            r.fail(S, f"manifest.timezone {m.get('timezone')!r} is not a fixed ±HH:MM offset")
            self.tz = timezone.utc
        self.ps = parse_iso(m.get("period_start"))
        self.pe = parse_iso(m.get("period_end"))
        if not self.ps or not self.pe or self.ps >= self.pe:
            r.fail(S, "bad period_start/period_end")
            self.ok_parse = False
            return
        if m.get("period_end_exclusive") is not True:
            r.warn(S, "manifest.period_end_exclusive is not true")
        rnd = m.get("rounding", {})
        if rnd.get("mode") != "ROUND_HALF_UP":
            r.fail(S, f"unexpected rounding mode {rnd}")
        pol_txt = json.dumps(self.policy)
        for needle in ("latest earlier date", "ROUND_HALF_UP", "unapplied_cash", "fx_gain_loss",
                       "issued_at"):
            if needle not in pol_txt:
                r.warn(S, f"policy.json does not mention {needle!r}")
        self.accounts = set(self.policy.get("accounts", []))

        # ---- ledger
        self.ledger = self.parse_ledger(os.path.join(self.pub, "ledger.csv"))

        # ---- invoices
        self.invoices = {}
        self.inv_list = []
        with open(os.path.join(self.pub, "invoices.jsonl"), encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                if not line.strip():
                    r.fail(S, f"invoices.jsonl line {ln} empty")
                    continue
                o = json.loads(line)
                if set(o) != INVOICE_KEYS:
                    r.fail(S, f"invoice line {ln}: keys {sorted(set(o) ^ INVOICE_KEYS)} differ")
                bad = False
                for k in ("doc_id", "invoice_id", "entity_id", "currency"):
                    if not isinstance(o.get(k), str) or not o.get(k):
                        r.fail(S, f"invoice line {ln}: {k}={o.get(k)!r}")
                        bad = True
                ia = parse_iso(o.get("issued_at"))
                if ia is None:
                    r.fail(S, f"invoice {o.get('invoice_id')}: bad issued_at {o.get('issued_at')!r}")
                    bad = True
                da = None
                if o.get("due_at") is not None:
                    da = parse_iso(o.get("due_at"))
                    if da is None:
                        r.fail(S, f"invoice {o.get('invoice_id')}: bad due_at {o.get('due_at')!r}")
                if o.get("currency") not in self.exps:
                    r.fail(S, f"invoice {o.get('invoice_id')}: currency {o.get('currency')} not in manifest")
                    bad = True
                if not is_int(o.get("total_minor")):
                    r.fail(S, f"invoice {o.get('invoice_id')}: total_minor not int")
                    bad = True
                li = o.get("line_items")
                if not isinstance(li, list) or not li:
                    r.fail(S, f"invoice {o.get('invoice_id')}: line_items empty/not list")
                    li = []
                for it in li:
                    for k in ("quantity", "unit_price_minor", "amount_minor", "line_no"):
                        if not is_int(it.get(k)):
                            r.fail(S, f"invoice {o.get('invoice_id')}: line {k} not int")
                if bad:
                    continue
                o["_ia"], o["_da"] = ia, da
                if o["invoice_id"] in self.invoices:
                    r.fail(S, f"duplicate invoice_id key {o['invoice_id']}")
                self.invoices[o["invoice_id"]] = o
                self.inv_list.append(o)

        # ---- payments
        self.payments = {}
        with open(os.path.join(self.pub, "payments.jsonl"), encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                if not line.strip():
                    r.fail(S, f"payments.jsonl line {ln} empty")
                    continue
                o = json.loads(line)
                if set(o) != PAYMENT_KEYS:
                    r.fail(S, f"payment line {ln}: keys {sorted(set(o) ^ PAYMENT_KEYS)} differ")
                    continue
                ts = parse_iso(o["timestamp"])
                if ts is None:
                    r.fail(S, f"payment {o['payment_id']}: bad timestamp {o['timestamp']!r}")
                    continue
                if o["currency"] not in self.exps:
                    r.fail(S, f"payment {o['payment_id']}: currency {o['currency']} not in manifest")
                    continue
                if not is_int(o["amount_minor"]) or o["amount_minor"] == 0:
                    r.fail(S, f"payment {o['payment_id']}: amount_minor {o['amount_minor']!r}")
                    continue
                refs = o["invoice_refs"]
                if refs is not None:
                    if not isinstance(refs, list) or not refs:
                        r.fail(S, f"payment {o['payment_id']}: invoice_refs must be null or non-empty list, got {refs!r}")
                        continue
                    okr = True
                    for ref in refs:
                        if set(ref) != REF_KEYS or not isinstance(ref["invoice_id"], str) \
                                or not is_int(ref["amount_minor"]) or ref["amount_minor"] <= 0:
                            r.fail(S, f"payment {o['payment_id']}: bad ref {ref!r}")
                            okr = False
                    if not okr:
                        continue
                o["_ts"] = ts
                if o["payment_id"] in self.payments:
                    r.fail(S, f"duplicate payment_id key {o['payment_id']}")
                self.payments[o["payment_id"]] = o

        # ---- fx
        self.fx = defaultdict(dict)  # cur -> {date: Decimal}
        self.fx_raw_count = 0
        with open(os.path.join(self.pub, "fx.csv"), newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            header = next(rd)
            if header != FX_HEADER:
                r.fail(S, f"fx header {header}")
            for ln, raw in enumerate(rd, start=2):
                self.fx_raw_count += 1
                if len(raw) != 4:
                    r.fail(S, f"fx line {ln}: {len(raw)} fields")
                    continue
                d, src, tgt, rate = raw
                if not DATE_RE.match(d) or not RATE_RE.match(rate) or Decimal(rate) <= 0:
                    r.fail(S, f"fx line {ln}: bad date/rate {raw}")
                    continue
                if tgt != self.base or src not in self.exps or src == self.base:
                    r.fail(S, f"fx line {ln}: pair {src}->{tgt} not foreign->base")
                    continue
                dd = date.fromisoformat(d)
                if dd in self.fx[src]:
                    r.fail(S, f"fx line {ln}: duplicate rate {src} {d}")
                self.fx[src][dd] = Decimal(rate)
        self.fx_dates = {c: sorted(v) for c, v in self.fx.items()}
        # public fx ids: 'DATE:SRC:TGT' (one per fx.csv row)
        self.fx_keys = {f"{d.isoformat()}:{c}:{self.base}": (c, d) for c, v in self.fx.items() for d in v}

        # ---- id uniqueness / counts
        docs = [i["doc_id"] for i in self.inv_list] + [p["doc_id"] for p in self.payments.values()]
        for k, v in Counter(docs).items():
            if v > 1:
                r.fail(S, f"doc_id {k} used {v} times")
        for k, v in Counter(x["row_id"] for x in self.ledger).items():
            if v > 1:
                r.fail(S, f"row_id {k} used {v} times")
        self.doc_ids = set(docs)
        self.row_ids = {x["row_id"] for x in self.ledger}
        self.doc_of = {i["doc_id"]: ("inv", i["invoice_id"]) for i in self.inv_list}
        self.doc_of.update({p["doc_id"]: ("pay", p["payment_id"]) for p in self.payments.values()})
        self.row_by_id = {x["row_id"]: x for x in self.ledger}
        cnt = m.get("counts", {})
        actual = {"ledger_rows": len(self.ledger), "invoices": len(self.inv_list),
                  "payments": len(self.payments), "fx_rates": self.fx_raw_count,
                  "documents": len(docs)}
        for k, v in actual.items():
            if k in cnt and cnt[k] != v:
                r.fail(S, f"manifest.counts.{k}={cnt[k]} but file has {v}")
        # case-suggested size
        if not 200 <= len(self.ledger) <= 2000:
            r.warn(S, f"ledger has {len(self.ledger)} rows (case suggests 200-2000)")
        if not 20 <= len(docs) <= 200:
            r.warn(S, f"{len(docs)} documents (case suggests 20-200)")
        # leak scan over record payloads
        for fn in ("ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv"):
            with open(os.path.join(self.pub, fn), encoding="utf-8") as f:
                txt = f.read()
            for mm in set(LEAK_WORDS.findall(txt)):
                r.fail(S, f"possible ground-truth leak: {fn} contains {mm!r}")

    def parse_ledger(self, path, S="schema"):
        """Strictly parse a ledger CSV; deviations reported under section S."""
        r = self.r
        out = []
        with open(path, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            header = next(rd)
            if header != LEDGER_HEADER:
                r.fail(S, f"ledger header {header}")
            for ln, raw in enumerate(rd, start=2):
                if len(raw) != len(LEDGER_HEADER):
                    r.fail(S, f"ledger line {ln}: {len(raw)} fields")
                    continue
                d = dict(zip(LEDGER_HEADER, raw))
                for k, v in d.items():
                    if v != v.strip() or v.lower() in ("null", "none", "nan"):
                        r.fail(S, f"ledger {d['row_id']}: field {k}={v!r} (null must be empty)")
                ts = parse_iso(d["timestamp"])
                if ts is None:
                    r.fail(S, f"ledger {d['row_id']}: bad timestamp {d['timestamp']!r}")
                    continue
                if not INT_RE.match(d["debit_minor"]) or not INT_RE.match(d["credit_minor"]):
                    r.fail(S, f"ledger {d['row_id']}: non-integer/negative amounts")
                    continue
                dr, cr = int(d["debit_minor"]), int(d["credit_minor"])
                if (dr > 0) == (cr > 0):
                    r.fail(S, f"ledger {d['row_id']}: not one-sided (debit={dr}, credit={cr})")
                if d["currency"] != self.base:
                    r.fail(S, f"ledger {d['row_id']}: currency {d['currency']} != base")
                if d["account"] not in self.accounts:
                    r.fail(S, f"ledger {d['row_id']}: unknown account {d['account']}")
                if not d["row_id"] or not d["entity_id"]:
                    r.fail(S, f"ledger line {ln}: empty row_id/entity_id")
                out.append({
                    "row_id": d["row_id"], "ts": ts, "ts_raw": d["timestamp"],
                    "account": d["account"], "dr": dr, "cr": cr,
                    "entity": d["entity_id"] or None, "inv": d["invoice_id"] or None,
                    "pay": d["payment_id"] or None})
        return out

    # ------------------------------------------------------------------ fx helpers
    def local_date(self, dt):
        return dt.astimezone(self.tz).date()

    def rate(self, cur, d):
        if cur == self.base:
            return Decimal(1), d
        ds = self.fx_dates.get(cur, [])
        best = None
        for x in ds:
            if x <= d:
                best = x
            else:
                break
        if best is None:
            return None, None
        return self.fx[cur][best], best

    def convert(self, amount, cur, d):
        rt, used = self.rate(cur, d)
        if rt is None:
            return None
        scale = Decimal(10) ** (self.exps[self.base] - self.exps[cur])
        return half_up(Decimal(amount) * rt * scale)

    def in_period(self, dt):
        return self.ps <= dt < self.pe

    # ------------------------------------------------------------------ 2. equalities
    def equalities(self, ledger, section="equalities", quiet=False):
        r = self.r
        bad = []
        td = sum(x["dr"] for x in ledger)
        tc = sum(x["cr"] for x in ledger)
        if td != tc:
            bad.append(f"total debits {fmt(td)} != total credits {fmt(tc)} (diff {fmt(td - tc)})")
        journals = defaultdict(list)
        for x in ledger:
            if x["pay"]:
                k = ("payment", x["pay"])
            elif x["inv"]:
                k = ("invoice", x["inv"])
            else:
                k = ("ts+entity", x["ts"].isoformat(), x["entity"])
            journals[k].append(x)
        for k, rows in journals.items():
            d = sum(x["dr"] for x in rows) - sum(x["cr"] for x in rows)
            if d:
                bad.append(f"journal {k} unbalanced by {fmt(d)} ({[x['row_id'] for x in rows]})")
            if len({x["ts"] for x in rows}) > 1:
                bad.append(f"journal {k} rows have different timestamps")
            if len({x["entity"] for x in rows}) > 1:
                bad.append(f"journal {k} rows have different entity_id")
        if quiet:
            return bad, journals
        for b in bad:
            r.fail(section, b)
        # documents
        for inv in self.inv_list:
            s = sum(it["amount_minor"] for it in inv["line_items"])
            if s != inv["total_minor"]:
                r.fail(section, f"invoice {inv['invoice_id']} total {fmt(inv['total_minor'])} != sum lines {fmt(s)}")
            for it in inv["line_items"]:
                if it["quantity"] * it["unit_price_minor"] != it["amount_minor"]:
                    r.fail(section, f"invoice {inv['invoice_id']} line {it['line_no']}: qty*price != amount")
            if [it["line_no"] for it in inv["line_items"]] != list(range(1, len(inv["line_items"]) + 1)):
                r.warn(section, f"invoice {inv['invoice_id']} line_no not 1..n")
        for p in self.payments.values():
            if p["invoice_refs"] is not None:
                s = sum(x["amount_minor"] for x in p["invoice_refs"])
                if s != p["amount_minor"]:
                    r.fail(section, f"payment {p['payment_id']} refs sum {fmt(s)} != amount {fmt(p['amount_minor'])}")
                for ref in p["invoice_refs"]:
                    inv = self.invoices.get(ref["invoice_id"])
                    if inv is None:
                        r.fail(section, f"payment {p['payment_id']} references missing invoice {ref['invoice_id']}")
                    elif inv["currency"] != p["currency"]:
                        r.fail(section, f"payment {p['payment_id']} ({p['currency']}) pays invoice {inv['invoice_id']} in {inv['currency']}")
                if len({x["invoice_id"] for x in p["invoice_refs"]}) != len(p["invoice_refs"]):
                    r.warn(section, f"payment {p['payment_id']} references same invoice twice")
        for x in ledger:
            if x["inv"] and x["inv"] not in self.invoices:
                r.fail(section, f"ledger {x['row_id']} references missing invoice {x['inv']}")
            if x["pay"] and x["pay"] not in self.payments:
                r.fail(section, f"ledger {x['row_id']} references missing payment {x['pay']}")
            if x["inv"] and x["inv"] in self.invoices and x["entity"] != self.invoices[x["inv"]]["entity_id"]:
                r.fail(section, f"ledger {x['row_id']} entity {x['entity']} != invoice entity")
            if x["inv"] and x["pay"] and x["pay"] in self.payments:
                refs = self.payments[x["pay"]]["invoice_refs"] or []
                if x["inv"] not in {q["invoice_id"] for q in refs}:
                    r.fail(section, f"ledger {x['row_id']}: invoice {x['inv']} not in refs of {x['pay']}")
        return bad, journals

    # ------------------------------------------------------------------ 4. audit
    def audit(self, ledger, section="audit", record=True):
        """Return dict {(type, tuple(affected)): finding}. Anomalies outside the 4
        types are reported as type 'other'."""
        F = {}

        def add(typ, ids, amount, note, evidence=()):
            key = (typ, tuple(ids))
            if key in F:
                F[key]["amount"] += amount
                F[key]["note"] += "; " + note
            else:
                F[key] = {"type": typ, "ids": tuple(ids), "amount": amount, "note": note,
                          "evidence": list(evidence)}

        by_inv = defaultdict(list)   # invoice posting rows (no payment_id)
        by_pay = defaultdict(list)
        other_rows = []
        for x in ledger:
            if x["pay"]:
                by_pay[x["pay"]].append(x)
            elif x["inv"]:
                by_inv[x["inv"]].append(x)
            else:
                other_rows.append(x)
        self._by_inv, self._by_pay = by_inv, by_pay

        # every ledger row must lie within the period
        for x in ledger:
            if not self.in_period(x["ts"]):
                add("other", (x["row_id"],), 0, f"ledger row {x['row_id']} timestamp {x['ts_raw']} outside period")
            if x["ts"].utcoffset() != self.tz.utcoffset(None):
                add("other", (x["row_id"],), 0, f"ledger row {x['row_id']} not in company tz offset")

        # ---- expenses / other journals
        exp_j = defaultdict(list)
        for x in other_rows:
            exp_j[(x["ts"], x["entity"])].append(x)
        for k, rows in exp_j.items():
            accs = sorted((x["account"], "D" if x["dr"] else "C") for x in rows)
            if len(rows) != 2 or not (accs[0][0] == "cash" and accs[0][1] == "C"
                                      and accs[1][0].endswith(EXPENSE_ACCOUNTS_SUFFIX) and accs[1][1] == "D"):
                add("other", tuple(x["row_id"] for x in rows), 0, f"unexpected non-document journal shape {accs}")

        # ---- invoice postings
        posted = {}
        for iid, rows in by_inv.items():
            ar = [x for x in rows if x["account"] == "accounts_receivable"]
            rv = [x for x in rows if x["account"] == "revenue"]
            if len(rows) != 2 or len(ar) != 1 or len(rv) != 1 or ar[0]["dr"] == 0 or rv[0]["cr"] == 0 \
                    or ar[0]["dr"] != rv[0]["cr"]:
                add("other", (iid,), 0, f"invoice {iid} posting shape unexpected: "
                    f"{[(x['account'], x['dr'], x['cr']) for x in rows]}")
                continue
            posted[iid] = (ar[0]["dr"], ar[0], rv[0])
        self._posted = posted

        for inv in self.inv_list:
            iid = inv["invoice_id"]
            ia = inv["_ia"]
            if iid not in posted:
                if self.in_period(ia):
                    add("other", (iid,), inv["total_minor"], f"in-period invoice {iid} not posted")
                continue
            amt, ar, rv = posted[iid]
            if not self.in_period(ia):
                add("period_cutoff", (iid,), amt,
                    f"issued_at {inv['issued_at']} outside period but posted at {ar['ts_raw']}",
                    [inv["doc_id"], ar["row_id"]])
                continue
            exp = self.convert(inv["total_minor"], inv["currency"], self.local_date(ia))
            if exp is None:
                add("other", (iid,), 0, f"no fx rate available for invoice {iid}")
                continue
            if amt != exp:
                add("fx_conversion", (iid,), abs(amt - exp),
                    f"posted {fmt(amt)} vs policy {fmt(exp)} (rate date {self.local_date(ia)})",
                    [inv["doc_id"], ar["row_id"]])
            if ar["ts"] != ia:
                add("other", (iid,), 0, f"invoice {iid} posting timestamp {ar['ts_raw']} != issued_at {inv['issued_at']} (same period)")

        # ---- duplicates (strict content key)
        groups = defaultdict(list)
        loose = defaultdict(list)
        for inv in self.inv_list:
            key = (inv["entity_id"], inv["currency"], inv["total_minor"],
                   tuple((it["sku"], it["description"], it["quantity"], it["unit_price_minor"],
                          it["amount_minor"]) for it in inv["line_items"]))
            groups[key].append(inv)
            loose[(inv["entity_id"], inv["currency"], inv["total_minor"])].append(inv)
        self._dup_groups = {}
        for key, invs in groups.items():
            if len(invs) < 2:
                continue
            invs.sort(key=lambda i: i["_ia"])
            orig = invs[0]
            for dup in invs[1:]:
                self._dup_groups[dup["invoice_id"]] = orig
                if dup["invoice_id"] in posted and self.in_period(dup["_ia"]):
                    amt, ar, rv = posted[dup["invoice_id"]]
                    add("duplicate_invoice", (dup["invoice_id"],), amt,
                        f"same content as {orig['invoice_id']} (issued {orig['issued_at']}), later copy {dup['issued_at']} posted",
                        [orig["doc_id"], dup["doc_id"], ar["row_id"]])
                else:
                    add("other", (dup["invoice_id"],), 0, f"content duplicate of {orig['invoice_id']} exists but is not posted in period")
        self._loose_same_amount = sum(1 for v in loose.values() if len(v) > 1) - \
            sum(1 for v in groups.values() if len(v) > 1)

        # ---- payments
        refs_by_inv = defaultdict(int)
        for pid, p in self.payments.items():
            rows = by_pay.get(pid, [])
            ts = p["_ts"]
            if not rows:
                if self.in_period(ts):
                    add("other", (pid,), 0, f"in-period payment {pid} not posted")
                continue
            if not self.in_period(ts):
                add("other", (pid,), 0, f"payment {pid} timestamp {p['timestamp']} outside period but posted")
            if any(x["ts"] != ts for x in rows):
                add("other", (pid,), 0, f"payment {pid} ledger timestamp differs from document instant")
            cash = [x for x in rows if x["account"] == "cash"]
            cash_net = sum(x["dr"] - x["cr"] for x in cash)
            unap = sum(x["cr"] - x["dr"] for x in rows if x["account"] == "unapplied_cash")
            fxgl = [x for x in rows if x["account"] == "fx_gain_loss"]
            unexpected = [x for x in rows if x["account"] not in
                          ("cash", "accounts_receivable", "unapplied_cash", "fx_gain_loss")]
            if unexpected or len(cash) != 1:
                add("other", (pid,), 0, f"payment {pid} has unexpected rows/cash count")
            exp_cash = self.convert(p["amount_minor"], p["currency"], self.local_date(ts))
            if exp_cash is None:
                add("other", (pid,), 0, f"no fx rate for payment {pid}")
                continue
            if cash_net != exp_cash:
                add("fx_conversion", (pid,), abs(cash_net - exp_cash),
                    f"cash {fmt(cash_net)} vs policy {fmt(exp_cash)} (rate date {self.local_date(ts)})",
                    [p["doc_id"]] + [x["row_id"] for x in cash])
            if p["invoice_refs"] is None:
                if any(x["account"] == "accounts_receivable" for x in rows):
                    add("other", (pid,), 0, f"unreferenced payment {pid} credits AR")
                if unap != cash_net:
                    add("other", (pid,), 0, f"unreferenced payment {pid}: unapplied {fmt(unap)} != cash {fmt(cash_net)}")
                if fxgl:
                    add("other", (pid,), 0, f"unreferenced payment {pid} has fx_gain_loss rows")
                continue
            if p["amount_minor"] < 0:
                add("other", (pid,), 0, f"negative payment {pid} with invoice refs")
            ar_by = defaultdict(int)
            ar_rows_by = defaultdict(list)
            for x in rows:
                if x["account"] == "accounts_receivable":
                    ar_by[x["inv"]] += x["cr"] - x["dr"]
                    ar_rows_by[x["inv"]].append(x["row_id"])
            ref_amt = defaultdict(int)
            for ref in p["invoice_refs"]:
                ref_amt[ref["invoice_id"]] += ref["amount_minor"]
                refs_by_inv[ref["invoice_id"]] += ref["amount_minor"]
            shortfall_total = 0
            exp_ar_total = 0
            for iid, ra in ref_amt.items():
                inv = self.invoices.get(iid)
                if inv is None:
                    continue
                if inv["_ia"] > ts:
                    add("other", (pid, iid), 0, f"payment {pid} ({p['timestamp']}) precedes invoice {iid} issue ({inv['issued_at']})")
                exp_ar = self.convert(ra, inv["currency"], self.local_date(inv["_ia"]))
                exp_ar_total += exp_ar
                got = ar_by.get(iid, 0)
                if got < exp_ar:
                    shortfall_total += exp_ar - got
                    unap_rows = [x["row_id"] for x in rows if x["account"] == "unapplied_cash"]
                    add("payment_allocation", (pid, iid), exp_ar - got,
                        f"AR credit {fmt(got)} vs expected {fmt(exp_ar)}" + (" (missing)" if iid not in ar_by else ""),
                        [p["doc_id"]] + ar_rows_by.get(iid, []) + unap_rows)
                elif got > exp_ar:
                    add("other", (pid, iid), got - exp_ar, f"AR over-credited by {fmt(got - exp_ar)}")
            for iid in ar_by:
                if iid not in ref_amt:
                    add("other", (pid, iid), 0, f"AR credit for unreferenced invoice {iid}")
            if unap != shortfall_total:
                add("other", (pid,), abs(unap - shortfall_total),
                    f"unapplied_cash {fmt(unap)} != allocation shortfall {fmt(shortfall_total)}")
            # fx gain/loss must be the residual at policy cash vs policy AR
            fx_net = sum(x["cr"] - x["dr"] for x in fxgl)
            if p["currency"] == self.base and fxgl:
                add("other", (pid,), 0, f"base-currency payment {pid} has fx_gain_loss rows")
            if len(fxgl) > 1:
                add("other", (pid,), 0, f"payment {pid} has {len(fxgl)} fx_gain_loss rows")
        # over-payment of invoices across all payments
        for iid, s in refs_by_inv.items():
            inv = self.invoices.get(iid)
            if inv and s > inv["total_minor"]:
                add("other", (iid,), s - inv["total_minor"], f"invoice {iid} paid {fmt(s)} > total {fmt(inv['total_minor'])}")
        return F

    # ------------------------------------------------------------------ 3. truth
    def check_truth(self, mine):
        T = "truth"
        r = self.r
        with open(os.path.join(self.hid, "truth.json"), encoding="utf-8") as f:
            self.truth = truth = json.load(f)
        if truth.get("episode_id") != self.manifest.get("episode_id"):
            r.fail(T, "truth.episode_id != manifest.episode_id")
        issues = truth.get("issues", [])
        if len(issues) > 12:
            r.warn(T, f"{len(issues)} issues (case suggests 0-12)")
        if len({i['issue_id'] for i in issues}) != len(issues):
            r.fail(T, "issue_id not unique")
        tkeys = set()
        for iss in issues:
            iid, typ, ids = iss.get("issue_id"), iss.get("type"), tuple(iss.get("affected_ids", []))
            amt = iss.get("expected_amount_minor")
            if typ not in ISSUE_TYPES:
                r.fail(T, f"{iid}: unknown type {typ}")
                continue
            if iss.get("currency") != self.base:
                r.fail(T, f"{iid}: currency {iss.get('currency')} != base {self.base}")
            if not is_int(amt) or amt <= 0:
                r.fail(T, f"{iid}: expected_amount_minor {amt!r} not positive int")
            # reporting conventions
            conv_ok = True
            if typ in ("duplicate_invoice", "period_cutoff"):
                conv_ok = len(ids) == 1 and ids[0] in self.invoices
            elif typ == "payment_allocation":
                conv_ok = len(ids) == 2 and ids[0] in self.payments and ids[1] in self.invoices
            elif typ == "fx_conversion":
                conv_ok = len(ids) == 1 and (ids[0] in self.invoices or ids[0] in self.payments)
            if not conv_ok:
                r.fail(T, f"{iid}: affected_ids {list(ids)} violate reporting convention for {typ}")
            key = (typ, ids)
            if key in tkeys:
                r.fail(T, f"{iid}: duplicate truth key {key}")
            tkeys.add(key)
            f = mine.get(key)
            if f is None:
                r.fail(T, f"{iid} {typ} {list(ids)} amount {fmt(amt)}: NOT reproducible from public records")
            elif f["amount"] != amt:
                r.fail(T, f"{iid} {typ} {list(ids)}: truth amount {fmt(amt)} != recomputed {fmt(f['amount'])} ({f['note']})")
            # evidence
            es = iss.get("evidence_sets")
            if not isinstance(es, list) or not es:
                r.fail(T, f"{iid}: no evidence_sets")
                continue
            any_ok = False
            for ev in es:
                if not isinstance(ev, list) or not ev or len(ev) > 10:
                    r.fail(T, f"{iid}: evidence set {ev!r} empty or >10 ids")
                    continue
                missing = [e for e in ev if e not in self.doc_ids and e not in self.row_ids
                           and e not in self.fx_keys]
                if missing:
                    r.fail(T, f"{iid}: evidence ids not in public records: {missing}")
                    continue
                ok, why, warns = self.evidence_sufficient(typ, ids, amt, ev)
                for w in warns:
                    r.warn("evidence", f"{iid} {typ} {list(ids)} set {ev}: {w}")
                if ok:
                    any_ok = True
                else:
                    r.fail("evidence", f"{iid} {typ} {list(ids)} set {ev}: insufficient - {why}")
            if not any_ok:
                r.fail("evidence", f"{iid}: no evidence set independently demonstrates the issue")
            # interaction annotation consistency
            if iss.get("interaction"):
                partners = [o for o in issues if o is not iss and o.get("interaction") == iss["interaction"]
                            and set(o.get("affected_ids", [])) & set(ids)]
                if not partners:
                    r.warn(T, f"{iid}: interaction {iss['interaction']} but no partner issue sharing an id")
        # diff with my audit
        for key, f in sorted(mine.items()):
            if f["type"] == "other":
                continue
            if key not in tkeys:
                r.fail("audit", f"UNDECLARED {f['type']} {list(f['ids'])} amount {fmt(f['amount'])}: {f['note']}")
        for key, f in sorted(mine.items()):
            if f["type"] == "other":
                lvl = "warn" if f["amount"] == 0 and ("not posted in period" in f["note"] or
                                                       "same period" in f["note"]) else "fail"
                getattr(r, lvl)("audit", f"anomaly {list(f['ids'])}: {f['note']}")
        return issues

    def policy_fx_key(self, cur, instant):
        """fx id of the rate the policy prescribes for `instant` (None for base currency)."""
        if cur == self.base:
            return None
        _, used = self.rate(cur, self.local_date(instant))
        return None if used is None else f"{used.isoformat()}:{cur}:{self.base}"

    def convert_with_key(self, amount, cur, key):
        c, d = self.fx_keys[key]
        assert c == cur
        scale = Decimal(10) ** (self.exps[self.base] - self.exps[cur])
        return half_up(Decimal(amount) * self.fx[c][d] * scale)

    def evidence_sufficient(self, typ, ids, amt, ev):
        """Strict: the claimed (type, ids, amount) must be demonstrable from the cited
        records ALONE (doc_ids, ledger rows, fx rate rows), using policy/manifest only as
        rules.  Absence of a record (e.g. no AR credit in a 'missing' allocation) is
        accepted only if a cited record corroborates the amount."""
        warns = []
        docs = {self.doc_of[e] for e in ev if e in self.doc_of}
        rows = [self.row_by_id[e] for e in ev if e in self.row_by_id]
        fxk = [e for e in ev if e in self.fx_keys]
        inv_docs = {v for k, v in docs if k == "inv"}
        pay_docs = {v for k, v in docs if k == "pay"}

        def need_rate(cur, instant):
            """Return (key, None) if the cited fx rows contain the policy rate, else (None, why)."""
            if cur == self.base:
                return None, None
            want = self.policy_fx_key(cur, instant)
            if want in fxk:
                return want, None
            wrong = [k for k in fxk if self.fx_keys[k][0] == cur]
            if wrong:
                return None, f"cited fx row(s) {wrong} are not the policy rate {want}"
            return None, f"policy fx rate {want} not cited"

        if typ == "duplicate_invoice":
            dup = ids[0]
            if dup not in inv_docs:
                return False, "affected (later) invoice document not cited", warns
            d = self.invoices[dup]
            twins = [i for i in inv_docs if i != dup and self._same_content(self.invoices[i], d)
                     and self.invoices[i]["_ia"] < d["_ia"]]
            if not twins:
                return False, "no earlier identical invoice document cited", warns
            post = [x for x in rows if x["inv"] == dup and not x["pay"]
                    and ((x["account"] == "accounts_receivable" and x["dr"]) or (x["account"] == "revenue" and x["cr"]))]
            if not post:
                return False, "copy's AR/revenue posting row not cited (booking and amount not shown)", warns
            if (post[0]["dr"] or post[0]["cr"]) != amt:
                return False, f"cited posting {fmt(post[0]['dr'] or post[0]['cr'])} != expected {fmt(amt)}", warns
            return True, "", warns
        if typ == "period_cutoff":
            iid = ids[0]
            if iid not in inv_docs:
                return False, "invoice document not cited", warns
            d = self.invoices[iid]
            if self.in_period(d["_ia"]):
                return False, "cited document is in period", warns
            post = [x for x in rows if x["inv"] == iid and not x["pay"]]
            if not post:
                return False, "no posting row cited", warns
            x = post[0]
            if not self.in_period(x["ts"]):
                return False, "posting row not in period", warns
            if (x["dr"] or x["cr"]) != amt:
                return False, "posting row amount != expected", warns
            return True, "", warns
        if typ == "fx_conversion":
            oid = ids[0]
            if oid in self.invoices:
                if oid not in inv_docs:
                    return False, "invoice document not cited", warns
                d = self.invoices[oid]
                post = [x for x in rows if x["inv"] == oid and not x["pay"]]
                if not post:
                    return False, "no AR/revenue posting row cited", warns
                key, why = need_rate(d["currency"], d["_ia"])
                if why:
                    return False, why, warns
                exp = self.convert_with_key(d["total_minor"], d["currency"], key) if key else d["total_minor"]
                got = post[0]["dr"] or post[0]["cr"]
            else:
                if oid not in pay_docs:
                    return False, "payment document not cited", warns
                p = self.payments[oid]
                cash = [x for x in rows if x["pay"] == oid and x["account"] == "cash"]
                if not cash:
                    return False, "cash row not cited", warns
                key, why = need_rate(p["currency"], p["_ts"])
                if why:
                    return False, why, warns
                exp = self.convert_with_key(p["amount_minor"], p["currency"], key) if key else p["amount_minor"]
                got = cash[0]["dr"] - cash[0]["cr"]
            if abs(got - exp) != amt:
                return False, f"|posted {fmt(got)} - policy {fmt(exp)}| != {fmt(amt)}", warns
            return True, "", warns
        if typ == "payment_allocation":
            pid, iid = ids
            if pid not in pay_docs:
                return False, "payment document not cited", warns
            p = self.payments[pid]
            refs = defaultdict(int)
            for x in p["invoice_refs"] or []:
                refs[x["invoice_id"]] += x["amount_minor"]
            if iid not in refs:
                return False, "payment document does not reference the invoice", warns
            if iid not in inv_docs:
                return False, "invoice document not cited (needed for its currency/issue date)", warns
            inv = self.invoices[iid]
            key, why = need_rate(inv["currency"], inv["_ia"])
            if why:
                return False, why, warns
            exp_ar = self.convert_with_key(refs[iid], inv["currency"], key) if key else refs[iid]
            ar = [x for x in rows if x["pay"] == pid and x["inv"] == iid and x["account"] == "accounts_receivable"]
            un = [x for x in rows if x["pay"] == pid and x["account"] == "unapplied_cash"]
            if not un:
                return False, "payment's unapplied_cash row not cited", warns
            actually_credited = [x for x in self._by_pay.get(pid, []) if x["inv"] == iid
                                 and x["account"] == "accounts_receivable"]
            if actually_credited and not ar:
                return False, "invoice was partly credited but the AR credit row is not cited", warns
            got = sum(x["cr"] - x["dr"] for x in ar)
            if exp_ar - got != amt:
                return False, f"policy AR {fmt(exp_ar)} - cited AR credit {fmt(got)} != {fmt(amt)}", warns
            if sum(x["cr"] - x["dr"] for x in un) != amt:
                warns.append(f"unapplied_cash {fmt(sum(x['cr'] - x['dr'] for x in un))} != shortfall {fmt(amt)}")
            return True, "", warns
        return False, "unknown type", warns

    def _by_pay_ar_invoices(self, pid):
        return {x["inv"] for x in self._by_pay.get(pid, []) if x["account"] == "accounts_receivable"}

    @staticmethod
    def _same_content(a, b):
        f = lambda i: (i["entity_id"], i["currency"], i["total_minor"],
                       tuple((it["sku"], it["description"], it["quantity"], it["unit_price_minor"])
                             for it in i["line_items"]))
        return f(a) == f(b)

    # ------------------------------------------------------------------ 5. provenance
    def check_provenance(self):
        P = "pre-error"
        r = self.r
        with open(os.path.join(self.hid, "provenance.json"), encoding="utf-8") as f:
            prov = json.load(f)
        if prov.get("episode_id") != self.manifest.get("episode_id"):
            r.fail(P, "provenance.episode_id != manifest.episode_id")
        changes = prov.get("changes", [])
        t_ids = {i["issue_id"] for i in self.truth.get("issues", [])}
        c_ids = {c.get("issue_id") for c in changes}
        if t_ids != c_ids:
            r.fail(P, f"truth issues {sorted(t_ids)} vs provenance changes {sorted(c_ids)} differ")
        before, after = {}, {}
        for c in changes:
            for x in c.get("rows_before", []):
                if x["row_id"] in before and before[x["row_id"]] != x:
                    r.fail(P, f"conflicting rows_before for {x['row_id']}")
                before[x["row_id"]] = x
            for x in c.get("rows_after", []):
                after.setdefault(x["row_id"], x)
        # rows_after must describe the public (final) ledger
        for rid, x in after.items():
            pub = self.row_by_id.get(rid)
            if pub is None:
                r.fail(P, f"rows_after {rid} not in public ledger")
                continue
            if (pub["account"], pub["dr"], pub["cr"], pub["ts_raw"]) != \
                    (x["account"], x["debit_minor"], x["credit_minor"], x["timestamp"]):
                r.fail(P, f"rows_after {rid} differs from public ledger row")
        # reconstruct clean ledger
        clean = []
        for x in self.ledger:
            rid = x["row_id"]
            if rid in before:
                b = before[rid]
                y = dict(x)
                y.update(account=b["account"], dr=b["debit_minor"], cr=b["credit_minor"],
                         ts=datetime.fromisoformat(b["timestamp"]), ts_raw=b["timestamp"])
                clean.append(y)
            elif rid in after:
                continue  # row introduced by an error
            else:
                clean.append(x)
        # rows removed by an error (e.g. 'missing' allocation): restore them, taking
        # payment/invoice links from the change record and entity from the journal
        for c in changes:
            for b in c.get("rows_before", []):
                rid = b["row_id"]
                if rid in self.row_by_id:
                    continue
                pay, inv = c.get("payment"), c.get("invoice")
                if not pay:
                    r.fail(P, f"rows_before {rid} deleted but change has no payment link")
                    continue
                ents = {x["entity"] for x in self.ledger if x["pay"] == pay}
                if len(ents) != 1:
                    r.fail(P, f"cannot infer entity for deleted row {rid}")
                    continue
                clean.append({"row_id": rid, "ts": datetime.fromisoformat(b["timestamp"]),
                              "ts_raw": b["timestamp"], "account": b["account"],
                              "dr": b["debit_minor"], "cr": b["credit_minor"],
                              "entity": ents.pop(),
                              "inv": inv if b["account"] == "accounts_receivable" else None,
                              "pay": pay})
                r.info(P, f"restored deleted row {rid} ({b['account']}) for {pay}/{inv}")
        # Prefer the exported pre-error ledger; cross-check it against the provenance
        # reconstruction so both artefacts tell the same causal story.
        lb_path = os.path.join(self.hid, "ledger_before.csv")
        if os.path.exists(lb_path):
            lb = self.parse_ledger(lb_path, P)
            key = lambda x: (x["row_id"], x["ts"], x["account"], x["dr"], x["cr"], x["entity"], x["inv"], x["pay"])
            a, b = {key(x) for x in lb}, {key(x) for x in clean}
            if a != b:
                r.fail(P, f"ledger_before.csv differs from provenance reconstruction: "
                          f"{len(a - b)} rows only in ledger_before, {len(b - a)} only in reconstruction "
                          f"(e.g. {sorted(a ^ b)[:2]})")
            r.info(P, f"using ledger_before.csv ({len(lb)} rows) for pre-error checks")
            clean = lb
        else:
            r.info(P, "ledger_before.csv absent; using provenance reconstruction")
        bad, _ = self.equalities(clean, quiet=True)
        for b in bad:
            r.fail(P, f"pre-error ledger: {b}")
        saved = (self._by_inv, self._by_pay, self._posted, self._dup_groups, self._loose_same_amount)
        f = self.audit(clean, record=False)
        (self._by_inv, self._by_pay, self._posted, self._dup_groups, self._loose_same_amount) = saved
        for key, v in sorted(f.items()):
            if v["type"] == "other" and "not posted in period" in v["note"]:
                continue
            r.fail(P, f"pre-error ledger is not clean: {v['type']} {list(v['ids'])} {v['note']}")
        # distractors from provenance must not be flagged on the public data
        self._distractors = prov.get("distractors", [])
        return prov

    # ------------------------------------------------------------------ 6. realism
    def realism(self, mine):
        R = "realism"
        r = self.r
        base_usd = USD_VALUE.get(self.base)
        # FX level & volatility
        for cur, ds in self.fx_dates.items():
            vals = [self.fx[cur][d] for d in ds]
            if cur in USD_VALUE and base_usd:
                implied = USD_VALUE[cur] / base_usd
                for d, v in zip(ds, vals):
                    if abs(math.log(float(v) / implied)) > math.log(1.30):
                        r.warn(R, f"fx {cur}->{self.base} {d} = {v} far from plausible ~{implied:.6g}")
                        break
            for (d0, v0), (d1, v1) in zip(zip(ds, vals), zip(ds[1:], vals[1:])):
                ch = abs(float(v1 / v0) - 1)
                if ch > 0.03:
                    r.warn(R, f"fx {cur} jumps {ch:.1%} from {d0} to {d1}")
                if (d1 - d0).days > 4:
                    r.warn(R, f"fx {cur} gap of {(d1 - d0).days} days {d0}..{d1}")
            wk = [d for d in ds if d.weekday() >= 5]
            if wk:
                r.info(R, f"fx {cur} has weekend rates {wk[:3]}")
            # coverage: every needed lookup has a rate on/before
        need = [(i["currency"], self.local_date(i["_ia"])) for i in self.inv_list] + \
               [(p["currency"], self.local_date(p["_ts"])) for p in self.payments.values()]
        for cur, d in need:
            if cur != self.base:
                rt, used = self.rate(cur, d)
                if rt is None:
                    r.fail(R, f"no {cur} rate on or before {d}")
                elif (d - used).days > 4:
                    r.warn(R, f"{cur} rate for {d} falls back {(d - used).days} days to {used}")
        # documents
        for inv in self.inv_list:
            if inv["total_minor"] <= 0:
                r.fail(R, f"invoice {inv['invoice_id']} non-positive total")
            if inv["_da"] is not None and inv["_da"] < inv["_ia"]:
                r.fail(R, f"invoice {inv['invoice_id']} due_at before issued_at")
            if inv["_da"] is not None and (inv["_da"] - inv["_ia"]).days > 120:
                r.warn(R, f"invoice {inv['invoice_id']} payment term {(inv['_da'] - inv['_ia']).days} days")
            for it in inv["line_items"]:
                if it["quantity"] <= 0 or it["unit_price_minor"] <= 0:
                    r.fail(R, f"invoice {inv['invoice_id']} line {it['line_no']} non-positive qty/price")
            if inv["_ia"] < self.ps - timedelta(days=62):
                r.warn(R, f"invoice {inv['invoice_id']} issued long before period ({inv['issued_at']})")
            if inv["_ia"] >= self.pe + timedelta(days=31):
                r.warn(R, f"invoice {inv['invoice_id']} issued long after period ({inv['issued_at']})")
            off = inv["_ia"].utcoffset()
            if abs(off) > timedelta(hours=14):
                r.fail(R, f"invoice {inv['invoice_id']} impossible offset")
        for p in self.payments.values():
            if not self.in_period(p["_ts"]):
                r.warn(R, f"payment {p['payment_id']} outside period ({p['timestamp']})")
            for ref in p["invoice_refs"] or []:
                inv = self.invoices.get(ref["invoice_id"])
                if inv and inv["_ia"] > p["_ts"]:
                    r.fail(R, f"payment {p['payment_id']} ({p['timestamp']}) before invoice {inv['invoice_id']} issued ({inv['issued_at']})")
        # refunds must follow a matching unreferenced receipt
        pay_entity = {}
        for x in self.ledger:
            if x["pay"]:
                pay_entity[x["pay"]] = x["entity"]
        unref = [p for p in self.payments.values() if p["invoice_refs"] is None]
        for p in unref:
            if p["amount_minor"] < 0:
                src = [q for q in unref if q["amount_minor"] == -p["amount_minor"] and q["currency"] == p["currency"]
                       and pay_entity.get(q["payment_id"]) == pay_entity.get(p["payment_id"]) and q["_ts"] < p["_ts"]]
                if not src:
                    r.warn(R, f"refund {p['payment_id']} has no earlier matching unapplied receipt")
        # magnitudes in USD
        if base_usd:
            exp_by = defaultdict(list)
            for x in self.ledger:
                if x["account"].endswith(EXPENSE_ACCOUNTS_SUFFIX) and x["dr"]:
                    exp_by[x["account"]].append(x["dr"] / 10 ** self.exps[self.base] * base_usd)
            for acc, v in sorted(exp_by.items()):
                v.sort()
                med = v[len(v) // 2]
                if med < 50:
                    r.warn(R, f"{acc}: median {med:.2f} USD-equivalent (min {v[0]:.2f}, max {v[-1]:.2f}) - implausibly small "
                              f"in {self.base}")
            rev_usd = []
            for inv in self.inv_list:
                if inv["currency"] in USD_VALUE:
                    rev_usd.append((inv["total_minor"] / 10 ** self.exps[inv["currency"]] * USD_VALUE[inv["currency"]],
                                    inv["invoice_id"], inv["currency"]))
            rev_usd.sort()
            if rev_usd and (rev_usd[0][0] < 5 or rev_usd[-1][0] > 20_000_000):
                r.warn(R, f"invoice USD-equivalent range {rev_usd[0][0]:.2f} ({rev_usd[0][2]}) .. {rev_usd[-1][0]:,.0f} ({rev_usd[-1][2]})")
            exp_med = sorted(x for v in exp_by.values() for x in v)
            if exp_med and rev_usd:
                em = exp_med[len(exp_med) // 2]
                rm = rev_usd[len(rev_usd) // 2][0]
                if rm / max(em, 1e-9) > 1000:
                    r.warn(R, f"median invoice ~{rm:,.0f} USD vs median expense ~{em:,.2f} USD (ratio {rm / em:,.0f}x)")
        # distractors should not be flagged (other than via declared issues)
        flagged_ids = {i for k, f in mine.items() for i in f["ids"]}
        truth_ids = {i for iss in self.truth.get("issues", []) for i in iss["affected_ids"]}
        for d in getattr(self, "_distractors", []):
            ids = [d.get(k) for k in ("invoice", "payment", "of") if d.get(k)] + d.get("invoices", []) + d.get("payments", [])
            hit = [i for i in ids if i in flagged_ids and i not in truth_ids]
            if hit:
                r.fail(R, f"distractor {d.get('kind')} {ids} got flagged by my audit: {hit}")
        self.r.info(R, f"{self._loose_same_amount} same-entity/same-amount invoice groups that differ in content (recurring distractors)")

    # ------------------------------------------------------------------ run
    def run(self):
        self.load()
        if not self.ok_parse:
            return self.r
        self.equalities(self.ledger)
        mine = self.audit(self.ledger)
        self.mine = mine
        self.check_truth(mine)
        try:
            self.check_provenance()
        except FileNotFoundError:
            self.r.fail("pre-error", "provenance.json missing")
        self.realism(mine)
        return self.r


def main(argv):
    verbose = "-v" in argv
    names = [a for a in argv if not a.startswith("-")]
    eps = []
    root_eps = os.path.join(REPO, "episodes")
    splits = sorted(x for x in os.listdir(root_eps) if os.path.isdir(os.path.join(root_eps, x))) \
        if os.path.isdir(root_eps) else []
    for split in splits:          # любые сплиты: train, dev или экспорт жюри (export --split ...)
        d = os.path.join(REPO, "episodes", split)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if names and n not in names:
                continue
            eps.append(Episode(split, n))
    if not eps:
        print("no episodes found")
        return 2
    sections = ["schema", "equalities", "truth", "evidence", "audit", "pre-error", "realism"]
    reports = []
    for e in eps:
        try:
            rep = e.run()
        except Exception as ex:  # pragma: no cover - report crashes as failures
            import traceback
            e.r.fail("schema", f"verifier crashed: {ex!r}\n{traceback.format_exc()}")
            rep = e.r
        nt = len(getattr(e, "truth", {}).get("issues", [])) if hasattr(e, "truth") else "?"
        nm = len([f for f in getattr(e, "mine", {}).values() if f["type"] != "other"])
        reports.append((rep, nt, nm))
    w = max(len(r.name) for r, _, _ in reports)
    hdr = f"{'episode':<{w}}  {'truth':>5} {'mine':>4}  " + " ".join(f"{s:>10}" for s in sections) + "  RESULT"
    print(hdr)
    print("-" * len(hdr))
    for rep, nt, nm in reports:
        print(f"{rep.name:<{w}}  {nt:>5} {nm:>4}  " + " ".join(f"{rep.status(s):>10}" for s in sections)
              + f"  {'FAIL' if rep.failed else 'PASS'}")
    print()
    for rep, _, _ in reports:
        lines = [(s, l, t) for s in sections for l, t in rep.msgs.get(s, []) if verbose or l != "INFO"]
        if not lines:
            continue
        print(f"== {rep.name}")
        for s, l, t in lines:
            print(f"  [{l}] {s}: {t}")
    nfail = sum(1 for r, _, _ in reports if r.failed)
    print(f"\n{len(reports) - nfail}/{len(reports)} episodes PASS")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
