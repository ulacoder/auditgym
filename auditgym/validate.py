"""Проверка данных и эталона эпизода (то, что жюри проверяет в разделе «Данные и эталон»)."""
from __future__ import annotations

import hashlib
import re
from .analysis import audit, ledger_balance_report
from .config import ERROR_TYPES, MAX_ISSUES, canonical_json
from .generator import ACCOUNTS, MIN_MATERIAL_MINOR, Episode, generate
from .matcher import grade

SCHEMA = {
    "ledger": {"row_id": str, "timestamp": str, "account": str, "debit_minor": int, "credit_minor": int,
               "currency": str, "entity_id": (str, type(None)), "invoice_id": (str, type(None)),
               "payment_id": (str, type(None))},
    "invoices": {"doc_id": str, "invoice_id": str, "entity_id": str, "issued_at": str, "due_at": str,
                 "currency": str, "total_minor": int, "line_items": list},
    "payments": {"doc_id": str, "payment_id": str, "timestamp": str, "currency": str, "amount_minor": int,
                 "invoice_refs": (list, type(None))},
    "fx": {"date": str, "source_currency": str, "target_currency": str, "rate": str},
}
ISO_OFFSET = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
DEC = re.compile(r"^\d+\.\d+$")
LEAK_PATTERNS = re.compile(r"ISS-|truth|provenance|distractor|error|seed", re.I)


def tables_hash(tables: dict) -> str:
    return hashlib.sha256(canonical_json(tables).encode()).hexdigest()


def _schema_errors(tables: dict, manifest: dict) -> list[str]:
    errs = []
    for t, fields in SCHEMA.items():
        for r in tables[t]:
            if set(r) != set(fields):
                errs.append(f"{t}: fields {sorted(r)}")
                break
            for k, typ in fields.items():
                if not isinstance(r[k], typ) or isinstance(r[k], bool):
                    errs.append(f"{t}.{k}: bad type")
                    break
    for r in tables["ledger"]:
        if not ISO_OFFSET.match(r["timestamp"]):
            errs.append("ledger timestamp not ISO-8601 with offset")
        if r["account"] not in ACCOUNTS or r["currency"] != manifest["base_currency"]:
            errs.append("ledger account/currency")
        if (r["debit_minor"] > 0) == (r["credit_minor"] > 0) or min(r["debit_minor"], r["credit_minor"]) < 0:
            errs.append("ledger row must be one-sided and non-negative")
    for i in tables["invoices"]:
        if not (ISO_OFFSET.match(i["issued_at"]) and ISO_OFFSET.match(i["due_at"])):
            errs.append("invoice timestamp format")
        if i["total_minor"] != sum(l["amount_minor"] for l in i["line_items"]):
            errs.append("invoice total != sum(lines)")
        if i["currency"] not in manifest["currency_exponents"]:
            errs.append("invoice currency without exponent")
    for p in tables["payments"]:
        if not ISO_OFFSET.match(p["timestamp"]):
            errs.append("payment timestamp format")
        if p["invoice_refs"] is not None and sum(x["amount_minor"] for x in p["invoice_refs"]) != p["amount_minor"]:
            errs.append("payment refs do not sum to amount")
    for r in tables["fx"]:
        if not DEC.match(r["rate"]):
            errs.append("fx rate is not a decimal string")
    inv_ids = {i["invoice_id"] for i in tables["invoices"]}
    pay_ids = {p["payment_id"] for p in tables["payments"]}
    for r in tables["ledger"]:
        if r["invoice_id"] is not None and r["invoice_id"] not in inv_ids:
            errs.append("ledger references unknown invoice")
        if r["payment_id"] is not None and r["payment_id"] not in pay_ids:
            errs.append("ledger references unknown payment")
    for p in tables["payments"]:
        for x in p["invoice_refs"] or []:
            if x["invoice_id"] not in inv_ids:
                errs.append("payment references unknown invoice")
    return sorted(set(errs))[:20]


def evidence_recompute(issue: dict, tables: dict, manifest: dict) -> int | None:
    """Сумма ошибки, пересчитанная ТОЛЬКО по записям одного набора доказательств
    (+ таблица курсов и manifest как справочные данные политики). None — набор недостаточен."""
    from .money import FxTable, convert_minor, parse_iso, tz_from_str
    base = manifest["base_currency"]
    tz = tz_from_str(manifest["timezone"])
    docs = {d["doc_id"]: d for d in tables["invoices"] + tables["payments"]}
    rows = {r["row_id"]: r for r in tables["ledger"]}
    fxrows = {f"{r['date']}:{r['source_currency']}:{r['target_currency']}": r for r in tables["fx"]}
    fx = FxTable(tables["fx"])
    aff = set(issue["affected_ids"])
    results = []
    for ev in issue["evidence_sets"]:
        D = [docs[i] for i in ev if i in docs]
        L = [rows[i] for i in ev if i in rows]
        X = [fxrows[i] for i in ev if i in fxrows]
        val = None
        t = issue["type"]
        if t == "duplicate_invoice" and len(D) == 2 and len(L) == 1:
            a, b = sorted(D, key=lambda d: parse_iso(d["issued_at"]))
            same = all(a[k] == b[k] for k in ("entity_id", "currency", "total_minor", "line_items"))
            r = L[0]
            if same and {b["invoice_id"]} == aff and r["invoice_id"] == b["invoice_id"] and r["payment_id"] is None:
                val = r["debit_minor"] or r["credit_minor"]
        elif t == "payment_allocation" and len(D) == 2 and 1 <= len(L) <= 2 and len(X) <= 1:
            p = next((d for d in D if "payment_id" in d), None)
            inv = next((d for d in D if "invoice_id" in d and "payment_id" not in d), None)
            un = next((r for r in L if r["account"] == "unapplied_cash"), None)
            ar = next((r for r in L if r["account"] == "accounts_receivable"), None)
            if p and inv and un and aff == {p["payment_id"], inv["invoice_id"]} and un["payment_id"] == p["payment_id"]:
                ref = next((x for x in (p.get("invoice_refs") or []) if x["invoice_id"] == inv["invoice_id"]), None)
                if ref is not None and (inv["currency"] == base) == (not X):
                    from decimal import Decimal
                    rate = Decimal(X[0]["rate"]) if X else Decimal(1)
                    want = (fx.rate_on_or_before(inv["currency"], base, parse_iso(inv["issued_at"]).astimezone(tz).date())[0]
                            .isoformat() if X else None)
                    if not X or X[0]["date"] == want:
                        exp = convert_minor(ref["amount_minor"], inv["currency"], rate, base)
                        got = (ar["credit_minor"] - ar["debit_minor"]) if ar and ar["payment_id"] == p["payment_id"] else 0
                        if exp - got == un["credit_minor"]:
                            val = exp - got
        elif t == "fx_conversion" and len(D) == 1 and len(L) == 1 and len(X) == 1:
            d, r, x = D[0], L[0], X[0]
            ts = d.get("issued_at") or d.get("timestamp")
            amount = d.get("total_minor", d.get("amount_minor"))
            want = fx.rate_on_or_before(d["currency"], base, parse_iso(ts).astimezone(tz).date())[0].isoformat()
            ident = d.get("invoice_id") or d.get("payment_id")
            if x["date"] == want and x["source_currency"] == d["currency"] and {ident} == aff:
                from decimal import Decimal
                exp = convert_minor(amount, d["currency"], Decimal(x["rate"]), base)
                val = abs((r["debit_minor"] or r["credit_minor"]) - exp)
        elif t == "period_cutoff" and len(D) == 1 and len(L) == 1:
            d, r = D[0], L[0]
            inst = parse_iso(d["issued_at"])
            outside = not (parse_iso(manifest["period_start"]) <= inst < parse_iso(manifest["period_end"]))
            if outside and r["invoice_id"] == d["invoice_id"] and {d["invoice_id"]} == aff:
                val = r["debit_minor"] or r["credit_minor"]
        results.append(val)
    if results and all(v is not None and v == results[0] for v in results):
        return results[0]
    return None


def validate_episode(ep: Episode, seed: int | None = None, config: dict | None = None) -> dict:
    checks: dict[str, bool] = {}
    notes: dict[str, object] = {}
    m = ep.manifest
    se = _schema_errors(ep.tables, m)
    checks["schema"] = not se
    notes["schema_errors"] = se
    c = m["counts"]
    checks["manifest_counts"] = (c["ledger_rows"] == len(ep.tables["ledger"]) and c["invoices"] == len(ep.tables["invoices"])
                                 and c["payments"] == len(ep.tables["payments"]) and c["fx_rates"] == len(ep.tables["fx"]))
    checks["size_in_range"] = 200 <= c["ledger_rows"] <= 2000 and 20 <= c["documents"] <= 200
    pre = ledger_balance_report(ep.clean_tables["ledger"])
    post = ledger_balance_report(ep.tables["ledger"])
    checks["books_balanced_before_errors"] = pre["balanced"]
    checks["books_balanced_after_errors"] = post["balanced"]
    clean_findings = audit(ep.clean_tables, m)
    checks["no_discrepancy_before_errors"] = not clean_findings
    found = audit(ep.tables, m)
    g = grade({"episode_id": ep.episode_id, "findings": found[:30]}, ep.truth, 0)
    checks["declared_discrepancies_reproduce_from_public_records"] = g["f1"] == 1.0 and len(found) <= 30
    notes["independent_audit"] = {"tp": g["tp"], "fp": g["fp"], "fn": g["fn"]}
    public_ids = ({r["row_id"] for r in ep.tables["ledger"]} | {i["doc_id"] for i in ep.tables["invoices"]}
                  | {p["doc_id"] for p in ep.tables["payments"]}
                  | {f"{r['date']}:{r['source_currency']}:{r['target_currency']}" for r in ep.tables["fx"]})
    all_ids = (public_ids | {i["invoice_id"] for i in ep.tables["invoices"]}
               | {p["payment_id"] for p in ep.tables["payments"]})
    iss = ep.truth["issues"]
    checks["evidence_available"] = all(all(set(s) <= public_ids for s in t["evidence_sets"]) and t["evidence_sets"]
                                       for t in iss)
    checks["amount_recomputable_from_evidence_only"] = all(
        evidence_recompute(t, ep.tables, m) == t["expected_amount_minor"] for t in iss)
    checks["affected_ids_exist"] = all(set(t["affected_ids"]) <= all_ids for t in iss)
    checks["issue_limits"] = (len(iss) <= MAX_ISSUES and all(t["type"] in ERROR_TYPES for t in iss)
                              and all(t["expected_amount_minor"] >= MIN_MATERIAL_MINOR for t in iss)
                              and len({(t["type"], tuple(sorted(t["affected_ids"]))) for t in iss}) == len(iss))
    checks["no_truth_markers_in_public_data"] = not LEAK_PATTERNS.search(canonical_json(ep.tables) + canonical_json(m))
    if seed is not None:
        ep2 = generate(seed, config)
        checks["reproducible"] = (tables_hash(ep2.tables) == tables_hash(ep.tables)
                                  and canonical_json(ep2.truth) == canonical_json(ep.truth))
    ok = all(checks.values())
    return {"episode_id": ep.episode_id, "ok": ok, "checks": checks, "notes": notes,
            "n_issues": len(iss), "types": sorted({t["type"] for t in iss}),
            "interactions": sorted({t["interaction"] for t in iss if t["interaction"]}),
            "ledger_rows": c["ledger_rows"], "documents": c["documents"]}
