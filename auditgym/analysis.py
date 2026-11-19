"""Независимая сверка по ОТКРЫТЫМ записям.

Используется в двух местах:
* валидатор данных: каждая объявленная в эталоне ошибка должна воспроизводиться
  из открытых записей, а в книгах «до ошибок» расхождений быть не должно;
* скриптовая политика-аудитор (демо): доказывает, что задача решается из наблюдений.

Модуль не читает truth/provenance и не импортирует генератор.
"""
from __future__ import annotations

import json
from collections import defaultdict

from .money import FxTable, convert_minor, parse_iso, policy_rate, tz_from_str


def _f(type_, affected, amount, currency, evidence, explanation):
    return {"type": type_, "affected_ids": affected, "claimed_amount_minor": int(amount),
            "currency": currency, "evidence_ids": evidence, "explanation": explanation[:1000]}


def ledger_balance_report(ledger: list[dict]) -> dict:
    """Бухгалтерские равенства: итог дебета = итог кредита и баланс каждой проводки."""
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    td = tc = 0
    for r in ledger:
        if r["payment_id"]:
            k = "P:" + r["payment_id"]
        elif r["invoice_id"]:
            k = "I:" + r["invoice_id"]
        else:
            k = f"E:{r['timestamp']}:{r['entity_id']}"
        groups[k][0] += r["debit_minor"]
        groups[k][1] += r["credit_minor"]
        td += r["debit_minor"]
        tc += r["credit_minor"]
    unbalanced = sorted(k for k, (d, c) in groups.items() if d != c)
    return {"total_debit": td, "total_credit": tc, "balanced": td == tc and not unbalanced,
            "unbalanced_groups": unbalanced[:20]}


def audit(tables: dict, manifest: dict) -> list[dict]:
    base = manifest["base_currency"]
    tz = tz_from_str(manifest["timezone"])
    start, end = parse_iso(manifest["period_start"]), parse_iso(manifest["period_end"])
    fx = FxTable(tables["fx"])
    invoices = {i["invoice_id"]: i for i in tables["invoices"]}
    payments = {p["payment_id"]: p for p in tables["payments"]}
    inv_rows: dict[str, list] = defaultdict(list)
    pay_rows: dict[str, list] = defaultdict(list)
    for r in tables["ledger"]:
        if r["payment_id"]:
            pay_rows[r["payment_id"]].append(r)
        elif r["invoice_id"]:
            inv_rows[r["invoice_id"]].append(r)

    def rate(cur, ts):
        return policy_rate(fx, cur, base, parse_iso(ts), tz)

    def fx_key(cur, ts):
        d = fx.rate_on_or_before(cur, base, parse_iso(ts).astimezone(tz).date())[0]
        return f"{d.isoformat()}:{cur}:{base}"

    out: list[dict] = []
    cutoff: set[str] = set()
    posted: list[tuple] = []
    for iid in sorted(inv_rows):
        inv = invoices.get(iid)
        rows = inv_rows[iid]
        ar = [r for r in rows if r["account"] == "accounts_receivable" and r["debit_minor"] > 0]
        if inv is None or not ar:
            continue
        ar = sorted(ar, key=lambda r: r["row_id"])[0]
        inst = parse_iso(inv["issued_at"])
        if not (start <= inst < end):
            cutoff.add(iid)
            out.append(_f("period_cutoff", [iid], ar["debit_minor"], base, [inv["doc_id"], ar["row_id"]],
                          f"Invoice issued {inv['issued_at']} is outside the period "
                          f"[{manifest['period_start']}, {manifest['period_end']}) but posted on {ar['timestamp']}."))
            continue
        expected = convert_minor(inv["total_minor"], inv["currency"], rate(inv["currency"], inv["issued_at"]), base)
        if ar["debit_minor"] != expected:
            out.append(_f("fx_conversion", [iid], abs(ar["debit_minor"] - expected), base,
                          [inv["doc_id"], ar["row_id"]] + ([fx_key(inv["currency"], inv["issued_at"])]
                                                           if inv["currency"] != base else []),
                          f"AR posted {ar['debit_minor']} but policy-rate conversion gives {expected}."))
        posted.append((inv, ar))

    groups: dict[tuple, list] = defaultdict(list)
    for inv, ar in posted:
        lines = json.dumps(inv["line_items"], sort_keys=True)
        groups[(inv["entity_id"], inv["currency"], inv["total_minor"], lines)].append((inv, ar))
    for key in sorted(groups, key=lambda k: json.dumps(k)):
        g = sorted(groups[key], key=lambda x: (parse_iso(x[0]["issued_at"]), x[0]["doc_id"]))
        first = g[0][0]
        for inv, ar in g[1:]:
            out.append(_f("duplicate_invoice", [inv["invoice_id"]], ar["debit_minor"], base,
                          [first["doc_id"], inv["doc_id"], ar["row_id"]],
                          f"Same entity, currency, total and line items (incl. PO) as {first['invoice_id']}; later copy."))

    for pid in sorted(payments):
        p = payments[pid]
        rows = pay_rows.get(pid, [])
        if p["invoice_refs"] is None:
            continue
        cash = [r for r in rows if r["account"] == "cash" and r["debit_minor"] > 0]
        if cash:
            exp_cash = convert_minor(p["amount_minor"], p["currency"], rate(p["currency"], p["timestamp"]), base)
            if cash[0]["debit_minor"] != exp_cash:
                out.append(_f("fx_conversion", [pid], abs(cash[0]["debit_minor"] - exp_cash), base,
                              [p["doc_id"], cash[0]["row_id"]] + ([fx_key(p["currency"], p["timestamp"])]
                                                                  if p["currency"] != base else []),
                              f"Cash posted {cash[0]['debit_minor']} but policy-rate conversion gives {exp_cash}."))
        unapplied = [r for r in rows if r["account"] == "unapplied_cash"]
        for ref in p["invoice_refs"]:
            inv = invoices.get(ref["invoice_id"])
            if inv is None:
                continue
            exp = convert_minor(ref["amount_minor"], inv["currency"], rate(inv["currency"], inv["issued_at"]), base)
            got = sum(r["credit_minor"] - r["debit_minor"] for r in rows
                      if r["account"] == "accounts_receivable" and r["invoice_id"] == ref["invoice_id"])
            if got < exp:
                ar_rows = [r["row_id"] for r in rows
                           if r["account"] == "accounts_receivable" and r["invoice_id"] == ref["invoice_id"]]
                ev = ([p["doc_id"], inv["doc_id"]] + ([unapplied[0]["row_id"]] if unapplied else []) + ar_rows[:1]
                      + ([fx_key(inv["currency"], inv["issued_at"])] if inv["currency"] != base else []))
                out.append(_f("payment_allocation", [pid, ref["invoice_id"]], exp - got, base, ev,
                              f"Remittance applies {ref['amount_minor']} {inv['currency']} to {ref['invoice_id']}; "
                              f"AR credited {got} vs {exp} expected."))
    return out
