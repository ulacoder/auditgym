"""Конфигурация эпизода: значения по умолчанию, слияние и проверка."""
from __future__ import annotations

import copy
import json

ERROR_TYPES = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")
INTERACTIONS = ("fx_invoice_with_partial_payment", "partial_fx_payment_at_boundary")
MAX_ISSUES = 12

DEFAULT_CONFIG: dict = {
    "config_version": "default-v1",
    "base_currency": None,          # None -> выбирается по seed
    "timezone": None,               # None -> выбирается по seed, напр. "+05:00"
    "period": None,                 # None -> выбирается по seed, напр. "2026-03"
    "n_foreign_currencies": [1, 3],
    "foreign_customer_share": 0.4,
    "n_customers": [8, 18],
    "n_invoices": [35, 65],
    "n_expenses": [25, 60],
    "errors": {t: 0 for t in ERROR_TYPES},
    "interactions": {k: 0 for k in INTERACTIONS},
    "error_options": {
        "payment_allocation_modes": ["partial", "missing"],
        "fx_targets": ["invoice", "payment"],
        "fx_kinds": ["prev_business_day", "next_business_day", "due_date", "naive_date"],
        "cutoff_tz_trap_share": 0.5,
    },
    "distractors": {
        "recurring_amounts": [2, 4],      # законные повторяющиеся суммы
        "legit_partial_share": 0.12,      # законные частичные оплаты
        "split_payment_share": 0.13,      # оплата одного счёта двумя платежами
        "multi_invoice_payment_share": 0.3,
        "unreferenced_payments": [2, 5],  # платежи без ссылок -> unapplied_cash
        "refund_share": 0.5,              # доля возвратов по ним
        "next_period_unposted": [1, 3],   # счета следующего периода, корректно не проведённые
        "tz_boundary_invoices": [2, 4],   # счета у границы периода с «обманчивым» смещением
    },
    # Безвредные преобразования (для парных проверок): меняют ID и порядок, не историю.
    "id_salt": "",
    "order_seed": None,
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_config(config: dict | None) -> dict:
    cfg = _merge(DEFAULT_CONFIG, config or {})
    unknown = set(cfg) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for t in cfg["errors"]:
        if t not in ERROR_TYPES:
            raise ValueError(f"unknown error type {t}")
    for t in cfg["interactions"]:
        if t not in INTERACTIONS:
            raise ValueError(f"unknown interaction {t}")
    n = sum(int(v) for v in cfg["errors"].values()) + 2 * sum(int(v) for v in cfg["interactions"].values())
    if n > MAX_ISSUES:
        raise ValueError(f"too many issues: {n} > {MAX_ISSUES}")
    if any(int(v) < 0 for v in list(cfg["errors"].values()) + list(cfg["interactions"].values())):
        raise ValueError("negative error count")
    return cfg


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
