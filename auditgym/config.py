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
    "n_invoices": [30, 55],
    "n_expenses": [45, 85],
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


FX_KINDS = ("prev_business_day", "next_business_day", "due_date", "naive_date")
ALLOC_MODES = ("partial", "missing")
FX_TARGETS = ("invoice", "payment")
LATEST_PERIOD = (2026, 10)        # только завершённые периоды: никаких дат в будущем
EARLIEST_PERIOD = (2020, 1)


def _is_int(v) -> bool:
    return type(v) is int


def _is_share(v) -> bool:
    return type(v) in (int, float) and 0 <= v <= 1


def _range_ok(v, lo=0, hi=None) -> bool:
    a, b = (v if isinstance(v, (list, tuple)) and len(v) == 2 else (v, v))
    return _is_int(a) and _is_int(b) and lo <= a <= b and (hi is None or b <= hi)


def _subset(name, v, allowed):
    if not isinstance(v, list) or not v or not all(isinstance(x, str) and x in allowed for x in v):
        raise ValueError(f"{name} must be a non-empty list of {list(allowed)}")


def resolve_config(config: dict | None) -> dict:
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be an object")
    cfg = _merge(DEFAULT_CONFIG, config or {})
    unknown = set(cfg) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for sect in ("errors", "interactions", "error_options", "distractors"):
        if not isinstance(cfg[sect], dict):
            raise ValueError(f"{sect} must be an object")
        bad = set(cfg[sect]) - set(DEFAULT_CONFIG[sect])
        if bad:
            raise ValueError(f"unknown keys in {sect}: {sorted(bad)}")
    counts = list(cfg["errors"].values()) + list(cfg["interactions"].values())
    if not all(_is_int(v) for v in counts):
        raise ValueError("error and interaction counts must be integers")
    if any(v < 0 for v in counts):
        raise ValueError("negative error count")
    n = sum(cfg["errors"].values()) + 2 * sum(cfg["interactions"].values())
    if n > MAX_ISSUES:
        raise ValueError(f"too many issues: {n} > {MAX_ISSUES}")
    # границы размеров: эпизод не может «повесить» сервер
    for key, hi in (("n_customers", 80), ("n_invoices", 160), ("n_expenses", 800), ("n_foreign_currencies", 6)):
        if not _range_ok(cfg[key], 1, hi):
            raise ValueError(f"{key} must be integers within 1..{hi}")
    if not _is_share(cfg["foreign_customer_share"]):
        raise ValueError("foreign_customer_share must be a number in 0..1")
    from .money import CURRENCY_EXPONENT
    if cfg["base_currency"] is not None and cfg["base_currency"] not in CURRENCY_EXPONENT:
        raise ValueError(f"base_currency must be one of {sorted(CURRENCY_EXPONENT)}")
    tz = cfg["timezone"]
    if tz is not None:
        import re
        m = re.fullmatch(r"([+-])(\d\d):(\d\d)", tz) if isinstance(tz, str) else None
        if not m or int(m.group(2)) > 14 or int(m.group(3)) not in (0, 15, 30, 45) \
                or (int(m.group(2)) == 14 and int(m.group(3)) != 0) or (tz == "-00:00"):
            raise ValueError("timezone must look like +05:00 (UTC offset, up to ±14:00)")
    per = cfg["period"]
    if per is not None:
        import re
        m = re.fullmatch(r"(\d{4})-(\d{2})", per) if isinstance(per, str) else None
        if not m or not 1 <= int(m.group(2)) <= 12 or \
                not EARLIEST_PERIOD <= (int(m.group(1)), int(m.group(2))) <= LATEST_PERIOD:
            raise ValueError("period must be YYYY-MM of a completed month between 2020-01 and 2026-10")
    eo = cfg["error_options"]
    _subset("error_options.payment_allocation_modes", eo["payment_allocation_modes"], ALLOC_MODES)
    _subset("error_options.fx_targets", eo["fx_targets"], FX_TARGETS)
    _subset("error_options.fx_kinds", eo["fx_kinds"], FX_KINDS)
    if eo["fx_targets"] == ["payment"] and eo["fx_kinds"] == ["due_date"]:
        raise ValueError("fx_kinds: payments have no due date; add another kind or the 'invoice' target")
    if not _is_share(eo["cutoff_tz_trap_share"]):
        raise ValueError("error_options.cutoff_tz_trap_share must be a number in 0..1")
    for k, v in cfg["distractors"].items():
        ok = _is_share(v) if k.endswith("_share") else _range_ok(v, 0, 50)
        if not ok:
            raise ValueError(f"distractors.{k} must be " + ("a number in 0..1" if k.endswith("_share")
                                                         else "an integer or [lo, hi] with 0 <= lo <= hi <= 50"))
    if not isinstance(cfg["id_salt"], str) or len(cfg["id_salt"]) > 64:
        raise ValueError("id_salt must be a string (<= 64 chars)")
    if cfg["order_seed"] is not None and not _is_int(cfg["order_seed"]):
        raise ValueError("order_seed must be an integer or null")
    if not isinstance(cfg["config_version"], str):
        raise ValueError("config_version must be a string")
    return cfg


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
