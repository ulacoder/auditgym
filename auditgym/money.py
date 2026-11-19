"""Деньги, валюты, время и курсы.

Все суммы — целые минимальные единицы. Курсы — Decimal из десятичных строк.
Округление: ROUND_HALF_UP один раз, после конвертации всей суммы документа
или всей суммы распределения в минимальные единицы базовой валюты.
"""
from __future__ import annotations

import bisect
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

CURRENCY_EXPONENT = {
    "USD": 2, "EUR": 2, "GBP": 2, "KZT": 2, "CHF": 2, "CNY": 2,
    "JPY": 0, "KWD": 3,
}
# Вымышленные «опорные» курсы к USD, только для правдоподобного масштаба.
USD_ANCHOR = {
    "USD": Decimal("1"), "EUR": Decimal("1.08"), "GBP": Decimal("1.27"),
    "KZT": Decimal("0.0021"), "CHF": Decimal("1.12"), "CNY": Decimal("0.138"),
    "JPY": Decimal("0.0067"), "KWD": Decimal("3.25"),
}
BASE_CURRENCIES = ["USD", "EUR", "GBP", "KZT"]
ROUNDING_MODE = "ROUND_HALF_UP"
RATE_DECIMALS = 8


def convert_minor(amount_minor: int, src: str, rate: Decimal, base: str) -> int:
    """Конвертация суммы в минимальных единицах src в минимальные единицы base."""
    if src == base:
        return int(amount_minor)
    e = CURRENCY_EXPONENT[base] - CURRENCY_EXPONENT[src]
    v = Decimal(int(amount_minor)) * rate * (Decimal(10) ** e)
    return int(v.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def fmt_rate(r: Decimal) -> str:
    return str(r.quantize(Decimal(1).scaleb(-RATE_DECIMALS), rounding=ROUND_HALF_UP))


def tz_from_str(s: str) -> timezone:
    """'+05:00' -> timezone(+5h)."""
    sign = -1 if s[0] == "-" else 1
    hh, mm = s[1:].split(":")
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


def tz_to_str(tz: timezone) -> str:
    off = tz.utcoffset(None)
    total = int(off.total_seconds() // 60)
    sign = "-" if total < 0 else "+"
    total = abs(total)
    return f"{sign}{total // 60:02d}:{total % 60:02d}"


def iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError("timestamp without offset")
    return dt


class FxTable:
    """Таблица курсов source->target с правилом «последняя дата не позже»."""

    def __init__(self, rows: list[dict]):
        self._by_pair: dict[tuple[str, str], tuple[list[date], list[Decimal]]] = {}
        tmp: dict[tuple[str, str], list[tuple[date, Decimal]]] = {}
        for r in rows:
            key = (r["source_currency"], r["target_currency"])
            tmp.setdefault(key, []).append((date.fromisoformat(r["date"]), Decimal(r["rate"])))
        for k, lst in tmp.items():
            lst.sort()
            self._by_pair[k] = ([d for d, _ in lst], [v for _, v in lst])

    def dates(self, src: str, tgt: str) -> list[date]:
        return list(self._by_pair.get((src, tgt), ([], []))[0])

    def rate_on_or_before(self, src: str, tgt: str, d: date) -> tuple[date, Decimal]:
        ds, vs = self._by_pair[(src, tgt)]
        i = bisect.bisect_right(ds, d) - 1
        if i < 0:
            raise KeyError(f"no rate for {src}->{tgt} on or before {d}")
        return ds[i], vs[i]

    def exact(self, src: str, tgt: str, d: date) -> Decimal | None:
        ds, vs = self._by_pair.get((src, tgt), ([], []))
        i = bisect.bisect_left(ds, d)
        if i < len(ds) and ds[i] == d:
            return vs[i]
        return None


def policy_rate_date(instant: datetime, company_tz: timezone) -> date:
    """Дата курса по политике: календарная дата момента в часовом поясе компании."""
    return instant.astimezone(company_tz).date()


def policy_rate(fx: FxTable, src: str, base: str, instant: datetime, company_tz: timezone) -> Decimal:
    if src == base:
        return Decimal(1)
    return fx.rate_on_or_before(src, base, policy_rate_date(instant, company_tz))[1]
