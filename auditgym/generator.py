"""Генератор компаний.

Порядок работы (причинная модель):
1. Скрытая история: компания, контрагенты, счета, платежи, расходы, курсы.
2. Корректная проводка истории в ledger (до ошибок) — снимок `clean`.
3. Для каждой запланированной ошибки создаётся собственная цель (обычный на вид
   счёт/платёж) и записывается причинное изменение (override проводки или лишний
   документ). Ошибочный ledger строится теми же функциями проводки с override.
4. Эталон (truth) описывает реальное расхождение: сумму в базовой валюте
   и наборы доступных записей, достаточные для проверки.
5. Публичные ID получаются хешированием канонических ключей с солью; порядок
   строк — хронологический или перемешанный (order_seed).

Все случайные потоки независимы: история зависит только от seed, каждая ошибка —
от своего потока, поэтому включение одной ошибки не меняет остальную компанию.
"""
from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from .config import canonical_json, resolve_config
from .money import (
    BASE_CURRENCIES, CURRENCY_EXPONENT, USD_ANCHOR, FxTable, convert_minor, fmt_rate,
    iso, policy_rate_date, tz_from_str, tz_to_str,
)
from .version import ENV_VERSION, SCHEMA_VERSION

COMPANY_OFFSETS = ["+05:00", "+06:00", "+03:00", "+01:00", "+00:00", "-05:00", "+09:00", "-03:00"]
DOC_OFFSETS = ["+00:00", "+01:00", "+03:00", "+05:00", "+06:00", "+09:00", "-05:00", "-08:00"]
CATALOG = [
    ("SKU-AUD", "Audit support hours"), ("SKU-LIC", "Software licence seat"),
    ("SKU-CLD", "Cloud storage plan"), ("SKU-CON", "Consulting day"),
    ("SKU-TRN", "Training session"), ("SKU-HW1", "Hardware module"),
    ("SKU-MNT", "Maintenance package"), ("SKU-DSG", "Design sprint"),
    ("SKU-LOG", "Logistics service"), ("SKU-DAT", "Data processing batch"),
]
NAMES_A = ["Altyn", "Nord", "Steppe", "Blue", "Silk", "Vega", "Orion", "Tengri", "Kite", "Polar",
           "Aral", "Saryarka", "Lumen", "Delta", "Quartz", "Maple", "Zenit", "Arman"]
NAMES_B = ["Logistics", "Foods", "Systems", "Trading", "Labs", "Textiles", "Energy", "Retail",
           "Media", "Build", "Agro", "Pharma", "Motors", "Studio"]
EXPENSE_ACCOUNTS = ["rent_expense", "payroll_expense", "software_expense", "utilities_expense",
                    "travel_expense"]
ACCOUNTS = ["cash", "accounts_receivable", "revenue", "unapplied_cash", "fx_gain_loss"] + EXPENSE_ACCOUNTS
MIN_MATERIAL_MINOR = 100


def sub_rng(seed: int, stream: str) -> random.Random:
    h = hashlib.sha256(f"{int(seed)}|{stream}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# ----------------------------------------------------------------------------- модели
@dataclass
class Entity:
    key: str
    name: str
    kind: str              # customer | vendor
    currency: str


@dataclass
class Invoice:
    key: str
    entity: str
    issued_at: datetime    # с «отображаемым» смещением документа
    due_at: datetime
    currency: str
    total: int
    lines: list
    tags: list = field(default_factory=list)


@dataclass
class Payment:
    key: str
    entity: str
    timestamp: datetime
    currency: str
    amount: int
    refs: list | None      # [[inv_key, amount_minor], ...] или None
    tags: list = field(default_factory=list)


@dataclass
class Expense:
    key: str
    vendor: str
    ts: datetime
    account: str
    amount: int


@dataclass
class Row:
    key: str
    ts: datetime
    account: str
    debit: int
    credit: int
    entity: str | None
    invoice: str | None
    payment: str | None


@dataclass
class Overrides:
    inv_rate: dict = field(default_factory=dict)     # inv_key -> Decimal
    inv_ts: dict = field(default_factory=dict)       # inv_key -> datetime (дата проводки)
    pay_rate: dict = field(default_factory=dict)     # pay_key -> Decimal
    alloc_shift: dict = field(default_factory=dict)  # (pay_key, inv_key) -> minor


@dataclass
class Episode:
    episode_id: str
    seed: int
    config: dict
    manifest: dict
    policy: dict
    tables: dict           # публичные таблицы (list[dict])
    truth: dict            # закрыто
    provenance: dict       # закрыто
    clean_tables: dict     # закрыто: книги до внесения ошибок


# ----------------------------------------------------------------------------- генератор
class _Gen:
    def __init__(self, seed: int, cfg: dict):
        self.seed = int(seed)
        self.cfg = cfg
        R = sub_rng(seed, "setting")
        self.base = cfg["base_currency"] or R.choice(BASE_CURRENCIES)
        self.tz_str = cfg["timezone"] or R.choice(COMPANY_OFFSETS)
        self.tz = tz_from_str(self.tz_str)
        if cfg["period"]:
            y, m = (int(x) for x in cfg["period"].split("-"))
        else:
            y, m = R.choice([2025, 2026]), R.randint(1, 12)
        self.start = datetime(y, m, 1, tzinfo=self.tz)
        self.end = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=self.tz)
        nf = R.randint(*_rng_pair(cfg["n_foreign_currencies"]))
        pool = [c for c in CURRENCY_EXPONENT if c != self.base]
        self.foreign = sorted(R.sample(pool, nf))
        self.company_name = f"{R.choice(NAMES_A)} {R.choice(NAMES_B)} LLP"
        self._build_fx(sub_rng(seed, "fx"))
        self.entities: dict[str, Entity] = {}
        self.invoices: dict[str, Invoice] = {}
        self.payments: dict[str, Payment] = {}
        self.expenses: list[Expense] = []
        self.error_invoices: dict[str, Invoice] = {}   # существуют только в ошибочном мире
        self.ov = Overrides()
        self.issues: list[dict] = []
        self.changes: list[dict] = []
        self.distractor_log: list[dict] = []
        self._po_used: set[int] = set()
        self._cnt: dict[str, int] = {}

    # ---------------------------------------------------------------- утилиты
    def _next(self, prefix: str) -> str:
        self._cnt[prefix] = self._cnt.get(prefix, 0) + 1
        return f"{prefix}_{self._cnt[prefix]:04d}"

    def _instant(self, R, a: datetime, b: datetime) -> datetime:
        span = int((b - a).total_seconds())
        return (a + timedelta(seconds=R.randrange(max(span, 1)))).astimezone(self.tz)

    def _display(self, R, instant: datetime, want_date_shift: bool = False) -> datetime:
        """Выбор смещения документа. want_date_shift: дата «на бумаге» != дате в поясе компании."""
        if want_date_shift:
            local = instant.astimezone(self.tz).date()
            opts = [o for o in DOC_OFFSETS if instant.astimezone(tz_from_str(o)).date() != local]
            if opts:
                return instant.astimezone(tz_from_str(R.choice(opts)))
        if R.random() < 0.5:
            return instant.astimezone(self.tz)
        return instant.astimezone(tz_from_str(R.choice(DOC_OFFSETS)))

    def _po(self, R) -> int:
        while True:
            po = R.randint(10000, 99999)
            if po not in self._po_used:
                self._po_used.add(po)
                return po

    def _lines(self, R, currency: str, big: bool = False, po: int | None = None, n: int | None = None):
        po = po if po is not None else self._po(R)
        lines = []
        for i in range(n or R.randint(1, 4)):
            sku, name = R.choice(CATALOG)
            usd = R.uniform(800, 6000) if big else R.uniform(15, 1500)
            major = Decimal(str(usd)) / USD_ANCHOR[currency]
            exp = CURRENCY_EXPONENT[currency]
            unit = int(major.quantize(Decimal(1))) * 10 ** exp
            unit = max(unit, 10 ** exp)
            qty = R.randint(1, 12) if not big else R.randint(2, 10)
            lines.append({"line_no": i + 1, "sku": sku, "description": f"{name} / PO-{po}",
                          "quantity": qty, "unit_price_minor": unit, "amount_minor": unit * qty})
        return lines

    def _customer(self, R, currency: str | None = None, foreign: bool | None = None) -> Entity:
        cands = [e for e in self.entities.values() if e.kind == "customer"]
        if currency:
            cands = [e for e in cands if e.currency == currency]
        elif foreign is True:
            cands = [e for e in cands if e.currency != self.base]
        elif foreign is False:
            cands = [e for e in cands if e.currency == self.base]
        return R.choice(sorted(cands, key=lambda e: e.key))

    def _make_invoice(self, R, ent: Entity, instant: datetime, big=False, tags=(), lines=None,
                      display_shift=False, register=True) -> Invoice:
        lines = lines if lines is not None else self._lines(R, ent.currency, big=big)
        disp = self._display(R, instant, display_shift)
        inv = Invoice(key=self._next("inv"), entity=ent.key, issued_at=disp,
                      due_at=disp + timedelta(days=R.choice([7, 14, 30, 45])),
                      currency=ent.currency, total=sum(l["amount_minor"] for l in lines),
                      lines=lines, tags=list(tags))
        if register:
            self.invoices[inv.key] = inv
        return inv

    def _make_payment(self, R, ent_key: str, currency: str, instant: datetime, amount: int,
                      refs, tags=(), display_shift=False) -> Payment:
        p = Payment(key=self._next("pay"), entity=ent_key, timestamp=self._display(R, instant, display_shift),
                    currency=currency, amount=amount, refs=refs, tags=list(tags))
        self.payments[p.key] = p
        return p

    def rate(self, cur: str, instant: datetime) -> Decimal:
        if cur == self.base:
            return Decimal(1)
        return self.fx.rate_on_or_before(cur, self.base, policy_rate_date(instant, self.tz))[1]

    # ---------------------------------------------------------------- курсы
    def _build_fx(self, R):
        d0 = self.start.date() - timedelta(days=10)
        d1 = self.end.date() + timedelta(days=10)
        rows = []
        for cur in self.foreign:
            r = USD_ANCHOR[cur] / USD_ANCHOR[self.base] * Decimal(str(R.uniform(0.95, 1.05)))
            d = d0
            while d <= d1:
                if d.weekday() < 5:
                    rows.append({"date": d.isoformat(), "source_currency": cur,
                                 "target_currency": self.base, "rate": fmt_rate(r)})
                    step = R.uniform(0.0015, 0.006) * R.choice([-1, 1])
                    r = r * Decimal(str(1 + step))
                d += timedelta(days=1)
        self.fx_rows = rows
        self.fx = FxTable(rows)

    # ---------------------------------------------------------------- история
    def build_history(self):
        cfg, R, dcfg = self.cfg, sub_rng(self.seed, "history"), self.cfg["distractors"]
        nc = R.randint(*_rng_pair(cfg["n_customers"]))
        for i in range(nc):
            cur = self.base
            if R.random() < cfg["foreign_customer_share"]:
                cur = R.choice(self.foreign)
            k = self._next("cust")
            self.entities[k] = Entity(k, f"{R.choice(NAMES_A)} {R.choice(NAMES_B)}", "customer", cur)
        for cur in [self.base] + self.foreign:   # хотя бы один клиент на валюту
            if not any(e.currency == cur and e.kind == "customer" for e in self.entities.values()):
                k = self._next("cust")
                self.entities[k] = Entity(k, f"{R.choice(NAMES_A)} {R.choice(NAMES_B)}", "customer", cur)
        for i in range(R.randint(3, 6)):
            k = self._next("vend")
            self.entities[k] = Entity(k, f"{R.choice(NAMES_A)} {R.choice(NAMES_B)}", "vendor", self.base)

        # обычные счета
        for _ in range(R.randint(*_rng_pair(cfg["n_invoices"]))):
            ent = self._customer(R)
            self._make_invoice(R, ent, self._instant(R, self.start, self.end - timedelta(hours=1)))

        # отвлекающие: счета у границ периода с «обманчивым» смещением, корректно проведённые
        for _ in range(R.randint(*_rng_pair(dcfg["tz_boundary_invoices"]))):
            ent = self._customer(R)
            if R.random() < 0.5:
                inst = self._instant(R, self.end - timedelta(hours=10), self.end - timedelta(minutes=5))
            else:
                inst = self._instant(R, self.start + timedelta(minutes=5), self.start + timedelta(hours=10))
            inv = self._make_invoice(R, ent, inst, display_shift=True, tags=["distractor:tz_boundary"])
            self.distractor_log.append({"kind": "tz_boundary_invoice", "invoice": inv.key})

        # отвлекающие: законные повторяющиеся суммы (тот же клиент и сумма, другой заказ)
        for _ in range(R.randint(*_rng_pair(dcfg["recurring_amounts"]))):
            ent = self._customer(R)
            tmpl = self._lines(R, ent.currency, n=R.randint(1, 2))
            t0 = self._instant(R, self.start, self.start + timedelta(days=8))
            keys = []
            for j in range(R.randint(2, 3)):
                po = self._po(R)
                lines = copy.deepcopy(tmpl)
                for l in lines:
                    l["description"] = l["description"].rsplit("/ PO-", 1)[0] + f"/ PO-{po}"
                inst = min(t0 + timedelta(days=7 * j, hours=R.randint(0, 20)), self.end - timedelta(hours=2))
                keys.append(self._make_invoice(R, ent, inst, lines=lines, tags=["distractor:recurring"]).key)
            self.distractor_log.append({"kind": "recurring_amount", "invoices": keys})

        # платежи по обычным счетам
        full_single: list[Payment] = []
        for inv in sorted(self.invoices.values(), key=lambda i: i.key):
            inst0 = inv.issued_at + timedelta(hours=2)
            if inst0 >= self.end - timedelta(days=1):
                continue
            r = R.random()
            if r < 0.18:
                continue  # не оплачен
            r -= 0.18
            if r < dcfg["legit_partial_share"]:
                amt = max(1, int(inv.total * R.uniform(0.3, 0.8)))
                p = self._make_payment(R, inv.entity, inv.currency, self._instant(R, inst0, self.end),
                                       amt, [[inv.key, amt]], tags=["distractor:legit_partial"])
                self.distractor_log.append({"kind": "legit_partial_payment", "payment": p.key, "invoice": inv.key})
                continue
            r -= dcfg["legit_partial_share"]
            if r < dcfg["split_payment_share"] and inv.total > 10:
                a1 = int(inv.total * R.uniform(0.3, 0.7))
                t1 = self._instant(R, inst0, self.end - timedelta(hours=1))
                t2 = self._instant(R, t1 + timedelta(minutes=30), self.end)
                p1 = self._make_payment(R, inv.entity, inv.currency, t1, a1, [[inv.key, a1]], tags=["distractor:split"])
                p2 = self._make_payment(R, inv.entity, inv.currency, t2, inv.total - a1,
                                        [[inv.key, inv.total - a1]], tags=["distractor:split"])
                self.distractor_log.append({"kind": "split_payment", "payments": [p1.key, p2.key], "invoice": inv.key})
                continue
            full_single.append(self._make_payment(R, inv.entity, inv.currency, self._instant(R, inst0, self.end),
                                                  inv.total, [[inv.key, inv.total]]))
        # объединение части полных оплат одного клиента в один платёж по нескольким счетам
        by_ent: dict[tuple, list[Payment]] = {}
        for p in full_single:
            by_ent.setdefault((p.entity, p.currency), []).append(p)
        for key in sorted(by_ent):
            ps = by_ent[key]
            while len(ps) >= 2 and R.random() < dcfg["multi_invoice_payment_share"]:
                a, b = ps.pop(), ps.pop()
                keep, drop = (a, b) if a.timestamp >= b.timestamp else (b, a)
                keep.refs = keep.refs + drop.refs
                keep.amount += drop.amount
                keep.tags.append("distractor:multi_invoice")
                del self.payments[drop.key]
                self.distractor_log.append({"kind": "multi_invoice_payment", "payment": keep.key})

        # платежи без ссылок (законно в unapplied_cash) и возвраты по ним
        n_unref = R.randint(*_rng_pair(dcfg["unreferenced_payments"]))
        for j in range(n_unref):
            ent = self._customer(R, currency=self.base)
            amt = R.randint(50, 3000) * 10 ** CURRENCY_EXPONENT[self.base]
            t = self._instant(R, self.start, self.end - timedelta(days=2))
            p = self._make_payment(R, ent.key, self.base, t, amt, None, tags=["distractor:unreferenced"])
            self.distractor_log.append({"kind": "unreferenced_payment", "payment": p.key})
            # хотя бы один возврат, если возвраты включены
            if R.random() < dcfg["refund_share"] or (j == 0 and dcfg["refund_share"] > 0):
                q = self._make_payment(R, ent.key, self.base, self._instant(R, t + timedelta(hours=1), self.end),
                                       -amt, None, tags=["distractor:refund"])
                self.distractor_log.append({"kind": "refund", "payment": q.key, "of": p.key})

        # расходы (шум)
        vendors = sorted([e for e in self.entities.values() if e.kind == "vendor"], key=lambda e: e.key)
        for _ in range(R.randint(*_rng_pair(cfg["n_expenses"]))):
            v = R.choice(vendors)
            self.expenses.append(Expense(self._next("exp"), v.key, self._instant(R, self.start, self.end),
                                         R.choice(EXPENSE_ACCOUNTS),
                                         R.randint(20, 4000) * 10 ** CURRENCY_EXPONENT[self.base]))

        # счета следующего периода в выгрузке документов, корректно не проведённые
        self.unposted: set[str] = set()
        for _ in range(R.randint(*_rng_pair(dcfg["next_period_unposted"]))):
            ent = self._customer(R)
            inst = self._instant(R, self.end, self.end + timedelta(days=3))
            inv = self._make_invoice(R, ent, inst, display_shift=R.random() < 0.5, tags=["distractor:next_period"])
            self.unposted.add(inv.key)
            self.distractor_log.append({"kind": "next_period_unposted_invoice", "invoice": inv.key})

    # ---------------------------------------------------------------- проводки
    def post(self, with_errors: bool) -> list[Row]:
        ov = self.ov if with_errors else Overrides()
        invs = dict(self.invoices)
        if with_errors:
            invs.update(self.error_invoices)
        rows: list[Row] = []
        for inv in invs.values():
            if inv.key in self.unposted:
                continue
            rate = ov.inv_rate.get(inv.key, self.rate(inv.currency, inv.issued_at))
            amt = convert_minor(inv.total, inv.currency, rate, self.base)
            ts = ov.inv_ts.get(inv.key, inv.issued_at).astimezone(self.tz)
            rows.append(Row(f"{inv.key}:ar", ts, "accounts_receivable", amt, 0, inv.entity, inv.key, None))
            rows.append(Row(f"{inv.key}:rev", ts, "revenue", 0, amt, inv.entity, inv.key, None))
        for p in self.payments.values():
            ts = p.timestamp.astimezone(self.tz)
            rate = ov.pay_rate.get(p.key, self.rate(p.currency, p.timestamp))
            cash = convert_minor(abs(p.amount), p.currency, rate, self.base)
            if p.refs is None:
                if p.amount >= 0:
                    rows.append(Row(f"{p.key}:cash", ts, "cash", cash, 0, p.entity, None, p.key))
                    rows.append(Row(f"{p.key}:unapplied", ts, "unapplied_cash", 0, cash, p.entity, None, p.key))
                else:
                    rows.append(Row(f"{p.key}:unapplied", ts, "unapplied_cash", cash, 0, p.entity, None, p.key))
                    rows.append(Row(f"{p.key}:cash", ts, "cash", 0, cash, p.entity, None, p.key))
                continue
            rows.append(Row(f"{p.key}:cash", ts, "cash", cash, 0, p.entity, None, p.key))
            credited, unapplied = 0, 0
            for inv_key, alloc in p.refs:
                inv = invs[inv_key]
                credit = convert_minor(alloc, inv.currency, self.rate(inv.currency, inv.issued_at), self.base)
                shift = ov.alloc_shift.get((p.key, inv_key), 0)
                if credit - shift > 0:
                    rows.append(Row(f"{p.key}:ar:{inv_key}", ts, "accounts_receivable", 0, credit - shift,
                                    p.entity, inv_key, p.key))
                credited += credit - shift
                unapplied += shift
            if unapplied:
                rows.append(Row(f"{p.key}:unapplied", ts, "unapplied_cash", 0, unapplied, p.entity, None, p.key))
            diff = cash - credited - unapplied
            if diff > 0:
                rows.append(Row(f"{p.key}:fx", ts, "fx_gain_loss", 0, diff, p.entity, None, p.key))
            elif diff < 0:
                rows.append(Row(f"{p.key}:fx", ts, "fx_gain_loss", -diff, 0, p.entity, None, p.key))
        for e in self.expenses:
            ts = e.ts.astimezone(self.tz)
            rows.append(Row(f"{e.key}:dr", ts, e.account, e.amount, 0, e.vendor, None, None))
            rows.append(Row(f"{e.key}:cr", ts, "cash", 0, e.amount, e.vendor, None, None))
        return rows

    # ---------------------------------------------------------------- ошибки
    def _issue(self, type_, affected, amount, evidence, interaction=None, **detail):
        iss = {"type": type_, "affected": affected, "amount": int(amount), "evidence": evidence,
               "interaction": interaction}
        self.issues.append(iss)
        self.changes.append({"issue_index": len(self.issues) - 1, "type": type_,
                             "interaction": interaction, **detail})

    def _wrong_rate(self, R, cur: str, instant: datetime, disp: datetime, due: datetime | None, kinds):
        """Подбор неверного курса. Возвращает (kind, wrong_date, wrong_rate) или None."""
        correct_d, correct_r = self.fx.rate_on_or_before(cur, self.base, policy_rate_date(instant, self.tz))
        dates = self.fx.dates(cur, self.base)
        kinds = list(kinds)
        R.shuffle(kinds)
        for kind in kinds:
            if kind == "prev_business_day":
                cand = [d for d in dates if d < correct_d]
                wd = cand[-1] if cand else None
            elif kind == "next_business_day":
                cand = [d for d in dates if d > correct_d]
                wd = cand[0] if cand else None
            elif kind == "due_date":
                wd = self.fx.rate_on_or_before(cur, self.base, due.astimezone(self.tz).date())[0] if due else None
            elif kind == "naive_date":
                nd = disp.date()   # дата «как написано» в смещении документа
                wd = self.fx.rate_on_or_before(cur, self.base, nd)[0] if nd != instant.astimezone(self.tz).date() else None
            else:
                wd = None
            if wd is None or wd == correct_d:
                continue
            wr = self.fx.exact(cur, self.base, wd)
            if wr is None or wr == correct_r:
                continue
            return kind, wd, wr
        return None

    def _material(self, a: int, b: int) -> bool:
        return abs(a - b) >= MIN_MATERIAL_MINOR

    def inject_errors(self):
        cfg = self.cfg
        opts = cfg["error_options"]
        plan = []
        for name in sorted(cfg["interactions"]):
            plan += [("ix", name, i) for i in range(int(cfg["interactions"][name]))]
        for t in sorted(cfg["errors"]):
            plan += [("err", t, i) for i in range(int(cfg["errors"][t]))]
        for kind, name, i in plan:
            R = sub_rng(self.seed, f"error:{name}:{i}")
            for attempt in range(40):
                ok = getattr(self, f"_inject_{name}")(R, opts, attempt)
                if ok:
                    break
            else:
                raise RuntimeError(f"could not inject {name}#{i}")

    # -- отдельные типы
    def _inject_duplicate_invoice(self, R, opts, attempt, interaction=None):
        ent = self._customer(R)
        t0 = self._instant(R, self.start, self.end - timedelta(days=3))
        orig = self._make_invoice(R, ent, t0, tags=["error_target:duplicate_original"])
        if R.random() < 0.5 and t0 + timedelta(days=1) < self.end:
            self._make_payment(R, ent.key, ent.currency, self._instant(R, t0 + timedelta(hours=3), self.end),
                               orig.total, [[orig.key, orig.total]])
        t1 = (orig.issued_at + timedelta(minutes=R.randint(10, 60 * 48))).astimezone(self.tz)
        disp1 = self._display(R, t1)
        dup = Invoice(key=self._next("inv"), entity=ent.key, issued_at=disp1,
                      due_at=disp1 + (orig.due_at - orig.issued_at), currency=orig.currency,
                      total=orig.total, lines=copy.deepcopy(orig.lines), tags=["error:duplicate_copy"])
        self.error_invoices[dup.key] = dup
        amt = convert_minor(dup.total, dup.currency, self.rate(dup.currency, dup.issued_at), self.base)
        self._issue("duplicate_invoice", [("invoice", dup.key)], amt,
                    [[("doc", orig.key), ("doc", dup.key)]], interaction,
                    cause="invoice entered twice; later copy posted to AR/revenue",
                    original=orig.key, duplicate=dup.key, delay_minutes=int((t1 - orig.issued_at).total_seconds() // 60))
        return True

    def _alloc_target(self, R, foreign: bool | None, big=False, pay_window=None, inv_window=None,
                      pay_shift=False, multi=False):
        ent = self._customer(R, foreign=foreign)
        a, b = inv_window or (self.start, self.end - timedelta(days=4))
        inv = self._make_invoice(R, ent, self._instant(R, a, b), big=big, tags=["error_target"])
        pa, pb = pay_window or (inv.issued_at + timedelta(hours=3), self.end - timedelta(minutes=10))
        refs = [[inv.key, inv.total]]
        amount = inv.total
        if multi:
            other = self._make_invoice(R, ent, self._instant(R, a, b), tags=["error_target:co_invoice"])
            refs.append([other.key, other.total])
            amount += other.total
            pa = max(pa, other.issued_at + timedelta(hours=3))
        p = self._make_payment(R, ent.key, ent.currency, self._instant(R, pa, pb), amount, refs,
                               tags=["error_target"], display_shift=pay_shift)
        return ent, inv, p

    def _shift(self, R, p: Payment, inv: Invoice, mode: str) -> int:
        alloc = dict((k, a) for k, a in p.refs)[inv.key]
        credit = convert_minor(alloc, inv.currency, self.rate(inv.currency, inv.issued_at), self.base)
        if mode == "missing":
            return credit
        return max(MIN_MATERIAL_MINOR, int(credit * R.uniform(0.2, 0.8)))

    def _apply_alloc(self, R, p, inv, mode, interaction=None):
        shift = self._shift(R, p, inv, mode)
        alloc = dict((k, a) for k, a in p.refs)[inv.key]
        credit = convert_minor(alloc, inv.currency, self.rate(inv.currency, inv.issued_at), self.base)
        if shift > credit or shift < MIN_MATERIAL_MINOR:
            return False
        self.ov.alloc_shift[(p.key, inv.key)] = shift
        self._issue("payment_allocation", [("payment", p.key), ("invoice", inv.key)], shift,
                    [[("doc", p.key), ("row", f"{p.key}:unapplied")]], interaction,
                    cause=f"{mode} allocation: part of AR credit posted to unapplied_cash",
                    mode=mode, payment=p.key, invoice=inv.key, expected_ar_credit=credit,
                    posted_ar_credit=credit - shift)
        return True

    def _inject_payment_allocation(self, R, opts, attempt, interaction=None):
        mode = R.choice(opts["payment_allocation_modes"])
        _, inv, p = self._alloc_target(R, foreign=None, multi=R.random() < 0.3)
        return self._apply_alloc(R, p, inv, mode)

    def _apply_fx_invoice(self, R, inv: Invoice, kinds, interaction=None):
        w = self._wrong_rate(R, inv.currency, inv.issued_at, inv.issued_at, inv.due_at, kinds)
        if not w:
            return False
        kind, wd, wr = w
        good = convert_minor(inv.total, inv.currency, self.rate(inv.currency, inv.issued_at), self.base)
        bad = convert_minor(inv.total, inv.currency, wr, self.base)
        if not self._material(good, bad):
            return False
        self.ov.inv_rate[inv.key] = wr
        self._issue("fx_conversion", [("invoice", inv.key)], abs(bad - good),
                    [[("doc", inv.key), ("row", f"{inv.key}:ar")], [("doc", inv.key), ("row", f"{inv.key}:rev")]],
                    interaction, cause=f"invoice converted at wrong rate ({kind})", kind=kind, invoice=inv.key,
                    wrong_rate_date=wd.isoformat(), wrong_rate=fmt_rate(wr),
                    correct_rate_date=self.fx.rate_on_or_before(inv.currency, self.base,
                                                                policy_rate_date(inv.issued_at, self.tz))[0].isoformat(),
                    posted_minor=bad, correct_minor=good)
        return True

    def _apply_fx_payment(self, R, p: Payment, kinds, interaction=None):
        w = self._wrong_rate(R, p.currency, p.timestamp, p.timestamp, None, kinds)
        if not w:
            return False
        kind, wd, wr = w
        good = convert_minor(p.amount, p.currency, self.rate(p.currency, p.timestamp), self.base)
        bad = convert_minor(p.amount, p.currency, wr, self.base)
        if not self._material(good, bad):
            return False
        self.ov.pay_rate[p.key] = wr
        self._issue("fx_conversion", [("payment", p.key)], abs(bad - good),
                    [[("doc", p.key), ("row", f"{p.key}:cash")]], interaction,
                    cause=f"payment converted at wrong rate ({kind})", kind=kind, payment=p.key,
                    wrong_rate_date=wd.isoformat(), wrong_rate=fmt_rate(wr), posted_minor=bad, correct_minor=good)
        return True

    def _inject_fx_conversion(self, R, opts, attempt, interaction=None):
        target = R.choice(opts["fx_targets"])
        kinds = list(opts["fx_kinds"])
        if target == "invoice":
            ent = self._customer(R, foreign=True)
            shift = "naive_date" in kinds and R.random() < 0.5
            if shift:  # момент рядом с полуночью, чтобы «дата на бумаге» отличалась
                d = self.start + timedelta(days=R.randint(1, 25))
                inst = self._instant(R, d - timedelta(hours=4), d + timedelta(hours=4))
            else:
                inst = self._instant(R, self.start, self.end - timedelta(days=2))
            inv = self._make_invoice(R, ent, inst, big=True, display_shift=shift, tags=["error_target"])
            if R.random() < 0.4 and inv.issued_at + timedelta(days=1) < self.end:
                self._make_payment(R, ent.key, ent.currency, self._instant(R, inv.issued_at + timedelta(hours=2), self.end),
                                   inv.total, [[inv.key, inv.total]])
            if self._apply_fx_invoice(R, inv, kinds):
                return True
            self._retract_invoice(inv)
            return False
        _, inv, p = self._alloc_target(R, foreign=True, big=True, pay_shift=R.random() < 0.5)
        if self._apply_fx_payment(R, p, [k for k in kinds if k != "due_date"]):
            return True
        self._retract_invoice(inv, p)
        return False

    def _retract_invoice(self, inv, p=None):
        """Неудачная попытка: цель остаётся обычным корректным документом (не ошибка)."""
        inv.tags = [t for t in inv.tags if not t.startswith("error")]
        if p is not None:
            p.tags = [t for t in p.tags if not t.startswith("error")]

    def _inject_period_cutoff(self, R, opts, attempt, interaction=None):
        ent = self._customer(R, foreign=False if R.random() < 0.7 else None)
        trap = R.random() < opts["cutoff_tz_trap_share"]
        # «ловушка» возможна, только если есть смещение документа западнее пояса компании
        gap = self.tz.utcoffset(None) - min(tz_from_str(o).utcoffset(None) for o in DOC_OFFSETS)
        if trap and gap > timedelta(minutes=30):
            inst = self._instant(R, self.end + timedelta(minutes=1), self.end + gap - timedelta(minutes=10))
        else:
            inst = self._instant(R, self.end, self.end + timedelta(days=2))
        inv = self._make_invoice(R, ent, inst, register=False, tags=["error:cutoff"])
        cause = "next-period invoice backdated into period"
        if trap:
            # документ записан в смещении, где «дата на бумаге» ещё в периоде,
            # а система проводок прочитала локальное время без смещения
            opts_off = [o for o in DOC_OFFSETS
                        if inst.astimezone(tz_from_str(o)).replace(tzinfo=self.tz) < self.end]
            if opts_off:
                disp = inst.astimezone(tz_from_str(R.choice(opts_off)))
                inv.issued_at = disp
                inv.due_at = disp + timedelta(days=30)
                ledger_ts = disp.replace(tzinfo=self.tz)
                cause = "offset dropped: document wall-clock time read in company timezone"
            else:
                trap = False
        if not trap:
            ledger_ts = self._instant(R, self.end - timedelta(hours=8), self.end - timedelta(minutes=1))
        if not (self.start <= ledger_ts < self.end):
            return False
        self.error_invoices[inv.key] = inv
        self.ov.inv_ts[inv.key] = ledger_ts
        amt = convert_minor(inv.total, inv.currency, self.rate(inv.currency, inv.issued_at), self.base)
        self._issue("period_cutoff", [("invoice", inv.key)], amt,
                    [[("doc", inv.key), ("row", f"{inv.key}:ar")], [("doc", inv.key), ("row", f"{inv.key}:rev")]],
                    interaction, cause=cause, invoice=inv.key, tz_trap=trap,
                    document_instant=iso(inst.astimezone(self.tz)), ledger_timestamp=iso(ledger_ts))
        return True

    # -- взаимодействующие сценарии
    def _inject_fx_invoice_with_partial_payment(self, R, opts, attempt):
        """Счёт в иностранной валюте проведён по неверному курсу, а его оплата
        распределена частично. Сумма недоразнесения считается по ВЕРНОМУ курсу счёта,
        поэтому без исправления FX-ошибки вторую ошибку правильно не оценить."""
        name = "fx_invoice_with_partial_payment"
        _, inv, p = self._alloc_target(R, foreign=True, big=True)
        kinds = [k for k in opts["fx_kinds"] if k != "naive_date"] or ["prev_business_day"]
        snap = (dict(self.ov.inv_rate), len(self.issues), len(self.changes))
        if not self._apply_fx_invoice(R, inv, kinds, interaction=name):
            self._retract_invoice(inv, p)
            return False
        if not self._apply_alloc(R, p, inv, "partial", interaction=name):
            self.ov.inv_rate, self.issues[:], self.changes[:] = snap[0], self.issues[:snap[1]], self.changes[:snap[2]]
            self._retract_invoice(inv, p)
            return False
        return True

    def _inject_partial_fx_payment_at_boundary(self, R, opts, attempt):
        """Частичный платёж в иностранной валюте в последний день периода; смещение
        документа сдвигает «дату на бумаге» на предыдущий день, и курс взят по ней."""
        name = "partial_fx_payment_at_boundary"
        last_day = self.end - timedelta(days=1)
        pw = (last_day + timedelta(minutes=5), last_day + timedelta(hours=10))
        _, inv, p = self._alloc_target(R, foreign=True, big=True, pay_window=pw,
                                       inv_window=(self.start, self.end - timedelta(days=5)), pay_shift=True)
        snap = (dict(self.ov.pay_rate), len(self.issues), len(self.changes))
        if not self._apply_fx_payment(R, p, ["naive_date"] if attempt < 20 else ["naive_date", "prev_business_day"],
                                      interaction=name):
            self._retract_invoice(inv, p)
            return False
        if not self._apply_alloc(R, p, inv, "partial", interaction=name):
            self.ov.pay_rate, self.issues[:], self.changes[:] = snap[0], self.issues[:snap[1]], self.changes[:snap[2]]
            self._retract_invoice(inv, p)
            return False
        return True


def _rng_pair(v):
    if isinstance(v, (list, tuple)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


# ----------------------------------------------------------------------------- рендер
class IdMapper:
    PREFIX = {"cust": "C", "vend": "V", "invoice": "INV", "payment": "PAY", "doc": "D", "row": "L"}

    def __init__(self, seed: int, salt: str):
        self.seed, self.salt = seed, salt
        self.map: dict[tuple[str, str], str] = {}
        self.used: set[str] = set()

    def get(self, kind: str, canon: str) -> str:
        k = (kind, canon)
        if k not in self.map:
            h = hashlib.sha256(f"{self.seed}|{self.salt}|{kind}|{canon}".encode()).hexdigest().upper()
            n = 8
            while f"{self.PREFIX[kind]}-{h[:n]}" in self.used:
                n += 1
            pid = f"{self.PREFIX[kind]}-{h[:n]}"
            self.used.add(pid)
            self.map[k] = pid
        return self.map[k]

    def ent(self, canon: str | None) -> str | None:
        if canon is None:
            return None
        return self.get("cust" if canon.startswith("cust") else "vend", canon)


def _render(g: _Gen, rows: list[Row], invoices: list[Invoice], payments: list[Payment], M: IdMapper,
            order_seed) -> dict:
    ledger = [{
        "row_id": M.get("row", r.key), "timestamp": iso(r.ts), "account": r.account,
        "debit_minor": r.debit, "credit_minor": r.credit, "currency": g.base,
        "entity_id": M.ent(r.entity),
        "invoice_id": M.get("invoice", r.invoice) if r.invoice else None,
        "payment_id": M.get("payment", r.payment) if r.payment else None,
    } for r in rows]
    inv_out = [{
        "doc_id": M.get("doc", i.key), "invoice_id": M.get("invoice", i.key), "entity_id": M.ent(i.entity),
        "issued_at": iso(i.issued_at), "due_at": iso(i.due_at), "currency": i.currency,
        "total_minor": i.total, "line_items": copy.deepcopy(i.lines),
    } for i in invoices]
    pay_out = [{
        "doc_id": M.get("doc", p.key), "payment_id": M.get("payment", p.key), "timestamp": iso(p.timestamp),
        "currency": p.currency, "amount_minor": p.amount,
        "invoice_refs": None if p.refs is None else
        [{"invoice_id": M.get("invoice", k), "amount_minor": a} for k, a in p.refs],
    } for p in payments]
    fx = copy.deepcopy(g.fx_rows)

    def chron(lst, tkey, idkey):
        return sorted(lst, key=lambda x: (datetime.fromisoformat(x[tkey]), x[idkey]))
    ledger = chron(ledger, "timestamp", "row_id")
    inv_out = chron(inv_out, "issued_at", "doc_id")
    pay_out = chron(pay_out, "timestamp", "doc_id")
    if order_seed is not None:
        for name, lst in (("ledger", ledger), ("invoices", inv_out), ("payments", pay_out), ("fx", fx)):
            sub_rng(int(order_seed), f"order:{name}").shuffle(lst)
    return {"ledger": ledger, "invoices": inv_out, "payments": pay_out, "fx": fx}


def build_policy(g: _Gen | None = None) -> dict:
    return {
        "rate_source": "fx.csv; rate = units of target_currency per 1 unit of source_currency",
        "rate_date_rule": "calendar date of the transaction instant in the company timezone "
                          "(manifest.timezone); if fx.csv has no rate for that date, use the latest earlier date",
        "invoice_rate_event": "issued_at",
        "payment_rate_event": "timestamp",
        "ar_credit_rule": "AR is credited for each invoice_ref at the invoice's policy rate; "
                          "difference with the cash debit is posted to fx_gain_loss",
        "unreferenced_payments": "payments with invoice_refs = null are posted to unapplied_cash; "
                                 "negative amount = refund (debit unapplied_cash, credit cash)",
        "period_boundary": "an event belongs to the period iff period_start <= instant < period_end "
                           "(instants compared with their offsets)",
        "rounding": {"mode": "ROUND_HALF_UP", "unit": "minor unit of base currency",
                     "stage": "once per converted document total or per invoice_ref allocation"},
        "money": "integer minor units; exponent per currency in manifest.currency_exponents",
        "ledger_currency": "all ledger rows are in base currency",
        "accounts": ACCOUNTS,
        "reporting": {
            "claimed_amount_minor": "absolute misstatement in base-currency minor units",
            "currency": "base currency",
            "affected_ids": {
                "duplicate_invoice": "[invoice_id of the later-issued copy]",
                "payment_allocation": "[payment_id, invoice_id]",
                "fx_conversion": "[invoice_id] or [payment_id] whose conversion is wrong",
                "period_cutoff": "[invoice_id]",
            },
            "evidence_ids": "doc_id and/or ledger row_id values that prove the finding",
        },
        "tolerance": {"amount_abs_minor": 1, "amount_rel": "0.0005"},
    }


TASK_TEXT = (
    "You are auditing one company for one reporting period. Records are read-only and reachable "
    "only through tools. Find misstatements of these types: duplicate_invoice, payment_allocation "
    "(missing or partial application of a payment to the invoices it references), fx_conversion "
    "(wrong rate date or rate), period_cutoff (invoice of another period posted in this period). "
    "Some companies are clean; legitimate split payments, refunds, unreferenced payments and "
    "recurring amounts are not errors. Submit findings once: "
    "{episode_id, findings:[{type, affected_ids, claimed_amount_minor, currency, evidence_ids, explanation}]}. "
    "Limits: 30 findings, 10 evidence ids, 1000 explanation chars. Budget: 100 steps including submit."
)


def generate(seed: int, config: dict | None = None) -> Episode:
    cfg = resolve_config(config)
    g = _Gen(seed, cfg)
    g.build_history()
    g.inject_errors()
    # Книги «до ошибок»: та же история (включая цели ошибок), без override и лишних документов.
    clean_rows = g.post(with_errors=False)
    err_rows = g.post(with_errors=True)

    M = IdMapper(seed, str(cfg["id_salt"]))
    inv_err = list(g.invoices.values()) + list(g.error_invoices.values())
    tables = _render(g, err_rows, inv_err, list(g.payments.values()), M, cfg["order_seed"])
    clean_tables = _render(g, clean_rows, list(g.invoices.values()), list(g.payments.values()), M, None)

    episode_id = "EP-" + hashlib.sha256(f"{seed}|{canonical_json(cfg)}".encode()).hexdigest()[:12].upper()

    def pub(kind, canon):
        return M.get({"invoice": "invoice", "payment": "payment", "doc": "doc", "row": "row"}[kind], canon)

    issues = []
    for i, iss in enumerate(g.issues):
        issues.append({
            "issue_id": f"ISS-{i + 1:02d}",
            "type": iss["type"],
            "affected_ids": [pub(k, c) for k, c in iss["affected"]],
            "expected_amount_minor": iss["amount"],
            "currency": g.base,
            "evidence_sets": [[pub(k, c) for k, c in s] for s in iss["evidence"]],
            "interaction": iss["interaction"],
        })
    truth = {"episode_id": episode_id, "issues": issues}

    rows_by_key = {r.key: r for r in err_rows}
    clean_by_key = {r.key: r for r in clean_rows}

    def row_view(r):
        return None if r is None else {"row_id": M.get("row", r.key), "account": r.account,
                                        "debit_minor": r.debit, "credit_minor": r.credit, "timestamp": iso(r.ts)}

    changes = []
    for ch in g.changes:
        ch = dict(ch)
        keys = [v for k, v in ch.items() if k in ("invoice", "payment", "original", "duplicate")]
        touched = sorted({rk for rk in set(rows_by_key) | set(clean_by_key)
                          if any(rk.startswith(k + ":") or rk.endswith(":" + k) for k in keys)})
        ch["rows_before"] = [row_view(clean_by_key.get(k)) for k in touched if k in clean_by_key]
        ch["rows_after"] = [row_view(rows_by_key.get(k)) for k in touched if k in rows_by_key]
        for k in ("invoice", "duplicate", "original"):
            if k in ch:
                ch[k] = M.get("invoice", ch[k])
        if "payment" in ch:
            ch["payment"] = M.get("payment", ch["payment"])
        ch["issue_id"] = issues[ch.pop("issue_index")]["issue_id"]
        changes.append(ch)

    def pub_any(c):
        if c.startswith("inv"):
            return M.get("invoice", c)
        if c.startswith("pay"):
            return M.get("payment", c)
        return c
    dlog = [{k: ([pub_any(x) for x in v] if isinstance(v, list) else pub_any(v) if isinstance(v, str) and k != "kind" else v)
             for k, v in d.items()} for d in g.distractor_log]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "env_version": ENV_VERSION,
        "episode_id": episode_id,
        "company": g.company_name,
        "base_currency": g.base,
        "currency_exponents": {c: CURRENCY_EXPONENT[c] for c in sorted({g.base, *g.foreign})},
        "period_start": iso(g.start),
        "period_end": iso(g.end),
        "period_end_exclusive": True,
        "timezone": g.tz_str,
        "counts": {
            "ledger_rows": len(tables["ledger"]), "invoices": len(tables["invoices"]),
            "payments": len(tables["payments"]), "fx_rates": len(tables["fx"]),
            "documents": len(tables["invoices"]) + len(tables["payments"]),
        },
        "rounding": {"mode": "ROUND_HALF_UP", "unit": "minor unit of base currency"},
        "null_encoding": "JSON null; empty field in CSV",
    }
    provenance = {
        "seed": int(seed), "env_version": ENV_VERSION, "config": cfg, "episode_id": episode_id,
        "setting": {"base": g.base, "foreign": g.foreign, "timezone": g.tz_str,
                    "period_start": iso(g.start), "period_end": iso(g.end)},
        "history": {"invoices": len(g.invoices), "payments": len(g.payments), "expenses": len(g.expenses),
                    "unposted_next_period": len(g.unposted)},
        "changes": changes,
        "distractors": dlog,
    }
    return Episode(episode_id, int(seed), cfg, manifest, build_policy(g), tables, truth, provenance, clean_tables)
