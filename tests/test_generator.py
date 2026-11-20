"""Генератор: независимые типы ошибок, сочетания, чистые случаи, отвлекающие случаи, равенства."""
import pytest

from auditgym.analysis import audit, ledger_balance_report
from auditgym.config import ERROR_TYPES, resolve_config
from auditgym.generator import generate
from auditgym.matcher import grade
from auditgym.validate import validate_episode


@pytest.mark.parametrize("etype", ERROR_TYPES)
def test_each_error_type_is_independently_configurable(etype):
    ep = generate(77, {"errors": {etype: 2}})
    assert [t["type"] for t in ep.truth["issues"]] == [etype, etype]
    assert validate_episode(ep)["ok"]


@pytest.mark.parametrize("name", ["fx_invoice_with_partial_payment", "partial_fx_payment_at_boundary"])
def test_interaction_scenarios_produce_linked_pairs(name):
    for seed in range(5):
        ep = generate(seed, {"interactions": {name: 1}})
        iss = ep.truth["issues"]
        assert sorted(t["type"] for t in iss) == ["fx_conversion", "payment_allocation"]
        assert all(t["interaction"] == name for t in iss)
        ids = [set(t["affected_ids"]) for t in iss]
        assert ids[0] & ids[1], "both errors must touch the same document"
        assert validate_episode(ep)["ok"]


def test_boundary_scenario_payment_is_on_last_business_day_of_period():
    from datetime import timedelta
    from auditgym.money import parse_iso, tz_from_str
    for seed in range(12):
        ep = generate(seed, {"interactions": {"partial_fx_payment_at_boundary": 1}})
        m = ep.manifest
        pid = [t for t in ep.truth["issues"] if t["type"] == "fx_conversion"][0]["affected_ids"][0]
        p = next(x for x in ep.tables["payments"] if x["payment_id"] == pid)
        tz = tz_from_str(m["timezone"])
        local = parse_iso(p["timestamp"]).astimezone(tz)
        end = parse_iso(m["period_end"])
        last = end - timedelta(days=1)
        while last.astimezone(tz).weekday() >= 5:      # в выходные курсов нет — последний рабочий день
            last -= timedelta(days=1)
        assert last <= local < last + timedelta(days=1)
        assert p["currency"] != m["base_currency"]
        # платёж «на бумаге» датирован предыдущим днём (ловушка naive_date)
        assert parse_iso(p["timestamp"]).date() < local.date()


def test_clean_company_has_no_issues_and_no_discrepancies():
    for seed in range(10):
        ep = generate(seed, {})
        assert ep.truth["issues"] == []
        assert audit(ep.tables, ep.manifest) == []


def test_distractors_present():
    ep = generate(11, {})
    kinds = {d["kind"] for d in ep.provenance["distractors"]}
    for k in ("recurring_amount", "unreferenced_payment", "tz_boundary_invoice", "next_period_unposted_invoice"):
        assert k in kinds
    assert any(p["amount_minor"] < 0 for p in ep.tables["payments"])          # возвраты
    assert any(p["invoice_refs"] is None for p in ep.tables["payments"])      # явные null
    assert any(r["account"] == "unapplied_cash" for r in ep.tables["ledger"])


def test_books_balance_before_and_after_errors():
    for seed in range(10):
        ep = generate(seed, {"errors": {t: 2 for t in ERROR_TYPES}})
        assert ledger_balance_report(ep.clean_tables["ledger"])["balanced"]
        assert ledger_balance_report(ep.tables["ledger"])["balanced"]


def test_truth_reproduces_from_public_records_only():
    for seed in range(15):
        ep = generate(seed, {"errors": {t: 1 for t in ERROR_TYPES},
                             "interactions": {"fx_invoice_with_partial_payment": 1}})
        g = grade({"episode_id": ep.episode_id, "findings": audit(ep.tables, ep.manifest)}, ep.truth, 0)
        assert g["f1"] == 1.0


def test_variation_across_seeds():
    eps = [generate(s, {"errors": {"fx_conversion": 1}}) for s in range(12)]
    assert len({e.manifest["base_currency"] for e in eps}) >= 3
    assert len({e.manifest["timezone"] for e in eps}) >= 4
    assert len({e.manifest["period_start"] for e in eps}) >= 6
    kinds = set()
    for s in range(30):
        kinds |= {c["kind"] for c in generate(s, {"errors": {"fx_conversion": 2}}).provenance["changes"]}
    assert len(kinds) >= 3


def test_history_independent_of_error_configuration():
    """Включение ошибки не меняет остальную компанию (независимые потоки случайности)."""
    a = generate(5, {})
    b = generate(5, {"errors": {"duplicate_invoice": 1}})
    a_ids = {i["invoice_id"] for i in a.tables["invoices"]}
    b_ids = {i["invoice_id"] for i in b.tables["invoices"]}
    assert a_ids < b_ids


def test_config_validation():
    with pytest.raises(ValueError):
        resolve_config({"errors": {"embezzlement": 1}})
    with pytest.raises(ValueError):
        resolve_config({"errors": {t: 4 for t in ERROR_TYPES}})
    with pytest.raises(ValueError):
        resolve_config({"bogus": 1})


def test_sizes_within_suggested_range():
    for seed in range(10):
        c = generate(seed, {"errors": {t: 2 for t in ERROR_TYPES}}).manifest["counts"]
        assert 200 <= c["ledger_rows"] <= 2000 and 20 <= c["documents"] <= 200
