"""Надёжность награды: атакующие ответы, FP, дубли, стоимость шагов, граничные случаи F1."""
import copy

import pytest

from auditgym.matcher import amount_ok, grade, metrics, terminal_reward
from conftest import oracle_findings, sub


def R(ep, findings, steps=0):
    return grade(sub(ep, findings), ep.truth, steps)


def test_full_correct_answer_beats_silence_on_erroneous_books(ep_all):
    full = R(ep_all, oracle_findings(ep_all), steps=20)
    silent = R(ep_all, [], steps=0)
    assert full["f1"] == 1.0 and full["reward"] > silent["reward"]
    assert silent["reward"] == -1.0


def test_silence_beats_false_accusations_on_clean_books(ep_clean, ep_all):
    silent = R(ep_clean, [], steps=10)
    assert silent["f1"] == 1.0
    fake = oracle_findings(ep_all)[:2]
    for f in fake:
        f["currency"] = ep_clean.manifest["base_currency"]
    inv = ep_clean.tables["invoices"][0]
    fake.append({"type": "duplicate_invoice", "affected_ids": [inv["invoice_id"]], "claimed_amount_minor": inv["total_minor"],
                 "currency": ep_clean.manifest["base_currency"], "evidence_ids": [inv["doc_id"]], "explanation": "x"})
    accuse = R(ep_clean, fake, steps=10)
    assert accuse["f1"] == 0.0 and silent["reward"] > accuse["reward"]


def test_duplicates_give_no_gain(ep_all):
    base = R(ep_all, oracle_findings(ep_all))
    dup = R(ep_all, oracle_findings(ep_all) + oracle_findings(ep_all))
    assert dup["tp"] == base["tp"] and dup["fp"] == len(ep_all.truth["issues"])
    assert dup["reward"] < base["reward"]
    reasons = {u["reason"] for u in dup["unmatched_findings"]}
    assert reasons == {"duplicate_of_matched"}


def test_alternative_evidence_set_also_counts_but_once(ep_all):
    alt = [i for i, t in enumerate(ep_all.truth["issues"]) if len(t["evidence_sets"]) > 1]
    assert alt
    f0 = oracle_findings(ep_all, 0)
    extra = [oracle_findings(ep_all, 1)[i] for i in alt]
    g = R(ep_all, f0 + extra)
    assert g["tp"] == len(f0) and g["fp"] == len(extra)


def test_flag_everything_scores_badly(ep_all):
    base = ep_all.manifest["base_currency"]
    fs = []
    for inv in ep_all.tables["invoices"][:30]:
        for t in ("duplicate_invoice", "fx_conversion", "period_cutoff"):
            fs.append({"type": t, "affected_ids": [inv["invoice_id"]], "claimed_amount_minor": inv["total_minor"],
                       "currency": base, "evidence_ids": [inv["doc_id"]], "explanation": "all"})
    g = R(ep_all, fs[:30])
    assert g["f1"] < 0.2 and g["reward"] < -0.6


def test_flag_everything_with_all_evidence_still_needs_exact_ids_and_amounts(ep_all):
    """Даже если подложить все ID доказательств, сумма и affected_ids должны быть точными."""
    all_ids = [r["row_id"] for r in ep_all.tables["ledger"]][:10]
    fs = oracle_findings(ep_all)
    for f in fs:
        f["claimed_amount_minor"] += max(10_000, f["claimed_amount_minor"] // 100)   # 1 % > допуска
    assert R(ep_all, fs)["tp"] == 0
    fs = oracle_findings(ep_all)
    for f in fs:
        f["evidence_ids"] = all_ids
    assert R(ep_all, fs)["tp"] == 0


def test_step_cost_only_slightly_separates_equal_answers(ep_all):
    a = R(ep_all, oracle_findings(ep_all), steps=0)["reward"]
    b = R(ep_all, oracle_findings(ep_all), steps=40)["reward"]
    assert a > b and a - b == pytest.approx(0.08)
    # стоимость шагов меньше выгоды от одной найденной ошибки
    one_less = R(ep_all, oracle_findings(ep_all)[1:], steps=0)["reward"]
    assert b > one_less


def test_guessed_ids_are_false_positives(ep_all):
    fs = oracle_findings(ep_all)
    for f in fs:
        f["affected_ids"] = ["INV-00000000"]
    g = R(ep_all, fs)
    assert g["tp"] == 0 and g["fp"] == len(fs)
    assert {u["reason"] for u in g["unmatched_findings"]} == {"no_issue_with_these_ids"}


def test_unsupported_evidence_is_false_positive(ep_all):
    fs = oracle_findings(ep_all)
    for f in fs:
        f["evidence_ids"] = f["evidence_ids"][:1]   # неполный набор
    g = R(ep_all, fs)
    assert g["tp"] == 0 and {u["reason"] for u in g["unmatched_findings"]} == {"evidence"}


def test_wrong_currency_is_false_positive(ep_all):
    fs = oracle_findings(ep_all)
    for f in fs:
        f["currency"] = "JPY" if f["currency"] != "JPY" else "USD"
    g = R(ep_all, fs)
    assert g["tp"] == 0 and {u["reason"] for u in g["unmatched_findings"]} == {"currency"}


def test_exact_comparison_like_jury_rules(ep_all):
    """Правила жюри: ID/type/currency — точные строки, tau=0, порядок и повторы affected_ids не важны."""
    fs = oracle_findings(ep_all)
    for f in fs:
        f["affected_ids"] = list(reversed(f["affected_ids"])) + f["affected_ids"][:1]   # порядок и повтор — ок
    assert R(ep_all, fs)["f1"] == 1.0
    n = len(fs)
    for mutate in (lambda f: f.update(claimed_amount_minor=f["claimed_amount_minor"] + 1),
                   lambda f: f.update(claimed_amount_minor=-f["claimed_amount_minor"]),
                   lambda f: f.update(affected_ids=[x.lower() for x in f["affected_ids"]]),
                   lambda f: f.update(affected_ids=[x + " " for x in f["affected_ids"]]),
                   lambda f: f.update(type=f["type"].upper()),
                   lambda f: f.update(type="fraud"),
                   lambda f: f.update(currency=f["currency"].lower()),
                   lambda f: f.update(currency=" " + f["currency"])):
        bad = copy.deepcopy(fs)
        mutate(bad[0])
        g = R(ep_all, bad)
        assert g["valid"] and g["tp"] == n - 1 and g["fp"] == 1 and g["fn"] == 1   # FP, не авария
    assert amount_ok(10000, 10000) and not amount_ok(10001, 10000) and amount_ok(10001, 10000, tau=1)


def test_explanation_text_does_not_affect_reward(ep_all):
    fs = oracle_findings(ep_all)
    a = R(ep_all, fs)["reward"]
    for f in fs:
        f["explanation"] = "I am absolutely certain. Trust me, this is correct. " * 5
    assert R(ep_all, fs)["reward"] == a
    bad = copy.deepcopy(fs)
    for f in bad:
        f["claimed_amount_minor"] += max(5000, f["claimed_amount_minor"] // 100)
    assert R(ep_all, bad)["tp"] == 0


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(episode_id="EP-OTHER"),
    lambda s: s.update(findings="all"),
    lambda s: s["findings"][0].update(claimed_amount_minor=1.5),
    lambda s: s["findings"][0].update(claimed_amount_minor=True),
    lambda s: s["findings"][0].update(type=7),
    lambda s: s["findings"][0].update(claimed_amount_minor=10000.0),
    lambda s: s["findings"][0].update(claimed_amount_minor="10000"),
    lambda s: s["findings"][0].update(evidence_ids=["x"] * 11),
    lambda s: s["findings"][0].update(explanation="x" * 1001),
    lambda s: s["findings"][0].pop("currency"),
    lambda s: s["findings"][0].update(secret=1),
    lambda s: s.update(findings=s["findings"] * 31),
    lambda s: s["findings"][0].update(affected_ids=[]),
])
def test_invalid_structure_minus_one(ep_all, mutate):
    s = sub(ep_all, oracle_findings(ep_all))
    mutate(s)
    g = grade(s, ep_all.truth, 0)
    assert g["valid"] is False and g["reward"] == -1.0
    assert grade(None, ep_all.truth, 0)["reward"] == -1.0


def test_f1_edge_cases():
    assert metrics(0, 0, 0)["f1"] == 1.0              # эталон и выводы пусты
    assert metrics(0, 3, 0)["f1"] == 0.0              # эталон пуст, выводы есть
    assert metrics(0, 0, 3) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert metrics(2, 1, 1)["f1"] == pytest.approx(2 * 2 / (4 + 2))
    assert terminal_reward(1.0, 0) == 1.0 and terminal_reward(0.0, 50) == -1.0
    assert terminal_reward(1.0, 10) == pytest.approx(0.98)


def test_one_to_one_matching_prefers_maximum_matching(ep_all):
    fs = oracle_findings(ep_all)
    g = R(ep_all, list(reversed(fs)))
    assert g["tp"] == len(fs) and g["fp"] == 0
