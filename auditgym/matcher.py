"""Общий matcher и терминальная награда (открытый код). Правила совпадают с опубликованными
правилами жюри FinTech (genesis-exe.com/fintech-checks.html, 20.11.2026).

Сравнение — точное, без нормализации:
  * ID, type и currency сравниваются как строки, с учётом регистра и пробелов;
  * affected_ids — точное множество (порядок и повторы внутри списка не важны);
  * claimed_amount_minor — целое JSON-число (не bool, не строка, не дробь);
  * допуск суммы tau = 0 по умолчанию: |claimed - expected| <= tau
    (tau публикуется в policy.json, ключ tolerance.amount_minor; у всех эпизодов 0).
Доказательства: evidence_ids — множество; в нём должен быть хотя бы один ПОЛНЫЙ допустимый набор,
а каждый ID должен существовать в эпизоде (недоступный ID делает вывод FP, ничего не раскрывая).
Структурные ошибки (отправка целиком, награда -1): не объект, лишние/недостающие ключи, чужой
episode_id, больше 30 выводов, больше 10 evidence_ids или 30 affected_ids (до удаления повторов),
explanation длиннее 1000 кодовых точек, сумма не целое число, поля не тех типов.
Хорошо сформированный, но неверный вывод — FP.
Неоднозначность: паросочетание максимальной мощности (алгоритм Куна); среди равных выбирается
лексикографически наименьшая последовательность пар (issue_id, индекс вывода) — правило жюри. Несопоставленные и
повторные выводы — FP, несопоставленные эталонные ошибки — FN.
"""
from __future__ import annotations

from fractions import Fraction
from decimal import Decimal

DEFAULT_TAU = 0
MAX_FINDINGS = 30
MAX_EVIDENCE = 10
MAX_AFFECTED = 30
MAX_EXPLANATION = 1000
STEP_COST = Decimal("0.002")


class InvalidSubmission(ValueError):
    pass


def _ids(v, field, limit):
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise InvalidSubmission(f"{field} must be a list of strings")
    if len(v) > limit:          # лимит проверяется до удаления повторов
        raise InvalidSubmission(f"{field} exceeds {limit}")
    return list(v)


def validate_submission(sub, episode_id: str) -> list[dict]:
    """Проверка структуры findings.json. Любая ошибка -> InvalidSubmission."""
    if not isinstance(sub, dict):
        raise InvalidSubmission("submission must be an object")
    if set(sub) != {"episode_id", "findings"}:
        raise InvalidSubmission("submission must have exactly keys episode_id, findings")
    if sub.get("episode_id") != episode_id:
        raise InvalidSubmission("episode_id mismatch")
    fs = sub.get("findings")
    if not isinstance(fs, list):
        raise InvalidSubmission("findings must be a list")
    if len(fs) > MAX_FINDINGS:
        raise InvalidSubmission(f"more than {MAX_FINDINGS} findings")
    out = []
    keys = {"type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"}
    for f in fs:
        if not isinstance(f, dict) or set(f) != keys:
            raise InvalidSubmission("finding must have exactly keys " + ",".join(sorted(keys)))
        if not isinstance(f["type"], str):
            raise InvalidSubmission("type must be a string")
        amt = f["claimed_amount_minor"]
        if not isinstance(amt, int) or isinstance(amt, bool):
            raise InvalidSubmission("claimed_amount_minor must be a JSON integer")
        if not isinstance(f["currency"], str):
            raise InvalidSubmission("currency must be a string")
        ex = f["explanation"]
        if not isinstance(ex, str) or len(ex) > MAX_EXPLANATION:
            raise InvalidSubmission(f"explanation must be a string of at most {MAX_EXPLANATION} code points")
        aff = _ids(f["affected_ids"], "affected_ids", MAX_AFFECTED)
        if not aff:
            raise InvalidSubmission("affected_ids must be non-empty")
        out.append({"type": f["type"], "affected_ids": aff, "claimed_amount_minor": amt,
                    "currency": f["currency"],
                    "evidence_ids": _ids(f["evidence_ids"], "evidence_ids", MAX_EVIDENCE),
                    "explanation": ex})
    return out


def amount_ok(claimed: int, expected: int, tau: int = DEFAULT_TAU) -> bool:
    return abs(claimed - expected) <= tau


def compare(f: dict, t: dict, visible: set | None = None, tau: int = DEFAULT_TAU) -> str | None:
    """None если пара совместима, иначе код причины (для диагностики)."""
    if f["type"] != t["type"]:
        return "type"
    if set(f["affected_ids"]) != set(t["affected_ids"]):
        return "affected_ids"
    if f["currency"] != t["currency"]:
        return "currency"
    if not amount_ok(f["claimed_amount_minor"], t["expected_amount_minor"], tau):
        return "amount"
    ev = set(f["evidence_ids"])
    if visible is not None and not ev <= visible:
        return "evidence_unavailable"
    if not any(set(s) <= ev for s in t["evidence_sets"]):
        return "evidence"
    return None


def match(findings: list[dict], issues: list[dict], visible: set | None = None, tau: int = DEFAULT_TAU) -> dict:
    issues = sorted(issues, key=lambda t: t["issue_id"])
    adj = [[j for j, t in enumerate(issues) if compare(f, t, visible, tau) is None] for f in findings]

    def max_card(banned_f: set, banned_i: set) -> int:
        owner = {}

        def aug(i, seen):
            for j in adj[i]:
                if j in banned_i or j in seen:
                    continue
                seen.add(j)
                if j not in owner or aug(owner[j], seen):
                    owner[j] = i
                    return True
            return False
        return sum(aug(i, set()) for i in range(len(findings)) if i not in banned_f)

    # максимальная мощность, затем детерминированный выбор (правило жюри): эталонные ошибки по
    # возрастанию issue_id, каждой — наименьший индекс вывода, при котором мощность сохраняется
    best = max_card(set(), set())
    matched, used_f, done_i = {}, set(), set()
    for j in range(len(issues)):
        if len(matched) == best:
            break
        for i in range(len(findings)):
            if i in used_f or j not in adj[i]:
                continue
            if len(matched) + 1 + max_card(used_f | {i}, done_i | {j}) == best:
                matched[i] = j
                used_f.add(i)
                break
        done_i.add(j)
    owner = [-1] * len(issues)
    for i, j in matched.items():
        owner[j] = i
    tp = len(matched)
    fp = len(findings) - tp
    fn = len(issues) - tp
    near = []
    for i, f in enumerate(findings):
        if i in matched:
            continue
        reasons = [compare(f, t, visible, tau) for t in issues if f["type"] == t["type"]
                   and set(f["affected_ids"]) == set(t["affected_ids"])]
        if not reasons:
            reason = "no_issue_with_these_ids"
        elif reasons[0] is None:
            reason = "duplicate_of_matched"
        else:
            reason = reasons[0]
        near.append({"finding_index": i, "type": f["type"], "reason": reason, "affected_ids": f["affected_ids"]})
    return {"tp": tp, "fp": fp, "fn": fn,
            "pairs": [{"finding_index": i, "issue_id": issues[j]["issue_id"]} for i, j in sorted(matched.items())],
            "missed_issue_ids": [issues[j]["issue_id"] for j in range(len(issues)) if owner[j] == -1],
            "unmatched_findings": near}


def metrics(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    if tp + fp + fn == 0:
        f1 = 1.0          # эталон и выводы пусты
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"precision": p, "recall": r, "f1": f1}


def terminal_reward(f1, inspection_steps: int) -> float:
    """clip(2*F1 - 1 - 0.002*steps, -1, 1), посчитано точно (Fraction) и округлено один раз.
    f1 может быть Fraction (точное значение из TP/FP/FN) или float."""
    f = f1 if isinstance(f1, Fraction) else Fraction(Decimal(str(f1)))
    v = 2 * f - 1 - Fraction(STEP_COST) * inspection_steps
    return float(min(Fraction(1), max(Fraction(-1), v)))


def exact_f1(tp: int, fp: int, fn: int) -> Fraction:
    return Fraction(1) if tp + fp + fn == 0 else Fraction(2 * tp, 2 * tp + fp + fn)


def visible_ids(tables: dict) -> set:
    """ID, которые агент может получить инструментами: row_id проводок, doc_id документов,
    ключи курсов DATE:SRC:TGT."""
    ids = {r["row_id"] for r in tables.get("ledger", [])}
    for t in ("invoices", "payments"):
        ids |= {d["doc_id"] for d in tables.get(t, [])}
    ids |= {f"{r['date']}:{r['source_currency']}:{r['target_currency']}" for r in tables.get("fx", [])}
    return ids


def tau_from_policy(policy: dict | None) -> int:
    v = ((policy or {}).get("tolerance") or {}).get("amount_minor", DEFAULT_TAU)
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else DEFAULT_TAU


def grade(submission, truth: dict, inspection_steps: int, visible: set | None = None,
          tau: int = DEFAULT_TAU) -> dict:
    """Полная оценка одной отправки. Неверная структура -> награда -1.
    visible — ID, доступные агенту (недоступный evidence-ID делает вывод FP); tau — допуск суммы."""
    try:
        fs = validate_submission(submission, truth["episode_id"])
    except InvalidSubmission as e:
        return {"valid": False, "error": str(e), "tp": 0, "fp": 0, "fn": len(truth["issues"]),
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "reward": -1.0,
                "inspection_steps": inspection_steps}
    m = match(fs, truth["issues"], visible, tau)
    mt = metrics(m["tp"], m["fp"], m["fn"])
    return {"valid": True, **m, **mt, "reward": terminal_reward(exact_f1(m["tp"], m["fp"], m["fn"]), inspection_steps),
            "inspection_steps": inspection_steps}
