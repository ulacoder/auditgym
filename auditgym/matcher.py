"""Общий matcher и терминальная награда (открытый код).

Нормализация
  * type: strip + lower;  ID и валюта: strip + upper;
  * affected_ids сравниваются как множества (порядок и повторы не важны);
  * claimed_amount_minor сравнивается по модулю.
Допуск суммы: |claimed| - expected| <= max(AMOUNT_ABS_TOL, ceil(expected * AMOUNT_REL_TOL)).
Доказательства: множество evidence_ids должно содержать хотя бы один ПОЛНЫЙ допустимый набор.
Совместимость пары (вывод, эталон) = совпали type, affected_ids, currency, сумма в допуске
и доказательства.
Неоднозначность: максимальное взаимно однозначное паросочетание (алгоритм Куна);
выводы обходятся в порядке отправки, эталонные ошибки — в порядке issue_id, поэтому
при равных вариантах выигрывает более ранний вывод. Несопоставленные и повторные
выводы — FP, несопоставленные эталонные ошибки — FN.
"""
from __future__ import annotations

import math
from decimal import Decimal

from .config import ERROR_TYPES

AMOUNT_ABS_TOL = 1
AMOUNT_REL_TOL = Decimal("0.0005")
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
    if len(v) > limit:
        raise InvalidSubmission(f"{field} exceeds {limit}")
    return [x.strip().upper() for x in v]


def validate_submission(sub, episode_id: str) -> list[dict]:
    """Проверка структуры findings.json. Любая ошибка -> InvalidSubmission."""
    if not isinstance(sub, dict):
        raise InvalidSubmission("submission must be an object")
    if set(sub) - {"episode_id", "findings"}:
        raise InvalidSubmission("unexpected top-level keys")
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
        t = f["type"]
        if not isinstance(t, str) or t.strip().lower() not in ERROR_TYPES:
            raise InvalidSubmission("unknown finding type")
        amt = f["claimed_amount_minor"]
        if not isinstance(amt, int) or isinstance(amt, bool):
            raise InvalidSubmission("claimed_amount_minor must be an integer")
        cur = f["currency"]
        if not isinstance(cur, str) or len(cur.strip()) != 3:
            raise InvalidSubmission("currency must be a 3-letter code")
        ex = f["explanation"]
        if not isinstance(ex, str) or len(ex) > MAX_EXPLANATION:
            raise InvalidSubmission(f"explanation must be a string <= {MAX_EXPLANATION} chars")
        aff = _ids(f["affected_ids"], "affected_ids", MAX_AFFECTED)
        if not aff:
            raise InvalidSubmission("affected_ids must be non-empty")
        out.append({"type": t.strip().lower(), "affected_ids": aff, "claimed_amount_minor": amt,
                    "currency": cur.strip().upper(),
                    "evidence_ids": _ids(f["evidence_ids"], "evidence_ids", MAX_EVIDENCE),
                    "explanation": ex})
    return out


def amount_ok(claimed: int, expected: int) -> bool:
    tol = max(AMOUNT_ABS_TOL, math.ceil(Decimal(abs(expected)) * AMOUNT_REL_TOL))
    return abs(abs(claimed) - abs(expected)) <= tol


def compare(f: dict, t: dict) -> str | None:
    """None если пара совместима, иначе код причины (для диагностики)."""
    if f["type"] != t["type"]:
        return "type"
    if set(f["affected_ids"]) != {x.upper() for x in t["affected_ids"]}:
        return "affected_ids"
    if f["currency"] != t["currency"].upper():
        return "currency"
    if not amount_ok(f["claimed_amount_minor"], t["expected_amount_minor"]):
        return "amount"
    ev = set(f["evidence_ids"])
    if not any({x.upper() for x in s} <= ev for s in t["evidence_sets"]):
        return "evidence"
    return None


def match(findings: list[dict], issues: list[dict]) -> dict:
    issues = sorted(issues, key=lambda t: t["issue_id"])
    adj = [[j for j, t in enumerate(issues) if compare(f, t) is None] for f in findings]
    owner = [-1] * len(issues)

    def try_assign(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] == -1 or try_assign(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(findings)):
        try_assign(i, set())
    matched = {owner[j]: j for j in range(len(issues)) if owner[j] != -1}
    tp = len(matched)
    fp = len(findings) - tp
    fn = len(issues) - tp
    near = []
    for i, f in enumerate(findings):
        if i in matched:
            continue
        reasons = [compare(f, t) for t in issues if f["type"] == t["type"]
                   and set(f["affected_ids"]) == {x.upper() for x in t["affected_ids"]}]
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


def terminal_reward(f1: float, inspection_steps: int) -> float:
    v = Decimal(str(f1)) * 2 - 1 - STEP_COST * inspection_steps
    return float(min(Decimal(1), max(Decimal(-1), v)))


def grade(submission, truth: dict, inspection_steps: int) -> dict:
    """Полная оценка одной отправки. Неверная структура -> награда -1."""
    try:
        fs = validate_submission(submission, truth["episode_id"])
    except InvalidSubmission as e:
        return {"valid": False, "error": str(e), "tp": 0, "fp": 0, "fn": len(truth["issues"]),
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "reward": -1.0,
                "inspection_steps": inspection_steps}
    m = match(fs, truth["issues"])
    mt = metrics(m["tp"], m["fp"], m["fn"])
    return {"valid": True, **m, **mt, "reward": terminal_reward(mt["f1"], inspection_steps),
            "inspection_steps": inspection_steps}
