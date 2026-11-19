"""Контракт reset/step, лимиты, завершение, неверный ввод."""
import json

import pytest

from auditgym.env import AuditEnv
from conftest import ALL, oracle_findings, sub


def mk(mode="train", **kw):
    return AuditEnv(mode=mode, time_limit_s=None, **kw)


def test_reset_returns_obs_info_json_serializable():
    env = mk()
    obs, info = env.reset(1, ALL)
    json.dumps(obs), json.dumps(info)
    assert {"task", "manifest", "policy", "tools", "steps_remaining"} <= set(obs)
    assert obs["steps_remaining"] == 100
    m = obs["manifest"]
    for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
              "counts", "rounding"):
        assert k in m
    assert info["episode_id"] == m["episode_id"]


def test_step_returns_five_tuple():
    env = mk()
    env.reset(1, ALL)
    out = env.step({"tool": "list_records", "args": {"table": "ledger"}})
    assert len(out) == 5
    obs, r, te, tr, info = out
    json.dumps(obs)
    assert r == 0.0 and te is False and tr is False
    assert obs["steps_remaining"] == 99


def test_paging_max_50_and_stable_cursors():
    env = mk()
    env.reset(1, ALL)
    o1, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 50}})
    assert len(o1["result"]["records"]) == 50
    c = o1["result"]["next_cursor"]
    o2, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": c}})
    o3, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": c}})
    assert o2["result"] == o3["result"]
    assert {r["row_id"] for r in o1["result"]["records"]}.isdisjoint(r["row_id"] for r in o2["result"]["records"])
    o4, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 51}})
    assert o4["error"]["code"] == "invalid_args"


def test_full_scan_covers_table_exactly_once():
    env = mk()
    obs, _ = env.reset(3, ALL)
    seen, cur = [], None
    while True:
        o, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": cur}})
        seen += [r["row_id"] for r in o["result"]["records"]]
        cur = o["result"]["next_cursor"]
        if cur is None:
            break
    assert len(seen) == len(set(seen)) == obs["manifest"]["counts"]["ledger_rows"]


@pytest.mark.parametrize("action,code", [
    ({"tool": "rm -rf"}, "unknown_tool"),
    ({"tool": "get_record", "args": {"table": "ledger", "id": "L-NOPE"}}, "unknown_id"),
    ({"tool": "get_document", "args": {"doc_id": "../../truth.json"}}, "unknown_id"),
    ({"tool": "list_records", "args": {"table": "truth"}}, "unknown_table"),
    ({"tool": "list_records", "args": {"table": "ledger", "cursor": "invoices@0"}}, "invalid_cursor"),
    ({"tool": "list_records", "args": {"table": "ledger", "cursor": "ledger@-5"}}, "invalid_cursor"),
    ({"tool": "calculator", "args": {"expression": "__import__('os')"}}, "calculator_error"),
    ("not json", "invalid_action"),
    ({"tool": "list_records", "args": [1, 2]}, "invalid_args"),
    ({"tool": "list_records", "args": {"table": "ledger"}, "extra": 1}, "invalid_action"),
])
def test_errors_are_bounded_and_consume_a_step(action, code):
    env = mk()
    env.reset(1, ALL)
    obs, r, te, tr, info = env.step(action)
    assert obs["error"]["code"] == code
    assert len(obs["error"]["message"]) <= 200
    assert obs["steps_remaining"] == 99 and not te and not tr and r == 0.0


def test_get_record_document_and_fx_ids():
    env = mk()
    obs, _ = env.reset(1, ALL)
    o, *_ = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 1}})
    inv = o["result"]["records"][0]
    o, *_ = env.step({"tool": "get_document", "args": {"doc_id": inv["doc_id"]}})
    assert o["result"]["doc_type"] == "invoice" and o["result"]["document"] == inv
    o, *_ = env.step({"tool": "get_record", "args": {"table": "invoices", "id": inv["invoice_id"]}})
    assert o["result"]["record"] == inv
    o, *_ = env.step({"tool": "list_records", "args": {"table": "fx", "limit": 1}})
    fx = o["result"]["records"][0]
    o, *_ = env.step({"tool": "get_record", "args": {"table": "fx", "id": f"{fx['date']}:{fx['source_currency']}:{fx['target_currency']}"}})
    assert o["result"]["record"] == fx


def test_records_are_read_only():
    env = mk()
    env.reset(1, ALL)
    o, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 1}})
    o["result"]["records"][0]["debit_minor"] = 10 ** 12
    o2, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 1}})
    assert o2["result"]["records"][0]["debit_minor"] != 10 ** 12


def test_calculator_arithmetic_only():
    env = mk()
    env.reset(1, ALL)
    ok = {"(1250 * 1.08345) / 3": "451.4375", "0.1+0.2": "0.3", "-(2-5)*4": "12", "10**3": "1000"}
    for expr, val in ok.items():
        o, *_ = env.step({"tool": "calculator", "args": {"expression": expr}})
        assert o["result"]["value"] == val, expr
    for bad in ["open('x')", "1/0", "2**1000", "x+1", "1e5", "[1]", "9" * 400, "", "lambda: 1"]:
        o, *_ = env.step({"tool": "calculator", "args": {"expression": bad}})
        assert o["error"]["code"] == "calculator_error", bad


def test_submit_terminates_and_resubmit_gives_nothing(ep_all):
    env = mk()
    env.reset(4242, {"errors": ALL["errors"], "interactions": ALL["interactions"]})
    obs, r, te, tr, info = env.step({"tool": "submit", "args": {"findings": sub(ep_all, oracle_findings(ep_all))}})
    assert te is True and tr is False and r == 1.0
    obs, r2, te2, tr2, _ = env.step({"tool": "submit", "args": {"findings": sub(ep_all, oracle_findings(ep_all))}})
    assert obs["error"]["code"] == "episode_finished" and r2 == 0.0 and te2 is True
    assert env.grader_result()["reward"] == 1.0


def test_step_limit_truncates_with_empty_answer():
    env = mk()
    env.reset(1, ALL)
    for i in range(99):
        _, r, te, tr, _ = env.step({"tool": "calculator", "args": {"expression": "1+1"}})
        assert not te and not tr
    obs, r, te, tr, info = env.step({"tool": "calculator", "args": {"expression": "1+1"}})
    assert tr is True and te is False
    g = env.grader_result()
    assert g["outcome"] == "step_limit" and g["tp"] == 0 and g["fn"] > 0
    assert r == -1.0  # пустой ответ на ошибочных книгах


def test_step_limit_on_clean_books_is_empty_answer():
    env = mk()
    env.reset(5, {})
    for _ in range(100):
        out = env.step({"tool": "calculator", "args": {"expression": "1"}})
    assert out[3] is True
    assert env.grader_result()["f1"] == 1.0
    assert abs(out[1] - (1 - 0.2)) < 1e-9


def test_submit_as_100th_step_is_terminated():
    env = mk()
    obs, info = env.reset(5, {})
    for _ in range(99):
        env.step({"tool": "calculator", "args": {"expression": "1"}})
    obs, r, te, tr, _ = env.step({"tool": "submit", "args": {"findings": {"episode_id": info["episode_id"], "findings": []}}})
    assert te is True and tr is False
    assert env.grader_result()["inspection_steps"] == 99


def test_time_limit_truncates():
    t = [0.0]
    env = AuditEnv(mode="train", time_limit_s=60, clock=lambda: t[0])
    env.reset(1, ALL)
    env.step({"tool": "list_records", "args": {"table": "fx"}})
    t[0] = 61.0
    obs, r, te, tr, info = env.step({"tool": "list_records", "args": {"table": "fx"}})
    assert tr is True and obs["error"]["code"] == "time_limit" and obs["result"] is None
    assert env.grader_result()["outcome"] == "time_limit"


def test_invalid_submission_gives_minus_one(ep_all):
    env = mk()
    env.reset(4242, ALL)
    obs, r, te, tr, info = env.step({"tool": "submit", "args": {"findings": {"episode_id": ep_all.episode_id,
                                                                             "findings": [{"type": "fraud"}]}}})
    assert te is True and r == -1.0 and obs["error"]["code"] == "invalid_submission"
    assert env.grader_result()["valid"] is False


def test_eval_mode_hides_reward_and_diagnostics(ep_all):
    env = AuditEnv(mode="eval", time_limit_s=None)
    env.reset(4242, ALL)
    obs, r, te, tr, info = env.step({"tool": "submit", "args": {"findings": sub(ep_all, oracle_findings(ep_all))}})
    assert r == 0.0 and info == {} and obs["result"] == {"accepted": True}
    assert env.grader_result()["reward"] == 1.0


def test_no_correctness_hints_during_episode(ep_all):
    env = mk()
    env.reset(4242, ALL)
    t = ep_all.truth["issues"][0]
    o, r, *_ = env.step({"tool": "get_record", "args": {"table": "invoices", "id": t["affected_ids"][-1]}})
    assert r == 0.0 and set(o) == {"step", "steps_remaining", "tool", "result", "error"}
