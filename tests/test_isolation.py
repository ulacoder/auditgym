"""Воспроизводимость, изоляция эпизодов, отсутствие утечек, обобщение (смена ID/порядка)."""
import json
import re

from auditgym.config import canonical_json
from auditgym.env import AuditEnv
from auditgym.generator import generate
from auditgym.harness import run_episode
from auditgym.policies import make_policy
from auditgym.server import RemoteEnv
from conftest import ALL, oracle_findings, sub


def _traj(seed, cfg, policy="scripted_auditor"):
    buf = []
    env = AuditEnv(mode="train", recorder=buf.append, time_limit_s=None)
    run_episode(env, make_policy(policy), seed, cfg)
    return canonical_json(buf)


def test_same_seed_config_actions_reproduce_everything():
    assert _traj(21, ALL) == _traj(21, ALL)
    assert _traj(21, ALL, "random_explorer") == _traj(21, ALL, "random_explorer")


def test_reset_isolates_episode_state():
    env = AuditEnv(mode="train", time_limit_s=None)
    o1, _ = env.reset(1, ALL)
    env.step({"tool": "list_records", "args": {"table": "ledger"}})
    env.reset(2, {})
    env.step({"tool": "submit", "args": {"findings": {"episode_id": "x", "findings": []}}})
    o1b, _ = env.reset(1, ALL)
    assert o1 == o1b
    o, *_ = env.step({"tool": "list_records", "args": {"table": "ledger"}})
    assert o["steps_remaining"] == 99


def test_different_seeds_differ():
    assert generate(1, ALL).tables != generate(2, ALL).tables


def test_no_truth_in_observations(ep_all):
    """Обходим ВСЕ записи всеми инструментами и ищем следы эталона."""
    env = AuditEnv(mode="train", time_limit_s=None, max_steps=10_000)
    buf = []
    env.recorder = buf.append
    env.reset(4242, ALL)
    for t in ("ledger", "invoices", "payments", "fx"):
        cur = None
        while True:
            o, *_ = env.step({"tool": "list_records", "args": {"table": t, "cursor": cur}})
            cur = o["result"]["next_cursor"]
            if cur is None:
                break
    for d in ep_all.tables["invoices"][:5]:
        env.step({"tool": "get_document", "args": {"doc_id": d["doc_id"]}})
    env.step({"tool": "get_record", "args": {"table": "ledger", "id": "../grader/truth.json"}})
    env.step({"tool": "calculator", "args": {"expression": "1/0"}})
    text = canonical_json([b["observation"] for b in buf])
    for iss in ep_all.truth["issues"]:
        assert iss["issue_id"] not in text
    for word in ("truth", "provenance", "issue_id", "expected_amount", "evidence_sets", "distractor", "4242",
                 "/home", "Traceback"):
        assert word not in text, word
    for ch in ep_all.provenance["changes"]:
        assert ch["cause"] not in text


def test_public_records_have_only_schema_fields(ep_all):
    assert set(ep_all.tables["ledger"][0]) == {"row_id", "timestamp", "account", "debit_minor", "credit_minor",
                                               "currency", "entity_id", "invoice_id", "payment_id"}
    assert set(ep_all.tables["invoices"][0]) == {"doc_id", "invoice_id", "entity_id", "issued_at", "due_at",
                                                 "currency", "total_minor", "line_items"}
    assert set(ep_all.tables["payments"][0]) == {"doc_id", "payment_id", "timestamp", "currency", "amount_minor",
                                                 "invoice_refs"}
    assert set(ep_all.tables["fx"][0]) == {"date", "source_currency", "target_currency", "rate"}


def test_ids_do_not_encode_error_status():
    ep = generate(9, {"errors": {"duplicate_invoice": 2, "period_cutoff": 2}})
    pat = re.compile(r"^(INV|PAY|D|L|C|V)-[0-9A-F]{8,}$")
    for t, k in (("invoices", "invoice_id"), ("invoices", "doc_id"), ("ledger", "row_id")):
        assert all(pat.match(r[k]) for r in ep.tables[t])


def test_id_and_order_transform_preserve_oracle_and_auditor_reward():
    cfg = dict(ALL)
    a = generate(31, cfg)
    b = generate(31, {**cfg, "id_salt": "other", "order_seed": 3})
    assert {i["invoice_id"] for i in a.tables["invoices"]}.isdisjoint(i["invoice_id"] for i in b.tables["invoices"])
    key = lambda ep: sorted((t["type"], t["expected_amount_minor"]) for t in ep.truth["issues"])
    assert key(a) == key(b)
    for ep, c in ((a, cfg), (b, {**cfg, "id_salt": "other", "order_seed": 3})):
        for p in ("oracle", "scripted_auditor"):
            env = AuditEnv(mode="train", time_limit_s=None)
            _, _, ret, _, _ = run_episode(env, make_policy(p, {ep.episode_id: ep.truth}.__getitem__), 31, c)
            assert ret > 0.95


def test_isolated_server_hides_reward_and_logs_to_grader(tmp_path, ep_all):
    log = tmp_path / "grader.jsonl"
    env = RemoteEnv(mode="eval", grader_log=str(log), time_limit_s=None)
    try:
        obs, info = env.reset(4242, ALL)
        assert "truth" not in json.dumps(obs)
        o, r, te, tr, inf = env.step({"tool": "submit", "args": {"findings": sub(ep_all, oracle_findings(ep_all))}})
        assert te and r == 0.0 and inf == {}
    finally:
        env.close()
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["result"]["reward"] == 1.0 and rec["episode_id"] == ep_all.episode_id


def test_trajectory_records_have_required_fields():
    buf = []
    env = AuditEnv(mode="train", recorder=buf.append, time_limit_s=None)
    run_episode(env, make_policy("scripted_auditor"), 3, ALL)
    for d in buf:
        assert {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"} <= set(d)
        json.dumps(d)
    assert buf[0]["step"] == 0 and buf[-1]["terminated"] is True
