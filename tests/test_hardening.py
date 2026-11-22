"""Регрессии по второму раунду независимого аудита: изоляция харнесса, тайм-ауты,
строгая валидация конфигов, нормализация и точная награда."""
import glob
import io
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from auditgym.config import resolve_config
from auditgym.env import AuditEnv
from auditgym.generator import NAMES_A, generate
from auditgym.harness import evaluate
from auditgym.matcher import exact_f1, grade, terminal_reward
from conftest import ALL, oracle_findings, sub

ROOT = Path(__file__).resolve().parent.parent
HOSTILE = str(Path(__file__).resolve().parent / "hostile")
SPLIT = [{"name": "t0", "seed": 101, "config": {"errors": {"fx_conversion": 1}}},
         {"name": "t1", "seed": 102, "config": {}}]


@pytest.fixture
def hostile_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", HOSTILE + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("AUDITGYM_SECRET", "test-private-secret")
    sys.path.insert(0, HOSTILE)
    yield
    sys.path.remove(HOSTILE)


# ---------------------------------------------------------------- харнесс и изоляция

def test_hanging_policy_is_cut_by_time_limit_and_run_continues(hostile_env):
    res = evaluate(["hostile_policies:Hang", "hostile_policies:HangInReset", "scripted_auditor"], SPLIT,
                   time_limit_s=1.0)
    for name in ("hostile_policies:Hang", "hostile_policies:HangInReset"):
        rows = res[name]["episodes"]
        assert all(r["policy_timeout"] for r in rows)
        assert all(r["result"]["outcome"] == "truncated:time_limit" or "time_limit" in json.dumps(r["result"])
                   for r in rows)
        assert all(r["time_s"] < 10 for r in rows)
    assert res["scripted_auditor"]["episodes"][0]["result"]["f1"] == 1.0


def test_policy_stdout_noise_and_exits_do_not_break_the_run(hostile_env):
    res = evaluate(["hostile_policies:Noisy", "hostile_policies:Exit", "hostile_policies:HardExit", "oracle"], SPLIT)
    noisy = res["hostile_policies:Noisy"]["episodes"]
    assert all(r["result"]["valid"] for r in noisy)            # print() не подменил ответ политики
    for name in ("hostile_policies:Exit", "hostile_policies:HardExit"):
        assert all(not r["result"]["valid"] and r["result"]["reward"] == -1.0 for r in res[name]["episodes"])
    assert all(r["result"]["reward"] == 1.0 for r in res["oracle"]["episodes"])


def test_self_declared_oracle_gets_no_truth(hostile_env):
    res = evaluate(["hostile_policies:SelfDeclaredOracle"], SPLIT)
    assert res["hostile_policies:SelfDeclaredOracle"]["episodes"][0]["result"]["tp"] == 0
    out = subprocess.run([sys.executable, "-m", "auditgym", "run", "--policy", "hostile_policies:SelfDeclaredOracle",
                          "--seed", "101", "--config", json.dumps({"errors": {"fx_conversion": 1}})],
                         capture_output=True, text=True, cwd=ROOT, env=os.environ, timeout=60)
    assert json.loads(out.stdout)["grader"]["tp"] == 0


def test_closed_run_leaves_no_files_and_no_secret_in_worker_env(hostile_env):
    before = set(glob.glob("/tmp/split_*") + glob.glob("/tmp/grader_*"))
    res = evaluate(["hostile_policies:Spy"], SPLIT)
    after = set(glob.glob("/tmp/split_*") + glob.glob("/tmp/grader_*"))
    assert after == before
    # от root /proc читается всегда; от обычного пользователя Spy ничего не находит
    if os.geteuid() != 0:
        assert all(r["result"]["valid"] for r in res["hostile_policies:Spy"]["episodes"])


def test_public_default_secret_is_not_a_closed_run(monkeypatch):
    monkeypatch.setenv("AUDITGYM_SECRET", "auditgym-open-v1")
    with pytest.raises(SystemExit):
        evaluate(["oracle"], SPLIT)


def test_hardened_process_hides_environ_from_same_user():
    if not sys.platform.startswith("linux") or os.geteuid() == 0:
        pytest.skip("needs Linux and a non-root user")
    code = "import sys,time; from auditgym._secure import harden; print(harden(), flush=True); time.sleep(20)"
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, cwd=ROOT,
                         env={**os.environ, "AUDITGYM_SECRET": "hide-me"})
    try:
        assert p.stdout.readline().strip() == "True"
        with pytest.raises(PermissionError):
            open(f"/proc/{p.pid}/environ", "rb").read()
    finally:
        p.kill()


# ---------------------------------------------------------------- сервер: один запуск закрытого эпизода

def _serve(lines, tmp_path, monkeypatch):
    from auditgym.server import serve
    monkeypatch.setenv("AUDITGYM_SECRET", "test-private-secret")
    split = tmp_path / "s.json"
    split.write_text(json.dumps({"config_version": "t", "episodes": SPLIT}))
    log = tmp_path / "g.jsonl"
    out = io.StringIO()
    serve("eval", str(log), episodes=str(split), stdin=io.StringIO("\n".join(json.dumps(x) for x in lines) + "\n"),
          stdout=out)
    return [json.loads(x) for x in out.getvalue().splitlines()], [json.loads(x) for x in log.read_text().splitlines()]


def test_closed_episode_cannot_be_restarted_and_abandoned_steps_count(tmp_path, monkeypatch):
    look = {"op": "step", "action": {"tool": "list_records", "args": {"table": "ledger"}}}
    replies, log = _serve([{"op": "reset", "index": 0}, look, look, look,
                           {"op": "reset", "index": 0}, {"op": "reset", "index": 1}], tmp_path, monkeypatch)
    assert "error" in replies[4]                                     # повторный reset того же эпизода
    # брошенный эпизод 0 закрыт пустым ответом с учётом 3 шагов; эпизод 1 закрыт при завершении сервера
    assert len(log) == 2
    assert log[0]["result"]["inspection_steps"] == 3 and log[0]["result"]["outcome"] == "abandoned"
    assert log[0]["result"]["fn"] == 1 and log[0]["result"]["reward"] == -1.0


# ---------------------------------------------------------------- среда

def test_malformed_submit_action_terminates_with_minus_one():
    env = AuditEnv(mode="train", time_limit_s=None)
    env.reset(5, ALL)
    for bad in ({"tool": "submit", "args": "x"}, {"tool": "submit", "args": {}, "extra": 1}):
        env.reset(5, ALL)
        _, r, te, tr, _ = env.step(bad)
        assert te and not tr and r == -1.0


def test_time_limit_does_not_count_the_rejected_action():
    t = {"now": 0.0}
    env = AuditEnv(mode="train", time_limit_s=60, clock=lambda: t["now"])
    env.reset(5, ALL)
    env.step({"tool": "list_records", "args": {"table": "ledger"}})
    t["now"] = 61.0
    _, _, te, tr, _ = env.step({"tool": "list_records", "args": {"table": "ledger"}})
    assert tr and not te
    assert env.grader_result()["inspection_steps"] == 1


# ---------------------------------------------------------------- оценщик

def test_float_amount_is_a_structural_error_like_jury_rules():
    ep = generate(7, ALL)
    f = oracle_findings(ep)
    f[0]["claimed_amount_minor"] = float(f[0]["claimed_amount_minor"])
    g = grade(sub(ep, f), ep.truth, 0)
    assert not g["valid"] and g["reward"] == -1.0


def test_unicode_lookalikes_are_not_normalized():
    ep = generate(7, ALL)
    f = oracle_findings(ep)
    f[0]["affected_ids"] = [a.lower().replace("i", "ı") for a in f[0]["affected_ids"]]
    f[1]["currency"] = f[1]["currency"].lower().replace("s", "ſ") if "S" in f[1]["currency"] else " " + f[1]["currency"]
    g = grade(sub(ep, f), ep.truth, 0)
    assert g["tp"] <= len(f) - 1


def test_reward_is_exact():
    assert terminal_reward(exact_f1(1, 1, 0), 0) == float(Fraction(1, 3))
    assert terminal_reward(exact_f1(2, 1, 1), 7) == float(2 * Fraction(4, 6) - 1 - Fraction(14, 1000))


# ---------------------------------------------------------------- конфиги и данные

@pytest.mark.parametrize("cfg", [
    {"errors": {"fx_conversion": 1.7}}, {"errors": {"fx_conversion": True}}, {"errors": {"fx_conversion": "2"}},
    {"distractors": {"recurring_amountz": [1, 2]}}, {"distractors": {"recurring_amounts": [3, 1]}},
    {"error_options": {"fx_kinds": []}}, {"error_options": {"fx_kinds": ["tomorrow"]}},
    {"error_options": {"payment_allocation_modes": []}},
    {"error_options": {"fx_targets": ["payment"], "fx_kinds": ["due_date"]}},
    {"timezone": "UTC"}, {"timezone": "+15:00"}, {"base_currency": "usd"}, {"base_currency": "RUB"},
    {"foreign_customer_share": 2}, {"period": "2026-12"}, {"period": "2027-01"}, {"period": "2026-13"},
])
def test_bad_configs_are_rejected_up_front(cfg):
    with pytest.raises(ValueError):
        resolve_config(cfg)


def test_valid_configs_never_crash_with_raw_exceptions():
    import random
    R = random.Random(3)
    for k in range(60):
        cfg = {"errors": {t: R.randint(0, 2) for t in ("duplicate_invoice", "payment_allocation", "fx_conversion",
                                                       "period_cutoff")},
               "interactions": {"fx_invoice_with_partial_payment": R.randint(0, 2),
                                "partial_fx_payment_at_boundary": R.randint(0, 2)},
               "error_options": {"fx_targets": R.choice([["invoice"], ["payment"], ["invoice", "payment"]]),
                                 "fx_kinds": R.sample(["prev_business_day", "next_business_day", "naive_date"],
                                                      R.randint(1, 3))}}
        try:
            generate(k, cfg)
        except ValueError:
            pass                      # допустимо: ясный отказ, а не падение


def test_boundary_scenario_adds_only_its_own_documents():
    for seed in range(15):
        a, b = generate(seed, {"interactions": {"partial_fx_payment_at_boundary": 1}}), generate(seed, {})
        extra = (len(a.tables["invoices"]) + len(a.tables["payments"])) - \
                (len(b.tables["invoices"]) + len(b.tables["payments"]))
        assert extra <= 3


def test_company_names_are_invented():
    for w in ("Orion", "Kite", "Silk", "Maple", "Polar", "Tengri"):
        assert w not in NAMES_A


def test_policy_runs_as_separate_os_user_when_available(hostile_env):
    from auditgym._secure import sandbox_spec
    spec = sandbox_spec()
    if spec is None:
        pytest.skip("needs root and the 'auditgym-policy' user (the Docker image has both)")
    res = evaluate(["hostile_policies:ShadowPlanter", "hostile_policies:SandboxProbe", "oracle", "scripted_auditor"],
                   SPLIT, policy_path=HOSTILE)
    rows = res["hostile_policies:SandboxProbe"]["episodes"]
    assert all(r["result"]["valid"] for r in rows), "policy could write the repo, read grader /proc or signal it"
    # политика, подбросившая пакет auditgym в cwd/HOME/tmp, не влияет на следующие политики
    assert all(r["result"]["reward"] == 1.0 for r in res["oracle"]["episodes"])
    assert all(r["result"]["f1"] == 1.0 for r in res["scripted_auditor"]["episodes"])


def test_evaluate_episode_ids_match_export(monkeypatch):
    from auditgym.harness import load_split
    monkeypatch.setenv("AUDITGYM_SECRET", "test-private-secret")
    eps = load_split(ROOT / "configs" / "dev.json")[:1]
    res = evaluate(["oracle"], eps)
    assert res["oracle"]["episodes"][0]["episode_id"] == generate(eps[0]["seed"], eps[0]["config"]).episode_id
