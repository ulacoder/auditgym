"""Red-team tests.

Section A: bug demonstrations. A1-A6 failed on the reviewed code and were fixed during the
review (kept as regression tests); A7 and A8 still FAIL and demonstrate open leaks.
Section B: tests that PASS and lock in attack resistance that was verified by hand.
Scripts with the full attacks live in redteam/.
"""
import hashlib
import io
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from auditgym import calculator, generator
from auditgym.config import ERROR_TYPES
from auditgym.env import AuditEnv
from auditgym.generator import COMPANY_OFFSETS, NAMES_A, NAMES_B, generate
from auditgym.matcher import grade
from auditgym.money import BASE_CURRENCIES, CURRENCY_EXPONENT
from auditgym.server import serve
from conftest import ALL, oracle_findings, sub

ROOT = Path(__file__).resolve().parent.parent
DEEP = "[" * 100_000 + "]" * 100_000


def _env(seed=4242, cfg=ALL, **kw):
    env = AuditEnv(mode="train", time_limit_s=None, **kw)
    obs, info = env.reset(seed, cfg)
    return env, obs, info


# =============================================================================== A. bugs

def test_A1_deeply_nested_findings_string_is_invalid_submission_not_crash():
    """submit(findings=<deep JSON string>) raises RecursionError out of env.step.
    Case 05: invalid structure must end the episode with reward -1."""
    env, _, _ = _env()
    obs, r, te, tr, info = env.step({"tool": "submit", "args": {"findings": DEEP}})
    assert te and r == -1.0 and obs["error"]["code"] == "invalid_submission"


def test_A2_deeply_nested_action_string_with_recorder_does_not_crash():
    """_parse catches RecursionError, but _record re-parses the action with json.loads
    and only catches ValueError -> crash whenever a recorder (trajectories) is attached."""
    env, _, _ = _env(recorder=lambda d: None)
    obs, r, te, tr, info = env.step(DEEP)
    assert obs["error"]["code"] == "invalid_action" and not (te or tr)


def test_A3_calculator_output_size_is_bounded():
    """A 30-char expression yields a ~250 KB observation (and ~1 MB for 123 chars):
    tiny magnitudes are not bounded and format(r, 'f') expands them."""
    x5 = "((((.1**12)**12)**12)**12)**12"
    for expr in (x5, "*".join([x5] * 4)):
        try:
            out = calculator.evaluate(expr)
        except calculator.CalcError:
            continue
        assert len(out) <= 1000, f"{len(expr)}-char expression -> {len(out)}-char result"


def test_A4_calculator_does_not_silently_round_large_constants():
    """Constants skip the MAX_ABS check and normalize() rounds them to 34 digits:
    '9'*40 evaluates to 10**40, i.e. a wrong answer instead of an error."""
    for expr in ("9" * 40, "12345678901234567890123456789012345678"):
        try:
            out = calculator.evaluate(expr)
        except calculator.CalcError:
            continue
        assert out == expr, f"{expr} -> {out}"


def _serve(lines, tmp_path, mode="eval"):
    log = tmp_path / "grader.jsonl"
    out = io.StringIO()
    serve(mode=mode, grader_log=str(log), time_limit_s=None,
          stdin=io.StringIO("\n".join(json.dumps(x) for x in lines) + "\n"), stdout=out)
    n = len(log.read_text().splitlines()) if log.exists() else 0
    return n, [json.loads(x) for x in out.getvalue().splitlines()]


def test_A5_client_cannot_suppress_grader_log_entry(tmp_path):
    """`_after_done` is taken from the (agent-side) request: setting it on the real
    terminal step means the episode is never written to the grader log
    (harness.evaluate then fails with KeyError). The server must track this itself."""
    ep = generate(5, {"errors": {"duplicate_invoice": 1}})
    n, _ = _serve([{"op": "reset", "seed": 5, "config": {"errors": {"duplicate_invoice": 1}}},
                   {"op": "step", "action": {"tool": "submit", "args": {"findings": sub(ep, [])}},
                    "_after_done": True}], tmp_path)
    assert n == 1


def test_A6_post_episode_steps_do_not_duplicate_grader_log(tmp_path):
    """A raw client (no `_after_done`) that keeps stepping after submit appends the
    same result again on every step: 1 graded episode -> 4 grader-log lines."""
    ep = generate(5, {})
    calc = {"tool": "calculator", "args": {"expression": "1"}}
    n, _ = _serve([{"op": "reset", "seed": 5, "config": {}},
                   {"op": "step", "action": {"tool": "submit", "args": {"findings": sub(ep, [])}}},
                   {"op": "step", "action": calc}, {"op": "step", "action": calc},
                   {"op": "step", "action": calc}], tmp_path)
    assert n == 1


def _attacker_fingerprint(seed, secret):
    def rng(stream):
        pre = f"{secret}|" if secret is not None else ""
        h = hashlib.sha256(f"{pre}{int(seed)}|{stream}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))
    R = rng("setting")
    base = R.choice(BASE_CURRENCIES)
    tz = R.choice(COMPANY_OFFSETS)
    y = R.choice([2025, 2026])
    m = R.randint(1, 12 if y == 2025 else 10)   # mirrors generator._Gen.__init__
    nf = R.randint(1, 3)
    foreign = tuple(sorted(R.sample([c for c in CURRENCY_EXPONENT if c != base], nf)))
    return base, tz, f"{y}-{m:02d}", foreign, f"{R.choice(NAMES_A)} {R.choice(NAMES_B)} LLP"


def test_A7_hidden_seed_not_recoverable_from_first_observation(monkeypatch):
    """The 'setting' stream depends only on the seed (plus a PUBLIC default secret unless
    the jury knows to export AUDITGYM_SECRET, which README never mentions); the manifest
    publishes its outputs. Brute force over 20k seeds takes ~0.3 s and finds the seed;
    redteam/seed_recovery.py then rebuilds the full truth (20/20 holdout episodes,
    reward ~0.98, with unknown id_salt/order_seed and unknown error config)."""
    hidden_seed = 13_579                       # 'closed' seed, jury config
    # закрытый прогон обязан задать секрет оценщика (serve/evaluate без него не стартуют)
    monkeypatch.setenv("AUDITGYM_SECRET", "jury-private-secret")
    env = AuditEnv(mode="eval")
    obs, _ = env.reset(hidden_seed, {"errors": {"duplicate_invoice": 1}, "id_salt": "jury", "order_seed": 3})
    man = obs["manifest"]
    want = (man["base_currency"], man["timezone"], man["period_start"][:7],
            tuple(sorted(c for c in man["currency_exponents"] if c != man["base_currency"])), man["company"])
    secret = getattr(generator, "DEFAULT_SECRET", None)   # only public knowledge
    found = [s for s in range(20_000) if _attacker_fingerprint(s, secret) == want]
    assert hidden_seed not in found


def test_A8_policy_in_official_harness_cannot_read_truth():
    """The policy runs inside harness.evaluate()'s process. Originally: Episode objects in gc
    and self-declared grader_only=True leaked truth (now fixed). Still open: evaluate()'s frame
    holds `episodes` (seeds+configs) and `hidden_secret`/`server_env` (AUDITGYM_SECRET), so
    FrameSnooper walks sys._getframe() and regenerates the truth, even with a jury secret."""
    from auditgym.harness import evaluate
    eps = [{"name": "x", "seed": 77, "config": {"errors": {"duplicate_invoice": 1, "period_cutoff": 1}}}]
    os.environ["AUDITGYM_SECRET"] = "jury-secret-not-known-to-agent"
    try:
        res = evaluate(["redteam.cheat_policy:GcSnooper", "redteam.cheat_policy:SelfDeclaredOracle",
                        "redteam.cheat_policy:FrameSnooper"], eps, isolated=True)
    finally:
        os.environ.pop("AUDITGYM_SECRET", None)
    for name, d in res.items():
        assert d["episodes"][0]["result"]["tp"] == 0, f"{name} read the truth"


# =============================================================================== B. resisted

def test_B1_spray_and_duplicate_tricks_do_not_beat_honest_partial(ep_all):
    t = ep_all.truth["issues"]
    honest = grade(sub(ep_all, oracle_findings(ep_all)[:2]), ep_all.truth, 0)["reward"]
    f = oracle_findings(ep_all)[:1]
    tricks = [
        f * 30,                                                            # duplicates
        [dict(f[0], affected_ids=f[0]["affected_ids"] * 5)] * 2,           # repeated ids
        [dict(f[0], affected_ids=[x.lower() + " " for x in f[0]["affected_ids"]])] * 2,
        [dict(f[0], affected_ids=sorted({i for x in t for i in x["affected_ids"]})[:10])],  # union of ids
    ]
    for tr in tricks:
        assert grade(sub(ep_all, tr), ep_all.truth, 0)["reward"] < honest


def test_B2_one_finding_never_matches_two_issues(ep_all):
    f = oracle_findings(ep_all)
    merged = dict(f[0], evidence_ids=sorted({e for x in f for e in x["evidence_ids"]})[:10])
    res = grade(sub(ep_all, [merged]), ep_all.truth, 0)
    assert res["tp"] <= 1


def test_B3_bad_values_are_invalid_not_crash(ep_all):
    base = oracle_findings(ep_all)[0]
    for mut in ({"claimed_amount_minor": 10 ** 5000}, {"claimed_amount_minor": -base["claimed_amount_minor"]},
                {"claimed_amount_minor": float(base["claimed_amount_minor"])}, {"claimed_amount_minor": True},
                {"currency": "usd "}, {"affected_ids": []}, {"affected_ids": [""]}, {"evidence_ids": [None]},
                {"type": "DUPLİCATE_INVOICE"}, {"explanation": "x" * 1001}):
        res = grade(sub(ep_all, [dict(base, **mut)]), ep_all.truth, 0)
        assert res["reward"] <= 1 and (not res["valid"] or res["tp"] <= 1)


@pytest.mark.parametrize("expr", ["1e999", "1_000", "١٢", "0x10", "__import__('os')", "(1).real", "9**9**9",
                                  "2**13", "0**-1", "0**0", "1/0", "lambda: 1", "[1]", "1if 1else 2", "1　+1"])
def test_B4_calculator_rejects_escapes(expr):
    with pytest.raises(calculator.CalcError):
        calculator.evaluate(expr)


@pytest.mark.parametrize("args", [{"table": "ledger", "limit": 0}, {"table": "ledger", "limit": 51},
                                  {"table": "ledger", "limit": True}, {"table": "ledger", "limit": 50.0},
                                  {"table": "ledger", "limit": "50"}, {"table": "ledger", "cursor": "ledger@١٢"},
                                  {"table": "ledger", "cursor": "ledger@+1"}, {"table": "ledger", "cursor": "fx@0"},
                                  {"table": "ledger", "cursor": "ledger@99999"}, {"table": ["ledger"]}])
def test_B5_list_records_edge_values_are_bounded_errors(args):
    env, _, _ = _env()
    obs, r, te, tr, _ = env.step({"tool": "list_records", "args": args})
    assert obs["error"] and obs["steps_remaining"] == 99 and r == 0 and not (te or tr)


def test_B6_fuzz_random_actions_never_crash():
    R = random.Random(0)
    atoms = [None, True, 0, -1, 51, 10 ** 30, 1.5, "", "ledger", "ledger@0", "1+2", "x" * 300, [], {}]
    env = AuditEnv(mode="train", time_limit_s=None, recorder=lambda d: json.dumps(d))
    for ep in range(20):
        env.reset(R.randint(0, 10 ** 6), {})
        for _ in range(100):
            tool = R.choice(["list_records", "get_record", "get_document", "calculator", "nope", None] * 5 + ["submit"])
            args = {k: R.choice(atoms) for k in R.sample(["table", "cursor", "limit", "id", "doc_id", "expression",
                                                          "findings"], R.randint(0, 3))}
            a = {"tool": tool, "args": args} if R.random() < 0.9 else R.choice(atoms)
            obs, r, te, tr, info = env.step(json.dumps(a) if R.random() < 0.2 else a)
            json.dumps([obs, r, te, tr, info], allow_nan=False)
            if te or tr:
                break


def test_B7_eval_mode_gives_no_signal_even_for_oracle(ep_all):
    env = AuditEnv(mode="eval", time_limit_s=None)
    _, info = env.reset(4242, ALL)
    good = env.step({"tool": "submit", "args": {"findings": sub(ep_all, oracle_findings(ep_all))}})
    env.reset(4242, ALL)
    bad = env.step({"tool": "submit", "args": {"findings": sub(ep_all, [])}})
    assert good == bad and good[1] == 0.0 and good[4] == {}


_DET = r"""
import sys, json, hashlib; sys.path.insert(0, %r)
from auditgym.env import AuditEnv
from auditgym.harness import run_episode
from auditgym.policies import make_policy
from auditgym.config import canonical_json
h = hashlib.sha256()
for seed, cfg in ((3, {"errors": {t: 1 for t in %r}, "interactions": {"fx_invoice_with_partial_payment": 1}}),
                  (4, {"id_salt": "z", "order_seed": 9})):
    for p in ("scripted_auditor", "random_explorer"):
        buf = []
        run_episode(AuditEnv(mode="train", recorder=buf.append, time_limit_s=None), make_policy(p), seed, cfg)
        h.update(canonical_json(buf).encode())
print(h.hexdigest())
"""


def test_B8_trajectories_identical_across_pythonhashseed():
    code = _DET % (str(ROOT), ERROR_TYPES)
    outs = {subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": s},
                           capture_output=True, text=True, check=True).stdout for s in ("0", "1", "4242")}
    assert len(outs) == 1
