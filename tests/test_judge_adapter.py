"""Проверки жюри FinTech (genesis-exe.com/fintech-checks.html, 20.11.2026) через judge_adapter.py:
контрактные тесты C01–C20, тесты matcher и награды 1–20, открытые примеры M01–M12, парные проверки."""
import copy
import csv
import json
import random
import shutil
from fractions import Fraction
from pathlib import Path

import pytest

import judge_adapter as JA

CLEAN = {"base_currency": "KZT", "errors": {t: 0 for t in ("duplicate_invoice", "payment_allocation",
                                                          "fx_conversion", "period_cutoff")}}
ONE = {"base_currency": "KZT", "errors": {"duplicate_invoice": 1, "payment_allocation": 0, "fx_conversion": 0,
                                          "period_cutoff": 0}}
ALL4 = {"errors": {"duplicate_invoice": 1, "payment_allocation": 1, "fx_conversion": 1, "period_cutoff": 1},
        "interactions": {"fx_invoice_with_partial_payment": 1, "partial_fx_payment_at_boundary": 1}}


def _gen(tmp_path, name, seed, cfg):
    d = tmp_path / name
    JA.generate_episode(seed, cfg, str(d))
    return d


def _closed(d):
    env = JA.make_env()
    obs, info = env.reset(None, {"mode": "closed", "episode_dir": str(d), "team_config": {}})
    return env, obs, info


def _act(env, name, **args):
    return env.step({"name": name, "args": args})


def _all_records(env, table):
    out, cur = [], None
    while True:
        o, *_ = _act(env, "list_records", table=table, cursor=cur, limit=50)
        out += o["result"]["records"]
        cur = o["result"]["next_cursor"]
        if cur is None:
            return out


def _truth(d):
    return json.loads((Path(d) / "truth.json").read_text())


def _oracle(d):
    t = _truth(d)
    return {"episode_id": t["episode_id"], "findings": [
        {"type": i["type"], "affected_ids": i["affected_ids"], "claimed_amount_minor": i["expected_amount_minor"],
         "currency": i["currency"], "evidence_ids": i["evidence_sets"][0], "explanation": "oracle"}
        for i in t["issues"]]}


# ----------------------------------------------------------------------------- блок 1

@pytest.fixture(scope="module")
def eps(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("jury")
    return {"C": _gen(tmp, "C", 11, CLEAN), "E": _gen(tmp, "E", 12, ONE), "A": _gen(tmp, "A", 13, ALL4)}


def test_c01_c03_reset_first_observation_and_manifest(eps):
    env, obs, info = _closed(eps["C"])
    json.dumps(obs), json.dumps(info)
    assert set(obs) == {"task", "manifest", "policy", "remaining_steps"} and obs["remaining_steps"] == 100
    m = obs["manifest"]
    for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
              "counts", "rounding"):
        assert k in m
    led = _all_records(env, "ledger")
    assert m["counts"]["ledger_rows"] == len(led)
    text = json.dumps(obs)
    assert "seed" not in json.dumps(info) and "truth" not in text and "issue_id" not in text


def test_c04_c06_paging(eps):
    env, obs, _ = _closed(eps["A"])
    o, r, te, tr, info = _act(env, "list_records", table="ledger", cursor=None, limit=50)
    assert type(te) is bool and type(tr) is bool and r == 0 and info == {}
    assert set(o) == {"result", "error", "remaining_steps"} and len(o["result"]["records"]) <= 50
    cur = o["result"]["next_cursor"]
    a, *_ = _act(env, "list_records", table="ledger", cursor=cur, limit=50)
    b, *_ = _act(env, "list_records", table="ledger", cursor=cur, limit=50)
    assert a["result"] == b["result"]
    env2, obs2, _ = _closed(eps["A"])
    rows = _all_records(env2, "ledger")
    ids = [x["row_id"] for x in rows]
    assert len(ids) == len(set(ids)) == obs2["manifest"]["counts"]["ledger_rows"] and len(ids) > 101


def test_c07_c14_records_documents_calculator_errors(eps):
    env, obs, _ = _closed(eps["C"])
    rows = _all_records(env, "ledger")
    docs = _all_records(env, "invoices")
    left = obs["remaining_steps"] - 0
    o, *_ = _act(env, "get_record", table="ledger", id=rows[0]["row_id"])
    assert o["result"] == rows[0] and o["error"] is None
    o, *_ = _act(env, "get_document", doc_id=docs[0]["doc_id"])
    assert o["result"] == docs[0]
    o, *_ = _act(env, "calculator", expression="0.1+0.2")
    assert o["result"] == "0.3"
    for bad in ("open('/etc/passwd').read()", "__import__('os').system('id')", "1;2"):
        o, *_ = _act(env, "calculator", expression=bad)
        assert o["result"] is None and o["error"]["code"]
    before = o["remaining_steps"]
    o, *_ = _act(env, "get_record", table="ledger", id="U-DOES-NOT-EXIST")
    assert o["error"] and o["remaining_steps"] == before - 1
    o, *_ = _act(env, "get_document", doc_id="U-DOES-NOT-EXIST")
    assert o["error"] and o["result"] is None
    o, *_ = env.step({"name": "rm_rf", "args": {}})
    assert o["error"] and o["remaining_steps"] == before - 3
    o, *_ = _act(env, "get_record", table="ledger", id=rows[1]["row_id"])
    assert o["result"] == rows[1] and o["remaining_steps"] == before - 4


def test_c15_submit_is_final(eps, tmp_path):
    env, obs, _ = _closed(eps["E"])
    o, r, te, tr, info = _act(env, "submit", findings=_oracle(eps["E"]))
    assert te and not tr and r == 0 and info == {} and o["result"] == {"accepted": True}
    o2, r2, te2, tr2, _ = _act(env, "submit", findings={"episode_id": "x", "findings": []})
    assert te2 and r2 == 0
    assert env._env.grader_result()["reward"] > 0.9      # оценщик зафиксировал первую отправку


def test_c16_c17_step_limit(eps):
    env, obs, _ = _closed(eps["E"])
    for _ in range(99):
        o, r, te, tr, _ = _act(env, "calculator", expression="1+1")
        assert not te and not tr
    o, r, te, tr, _ = _act(env, "submit", findings=_oracle(eps["E"]))
    assert te and not tr and o["result"] == {"accepted": True}
    g = env._env.grader_result()
    assert g["inspection_steps"] == 99 and g["tp"] == 1
    env, obs, _ = _closed(eps["E"])
    for k in range(100):
        o, r, te, tr, _ = _act(env, "calculator", expression="1+1")
    assert tr and not te and env._env.grader_result()["tp"] == 0
    o, r, te, tr, _ = _act(env, "submit", findings=_oracle(eps["E"]))
    assert env._env.grader_result()["tp"] == 0


def test_c19_closed_reward_zero_and_train_reward_visible(eps):
    env, obs, _ = _closed(eps["E"])
    for _ in range(3):
        o, r, te, tr, info = _act(env, "list_records", table="invoices", cursor=None, limit=5)
        assert r == 0 and info == {}
    o, r, te, tr, info = _act(env, "submit", findings=_oracle(eps["E"]))
    assert r == 0 and info == {} and "reward" not in json.dumps(o)
    env = JA.make_env()
    env.reset(None, {"mode": "train", "episode_dir": str(eps["E"]), "team_config": {}})
    o, r, te, tr, info = _act(env, "submit", findings=_oracle(eps["E"]))
    assert te and r == 1.0 and info["tp"] == 1


def test_c20_no_truth_in_any_policy_channel(eps):
    for key in ("C", "E", "A"):
        t = _truth(eps[key])
        env, obs, info = _closed(eps[key])
        seen = [json.dumps(obs), json.dumps(info)]
        for table in JA.TABLES:
            seen.append(json.dumps(_all_records(env, table)))
        for d in _all_records(env, "invoices")[:5] + _all_records(env, "payments")[:5]:
            o, r, te, tr, info = _act(env, "get_document", doc_id=d["doc_id"])
            seen.append(json.dumps([o, info]))
        o, *_ = _act(env, "get_record", table="ledger", id="../../truth.json")
        seen.append(json.dumps(o))
        text = "\n".join(seen)
        for issue in t["issues"]:
            assert issue["issue_id"] not in text
        for w in ("truth", "provenance", "expected_amount", "evidence_sets", str(Path(eps[key]))):
            assert w not in text


def test_repeated_reset_restarts_same_episode(eps):
    env, obs, _ = _closed(eps["A"])
    a, *_ = _act(env, "list_records", table="payments", cursor=None, limit=7)
    obs2, _ = env.reset(None, {"mode": "closed", "episode_dir": str(eps["A"]), "team_config": {}})
    assert obs2 == obs and obs2["remaining_steps"] == 100
    b, *_ = _act(env, "list_records", table="payments", cursor=None, limit=7)
    assert a == b


# ----------------------------------------------------------------------------- блок 3 и M01–M12

@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    """Общая фикстура жюри: ошибка A (тип T), записи R1/R2, 10000 KZT, доказательства {D1, D2}, tau=0."""
    d = tmp_path_factory.mktemp("fx") / "F"
    JA.generate_episode(21, CLEAN, str(d))
    ep_rows = list(csv.DictReader(open(d / "ledger.csv")))
    docs = [json.loads(l) for l in open(d / "invoices.jsonl")]
    t = _truth(d)
    ids = {"R1": ep_rows[0]["row_id"], "R2": ep_rows[1]["row_id"], "R3": ep_rows[2]["row_id"],
           "R4": ep_rows[3]["row_id"], "R5": ep_rows[4]["row_id"],
           "D1": docs[0]["doc_id"], "D2": docs[1]["doc_id"], "D3": docs[2]["doc_id"]}
    t["issues"] = [{"issue_id": "ISS-A", "type": "duplicate_invoice", "affected_ids": [ids["R1"]],
                    "expected_amount_minor": 10000, "currency": "KZT", "interaction": None,
                    "evidence_sets": [[ids["D1"], ids["D2"]]]}]
    (d / "truth.json").write_text(json.dumps(t))
    return d, ids, t


def _F(ids, t, **kw):
    f = {"type": "duplicate_invoice", "affected_ids": [ids["R1"]], "claimed_amount_minor": 10000,
         "currency": "KZT", "evidence_ids": [ids["D1"], ids["D2"]], "explanation": "short"}
    f.update(kw)
    return f


def _score(d, t, findings, steps=10, issues=None):
    if issues is not None:
        t2 = copy.deepcopy(t)
        t2["issues"] = issues
        dd = Path(str(d) + f"_{len(issues)}")
        if dd.exists():
            shutil.rmtree(dd)
        shutil.copytree(d, dd)
        (dd / "truth.json").write_text(json.dumps(t2))
        d = dd
    return JA.score_episode(str(d), {"episode_id": t["episode_id"], "findings": findings}, steps)


def _close(x, y):
    return abs(x - float(y)) <= 1e-9


def test_matcher_block_1_to_20(fixture_dir):
    d, ids, t = fixture_dir
    A = t["issues"][0]
    B = dict(A, issue_id="ISS-B", affected_ids=[ids["R2"]], expected_amount_minor=20000)
    s = _score(d, t, [_F(ids, t)]); assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 0) and _close(s["return"], 0.98)
    s = _score(d, t, []); assert (s["tp"], s["fp"], s["fn"]) == (0, 0, 1) and s["return"] == -1
    s = _score(d, t, [], issues=[]); assert s["f1"] == 1 and _close(s["return"], 0.98)
    s = _score(d, t, [_F(ids, t)], issues=[]); assert (s["fp"], s["return"]) == (1, -1)
    s = _score(d, t, [_F(ids, t), _F(ids, t)]); assert (s["tp"], s["fp"]) == (1, 1) and _close(s["return"], Fraction(47, 150))
    for bad in (_F(ids, t, type="period_cutoff"), _F(ids, t, affected_ids=[ids["R3"]]),
                _F(ids, t, claimed_amount_minor=10001), _F(ids, t, currency="USD"),
                _F(ids, t, evidence_ids=[ids["D1"]]), _F(ids, t, evidence_ids=[ids["D1"], "U-NOPE"])):
        s = _score(d, t, [bad]); assert (s["tp"], s["fp"], s["fn"], s["return"]) == (0, 1, 1, -1)
    mass = [_F(ids, t, type=ty, affected_ids=[ids[r]]) for r in ("R1", "R2", "R3", "R4", "R5")
            for ty in ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")]
    s = _score(d, t, mass); assert (s["tp"], s["fp"], s["fn"]) == (1, 19, 0) and _close(s["return"], Fraction(-871, 1050))
    s = _score(d, t, [_F(ids, t)], issues=[A, B]); assert (s["tp"], s["fn"]) == (1, 1) and _close(s["return"], Fraction(47, 150))
    s = _score(d, t, [_F(ids, t), _F(ids, t, affected_ids=[ids["R5"]])], issues=[A, B])
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1) and _close(s["return"], -0.02)
    for steps, want in ((0, 1.0), (50, 0.90), (100, 0.80)):
        assert _close(_score(d, t, [_F(ids, t)], steps)["return"], want)
    many = [_F(ids, t) for _ in range(31)]
    s = _score(d, t, many); assert s["return"] == -1 and s["tp"] is None
    s = _score(d, t, [_F(ids, t, evidence_ids=[f"E{k}" for k in range(11)])]); assert s["return"] == -1 and s["f1"] is None
    s = _score(d, t, [_F(ids, t, explanation="x" * 1001)]); assert s["return"] == -1 and s["tp"] is None
    s = _score(d, t, [_F(ids, t, explanation="ы" * 1000)]); assert s["tp"] == 1     # 1000 кодовых точек — можно


def test_open_examples_m01_m12(fixture_dir):
    d, ids, t = fixture_dir
    A = dict(t["issues"][0], affected_ids=[ids["R1"], ids["R2"]])
    base = dict(affected_ids=[ids["R1"], ids["R2"]])
    s = _score(d, t, [_F(ids, t, affected_ids=[ids["R2"], ids["R1"], ids["R1"]])], issues=[A]); assert s["tp"] == 1
    s = _score(d, t, [_F(ids, t, affected_ids=[ids["R1"], ids["R2"], ids["R3"]])], issues=[A]); assert (s["fp"], s["fn"]) == (1, 1)
    s = _score(d, t, [_F(ids, t, evidence_ids=[ids["D2"], ids["D1"], ids["D3"]], **base)], issues=[A]); assert s["tp"] == 1
    s = _score(d, t, [_F(ids, t, evidence_ids=[ids["D1"], ids["D2"], "U-NOPE"], **base)], issues=[A]); assert (s["fp"], s["fn"]) == (1, 1)
    s = _score(d, t, [_F(ids, t, evidence_ids=[ids["D2"]] + [ids["D1"]] * 10, **base)], issues=[A]); assert s["return"] == -1
    for cur in ("kzt", " KZT", "KZT "):
        s = _score(d, t, [_F(ids, t, currency=cur, **base)], issues=[A]); assert (s["fp"], s["fn"]) == (1, 1)
    s = _score(d, t, [_F(ids, t, claimed_amount_minor=10001, **base)], issues=[A]); assert (s["fp"], s["fn"]) == (1, 1)
    for amt in ("10000", True, 10000.0):
        s = _score(d, t, [_F(ids, t, claimed_amount_minor=amt, **base)], issues=[A]); assert s["return"] == -1
    s = _score(d, t, [_F(ids, t, **base), _F(ids, t, **base)], issues=[A]); assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 0)


def test_m08_published_tau_one(fixture_dir, tmp_path):
    d, ids, t = fixture_dir
    dd = tmp_path / "tau1"
    shutil.copytree(d, dd)
    pol = json.loads((dd / "policy.json").read_text())
    pol["tolerance"]["amount_minor"] = 1
    (dd / "policy.json").write_text(json.dumps(pol))
    assert _score(dd, t, [_F(ids, t, claimed_amount_minor=10001)])["tp"] == 1
    assert _score(dd, t, [_F(ids, t, claimed_amount_minor=10002)])["tp"] == 0


def test_m10_m11_maximum_matching_and_tie_break():
    from auditgym.matcher import match
    def F(n):
        return {"type": "t", "affected_ids": ["R1"], "currency": "KZT", "claimed_amount_minor": n,
                "evidence_ids": ["D1"], "explanation": ""}
    A = {"issue_id": "A", "type": "t", "affected_ids": ["R1"], "currency": "KZT", "expected_amount_minor": 10000,
         "evidence_sets": [["D1"]]}
    B = dict(A, issue_id="B", expected_amount_minor=20000)
    m = match([F(15000), F(10000)], [B, A], tau=5000)        # F0 подходит A и B, F1 — только A
    assert m["tp"] == 2 and {(p["issue_id"], p["finding_index"]) for p in m["pairs"]} == {("A", 1), ("B", 0)}
    B2 = dict(A, issue_id="B")
    m = match([F(10000), F(10000)], [B2, A])
    assert [(p["issue_id"], p["finding_index"]) for p in sorted(m["pairs"], key=lambda p: p["issue_id"])] == [("A", 0), ("B", 1)]


# ----------------------------------------------------------------------------- блок 4

def _transform(src: Path, dst: Path, rename: bool, reorder: bool, seed: int):
    """Как у жюри: копия эпизода, перестановка записей и/или согласованная замена ID везде."""
    shutil.copytree(src, dst)
    R = random.Random(seed)
    mapping = {}
    if rename:
        led = list(csv.DictReader(open(src / "ledger.csv")))
        idset = {r["row_id"] for r in led} | {r["entity_id"] for r in led if r["entity_id"]}
        for f in ("invoices.jsonl", "payments.jsonl"):
            for l in open(src / f):
                o = json.loads(l)
                idset |= {o["doc_id"]} | {v for k, v in o.items() if k.endswith("_id") and isinstance(v, str)}
        for x in sorted(idset):
            mapping[x] = f"Z{R.randrange(10**9):09d}-{len(mapping)}"

    def sub(v):
        if isinstance(v, str):
            return mapping.get(v, v)
        if isinstance(v, list):
            return [sub(x) for x in v]
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        return v

    for f in ("invoices.jsonl", "payments.jsonl"):
        rows = [sub(json.loads(l)) for l in open(dst / f)]
        if reorder:
            R.shuffle(rows)
        (dst / f).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for f in ("ledger.csv", "fx.csv"):
        rd = list(csv.DictReader(open(dst / f)))
        fields = list(rd[0].keys())
        rows = [{k: sub(v) for k, v in r.items()} for r in rd]
        if reorder:
            R.shuffle(rows)
        with open(dst / f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    for f in ("truth.json", "provenance.json"):
        (dst / f).write_text(json.dumps(sub(json.loads((dst / f).read_text()))))
    if (dst / "ledger_before.csv").exists():
        (dst / "ledger_before.csv").unlink()
    return {v: k for k, v in mapping.items()}


@pytest.mark.parametrize("rename,reorder,seed,cfg", [
    (False, True, 31, CLEAN), (True, False, 32, ONE), (True, True, 33, ALL4), (True, True, 34, ALL4)])
def test_paired_checks(tmp_path, rename, reorder, seed, cfg):
    src = _gen(tmp_path, "orig", seed, cfg)
    inv = _transform(src, tmp_path / "trans", rename, reorder, seed)
    res = {}
    for name, d in (("orig", src), ("trans", tmp_path / "trans")):
        env, obs, _ = _closed(d)
        for _ in range(7):
            _act(env, "calculator", expression="2*3")
        _act(env, "submit", findings=_oracle(d))
        g = env._env.grader_result()
        res[name] = (g["tp"], g["fp"], g["fn"], g["f1"], g["reward"], g["inspection_steps"])
        assert JA.score_episode(str(d), _oracle(d), 7)["return"] == g["reward"]
    assert res["orig"] == res["trans"] and res["orig"][0] == len(_truth(src)["issues"])
    back = json.loads(json.dumps(_oracle(tmp_path / "trans")["findings"]))
    unmapped = [{**f, "affected_ids": [inv.get(x, x) for x in f["affected_ids"]],
                 "evidence_ids": [inv.get(x, x) for x in f["evidence_ids"]]} for f in back]
    assert unmapped == _oracle(src)["findings"]
    env1, o1, _ = _closed(src)
    env2, o2, _ = _closed(src)
    assert o1 == o2 and _all_records(env1, "ledger") == _all_records(env2, "ledger")
