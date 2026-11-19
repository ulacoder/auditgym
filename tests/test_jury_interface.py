"""Интерфейс для жюри: схемы, загрузка эпизодов с диска, сервер с закрытыми seed, секрет оценщика."""
import io
import json
import os
from pathlib import Path

import pytest

from auditgym.analysis import audit
from auditgym.env import AuditEnv
from auditgym.episode_io import export_episode, load_episode
from auditgym.generator import generate
from auditgym.harness import evaluate, run_episode
from auditgym.matcher import grade
from auditgym.policies import make_policy
from auditgym.server import serve
from auditgym.validate import SCHEMA, evidence_recompute
from conftest import ALL, oracle_findings, sub

ROOT = Path(__file__).resolve().parent.parent


def test_json_schemas_exist_and_agree_with_code(ep_all):
    sch = {p.name: json.loads(p.read_text()) for p in (ROOT / "schemas").glob("*.json")}
    for name in ("findings", "action", "observation", "manifest", "truth", "trajectory",
                 "ledger_row", "invoice", "payment", "fx_row"):
        assert f"{name}.schema.json" in sch
    for table, fname in (("ledger", "ledger_row"), ("invoices", "invoice"), ("payments", "payment"), ("fx", "fx_row")):
        assert set(sch[f"{fname}.schema.json"]["required"]) == set(SCHEMA[table])
    assert set(sch["findings.schema.json"]["properties"]["findings"]["items"]["required"]) == {
        "type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"}
    assert set(sch["manifest.schema.json"]["required"]) <= set(ep_all.manifest)
    assert set(sch["truth.schema.json"]["properties"]["issues"]["items"]["required"]) == set(ep_all.truth["issues"][0])


def test_export_load_roundtrip_and_env_from_disk(tmp_path, ep_all):
    export_episode(ep_all, tmp_path / "pub", tmp_path / "grd")
    assert not {"truth.json", "provenance.json", "ledger_before.csv"} & {p.name for p in (tmp_path / "pub").iterdir()}
    ep = load_episode(tmp_path / "pub", tmp_path / "grd")
    assert ep.tables == ep_all.tables and ep.truth == ep_all.truth
    env = AuditEnv(mode="train", time_limit_s=None)
    obs, info = env.reset(config={"episode_dir": str(tmp_path / "pub"), "grader_dir": str(tmp_path / "grd")})
    assert info["episode_id"] == ep_all.episode_id
    _, r, te, *_ = env.step({"tool": "submit", "args": {"findings": sub(ep_all, oracle_findings(ep_all))}})
    assert te and r == 1.0


def test_jury_can_permute_files_themselves(tmp_path, ep_all):
    """Жюри переставляет строки в выгруженных файлах — ответ oracle и аудитора не меняется."""
    export_episode(ep_all, tmp_path / "pub", tmp_path / "grd")
    led = (tmp_path / "pub" / "ledger.csv").read_text().splitlines()
    (tmp_path / "pub" / "ledger.csv").write_text("\n".join([led[0]] + list(reversed(led[1:]))) + "\n")
    env = AuditEnv(mode="train", time_limit_s=None)
    cfg = {"episode_dir": str(tmp_path / "pub"), "grader_dir": str(tmp_path / "grd")}
    _, _, ret, _, _ = run_episode(env, make_policy("scripted_auditor"), None, cfg)
    assert ret > 0.95


def test_server_with_hidden_episodes_rejects_seed_reset(tmp_path):
    split = tmp_path / "hidden.json"
    split.write_text(json.dumps({"config_version": "jury", "episodes": [
        {"name": "h0", "seed": 987654, "config": {"errors": {"fx_conversion": 1}}}]}))
    out = io.StringIO()
    lines = [{"op": "reset", "seed": 1, "config": {}}, {"op": "reset", "index": 0}, {"op": "oracle_truth"}]
    serve(mode="eval", episodes=str(split), time_limit_s=None, allow_public_secret=True,
          stdin=io.StringIO("\n".join(json.dumps(x) for x in lines) + "\n"), stdout=out)
    r = [json.loads(x) for x in out.getvalue().splitlines()]
    assert "error" in r[0] and "observation" in r[1] and "error" in r[2]
    assert "987654" not in json.dumps(r[1])


def test_only_grader_listed_policies_get_truth():
    eps = [{"name": "x", "seed": 77, "config": {"errors": {"duplicate_invoice": 1, "period_cutoff": 1}}}]
    res = evaluate(["oracle", "examples.my_policy:TruthOracle"], eps, isolated=True, allow_public_secret=True)
    assert res["oracle"]["episodes"][0]["result"]["reward"] == 1.0
    assert res["examples.my_policy:TruthOracle"]["episodes"][0]["result"]["tp"] == 0
    res = evaluate(["examples.my_policy:TruthOracle"], eps, isolated=True,
                   oracle_policies={"examples.my_policy:TruthOracle"}, allow_public_secret=True)
    assert res["examples.my_policy:TruthOracle"]["episodes"][0]["result"]["reward"] == 1.0


def test_secret_changes_ids_and_data(monkeypatch):
    a = generate(5, ALL)
    monkeypatch.setenv("AUDITGYM_SECRET", "jury-secret-xyz")
    b = generate(5, ALL)
    assert a.episode_id != b.episode_id
    assert {i["invoice_id"] for i in a.tables["invoices"]}.isdisjoint(i["invoice_id"] for i in b.tables["invoices"])
    assert a.tables["fx"] != b.tables["fx"]
    g = grade(sub(b, audit(b.tables, b.manifest)), b.truth, 0)
    assert g["f1"] == 1.0


def test_amount_recomputable_from_evidence_set_only(ep_all):
    for t in ep_all.truth["issues"]:
        assert evidence_recompute(t, ep_all.tables, ep_all.manifest) == t["expected_amount_minor"]
    # без ключа курса набор доказательств FX-ошибки недостаточен
    fx = next(t for t in ep_all.truth["issues"] if t["type"] == "fx_conversion")
    weak = dict(fx, evidence_sets=[[i for i in s if ":" not in i] for s in fx["evidence_sets"]])
    assert evidence_recompute(weak, ep_all.tables, ep_all.manifest) is None


def test_failure_diagnostics_attribute_false_positives_to_distractors():
    ep = generate(11, {})
    rec = next(d for d in ep.provenance["distractors"] if d["kind"] == "recurring_amount")
    inv = next(i for i in ep.tables["invoices"] if i["invoice_id"] == rec["invoices"][1])
    env = AuditEnv(mode="eval", time_limit_s=None)
    env.reset(11, {})
    env.step({"tool": "submit", "args": {"findings": sub(ep, [{
        "type": "duplicate_invoice", "affected_ids": [inv["invoice_id"]], "claimed_amount_minor": inv["total_minor"],
        "currency": ep.manifest["base_currency"], "evidence_ids": [inv["doc_id"]], "explanation": "same amount"}])}})
    g = env.grader_result()
    assert g["unmatched_findings"][0]["attribution"] == "distractor:recurring_amount"


def test_gymnasium_style_reset_kwargs():
    env = AuditEnv(mode="train", time_limit_s=None)
    o1, i1 = env.reset(seed=3, options={"config": ALL})
    o2, i2 = env.reset(3, ALL)
    assert i1 == i2 and o1 == o2
    obs, info = env.reset()          # seed=None: случайный эпизод
    assert obs["steps_remaining"] == 100


def test_get_record_accepts_doc_id(ep_all):
    env = AuditEnv(mode="train", time_limit_s=None)
    env.reset(4242, ALL)
    inv = ep_all.tables["invoices"][0]
    o, *_ = env.step({"tool": "get_record", "args": {"table": "invoices", "id": inv["doc_id"]}})
    assert o["result"]["record"] == inv


def test_manifest_marks_data_as_synthetic(ep_all):
    assert ep_all.manifest["synthetic"] is True and "fictional" in ep_all.manifest["disclaimer"]


def test_closed_run_requires_grader_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDITGYM_SECRET", raising=False)
    split = tmp_path / "hidden.json"
    split.write_text(json.dumps({"config_version": "jury", "episodes": [{"name": "h", "seed": 5, "config": {}}]}))
    with pytest.raises(SystemExit):
        serve(mode="eval", episodes=str(split), stdin=io.StringIO(""), stdout=io.StringIO())
    with pytest.raises(SystemExit):
        evaluate(["flag_all"], [{"name": "h", "seed": 5, "config": {}}], isolated=True)


def test_config_size_bounds():
    from auditgym.config import resolve_config
    for bad in ({"n_invoices": [100000, 100000]}, {"n_expenses": [5, 2]}, {"n_customers": True}):
        with pytest.raises(ValueError):
            resolve_config(bad)
