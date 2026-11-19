"""Выгрузка эпизода на диск и загрузка обратно.

Открытая часть (видит агент):   <dir>/manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv
Закрытая часть (только оценщик): <grader_dir>/truth.json, provenance.json, ledger_before.csv

Загрузка позволяет жюри подать в среду эпизод, который они сами изменили
(переставили строки, переименовали ID): AuditEnv().reset(config={"episode_dir": ..., "grader_dir": ...}).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .generator import Episode

LEDGER_COLS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency", "entity_id",
               "invoice_id", "payment_id"]
FX_COLS = ["date", "source_currency", "target_currency", "rate"]
_INT_COLS = {"debit_minor", "credit_minor"}


def _write_csv(path: Path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in cols})


def _read_csv(path: Path, int_cols=frozenset(), nullable=frozenset()):
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                if k in int_cols:
                    row[k] = int(v)
                elif k in nullable and v == "":
                    row[k] = None
                else:
                    row[k] = v
            out.append(row)
    return out


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def export_episode(ep: Episode, public_dir, grader_dir) -> None:
    pub, grd = Path(public_dir), Path(grader_dir)
    pub.mkdir(parents=True, exist_ok=True)
    grd.mkdir(parents=True, exist_ok=True)
    (pub / "manifest.json").write_text(json.dumps(ep.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (pub / "policy.json").write_text(json.dumps(ep.policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(pub / "ledger.csv", LEDGER_COLS, ep.tables["ledger"])
    _write_csv(pub / "fx.csv", FX_COLS, ep.tables["fx"])
    for t in ("invoices", "payments"):
        with open(pub / f"{t}.jsonl", "w", encoding="utf-8") as fh:
            for r in ep.tables[t]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (grd / "truth.json").write_text(json.dumps(ep.truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (grd / "provenance.json").write_text(json.dumps(ep.provenance, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    if ep.clean_tables:
        _write_csv(grd / "ledger_before.csv", LEDGER_COLS, ep.clean_tables["ledger"])


def load_episode(public_dir, grader_dir) -> Episode:
    pub, grd = Path(public_dir), Path(grader_dir)
    manifest = json.loads((pub / "manifest.json").read_text(encoding="utf-8"))
    policy = json.loads((pub / "policy.json").read_text(encoding="utf-8"))
    tables = {
        "ledger": _read_csv(pub / "ledger.csv", _INT_COLS, {"entity_id", "invoice_id", "payment_id"}),
        "invoices": _jsonl(pub / "invoices.jsonl"),
        "payments": _jsonl(pub / "payments.jsonl"),
        "fx": _read_csv(pub / "fx.csv"),
    }
    truth = json.loads((grd / "truth.json").read_text(encoding="utf-8"))
    prov_path = grd / "provenance.json"
    provenance = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {}
    if truth.get("episode_id") != manifest.get("episode_id"):
        raise ValueError("truth.json episode_id does not match manifest.json")
    clean = None
    before = grd / "ledger_before.csv"
    if before.exists():
        clean = {"ledger": _read_csv(before, _INT_COLS, {"entity_id", "invoice_id", "payment_id"})}
    return Episode(manifest["episode_id"], int(provenance.get("seed", -1)), provenance.get("config", {}),
                   manifest, policy, tables, truth, provenance, clean)
