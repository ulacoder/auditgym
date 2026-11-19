"""CLI: python -m auditgym <command>

check         единая команда проверки: тесты + валидация эпизодов + базовые политики + парные проверки
export        выгрузить открытые эпизоды (public/) и закрытые файлы оценщика (grader/)
trajectories  записать trajectories.jsonl (режим train)
evaluate      запустить политики на сплите и вывести P/R/F1/возврат/шаги/время
demo          показать работу среды на одном эпизоде скриптовой политикой
serve         изолированный сервер среды (JSON-строки по stdin/stdout)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from .harness import ROOT, evaluate, load_split, paired_checks, record_trajectories
from .policies import BASELINES

SPLITS = {"train": "configs/train.json", "dev": "configs/dev.json", "holdout_example": "configs/holdout_example.json"}


def _split(name):
    p = Path(name)
    if not p.exists():
        p = ROOT / SPLITS[name]
    return load_split(p)


def cmd_export(args):
    from .generator import generate
    out = Path(args.out)
    for split in args.splits:
        for e in _split(split):
            ep = generate(e["seed"], e["config"])
            pub = out / "episodes" / split / e["name"]
            grd = out / "grader_data" / split / e["name"]
            pub.mkdir(parents=True, exist_ok=True)
            grd.mkdir(parents=True, exist_ok=True)
            (pub / "manifest.json").write_text(json.dumps(ep.manifest, indent=2, ensure_ascii=False))
            (pub / "policy.json").write_text(json.dumps(ep.policy, indent=2, ensure_ascii=False))
            for t, fname, cols in (("ledger", "ledger.csv", ["row_id", "timestamp", "account", "debit_minor",
                                                             "credit_minor", "currency", "entity_id", "invoice_id",
                                                             "payment_id"]),
                                   ("fx", "fx.csv", ["date", "source_currency", "target_currency", "rate"])):
                with open(pub / fname, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=cols)
                    w.writeheader()
                    for r in ep.tables[t]:
                        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
            for t in ("invoices", "payments"):
                with open(pub / f"{t}.jsonl", "w", encoding="utf-8") as fh:
                    for r in ep.tables[t]:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            (grd / "truth.json").write_text(json.dumps(ep.truth, indent=2, ensure_ascii=False))
            (grd / "provenance.json").write_text(json.dumps(ep.provenance, indent=2, ensure_ascii=False))
            print(f"{split}/{e['name']}: {ep.episode_id} issues={len(ep.truth['issues'])} "
                  f"rows={ep.manifest['counts']['ledger_rows']} docs={ep.manifest['counts']['documents']}")


def cmd_trajectories(args):
    eps = _split(args.split)
    n = record_trajectories(eps, args.policies, args.out)
    print(f"wrote {n} steps to {args.out}")


def _table(res):
    lines = ["| policy | episodes | P | R | F1 (mean) | F1 (micro) | return | steps | time, s | invalid |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in res.items():
        s = d["summary"]
        lines.append(f"| {name} | {s['episodes']} | {s['mean_precision']:.3f} | {s['mean_recall']:.3f} | "
                     f"{s['mean_f1']:.3f} | {s['micro_f1']:.3f} | {s['mean_return']:+.3f} | {s['mean_steps']:.1f} | "
                     f"{s['mean_time_s']:.3f} | {s['invalid_submissions']} |")
    return "\n".join(lines)


def cmd_evaluate(args):
    res = evaluate(args.policies, _split(args.split), isolated=not args.in_process)
    print(_table(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_demo(args):
    from .env import AuditEnv
    from .policies import make_policy
    e = _split(args.split)[args.index]
    env = AuditEnv(mode="train")
    pol = make_policy("scripted_auditor")
    obs, info = env.reset(e["seed"], e["config"])
    m = obs["manifest"]
    print(f"episode {info['episode_id']}  company={m['company']}  base={m['base_currency']}  tz={m['timezone']}")
    print(f"period [{m['period_start']}, {m['period_end']})  counts={m['counts']}")
    pol.reset(obs, info)
    done = False
    while not done:
        a = pol.act(obs)
        obs, r, te, tr, inf = env.step(a)
        done = te or tr
        if a["tool"] == "list_records":
            res = obs["result"]
            print(f"step {obs['step']:>3}: list_records({a['args']['table']}, cursor={a['args']['cursor']}) "
                  f"-> {len(res['records'])} rows, next={res['next_cursor']}")
        else:
            fs = a["args"]["findings"]["findings"]
            print(f"step {obs['step']:>3}: submit {len(fs)} findings")
            for f in fs:
                print(f"         - {f['type']:<19} {','.join(f['affected_ids']):<28} "
                      f"{f['claimed_amount_minor']:>12} {f['currency']}  evidence={f['evidence_ids']}")
    g = inf["grading"]
    print(f"reward={r:+.3f}  P={g['precision']:.2f} R={g['recall']:.2f} F1={g['f1']:.2f}  "
          f"TP={g['tp']} FP={g['fp']} FN={g['fn']}  inspection_steps={g['inspection_steps']}")


def fingerprint() -> str:
    """SHA-256 всех открытых таблиц и эталонов сплитов: одинаков на любой машине и версии Python 3.11+."""
    import hashlib
    from .config import canonical_json
    from .generator import generate
    from .validate import tables_hash
    h = hashlib.sha256()
    for split in ("train", "dev", "holdout_example"):
        for e in _split(split):
            ep = generate(e["seed"], e["config"])
            h.update(tables_hash(ep.tables).encode())
            h.update(canonical_json(ep.truth).encode())
    return h.hexdigest()


def cmd_check(args):
    from .generator import generate
    from .validate import validate_episode
    t0 = time.time()
    report = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "sections": {}}
    ok_all = True
    if not args.skip_tests:
        print("[1/4] pytest ...", flush=True)
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")],
                           cwd=ROOT, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        print("      " + tail)
        report["sections"]["tests"] = {"ok": r.returncode == 0, "summary": tail}
        ok_all &= r.returncode == 0
        if r.returncode != 0:
            print(r.stdout[-4000:])
    print("[2/4] validating episodes (data & truth) ...", flush=True)
    val = {}
    for split in ("train", "dev", "holdout_example"):
        rows = []
        for e in _split(split):
            v = validate_episode(generate(e["seed"], e["config"]), e["seed"], e["config"])
            v["name"] = e["name"]
            rows.append(v)
        passed = sum(v["ok"] for v in rows)
        print(f"      {split}: {passed}/{len(rows)} episodes valid")
        for v in rows:
            if not v["ok"]:
                print(f"      FAIL {v['name']}: {[k for k, x in v['checks'].items() if not x]}")
        val[split] = {"passed": passed, "total": len(rows), "episodes": rows}
        ok_all &= passed == len(rows)
    report["sections"]["validation"] = val
    fp = fingerprint()
    ref = ROOT / "configs" / "FINGERPRINT"
    fp_ok = (not ref.exists()) or ref.read_text().strip() == fp
    print(f"      data fingerprint {fp[:16]}... {'matches configs/FINGERPRINT' if ref.exists() and fp_ok else 'MISMATCH' if not fp_ok else '(no reference)'}")
    report["sections"]["fingerprint"] = {"value": fp, "ok": fp_ok}
    ok_all &= fp_ok
    print("[3/4] baseline policies on holdout_example (isolated grader process) ...", flush=True)
    hold = _split("holdout_example")
    res = evaluate(list(BASELINES) + ["scripted_auditor"], hold, isolated=True)
    print(_table(res))
    s = {k: v["summary"] for k, v in res.items()}
    expectations = {
        "oracle_max_reward": abs(s["oracle"]["mean_return"] - 1.0) < 1e-9,
        "flag_all_is_bad": s["flag_all"]["mean_return"] < -0.5,
        "random_is_bad": s["random_explorer"]["mean_return"] < 0.0,
        "naive_below_auditor": s["naive_reconciler"]["mean_f1"] < s["scripted_auditor"]["mean_f1"],
        "auditor_solves_from_observations": s["scripted_auditor"]["micro_f1"] == 1.0,
    }
    for k, v in expectations.items():
        print(f"      {'ok ' if v else 'FAIL'} {k}")
    ok_all &= all(expectations.values())
    report["sections"]["baselines"] = {"results": res, "expectations": expectations}
    print("[4/4] paired checks: repeat seed/actions, permutation + ID change ...", flush=True)
    pc = paired_checks(hold)
    passed = sum(p["ok"] for p in pc)
    print(f"      {passed}/{len(pc)} paired checks passed")
    ok_all &= passed == len(pc)
    report["sections"]["paired_checks"] = {"passed": passed, "total": len(pc), "checks": pc}
    report["ok"] = bool(ok_all)
    report["elapsed_s"] = round(time.time() - t0, 2)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md = [f"# Check report\n", f"Overall: **{'PASS' if ok_all else 'FAIL'}** ({report['elapsed_s']} s)\n",
          "## Validation\n"] + [f"- {k}: {v['passed']}/{v['total']}" for k, v in val.items()] + \
         ["\n## Baselines (holdout_example, 20 episodes)\n", _table(res), "\n## Failure reasons (unmatched findings)\n"] + \
         [f"- {k}: {v['summary']['failure_reasons']}" for k, v in res.items()] + \
         ["\n## Per error type (found/total, FP)\n"] + \
         [f"- {k}: " + ", ".join(f"{t} {d['found']}/{d['total']} FP={d['fp']}" for t, d in v['summary']['per_type'].items())
          for k, v in res.items()] + \
         [f"\n## Paired checks\n\n{passed}/{len(pc)} passed"]
    (out / "check_report.md").write_text("\n".join(md) + "\n")
    print(f"\nOVERALL: {'PASS' if ok_all else 'FAIL'}  (report: {out / 'check_report.md'})")
    sys.exit(0 if ok_all else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="auditgym")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--report-dir", default=str(ROOT / "reports"))
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("export")
    p.add_argument("--out", default=str(ROOT))
    p.add_argument("--splits", nargs="+", default=["train", "dev"])
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("trajectories")
    p.add_argument("--split", default="train")
    p.add_argument("--policies", nargs="+", default=["scripted_auditor", "random_explorer", "naive_reconciler", "flag_all"])
    p.add_argument("--out", default=str(ROOT / "trajectories" / "trajectories.jsonl"))
    p.set_defaults(fn=cmd_trajectories)
    p = sub.add_parser("evaluate")
    p.add_argument("--split", default="dev")
    p.add_argument("--policies", nargs="+", default=list(BASELINES) + ["scripted_auditor"])
    p.add_argument("--in-process", action="store_true")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("demo")
    p.add_argument("--split", default="train")
    p.add_argument("--index", type=int, default=8)
    p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("serve")
    p.add_argument("--mode", default="eval", choices=["train", "eval"])
    p.add_argument("--grader-log")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--no-time-limit", action="store_true")
    p.set_defaults(fn=lambda a: __import__("auditgym.server", fromlist=["serve"]).serve(
        a.mode, a.grader_log, a.max_steps, None if a.no_time_limit else a.time_limit))
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
