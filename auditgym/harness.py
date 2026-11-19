"""Харнесс оценщика: запуск политик, траектории, парные проверки, отчёт."""
from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from pathlib import Path

from .config import canonical_json
from .env import AuditEnv
from .generator import generate
from .policies import BASELINES, make_policy
from .server import RemoteEnv

ROOT = Path(__file__).resolve().parent.parent


def load_split(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [{**e, "config": {**e.get("config", {}), "config_version": data["config_version"]}}
            for e in data["episodes"]]


def run_episode(env, policy, seed=None, config=None, max_steps=100, index=None):
    t0 = time.perf_counter()
    obs, info = env.reset(index=index) if index is not None else env.reset(seed, config)
    policy.reset(obs, info)
    done, steps, ret = False, 0, 0.0
    while not done:
        try:
            action = policy.act(obs)
        except Exception:
            # упавшая политика = неверная структура ответа (награда −1 у оценщика)
            action = {"tool": "submit", "args": {"findings": None}}
        obs, r, te, tr, inf = env.step(action)
        steps += 1
        ret += r
        done = te or tr
        if steps > max_steps + 1:
            raise RuntimeError("environment did not terminate")
    return info["episode_id"], steps, ret, time.perf_counter() - t0, inf


ORACLE_POLICIES = frozenset({"oracle"})


def evaluate(policy_names, episodes, isolated=True, mode="eval", oracle_policies=ORACLE_POLICIES,
             allow_public_secret=False) -> dict:
    """Одинаковые политики на одних эпизодах; награда берётся у оценщика.

    isolated=True — три процесса: оценщик со средой (serve --episodes), политика (policy-worker)
    и этот харнесс, который пересылает наблюдения и действия. Seed, config и секрет оценщика
    в процесс политики не попадают; эталон получает только политика из oracle_policies.
    """
    if isolated and not os.environ.get("AUDITGYM_SECRET") and not allow_public_secret:
        raise SystemExit("AUDITGYM_SECRET is not set: a closed run needs a grader secret "
                         "(or pass allow_public_secret / --allow-public-secret for open splits)")
    from .policy_worker import RemotePolicy
    out = {}
    for name in policy_names:
        rows = []
        grader_only = name in oracle_policies
        log = tempfile.NamedTemporaryFile(prefix="grader_", suffix=".jsonl", delete=False).name
        if isolated:
            split = tempfile.NamedTemporaryFile(prefix="split_", suffix=".json", delete=False, mode="w")
            json.dump({"config_version": "eval-run", "episodes": episodes}, split)
            split.close()
            env = RemoteEnv(mode=mode, grader_log=log, episodes=split.name, allow_oracle=grader_only,
                            allow_public_secret=allow_public_secret)
            policy_env = {k: v for k, v in os.environ.items() if k != "AUDITGYM_SECRET"}
            pol = RemotePolicy(name, policy_env)
        else:
            env = AuditEnv(mode=mode)
        try:
            for k, e in enumerate(episodes):
                t0 = time.perf_counter()
                if isolated:
                    obs, info = env.reset(index=k)
                    pol.set_truth(env.oracle_truth() if grader_only else None)
                    policy = pol
                    res = None
                else:
                    ep = generate(e["seed"], e["config"])
                    obs, info = env.reset_episode(ep)
                    policy = make_policy(name, (lambda eid, t=ep.truth: t) if grader_only else (lambda eid: None))
                policy.reset(obs, info)
                done, steps = False, 0
                while not done:
                    try:
                        action = policy.act(obs)
                    except Exception:
                        action = {"tool": "submit", "args": {"findings": None}}
                    obs, r, te, tr, _ = env.step(action)
                    steps += 1
                    done = te or tr
                if not isolated:
                    res = env.grader_result()
                rows.append({"name": e["name"], "episode_id": info["episode_id"], "steps": steps,
                             "time_s": time.perf_counter() - t0, "result": res})
        finally:
            if isolated:
                pol.close()
                env.close()
                os.unlink(split.name)
        if isolated:
            with open(log, encoding="utf-8") as fh:
                by_id = {}
                for line in fh:
                    d = json.loads(line)
                    by_id.setdefault(d["episode_id"], d["result"])
            for r in rows:
                r["result"] = by_id[r["episode_id"]]
        os.unlink(log)
        for r in rows:
            r["n_issues"] = len(r["result"].get("issue_types", {}))
            r["issue_types"] = r["result"].get("issue_types", {})
        out[name] = {"episodes": rows, "summary": summarize(rows)}
    return out


def summarize(rows) -> dict:
    res = [r["result"] for r in rows]
    tp, fp, fn = sum(x["tp"] for x in res), sum(x["fp"] for x in res), sum(x["fn"] for x in res)
    return {
        "episodes": len(rows),
        "mean_precision": statistics.mean(x["precision"] for x in res),
        "mean_recall": statistics.mean(x["recall"] for x in res),
        "mean_f1": statistics.mean(x["f1"] for x in res),
        "micro_f1": (2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else 1.0,
        "mean_return": statistics.mean(x["reward"] for x in res),
        "mean_steps": statistics.mean(r["steps"] for r in rows),
        "mean_time_s": statistics.mean(r["time_s"] for r in rows),
        "tp": tp, "fp": fp, "fn": fn,
        "invalid_submissions": sum(1 for x in res if not x["valid"]),
        "failure_reasons": _reasons(res),
        "per_type": _per_type(rows),
    }


def _per_type(rows) -> dict:
    """Разбор сбоев по типам: найдено/всего (recall) и число FP с этим типом."""
    out: dict[str, dict] = {}
    for r in rows:
        res, types = r["result"], r.get("issue_types", {})
        found = {p["issue_id"] for p in res.get("pairs", [])}
        for iid, t in types.items():
            d = out.setdefault(t, {"found": 0, "total": 0, "fp": 0})
            d["total"] += 1
            d["found"] += iid in found
        for u in res.get("unmatched_findings", []):
            out.setdefault(u["type"], {"found": 0, "total": 0, "fp": 0})["fp"] += 1
    return dict(sorted(out.items()))


def _reasons(res) -> dict:
    """Разбор ложных выводов: причина несовпадения и к чему относится вывод (ловушка, чистая запись...)."""
    c: dict[str, int] = {}
    for x in res:
        for u in x.get("unmatched_findings", []):
            key = u.get("attribution") or u["reason"]
            c[key] = c.get(key, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def record_trajectories(episodes, policy_names, path) -> int:
    """Траектории в режиме train (терминальная награда видна политике). Эталона в наблюдениях нет."""
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        def rec(d):
            nonlocal n
            fh.write(json.dumps({**d, "policy": current[0]}, ensure_ascii=False) + "\n")
            n += 1
        current = [None]
        env = AuditEnv(mode="train", recorder=rec)
        truths = {}
        for name in policy_names:
            current[0] = name
            for e in episodes:
                ep = generate(e["seed"], e["config"])
                truths[ep.episode_id] = ep.truth
                pol = make_policy(name, truths.__getitem__)
                run_episode(env, pol, e["seed"], e["config"])
    return n


def _transform(cfg, k):
    return {**cfg, "id_salt": f"pair-{k}", "order_seed": 7000 + k}


def paired_checks(episodes) -> list[dict]:
    """20 парных проверок: (1) повтор seed+действий даёт ту же траекторию;
    (2) смена ID и порядка сохраняет ответ oracle и награду (с учётом новых ID),
    а также награду скриптового аудитора."""
    out = []
    for k, e in enumerate(episodes):
        traj = []
        for _ in range(2):
            buf = []
            env = AuditEnv(mode="train", recorder=buf.append, time_limit_s=None)
            run_episode(env, make_policy("scripted_auditor"), e["seed"], e["config"])
            traj.append(canonical_json([{k2: v for k2, v in d.items()} for d in buf]))
        repeat_ok = traj[0] == traj[1]
        a = generate(e["seed"], e["config"])
        b = generate(e["seed"], _transform(e["config"], k))
        ids_changed = (set(i["invoice_id"] for i in a.tables["invoices"])
                       .isdisjoint(i["invoice_id"] for i in b.tables["invoices"]))
        order_changed = [r["debit_minor"] for r in a.tables["ledger"]] != [r["debit_minor"] for r in b.tables["ledger"]]

        def strip_ids(ep):
            return sorted((t["type"], t["expected_amount_minor"], t["currency"], t["interaction"] or "",
                           len(t["affected_ids"])) for t in ep.truth["issues"])
        truth_same = strip_ids(a) == strip_ids(b)
        rewards = {}
        for tag, ep, cfg in (("orig", a, e["config"]), ("transformed", b, _transform(e["config"], k))):
            for pname in ("oracle", "scripted_auditor"):
                env = AuditEnv(mode="train", time_limit_s=None)
                pol = make_policy(pname, {ep.episode_id: ep.truth}.__getitem__)
                _, _, ret, _, _ = run_episode(env, pol, e["seed"], cfg)
                rewards[f"{pname}_{tag}"] = ret
        inv_ok = (rewards["oracle_orig"] == rewards["oracle_transformed"] == 1.0
                  and rewards["scripted_auditor_orig"] == rewards["scripted_auditor_transformed"])
        out.append({"name": e["name"], "repeat_identical": repeat_ok, "ids_changed": ids_changed,
                    "order_changed": order_changed, "truth_invariant": truth_same, "reward_invariant": inv_ok,
                    "rewards": rewards,
                    "ok": repeat_ok and ids_changed and order_changed and truth_same and inv_ok})
    return out
