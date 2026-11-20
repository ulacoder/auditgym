"""Харнесс оценщика: запуск политик, траектории, парные проверки, отчёт."""
from __future__ import annotations

import json
import os
import sys
import statistics
import time
from pathlib import Path

from .config import canonical_json
from .env import AuditEnv
from .generator import generate
from .policies import BASELINES, make_policy
from .server import RemoteEnv

ROOT = Path(__file__).resolve().parent.parent


def split_from_dict(data: dict) -> list[dict]:
    """config_version берётся из файла сплита, если у эпизода нет своего: episode_id совпадает с export."""
    from .env import ENV_VERSION
    ev = data.get("env_version")
    if ev is not None and ev != ENV_VERSION:
        raise ValueError(f"split was made for env_version {ev}, this environment is {ENV_VERSION}")
    return [{**e, "config": {"config_version": data["config_version"], **e.get("config", {})}}
            for e in data["episodes"]]


def load_split(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return split_from_dict(json.load(fh))


def run_episode(env, policy, seed=None, config=None, max_steps=100, index=None, episode=None):
    t0 = time.perf_counter()
    if episode is not None:
        obs, info = env.reset_episode(episode)
    elif index is not None:
        obs, info = env.reset(index=index)
    else:
        obs, info = env.reset(seed, config)
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


TIME_GRACE_S = 1.0
_PROBE = {"tool": "list_records", "args": {"table": "invoices", "limit": 1}}


def evaluate(policy_names, episodes, isolated=True, mode="eval", oracle_policies=ORACLE_POLICIES,
             allow_public_secret=False, time_limit_s: float = 60.0, policy_path: str | None = None) -> dict:
    """Одинаковые политики на одних эпизодах; награда берётся у оценщика.

    isolated=True — три процесса: оценщик со средой (serve), политика (policy-worker) и этот
    харнесс, который пересылает наблюдения и действия.
      * seed/config закрытых эпизодов идут оценщику по анонимному каналу (не файлом и не в argv);
      * итоги оценщика возвращаются по отдельному каналу прямо в харнесс (файла журнала нет);
      * харнесс и оценщик недампируемы (PR_SET_DUMPABLE=0): /proc/<pid>/environ, mem и fd
        недоступны процессу политики, поэтому секрет оценщика ему не виден;
      * у политики крайний срок time_limit_s на эпизод: зависшая политика убивается, эпизод
        завершается как truncated(time_limit) с пустым ответом, следующий эпизод идёт с новым процессом;
      * код пакета auditgym хешируется до запуска и проверяется после каждой политики.
    Эталон получает только политика из oracle_policies.
    """
    from .generator import has_private_secret
    if not has_private_secret() and not allow_public_secret:
        raise SystemExit("AUDITGYM_SECRET is not set (or equals the public default): a closed run needs a grader secret "
                         "(or pass allow_public_secret / --allow-public-secret for open splits)")
    from ._secure import harden, kill_user_processes, package_digest, purge_user_files, sandbox_spec
    from .policy_worker import TIMEOUT, RemotePolicy
    sandbox = None
    if isolated:
        harden()
        sandbox = sandbox_spec()
        if sandbox is None:
            sys.stderr.write("auditgym: policies run as the same OS user as the grader. For full OS isolation run "
                             "evaluate as root with a user 'auditgym-policy' (the Docker image does this).\n")
    digest = package_digest()
    out = {}
    for name in policy_names:
        rows = []
        grader_only = name in oracle_policies
        if isolated:
            env = RemoteEnv(mode=mode, episodes_data={"config_version": "eval-run", "episodes": episodes},
                            collect_results=True, allow_oracle=grader_only,
                            allow_public_secret=allow_public_secret, time_limit_s=time_limit_s)
            policy_env = {k: v for k, v in os.environ.items() if k != "AUDITGYM_SECRET"}
            pol = RemotePolicy(name, policy_env, sandbox=sandbox, policy_path=policy_path)
        else:
            env = AuditEnv(mode=mode, time_limit_s=time_limit_s)
        grader_died = False
        try:
            for k, e in enumerate(episodes):
                t0 = time.perf_counter()
                if isolated and grader_died:
                    break
                if isolated:
                    try:
                        obs, info = env.reset(index=k)
                    except (RuntimeError, OSError):
                        grader_died = True      # процесс оценщика убит (например, политикой)
                        break
                    deadline = time.monotonic() + time_limit_s + TIME_GRACE_S
                    pol.set_truth(env.oracle_truth() if grader_only else None)
                    timed_out = pol.reset(obs, info, deadline) is TIMEOUT
                    res = None
                else:
                    ep = generate(e["seed"], e["config"])
                    obs, info = env.reset_episode(ep)
                    policy = make_policy(name, (lambda eid, t=ep.truth: t) if grader_only else (lambda eid: None))
                    policy.reset(obs, info)
                    timed_out = False
                done, steps = False, 0
                while not done:
                    if isolated:
                        action = _PROBE if timed_out else pol.act(obs, deadline)
                        if action is TIMEOUT:
                            timed_out, action = True, _PROBE
                        if timed_out:
                            # срок оценщика истёк раньше нашего (он стартовал до reset): шаг даст truncated
                            while time.monotonic() < deadline:
                                time.sleep(0.05)
                    else:
                        try:
                            action = policy.act(obs)
                        except Exception:
                            action = {"tool": "submit", "args": {"findings": None}}
                    try:
                        obs, r, te, tr, _ = env.step(action)
                    except (RuntimeError, OSError):
                        if not isolated:
                            raise
                        grader_died = True
                        break
                    steps += 1
                    done = te or tr
                    if steps > 1000:
                        raise RuntimeError("environment did not terminate")
                if grader_died:
                    break
                if not isolated:
                    res = env.grader_result()
                rows.append({"name": e["name"], "episode_id": info["episode_id"], "steps": steps,
                             "time_s": time.perf_counter() - t0, "result": res, "policy_timeout": timed_out})
        finally:
            if isolated:
                pol.close()
                if sandbox is not None:
                    kill_user_processes(sandbox["uid"])   # фоновые процессы политики не переживают её прогон
                    purge_user_files(sandbox["uid"])      # и её файлы во временных каталогах тоже
                try:
                    env.close()
                except OSError:
                    pass
        if isolated:
            by_id = {}
            for d in env.results():
                by_id.setdefault(d["episode_id"], d["result"])
            if grader_died:
                # оценщик убит до конца прогона: каждый эпизод без итога засчитывается как неверная отправка
                done_ids = {r["episode_id"] for r in rows}
                for e in episodes[len(rows):]:
                    ep = generate(e["seed"], e["config"])
                    rows.append({"name": e["name"], "episode_id": ep.episode_id, "steps": 0, "time_s": 0.0,
                                 "result": None, "policy_timeout": False})
                for r, e in zip(rows, episodes):
                    if r["episode_id"] not in by_id:
                        by_id[r["episode_id"]] = _invalid_result(generate(e["seed"], e["config"]), "grader_killed")
                del done_ids
            missing = [r["episode_id"] for r in rows if r["episode_id"] not in by_id]
            if missing:
                raise RuntimeError(f"grader returned no result for {missing}")
            for r in rows:
                r["result"] = by_id[r["episode_id"]]
                if grader_died:
                    r["grader_killed"] = True
        if package_digest() != digest:
            raise RuntimeError(f"auditgym package files changed while running policy {name!r}: run aborted")
        for r in rows:
            r["n_issues"] = len(r["result"].get("issue_types", {}))
            r["issue_types"] = r["result"].get("issue_types", {})
        out[name] = {"episodes": rows, "summary": summarize(rows)}
    return out


def _invalid_result(ep, why: str) -> dict:
    from .matcher import grade
    res = grade(None, ep.truth, 0)
    res.update(outcome=why, episode_id=ep.episode_id,
               issue_types={t["issue_id"]: t["type"] for t in ep.truth["issues"]},
               interactions={t["issue_id"]: t["interaction"] for t in ep.truth["issues"] if t.get("interaction")},
               unmatched_findings=[], missed_issue_ids=[t["issue_id"] for t in ep.truth["issues"]])
    return res


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
