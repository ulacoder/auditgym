"""Изолированный режим: среда и оценщик в отдельном процессе, обмен JSON-строками.

Запуск:  python -m auditgym serve --mode eval --episodes configs/hidden.json --grader-log grader.jsonl

Запросы (одна JSON-строка на запрос):
  {"op": "reset", "index": k}                 эпизод k из --episodes: seed и config остаются у оценщика
  {"op": "reset", "seed": 1, "config": {...}} только без --episodes (открытые эксперименты)
  {"op": "step", "action": {...}}
  {"op": "oracle_truth"}                      только при --allow-oracle (политика oracle организаторов)
  {"op": "close"}
Ответы: {"observation", "info"} | {"observation", "reward", "terminated", "truncated", "info"} | {"error"}

Итог оценки (награда, P/R/F1, диагностика) пишется ТОЛЬКО в --grader-log, который процесс
агента не читает. В режиме eval политика получает reward=0 и пустой info.
"""
from __future__ import annotations

import json
import subprocess
import sys

from .env import AuditEnv


def _load_episodes(path):
    from .harness import load_split
    return load_split(path)


def serve(mode: str = "eval", grader_log: str | None = None, max_steps: int = 100,
          time_limit_s: float | None = 60.0, stdin=None, stdout=None, episodes: str | None = None,
          allow_oracle: bool = False, allow_public_secret: bool = False) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    env = AuditEnv(mode=mode, max_steps=max_steps, time_limit_s=time_limit_s)
    hidden = _load_episodes(episodes) if episodes else None
    if hidden is not None:
        import os
        if not os.environ.get("AUDITGYM_SECRET") and not allow_public_secret:
            sys.stderr.write("auditgym serve: AUDITGYM_SECRET is not set. A closed run (--episodes) needs a grader "
                             "secret, otherwise small seeds can be brute-forced from public IDs. Set it, or pass "
                             "--allow-public-secret for open splits.\n")
            raise SystemExit(2)
    episode_id, logged = None, False
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "reset":
                if hidden is not None:
                    if "seed" in req or "config" in req:
                        raise PermissionError("seed/config reset disabled: use index")
                    e = hidden[int(req["index"])]
                    obs, info = env.reset(e["seed"], e["config"])
                else:
                    seed = req.get("seed")
                    if not isinstance(seed, int) or isinstance(seed, bool):
                        raise TypeError("seed must be an integer")
                    obs, info = env.reset(seed, req.get("config") or {})
                episode_id, logged = info["episode_id"], False
                out = {"observation": obs, "info": info}
            elif op == "step":
                obs, r, te, tr, info = env.step(req.get("action"))
                out = {"observation": obs, "reward": r, "terminated": te, "truncated": tr, "info": info}
                if (te or tr) and grader_log and not logged and env.grader_result() is not None:
                    with open(grader_log, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"episode_id": episode_id, "result": env.grader_result()},
                                            ensure_ascii=False) + "\n")
                    logged = True
            elif op == "oracle_truth":
                if not allow_oracle:
                    raise PermissionError("oracle disabled")
                out = {"truth": env._ep.truth}
            elif op == "close":
                break
            else:
                out = {"error": "unknown op"}
        except Exception as e:  # ограниченная ошибка протокола без деталей реализации
            out = {"error": f"protocol_error: {type(e).__name__}"}
        stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        stdout.flush()


class RemoteEnv:
    """Клиент к `python -m auditgym serve`: тот же контракт reset/step."""

    def __init__(self, mode="eval", grader_log=None, max_steps=100, time_limit_s=60.0, episodes=None,
                 allow_oracle=False, env=None, allow_public_secret=False):
        cmd = [sys.executable, "-m", "auditgym", "serve", "--mode", mode, "--max-steps", str(max_steps)]
        if grader_log:
            cmd += ["--grader-log", grader_log]
        if episodes:
            cmd += ["--episodes", str(episodes)]
        if allow_oracle:
            cmd += ["--allow-oracle"]
        if allow_public_secret:
            cmd += ["--allow-public-secret"]
        if time_limit_s is None:
            cmd += ["--no-time-limit"]
        else:
            cmd += ["--time-limit", str(time_limit_s)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)

    def _call(self, req):
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        if "error" in r:
            raise RuntimeError(r["error"])
        return r

    def reset(self, seed=None, config=None, index=None):
        req = {"op": "reset", "index": index} if index is not None else {"op": "reset", "seed": seed,
                                                                           "config": config or {}}
        r = self._call(req)
        return r["observation"], r["info"]

    def step(self, action):
        r = self._call({"op": "step", "action": action})
        return r["observation"], r["reward"], r["terminated"], r["truncated"], r["info"]

    def oracle_truth(self):
        return self._call({"op": "oracle_truth"})["truth"]

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"op": "close"}) + "\n")
            self.p.stdin.flush()
        except Exception:
            pass
        self.p.wait(timeout=10)
