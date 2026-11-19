"""Изолированный режим: среда и оценщик в отдельном процессе, обмен JSON-строками.

Запрос:  {"op": "reset", "seed": 1, "config": {...}} | {"op": "step", "action": {...}} | {"op": "close"}
Ответ:   {"observation": ..., "info": ...} | {"observation", "reward", "terminated", "truncated", "info"}

Итог оценки (награда, P/R/F1, диагностика) пишется ТОЛЬКО в --grader-log, который
процесс агента не читает. В режиме eval политика получает reward=0 и пустой info.
"""
from __future__ import annotations

import json
import subprocess
import sys

from .env import AuditEnv


def serve(mode: str = "eval", grader_log: str | None = None, max_steps: int = 100,
          time_limit_s: float | None = 60.0, stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    env = AuditEnv(mode=mode, max_steps=max_steps, time_limit_s=time_limit_s)
    episode_id = None
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "reset":
                obs, info = env.reset(int(req["seed"]), req.get("config") or {})
                episode_id = info["episode_id"]
                out = {"observation": obs, "info": info}
            elif op == "step":
                obs, r, te, tr, info = env.step(req.get("action"))
                out = {"observation": obs, "reward": r, "terminated": te, "truncated": tr, "info": info}
                if (te or tr) and grader_log and env.grader_result() is not None and not req.get("_after_done"):
                    with open(grader_log, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"episode_id": episode_id, "result": env.grader_result()}) + "\n")
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

    def __init__(self, mode="eval", grader_log=None, max_steps=100, time_limit_s=60.0):
        cmd = [sys.executable, "-m", "auditgym", "serve", "--mode", mode, "--max-steps", str(max_steps)]
        if grader_log:
            cmd += ["--grader-log", grader_log]
        if time_limit_s is None:
            cmd += ["--no-time-limit"]
        else:
            cmd += ["--time-limit", str(time_limit_s)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._done = False

    def _call(self, req):
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def reset(self, seed, config=None):
        r = self._call({"op": "reset", "seed": seed, "config": config or {}})
        self._done = False
        return r["observation"], r["info"]

    def step(self, action):
        r = self._call({"op": "step", "action": action, "_after_done": self._done})
        if r.get("terminated") or r.get("truncated"):
            self._done = True
        return r["observation"], r["reward"], r["terminated"], r["truncated"], r["info"]

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"op": "close"}) + "\n")
            self.p.stdin.flush()
        except Exception:
            pass
        self.p.wait(timeout=10)
