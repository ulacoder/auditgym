"""Процесс политики: получает ТОЛЬКО наблюдения и возвращает действия (JSON-строки по stdin/stdout).

Харнесс оценщика держит среду в одном процессе (server), политику — в другом (этот модуль),
и пересылает между ними наблюдения и действия. Политика не видит ни seed, ни config, ни
секрета оценщика, ни эталона (кроме политик, которым оценщик явно разрешил эталон: oracle).

Запросы: {"op":"reset","observation":..,"info":..,"truth":<только для oracle>} | {"op":"act","observation":..} | {"op":"close"}
Ответы:  {"ok":true} | {"action":{...}}
"""
from __future__ import annotations

import json
import sys

INVALID = {"tool": "submit", "args": {"findings": None}}


def main(policy_name: str) -> None:
    from .policies import make_policy
    truth = {"value": None}
    policy = make_policy(policy_name, lambda eid: truth["value"])
    out = sys.stdout
    for line in sys.stdin:
        req = json.loads(line)
        op = req.get("op")
        if op == "reset":
            truth["value"] = req.get("truth")
            try:
                policy = make_policy(policy_name, lambda eid: truth["value"])
                policy.reset(req["observation"], req["info"])
                resp = {"ok": True}
            except Exception:
                resp = {"ok": False}
        elif op == "act":
            try:
                action = policy.act(req["observation"])
                json.dumps(action)
            except Exception:
                action = INVALID          # упавшая политика = неверная структура ответа
            resp = {"action": action}
        elif op == "close":
            break
        else:
            resp = {"error": "unknown op"}
        out.write(json.dumps(resp, ensure_ascii=False) + "\n")
        out.flush()


class RemotePolicy:
    """Клиент к процессу политики: тот же интерфейс reset/act."""

    def __init__(self, policy_name: str, env: dict):
        import subprocess
        self.p = subprocess.Popen([sys.executable, "-m", "auditgym", "policy-worker", "--policy", policy_name],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        self._truth = None

    def set_truth(self, truth):
        self._truth = truth

    def _call(self, req):
        self.p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        return json.loads(line) if line else {"action": INVALID}

    def reset(self, observation, info):
        req = {"op": "reset", "observation": observation, "info": info}
        if self._truth is not None:
            req["truth"] = self._truth
        self._call(req)

    def act(self, observation):
        return self._call({"op": "act", "observation": observation}).get("action", INVALID)

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"op": "close"}) + "\n")
            self.p.stdin.flush()
        except Exception:
            pass
        self.p.wait(timeout=10)
