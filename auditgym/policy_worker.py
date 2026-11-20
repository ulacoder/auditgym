"""Процесс политики: получает ТОЛЬКО наблюдения и возвращает действия (JSON-строки по stdin/stdout).

Харнесс оценщика держит среду в одном процессе (server), политику — в другом (этот модуль),
и пересылает между ними наблюдения и действия. Политика не видит ни seed, ни config, ни
секрета оценщика, ни эталона (кроме политик, которым оценщик явно разрешил эталон: oracle).

Запросы: {"op":"reset","observation":..,"info":..,"truth":<только для oracle>} | {"op":"act","observation":..} | {"op":"close"}
Ответы:  {"ok":true} | {"action":{...}}
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

INVALID = {"tool": "submit", "args": {"findings": None}}
TIMEOUT = object()   # политика не ответила до крайнего срока эпизода


def main(policy_name: str) -> None:
    # Протокол идёт по отдельной копии stdout; print() внутри политики уходит в stderr
    # и не может сломать обмен с харнессом.
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    from .policies import make_policy
    truth = {"value": None}
    policy = None
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        op = req.get("op")
        if op == "reset":
            truth["value"] = req.get("truth")
            try:
                policy = make_policy(policy_name, lambda eid: truth["value"])
                policy.reset(req["observation"], req["info"])
                resp = {"ok": True}
            except BaseException:         # в т.ч. SystemExit/KeyboardInterrupt из кода политики
                resp = {"ok": False}
        elif op == "act":
            try:
                action = policy.act(req["observation"])
                json.dumps(action)
            except BaseException:
                action = INVALID          # упавшая политика = неверная структура ответа
            resp = {"action": action}
        elif op == "close":
            break
        else:
            resp = {"error": "unknown op"}
        try:
            data = json.dumps(resp, ensure_ascii=False)
        except Exception:
            data = json.dumps({"action": INVALID})
        proto.write(data + "\n")
        proto.flush()


class RemotePolicy:
    """Клиент к процессу политики: тот же интерфейс reset/act, но с крайним сроком.

    Если политика не ответила до deadline (time.monotonic()), процесс убивается и вызов
    возвращает TIMEOUT; для следующего эпизода процесс запускается заново. Упавший процесс
    (EOF, мусор вместо JSON) даёт неверную отправку (INVALID), как исключение в политике.
    """

    def __init__(self, policy_name: str, env: dict, sandbox=None, policy_path: str | None = None):
        self.policy_name, self.env = policy_name, env
        self.sandbox, self.policy_path = sandbox, policy_path
        self._truth = None
        self.p = None
        self._spawn()

    def _spawn(self):
        import subprocess
        from ._secure import LineReader, sandbox_preexec
        root = str(Path(__file__).resolve().parent.parent)
        kw = {}
        env = dict(self.env)
        if self.policy_path:
            env["PYTHONPATH"] = os.pathsep.join([self.policy_path, env.get("PYTHONPATH", "")]).strip(os.pathsep)
        if self.sandbox is not None:
            # отдельный пользователь ОС: нет записи в репозиторий, нет доступа к /proc и сигналов оценщика
            home = self.sandbox["home"]
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "LANG": "C.UTF-8",
                   "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHONPATH": os.pathsep.join(x for x in (self.policy_path or os.environ.get("PYTHONPATH", ""),
                                                             root) if x)}
            kw = dict(user=self.sandbox["uid"], group=self.sandbox["gid"], extra_groups=[], cwd=home,
                      preexec_fn=sandbox_preexec)
        else:
            kw = dict(start_new_session=True)
        self.p = subprocess.Popen([sys.executable, "-m", "auditgym", "policy-worker", "--policy", self.policy_name],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, **kw)
        self.reader = LineReader(self.p.stdout.fileno())
        self.dead = False

    def _kill(self):
        if self.p is not None and self.p.poll() is None:
            self.p.kill()
        if self.p is not None:
            self.p.wait()
        self.dead = True

    def ensure_alive(self):
        if self.dead or self.p.poll() is not None:
            self._kill()
            self._spawn()

    def set_truth(self, truth):
        self._truth = truth

    def _call(self, req, deadline=None):
        from ._secure import Timeout
        if self.dead:
            return None
        try:
            self.p.stdin.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
            self.p.stdin.flush()
            line = self.reader.readline(deadline)
        except Timeout:
            self._kill()
            return TIMEOUT
        except (OSError, ValueError):
            self._kill()
            return None
        if line is None:
            self._kill()
            return None
        try:
            r = json.loads(line)
            return r if isinstance(r, dict) else None
        except (ValueError, RecursionError, MemoryError):   # мусор или слишком глубокий JSON
            self._kill()
            return None

    def reset(self, observation, info, deadline=None):
        self.ensure_alive()
        req = {"op": "reset", "observation": observation, "info": info}
        if self._truth is not None:
            req["truth"] = self._truth
        r = self._call(req, deadline)
        return TIMEOUT if r is TIMEOUT else None

    def act(self, observation, deadline=None):
        r = self._call({"op": "act", "observation": observation}, deadline)
        if r is TIMEOUT:
            return TIMEOUT
        return (r or {}).get("action", INVALID)

    def close(self):
        try:
            if not self.dead and self.p.poll() is None:
                self.p.stdin.write(b'{"op": "close"}\n')
                self.p.stdin.flush()
                self.p.wait(timeout=10)
        except Exception:
            pass
        self._kill()
