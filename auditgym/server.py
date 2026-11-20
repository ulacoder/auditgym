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
import os
import subprocess
import sys
import threading

from ._secure import harden
from .env import AuditEnv


def _load_episodes(path):
    from .harness import load_split
    return load_split(path)


def _load_episodes_fd(fd: int):
    from .harness import split_from_dict
    with os.fdopen(fd, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    secret = data.pop("grader_secret", None)
    if secret:
        # секрет приходит по каналу, а не через окружение: его нет в /proc/<pid>/environ даже до harden()
        os.environ["AUDITGYM_SECRET"] = secret
    return split_from_dict(data)


def serve(mode: str = "eval", grader_log: str | None = None, max_steps: int = 100,
          time_limit_s: float | None = 60.0, stdin=None, stdout=None, episodes: str | None = None,
          allow_oracle: bool = False, allow_public_secret: bool = False,
          episodes_fd: int | None = None, grader_fd: int | None = None) -> None:
    harden()   # секрет и каналы оценщика недоступны процессам политики через /proc
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    env = AuditEnv(mode=mode, max_steps=max_steps, time_limit_s=time_limit_s)
    if episodes_fd is not None:
        hidden = _load_episodes_fd(episodes_fd)
    else:
        hidden = _load_episodes(episodes) if episodes else None
    if hidden is not None:
        from .generator import has_private_secret
        if not has_private_secret() and not allow_public_secret:
            sys.stderr.write("auditgym serve: AUDITGYM_SECRET is not set (or equals the public default). A closed run (--episodes) needs a grader "
                             "secret, otherwise small seeds can be brute-forced from public IDs. Set it, or pass "
                             "--allow-public-secret for open splits.\n")
            raise SystemExit(2)
    if grader_fd is not None:
        glog = os.fdopen(grader_fd, "w", encoding="utf-8")
    elif grader_log:
        glog = open(grader_log, "a", encoding="utf-8")
    else:
        glog = None
    state = {"episode_id": None, "logged": False}
    started: set[int] = set()

    def log_result():
        if glog is not None and not state["logged"] and env.grader_result() is not None:
            glog.write(json.dumps({"episode_id": state["episode_id"], "result": env.grader_result()},
                                  ensure_ascii=False) + "\n")
            glog.flush()
            state["logged"] = True

    def abandon_current():
        # незавершённый эпизод нельзя «переиграть»: он закрывается пустым ответом, шаги учитываются
        if state["episode_id"] is not None and env.abandon() is not None:
            log_result()

    try:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise TypeError("request must be an object")
                op = req.get("op")
                if op == "reset":
                    if hidden is not None:
                        if "seed" in req or "config" in req:
                            raise PermissionError("seed/config reset disabled: use index")
                        k = req["index"]
                        if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < len(hidden):
                            raise IndexError("bad index")
                        if k in started:
                            raise PermissionError("episode already started: each closed episode runs once")
                        abandon_current()
                        started.add(k)
                        e = hidden[k]
                        obs, info = env.reset(e["seed"], e["config"])
                    else:
                        seed = req.get("seed")
                        if not isinstance(seed, int) or isinstance(seed, bool):
                            raise TypeError("seed must be an integer")
                        abandon_current()
                        obs, info = env.reset(seed, req.get("config") or {})
                    state["episode_id"], state["logged"] = info["episode_id"], False
                    out = {"observation": obs, "info": info}
                elif op == "step":
                    obs, r, te, tr, info = env.step(req.get("action"))
                    out = {"observation": obs, "reward": r, "terminated": te, "truncated": tr, "info": info}
                    if te or tr:
                        log_result()
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
    finally:
        abandon_current()
        if glog is not None:
            glog.close()


class RemoteEnv:
    """Клиент к `python -m auditgym serve`: тот же контракт reset/step.

    episodes_data — закрытый список эпизодов: передаётся серверу через анонимный канал, а не файлом,
    поэтому seed/config не видны ни в файловой системе, ни в командной строке процесса.
    collect_results=True — итоги оценщика идут по отдельному каналу прямо в харнесс (results()),
    без файла, в который мог бы писать процесс политики.
    """

    def __init__(self, mode="eval", grader_log=None, max_steps=100, time_limit_s=60.0, episodes=None,
                 allow_oracle=False, env=None, allow_public_secret=False, episodes_data=None,
                 collect_results=False):
        cmd = [sys.executable, "-E", "-s", "-m", "auditgym", "serve", "--mode", mode, "--max-steps", str(max_steps)]
        pass_fds, close_after = [], []
        if grader_log:
            cmd += ["--grader-log", grader_log]
        self._results, self._reader = [], None
        if collect_results:
            gr, gw = os.pipe()
            cmd += ["--grader-fd", str(gw)]
            pass_fds.append(gw)
            close_after.append(gw)
            self._gr = gr
        ep_w = None
        if episodes_data is not None:
            er, ew = os.pipe()
            cmd += ["--episodes-fd", str(er)]
            pass_fds.append(er)
            close_after.append(er)
            ep_w = ew
        elif episodes:
            cmd += ["--episodes", str(episodes)]
        if allow_oracle:
            cmd += ["--allow-oracle"]
        if allow_public_secret:
            cmd += ["--allow-public-secret"]
        if time_limit_s is None:
            cmd += ["--no-time-limit"]
        else:
            cmd += ["--time-limit", str(time_limit_s)]
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        if episodes_data is not None:
            env = {k: v for k, v in (env if env is not None else os.environ).items() if k != "AUDITGYM_SECRET"}
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env,
                                  pass_fds=pass_fds, cwd=str(root))
        for fd in close_after:
            os.close(fd)
        if ep_w is not None:
            payload = dict(episodes_data)
            if os.environ.get("AUDITGYM_SECRET"):
                payload["grader_secret"] = os.environ["AUDITGYM_SECRET"]
            with os.fdopen(ep_w, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        if collect_results:
            def drain():
                with os.fdopen(self._gr, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            self._results.append(json.loads(line))
            self._reader = threading.Thread(target=drain, daemon=True)
            self._reader.start()

    def _call(self, req):
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("grader process exited")
        r = json.loads(line)
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

    def results(self) -> list[dict]:
        """Итоги оценщика (после close())."""
        return list(self._results)

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"op": "close"}) + "\n")
            self.p.stdin.flush()
        except Exception:
            pass
        try:
            self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        if self._reader is not None:
            self._reader.join(timeout=30)
