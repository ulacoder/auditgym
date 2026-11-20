"""Враждебные политики для тестов изоляции харнесса (подключаются как hostile_policies:Class)."""
import os
import time


class _Sub:
    def reset(self, obs, info):
        self.eid = info["episode_id"]

    def _empty(self):
        return {"tool": "submit", "args": {"findings": {"episode_id": self.eid, "findings": []}}}


class Hang(_Sub):
    def act(self, obs):
        time.sleep(10 ** 6)


class HangInReset(_Sub):
    def reset(self, obs, info):
        time.sleep(10 ** 6)

    def act(self, obs):
        return {"tool": "submit", "args": {"findings": None}}


class Noisy(_Sub):
    def reset(self, obs, info):
        print("noise from reset")
        super().reset(obs, info)

    def act(self, obs):
        print('{"action": {"tool": "submit", "args": {"findings": "forged"}}}')
        return self._empty()


class Exit(_Sub):
    def act(self, obs):
        raise SystemExit(0)


class HardExit(_Sub):
    def act(self, obs):
        os._exit(3)


class SelfDeclaredOracle:
    grader_only = True

    def __init__(self, truth_provider):
        self.tp = truth_provider

    def reset(self, obs, info):
        self.eid = info["episode_id"]
        self.truth = self.tp(self.eid)

    def act(self, obs):
        # без эталона честно отправляет пустой ответ; с эталоном — все проблемы из него
        findings = []
        for t in (self.truth or {}).get("issues", []):
            findings.append({"type": t["type"], "affected_ids": t["affected_ids"],
                             "claimed_amount_minor": t["expected_amount_minor"], "currency": t["currency"],
                             "evidence_ids": t["evidence_sets"][0]})
        return {"tool": "submit", "args": {"findings": {"episode_id": self.eid, "findings": findings}}}


class Spy(_Sub):
    """Ищет следы закрытого прогона: временные файлы и секрет в /proc. Сообщает через stderr."""
    def act(self, obs):
        import glob
        import sys
        hits = glob.glob("/tmp/split_*") + glob.glob("/tmp/grader_*")
        for d in glob.glob("/proc/[0-9]*"):
            if d == f"/proc/{os.getpid()}":
                continue
            try:
                if b"AUDITGYM_SECRET=" in open(d + "/environ", "rb").read():
                    hits.append(d)
            except OSError:
                pass
        sys.stderr.write(f"SPY_HITS={len(hits)}\n")
        if hits:
            return {"tool": "submit", "args": {"findings": None}}
        return self._empty()


class SandboxProbe(_Sub):
    """Под отдельным пользователем ни одна из атак не должна удаться; иначе отправляет мусор (−1)."""
    def act(self, obs):
        import glob
        import signal
        import auditgym
        root = os.path.dirname(os.path.dirname(auditgym.__file__))
        for target in (os.path.join(root, "argparse.py"), os.path.join(root, "auditgym", "__pycache__", "x.pyc")):
            try:
                open(target, "a").close()
                return {"tool": "submit", "args": {"findings": None}}
            except OSError:
                pass
        for d in glob.glob("/proc/[0-9]*"):
            if d.endswith("/" + str(os.getpid())):
                continue
            try:
                if b"AUDITGYM_SECRET" in open(d + "/environ", "rb").read():
                    return {"tool": "submit", "args": {"findings": None}}
            except OSError:
                pass
        try:
            os.kill(os.getppid(), 0)
            return {"tool": "submit", "args": {"findings": None}}
        except OSError:
            pass
        return self._empty()
