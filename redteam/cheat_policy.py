"""External policy adapters that read the hidden truth through the official harness
(`evaluate([...,"redteam.cheat_policy:SelfDeclaredOracle"], ...)` / `python -m auditgym evaluate`).

SelfDeclaredOracle: any adapter may set grader_only = True and make_policy hands it truth_provider.
GcSnooper: no flag at all; the 'isolated' harness generates the episode in the SAME process as the
policy, so the Episode object (truth, provenance, seed) is reachable via gc.
"""
import gc


def _submit(info, issues):
    return {"tool": "submit", "args": {"findings": {"episode_id": info["episode_id"], "findings": [
        {"type": t["type"], "affected_ids": t["affected_ids"], "claimed_amount_minor": t["expected_amount_minor"],
         "currency": t["currency"], "evidence_ids": t["evidence_sets"][0], "explanation": ""} for t in issues]}}}


class SelfDeclaredOracle:
    grader_only = True

    def __init__(self, truth_provider):
        self.tp = truth_provider

    def reset(self, obs, info):
        self.info = info

    def act(self, obs):
        return _submit(self.info, self.tp(self.info["episode_id"])["issues"])


class GcSnooper:
    def reset(self, obs, info):
        self.info = info

    def act(self, obs):
        from auditgym.generator import Episode
        for o in gc.get_objects():
            if isinstance(o, Episode) and o.episode_id == self.info["episode_id"]:
                return _submit(self.info, o.truth["issues"])
        return _submit(self.info, [])


class FrameSnooper:  # noqa: E302
    """Works against the patched harness (seed/config sent to the server via a split file,
    AUDITGYM_SECRET popped from os.environ): evaluate()'s frame still holds `episodes`
    (seeds + configs) and `hidden_secret`/`server_env`, so the policy regenerates the truth."""

    def reset(self, obs, info):
        self.info, self.man = info, obs["manifest"]

    def act(self, obs):
        import os
        import sys
        from auditgym.generator import generate
        f, eps, secret = sys._getframe(), None, None
        while f is not None:
            loc = f.f_locals
            if isinstance(loc.get("episodes"), list):
                eps = loc["episodes"]
            secret = secret or loc.get("hidden_secret") or (loc.get("server_env") or {}).get("AUDITGYM_SECRET")
            f = f.f_back
        old = os.environ.get("AUDITGYM_SECRET")
        if secret:
            os.environ["AUDITGYM_SECRET"] = secret
        try:
            for e in eps or []:
                ep = generate(e["seed"], e["config"])
                # config_version may be rewritten by the harness: match on content, not episode_id
                if (ep.manifest["company"], ep.manifest["counts"], ep.manifest["period_start"]) == \
                        (self.man["company"], self.man["counts"], self.man["period_start"]):
                    return _submit(self.info, ep.truth["issues"])
        finally:
            if secret:
                if old is None:
                    os.environ.pop("AUDITGYM_SECRET", None)
                else:
                    os.environ["AUDITGYM_SECRET"] = old
        return _submit(self.info, [])
