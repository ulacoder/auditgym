"""Hash of trajectories (scripted_auditor + random actions) over several episodes; run under various PYTHONHASHSEED."""
import sys, json, hashlib, random
sys.path.insert(0, "/home/claude/auditgym")
from auditgym.env import AuditEnv
from auditgym.harness import load_split, run_episode
from auditgym.policies import make_policy
from auditgym.config import canonical_json
h = hashlib.sha256()
eps = load_split("/home/claude/auditgym/configs/holdout_example.json")[:8]
for e in eps:
    for pname in ("scripted_auditor", "random_explorer", "naive_reconciler"):
        buf = []
        env = AuditEnv(mode="train", recorder=buf.append, time_limit_s=None)
        run_episode(env, make_policy(pname, lambda eid: None), e["seed"], e["config"])
        h.update(canonical_json(buf).encode())
    for salt in ("", "zz"):
        env = AuditEnv(mode="train", time_limit_s=None)
        obs, info = env.reset(e["seed"], {**e["config"], "id_salt": salt, "order_seed": 5})
        h.update(canonical_json([obs, info, env._ep.truth, env._ep.provenance, env._ep.tables]).encode())
print(h.hexdigest())
