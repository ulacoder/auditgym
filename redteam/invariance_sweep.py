"""Random seeds/configs: evidence ids exist, oracle=1.0, scripted reward invariant under id_salt/order_seed."""
import sys, random
sys.path.insert(0, "/home/claude/auditgym")
from auditgym.env import AuditEnv
from auditgym.generator import generate
from auditgym.harness import run_episode
from auditgym.policies import make_policy
from auditgym.config import ERROR_TYPES, INTERACTIONS
R = random.Random(1)
bad = []
for k in range(40):
    seed = R.randint(0, 10**9)
    cfg = {"errors": {t: R.randint(0, 2) for t in ERROR_TYPES}, "interactions": {i: R.randint(0, 1) for i in INTERACTIONS}}
    rew = []
    for tr in ({}, {"id_salt": f"s{k}", "order_seed": k}):
        c = {**cfg, **tr}
        ep = generate(seed, c)
        ids = {r["row_id"] for r in ep.tables["ledger"]} | {r["doc_id"] for t in ("invoices", "payments") for r in ep.tables[t]} \
            | {r["invoice_id"] for r in ep.tables["invoices"]} | {r["payment_id"] for r in ep.tables["payments"]}
        for t in ep.truth["issues"]:
            if not set(t["affected_ids"]) <= ids or not all(set(s) <= ids for s in t["evidence_sets"]):
                bad.append(("missing id", seed, cfg, t["issue_id"]))
        for p in ("oracle", "scripted_auditor"):
            _, _, ret, _, _ = run_episode(AuditEnv(mode="train", time_limit_s=None), make_policy(p, {ep.episode_id: ep.truth}.__getitem__), seed, c)
            rew.append(ret)
    if rew[0] != 1.0 or rew[2] != 1.0 or rew[1] != rew[3] or rew[1] < 0.9:
        bad.append(("reward", seed, cfg, rew))
print("episodes", 40, "problems", len(bad))
for b in bad[:10]:
    print(" ", b)
