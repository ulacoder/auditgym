"""Random-JSON fuzz of env.step + paired invariance sweep."""
import sys, json, random, traceback, string
sys.path.insert(0, "/home/claude/auditgym")
from auditgym.env import AuditEnv, TOOLS
from auditgym.config import ERROR_TYPES, INTERACTIONS
R = random.Random(0)
def rval(d=0):
    k = R.random()
    if d > 3 or k < 0.3:
        return R.choice([None, True, False, 0, -1, 50, 51, 10**20, 1.5, float("nan"), "", "ledger", "ledger@0",
                         "ledger@50", "fx@1", "1+2", "2**12", "INV-", "D-0", "x" * 500, "\u0000", "١٢", [], {}])
    if k < 0.6:
        return [rval(d + 1) for _ in range(R.randint(0, 4))]
    keys = ["table", "cursor", "limit", "id", "doc_id", "expression", "findings", "episode_id", "type",
            "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation", "x"]
    return {R.choice(keys): rval(d + 1) for _ in range(R.randint(0, 5))}
def raction(env):
    tool = R.choice(["list_records", "get_record", "get_document", "calculator", "nope", None] * 6 + ["submit"])
    a = {"tool": tool, "args": rval(1) if R.random() < 0.3 else {k: rval(2) for k in R.sample(["table", "cursor", "limit", "id", "doc_id", "expression", "findings"], R.randint(0, 3))}}
    if R.random() < 0.1:
        a["extra"] = 1
    if tool == "submit" and R.random() < 0.5:
        a["args"] = {"findings": {"episode_id": env._ep.episode_id if R.random() < 0.8 else "x",
                                  "findings": [{"type": R.choice(ERROR_TYPES + ("x",)), "affected_ids": rval(2),
                                                "claimed_amount_minor": rval(2), "currency": rval(2),
                                                "evidence_ids": rval(2), "explanation": rval(2)} for _ in range(R.randint(0, 3))]}}
    if R.random() < 0.2:
        try:
            return json.dumps(a)
        except Exception:
            pass
    return a
crashes = {}
n = 0
env = AuditEnv(mode="train", recorder=lambda d: json.dumps(d))
for ep in range(120):
    env.reset(R.randint(0, 10**6), {"errors": {"duplicate_invoice": R.randint(0, 2)}})
    for _ in range(120):
        a = raction(env)
        n += 1
        try:
            o, r, te, tr, info = env.step(a)
            json.dumps([o, r, te, tr, info], allow_nan=False)
            assert isinstance(r, float) and -1 <= r <= 1
            if te or tr:
                break
        except Exception as e:
            key = type(e).__name__ + ":" + str(e)[:80]
            crashes.setdefault(key, repr(a)[:200])
print("actions", n, "distinct crashes", len(crashes))
for k, v in crashes.items():
    print(" ", k, "<-", v)
