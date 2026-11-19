import os, sys
sys.path.insert(0, "/home/claude/auditgym")
from auditgym.harness import evaluate, load_split
eps = load_split("/home/claude/auditgym/configs/holdout_example.json")[5:9]
for secret in (None, "jury-secret-123"):
    if secret:
        os.environ["AUDITGYM_SECRET"] = secret
    res = evaluate(["redteam.cheat_policy:SelfDeclaredOracle", "redteam.cheat_policy:GcSnooper",
                    "redteam.cheat_policy:FrameSnooper", "flag_all"], eps, isolated=True)
    for k, v in res.items():
        s = v["summary"]
        print(f"secret={secret!s:16} {k}: mean_f1={s['mean_f1']:.3f} mean_return={s['mean_return']:+.3f} tp={s['tp']}")
