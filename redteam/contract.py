import sys, json
sys.path.insert(0, "/home/claude/auditgym")
from auditgym.env import AuditEnv
calc = {"tool": "calculator", "args": {"expression": "1+1"}}
def mk(mode="train", clock=None, **kw):
    env = AuditEnv(mode=mode, clock=clock, **kw) if clock else AuditEnv(mode=mode, **kw)
    obs, info = env.reset(11, {})
    return env, obs, info
# 99 explore + submit at 100
env, obs, info = mk()
for i in range(99):
    o, r, te, tr, inf = env.step(calc)
    assert not (te or tr), i
o, r, te, tr, inf = env.step({"tool": "submit", "args": {"findings": {"episode_id": info["episode_id"], "findings": []}}})
print("submit@100:", te, tr, r, inf["grading"]["inspection_steps"], o["steps_remaining"])
print("after done:", env.step(calc))
# 100 explores
env, obs, info = mk()
for i in range(100):
    o, r, te, tr, inf = env.step("garbage")
print("100 invalid:", te, tr, r, inf, o)
# time limit
t = [0.0]
env, obs, info = mk(clock=lambda: t[0])
env.step(calc); t[0] = 61
o, r, te, tr, inf = env.step({"tool": "submit", "args": {"findings": {"episode_id": info["episode_id"], "findings": []}}})
print("late submit:", te, tr, r, inf, o)
# eval mode
env, obs, info = mk(mode="eval")
o, r, te, tr, inf = env.step({"tool": "submit", "args": {"findings": {"episode_id": info["episode_id"], "findings": []}}})
print("eval submit:", r, te, tr, inf, o, "| reset info:", info)
# eval invalid submit
env, obs, info = mk(mode="eval")
print("eval invalid:", env.step({"tool": "submit", "args": {"findings": {"episode_id": "nope", "findings": []}}}))
print("reset obs keys:", list(obs), "manifest keys:", list(obs["manifest"]))
