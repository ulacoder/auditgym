import sys, json, traceback
sys.path.insert(0, "/home/claude/auditgym")
from auditgym.env import AuditEnv
from auditgym import calculator
x5 = "((((.1**12)**12)**12)**12)**12"
e = x5 + "*" + x5 + "*" + x5 + "*" + x5
print("expr len", len(e))
try:
    print("calc big len", len(calculator.evaluate(e)))
except Exception as ex:
    print("calc", type(ex).__name__, ex)

def run(name, action, recorder=None, mode="train"):
    env = AuditEnv(mode=mode, recorder=recorder)
    obs, info = env.reset(7, {"errors": {"duplicate_invoice": 1}})
    try:
        o, r, te, tr, inf = env.step(action)
        print(name, "->", r, te, tr, json.dumps(o)[:150], inf)
    except BaseException as ex:
        print(name, "UNCAUGHT", type(ex).__name__, str(ex)[:100])
    return env

deep = "[" * 100000 + "]" * 100000
run("deep action str", deep)
run("deep action str w/ recorder", deep, recorder=lambda d: None)
run("deep findings str", {"tool": "submit", "args": {"findings": deep}})
run("deep findings dict", {"tool": "submit", "args": {"findings": json.loads("[" * 500 + "]" * 500)}})
run("huge int json", '{"tool":"calculator","args":{"expression":"1"},"x":' + "1"*5000 + '}')
run("huge int findings", {"tool": "submit", "args": {"findings": '{"a":' + "1"*5000 + '}'}})
run("nan", '{"tool":"list_records","args":{"table":"ledger","limit":NaN}}')
run("limit 50.0", {"tool": "list_records", "args": {"table": "ledger", "limit": 50.0}})
run("limit True", {"tool": "list_records", "args": {"table": "ledger", "limit": True}})
run("cursor unicode", {"tool": "list_records", "args": {"table": "ledger", "cursor": "ledger@١٢"}})
run("cursor int", {"tool": "list_records", "args": {"table": "ledger", "cursor": 5}})
run("cursor neg", {"tool": "list_records", "args": {"table": "ledger", "cursor": "ledger@-0"}})
run("table list", {"tool": "list_records", "args": {"table": ["ledger"]}})
run("table dict", {"tool": "list_records", "args": {"table": {"a": 1}}})
run("id dict", {"tool": "get_record", "args": {"table": "ledger", "id": {"a": 1}}})
run("tool list", {"tool": ["submit"]})
run("args list", {"tool": "calculator", "args": [1]})
run("bytes", b'{"tool":"calculator"}')
run("None", None)
run("int", 5)
run("surrogate", '{"tool":"calculator","args":{"expression":"\\ud800"}}')
class Weird(dict):
    def __iter__(self): raise RuntimeError("boom")
run("weird dict", Weird(tool="calculator"))
run("findings with unhashable", {"tool": "submit", "args": {"findings": {"episode_id": [1], "findings": []}}})
run("findings with set", {"tool": "submit", "args": {"findings": {"episode_id": "x", "findings": [], 1: 2}}})
