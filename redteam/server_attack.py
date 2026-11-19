"""Raw-protocol attacks on `python -m auditgym serve` (agent speaks the protocol directly)."""
import json, subprocess, sys, tempfile, os
log = tempfile.mktemp(suffix=".jsonl")
p = subprocess.Popen([sys.executable, "-m", "auditgym", "serve", "--mode", "eval", "--grader-log", log],
                     cwd="/home/claude/auditgym", stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def call(req):
    p.stdin.write((req if isinstance(req, str) else json.dumps(req)) + "\n"); p.stdin.flush()
    return json.loads(p.stdout.readline())
def nlog():
    return sum(1 for _ in open(log)) if os.path.exists(log) else 0
r = call({"op": "reset", "seed": 5, "config": {"errors": {"duplicate_invoice": 1}}}); eid = r["info"]["episode_id"]
bad = {"tool": "submit", "args": {"findings": {"episode_id": eid, "findings": [{"type": "duplicate_invoice", "affected_ids": ["X"], "claimed_amount_minor": 1, "currency": "USD", "evidence_ids": [], "explanation": ""}]}}}
# 1. suppress grading of the real submit
print("submit with _after_done=true:", call({"op": "step", "action": bad, "_after_done": True})["terminated"], "log lines:", nlog())
# 2. re-reset same seed, submit again (second attempt), then spam steps after done -> duplicates
call({"op": "reset", "seed": 5, "config": {"errors": {"duplicate_invoice": 1}}})
call({"op": "step", "action": {"tool": "submit", "args": {"findings": {"episode_id": eid, "findings": []}}}})
for _ in range(3):
    call({"op": "step", "action": {"tool": "calculator", "args": {"expression": "1"}}})
print("after 1 graded submit + 3 post-done steps, log lines:", nlog())
# 3. mid-episode reset leaves no record
call({"op": "reset", "seed": 6, "config": {}})
call({"op": "step", "action": {"tool": "calculator", "args": {"expression": "1"}}})
call({"op": "reset", "seed": 6, "config": {}})
print("abandoned episode, log lines:", nlog())
# 4. odd requests
for req in ['{"op":"step"}', '[]', '{"op":"reset","seed":"7"}', '{"op":"reset","seed":7.9}', '{"op":"reset","seed":true}',
            '{"op":"reset","seed":1,"config":{"id_salt":"x","bogus":1}}', '{"op":"reset","seed":1,"config":{"errors":{"duplicate_invoice":1e9}}}']:  # NOTE: config n_invoices=[100000,100000] hangs the server (no size bounds in resolve_config)
    try:
        r = call(req); print(req[:60], "->", json.dumps(r)[:120])
    except Exception as e:
        print(req[:60], "-> client error", type(e).__name__)
p.stdin.close(); p.wait(timeout=60)
print(open(log).read()[:300])
