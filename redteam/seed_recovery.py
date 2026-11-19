"""LEAK: the hidden seed (and then the full truth) is recoverable from the FIRST observation alone.

The 'setting' RNG stream depends only on the seed, and the manifest publishes its outputs
(company name, base currency, timezone, period, foreign currencies). Brute-forcing the seed is
cheap; with the seed, the open-source generator reproduces the episode. The error config is
recovered by matching manifest['counts']; order_seed / config_version do not affect content.
Result: a policy that never reads a single record submits the exact truth at step 1.

usage: python3 redteam/seed_recovery.py [max_seed]
"""
import itertools, json, sys, time
sys.path.insert(0, "/home/claude/auditgym")
from auditgym.env import AuditEnv
from auditgym.generator import (COMPANY_OFFSETS, NAMES_A, NAMES_B, generate, sub_rng)
from auditgym.money import BASE_CURRENCIES, CURRENCY_EXPONENT
from auditgym.harness import load_split


def setting_fingerprint(seed):
    R = sub_rng(seed, "setting")
    base = R.choice(BASE_CURRENCIES)
    tz = R.choice(COMPANY_OFFSETS)
    y = R.choice([2025, 2026])
    m = R.randint(1, 12 if y == 2025 else 10)   # mirrors generator._Gen.__init__ (current version)
    nf = R.randint(1, 3)
    foreign = sorted(R.sample([c for c in CURRENCY_EXPONENT if c != base], nf))
    name = f"{R.choice(NAMES_A)} {R.choice(NAMES_B)} LLP"
    return base, tz, f"{y}-{m:02d}", tuple(foreign), name


def observed_fingerprint(man):
    foreign = tuple(sorted(c for c in man["currency_exponents"] if c != man["base_currency"]))
    return man["base_currency"], man["timezone"], man["period_start"][:7], foreign, man["company"]


def recover_seed(man, max_seed):
    want = observed_fingerprint(man)
    return [s for s in range(max_seed) if setting_fingerprint(s) == want]


def recover_configs(seed, man):
    out = []
    types = ["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"]
    ixs = ["fx_invoice_with_partial_payment", "partial_fx_payment_at_boundary"]
    for ev in itertools.product(range(3), repeat=4):
        for iv in itertools.product(range(2), repeat=2):
            cfg = {"errors": dict(zip(types, ev)), "interactions": dict(zip(ixs, iv))}
            try:
                ep = generate(seed, cfg)
            except Exception:
                continue
            if ep.manifest["counts"] == man["counts"]:
                out.append((cfg, ep))
    return out


def read_all(env, table):
    rows, cur, steps = [], None, 0
    while True:
        o, *_ = env.step({"tool": "list_records", "args": {"table": table, "cursor": cur, "limit": 50}})
        steps += 1
        rows += o["result"]["records"]
        cur = o["result"]["next_cursor"]
        if cur is None:
            return rows, steps


def attack(env, obs, max_seed):
    """Returns (findings, steps_used, recovered_seed). Uses only observations."""
    man = obs["manifest"]
    seeds = recover_seed(man, max_seed)
    if len(seeds) != 1:
        return [], 0, seeds
    cands = recover_configs(seeds[0], man)
    if not cands:
        return [], 0, seeds
    invs, n1 = read_all(env, "invoices")
    sig = lambda i: (i["issued_at"], i["total_minor"], i["currency"], json.dumps(i["line_items"]))
    obs_sig = sorted(sig(i) for i in invs)
    cands = [(c, ep) for c, ep in cands if sorted(sig(i) for i in ep.tables["invoices"]) == obs_sig] or cands
    cfg, ep = cands[0]
    pays, n2 = read_all(env, "payments")
    led, n3 = read_all(env, "ledger")
    m = {}
    for a_tab, b_tab, key, f in (("invoices", invs, "invoice_id", sig),
                                 ("payments", pays, "payment_id", lambda p: (p["timestamp"], p["amount_minor"], p["currency"]))):
        ob = {f(r): r for r in b_tab}
        for r in ep.tables[a_tab]:
            m[r[key]] = ob[f(r)][key]
            m[r["doc_id"]] = ob[f(r)]["doc_id"]
    lsig = lambda r, mp: (r["timestamp"], r["account"], r["debit_minor"], r["credit_minor"],
                          mp(r["invoice_id"]), mp(r["payment_id"]))
    ob = {lsig(r, lambda x: x): r["row_id"] for r in led}
    for r in ep.tables["ledger"]:
        m[r["row_id"]] = ob[lsig(r, lambda x: m.get(x, x))]
    tr = lambda ids: [m.get(x, x) for x in ids]   # fx keys DATE:SRC:TGT are not salted
    findings = [{"type": t["type"], "affected_ids": tr(t["affected_ids"]), "claimed_amount_minor": t["expected_amount_minor"],
                 "currency": t["currency"], "evidence_ids": tr(t["evidence_sets"][0]), "explanation": ""}
                for t in ep.truth["issues"]]
    return findings, n1 + n2 + n3, seeds


if __name__ == "__main__":
    max_seed = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20000
    t = time.perf_counter()
    for s in range(20000):
        setting_fingerprint(s)
    print(f"setting fingerprint: {(time.perf_counter() - t) / 20000 * 1e6:.1f} us/seed")
    for e in load_split("/home/claude/auditgym/configs/holdout_example.json")[:: (1 if "--all" in sys.argv else 4)]:
        env = AuditEnv(mode="train")
        obs, info = env.reset(e["seed"], e["config"])        # the jury's hidden seed/config
        t0 = time.perf_counter()
        findings, steps, seeds = attack(env, obs, max_seed)  # attacker sees only observations
        _, r, te, _, inf = env.step({"tool": "submit", "args": {"findings": {"episode_id": info["episode_id"], "findings": findings}}})
        print(f"{e['name']}: true seed {e['seed']} recovered {seeds} in {time.perf_counter() - t0:.1f}s, {steps} read steps; "
              f"issues={len(findings)} reward = {r} tp={inf['grading']['tp']} fn={inf['grading']['fn']}")
