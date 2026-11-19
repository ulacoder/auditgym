#!/usr/bin/env python3
"""Mutation self-test for verify.py: copies episodes into a temp tree, injects one
defect per copy, and checks verify.py reports FAIL for each (and PASS for the
untouched copy).  Never writes inside the repo.  Usage: python3 independent_check/selftest_mutations.py
"""
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VERIFY = os.path.join(HERE, "verify.py")


def rw_ledger(path, fn):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    fn(rows)
    with open(path, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def rw_jsonl(path, fn):
    with open(path) as f:
        objs = [json.loads(l) for l in f if l.strip()]
    fn(objs)
    with open(path, "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in objs)


def rw_json(path, fn):
    with open(path) as f:
        o = json.load(f)
    fn(o)
    with open(path, "w") as f:
        json.dump(o, f)


def first_invoice_rows(rows, base_only=False):
    hdr = rows[0]
    idx = [i for i, r in enumerate(rows[1:], 1) if r[7] and not r[8]]
    return idx


def m_fx_undeclared(pub, hid):  # shift a posted invoice amount on both AR and revenue
    def fn(rows):
        idx = first_invoice_rows(rows)
        inv = rows[idx[0]][7]
        for i in idx:
            if rows[i][7] == inv:
                for c in (3, 4):
                    if rows[i][c] != "0":
                        rows[i][c] = str(int(rows[i][c]) + 777)
    rw_ledger(os.path.join(pub, "ledger.csv"), fn)


def m_unbalanced(pub, hid):
    def fn(rows):
        i = first_invoice_rows(rows)[0]
        c = 3 if rows[i][3] != "0" else 4
        rows[i][c] = str(int(rows[i][c]) + 1)
    rw_ledger(os.path.join(pub, "ledger.csv"), fn)


def m_two_sided(pub, hid):
    def fn(rows):
        rows[1][3], rows[1][4] = "5", "5"
    rw_ledger(os.path.join(pub, "ledger.csv"), fn)


def m_null_string(pub, hid):
    def fn(rows):
        for r in rows[1:]:
            if not r[8]:
                r[8] = "null"
                break
    rw_ledger(os.path.join(pub, "ledger.csv"), fn)


def m_line_sum(pub, hid):
    def fn(objs):
        objs[0]["line_items"][0]["amount_minor"] += 1
        objs[0]["line_items"][0]["unit_price_minor"] = objs[0]["line_items"][0]["amount_minor"]
        objs[0]["line_items"][0]["quantity"] = 1
    rw_jsonl(os.path.join(pub, "invoices.jsonl"), fn)


def m_truth_amount(pub, hid):
    rw_json(os.path.join(hid, "truth.json"), lambda t: t["issues"][0].__setitem__(
        "expected_amount_minor", t["issues"][0]["expected_amount_minor"] + 5))


def m_truth_drop(pub, hid):
    rw_json(os.path.join(hid, "truth.json"), lambda t: t["issues"].pop(0))


def m_truth_bad_evidence(pub, hid):
    def fn(t):
        for s in t["issues"][0]["evidence_sets"]:
            s[:] = s[:1]
    rw_json(os.path.join(hid, "truth.json"), fn)


def m_truth_ghost_id(pub, hid):
    def fn(t):
        t["issues"][0]["evidence_sets"] = [["D-DEADBEEF"]]
    rw_json(os.path.join(hid, "truth.json"), fn)


def m_payment_ref_sum(pub, hid):
    def fn(objs):
        for p in objs:
            if p["invoice_refs"]:
                p["invoice_refs"][0]["amount_minor"] += 1
                break
    rw_jsonl(os.path.join(pub, "payments.jsonl"), fn)


def m_due_before_issue(pub, hid):
    def fn(objs):
        objs[0]["due_at"] = "2000-01-01T00:00:00+00:00"
    rw_jsonl(os.path.join(pub, "invoices.jsonl"), fn)


def m_no_offset(pub, hid):
    def fn(objs):
        objs[0]["issued_at"] = objs[0]["issued_at"][:19]
    rw_jsonl(os.path.join(pub, "invoices.jsonl"), fn)


def m_posted_duplicate(pub, hid):
    """Add an exact copy of a posted invoice (later) and post it: undeclared duplicate."""
    with open(os.path.join(pub, "invoices.jsonl")) as f:
        objs = [json.loads(l) for l in f]
    with open(os.path.join(pub, "ledger.csv"), newline="") as f:
        rows = list(csv.reader(f))
    src = objs[0]
    posted = [r for r in rows[1:] if r[7] == src["invoice_id"] and not r[8]]
    cp = dict(src, doc_id="D-ZZZZ0001", invoice_id="INV-ZZZZ0001")  # same instant -> same rate
    objs.append(cp)
    for k, r in enumerate(posted):
        rows.append([f"L-ZZZZ000{k}"] + r[1:7] + ["INV-ZZZZ0001", ""])
    with open(os.path.join(pub, "invoices.jsonl"), "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in objs)
    with open(os.path.join(pub, "ledger.csv"), "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    def fx_counts(m):
        m["counts"]["invoices"] += 1
        m["counts"]["documents"] += 1
        m["counts"]["ledger_rows"] += len(posted)
    rw_json(os.path.join(pub, "manifest.json"), fx_counts)


def _issue(t, typ):
    return next(i for i in t["issues"] if i["type"] == typ)


def m_fx_key_dropped(pub, hid):
    def fn(t):
        iss = _issue(t, "fx_conversion")
        iss["evidence_sets"] = [[e for e in s if e.count(":") != 2] for s in iss["evidence_sets"]]
    rw_json(os.path.join(hid, "truth.json"), fn)


def m_fx_key_wrong_date(pub, hid):
    """Replace the policy fx key by another existing rate row of the same pair."""
    with open(os.path.join(pub, "fx.csv")) as f:
        fx = list(csv.reader(f))[1:]
    def fn(t):
        iss = _issue(t, "fx_conversion")
        for s in iss["evidence_sets"]:
            for k, e in enumerate(s):
                if e.count(":") == 2:
                    d, src, tgt = e.split(":")
                    other = next(r for r in fx if r[1] == src and r[2] == tgt and r[0] != d)
                    s[k] = f"{other[0]}:{src}:{tgt}"
    rw_json(os.path.join(hid, "truth.json"), fn)


def m_dup_row_dropped(pub, hid):
    def fn(t):
        iss = _issue(t, "duplicate_invoice")
        iss["evidence_sets"] = [[e for e in s if not e.startswith("L-")] for s in iss["evidence_sets"]]
    rw_json(os.path.join(hid, "truth.json"), fn)


def m_alloc_invoice_doc_dropped(pub, hid):
    with open(os.path.join(pub, "invoices.jsonl")) as f:
        inv_docs = {json.loads(l)["doc_id"] for l in f}
    def fn(t):
        iss = _issue(t, "payment_allocation")
        iss["evidence_sets"] = [[e for e in s if e not in inv_docs] for s in iss["evidence_sets"]]
    rw_json(os.path.join(hid, "truth.json"), fn)


def m_ledger_before_unbalanced(pub, hid):
    def fn(rows):
        i = first_invoice_rows(rows)[0]
        c = 3 if rows[i][3] != "0" else 4
        rows[i][c] = str(int(rows[i][c]) + 1)
    rw_ledger(os.path.join(hid, "ledger_before.csv"), fn)


MUTANTS = {
    "fx_undeclared": ("train_00_clean", m_fx_undeclared),
    "unbalanced_row": ("train_00_clean", m_unbalanced),
    "two_sided_row": ("train_00_clean", m_two_sided),
    "null_string_in_csv": ("train_00_clean", m_null_string),
    "line_items_sum": ("train_00_clean", m_line_sum),
    "payment_refs_sum": ("train_00_clean", m_payment_ref_sum),
    "due_before_issue": ("train_00_clean", m_due_before_issue),
    "timestamp_without_offset": ("train_00_clean", m_no_offset),
    "posted_duplicate": ("train_00_clean", m_posted_duplicate),
    "truth_amount_off": ("train_04_all_types", m_truth_amount),
    "truth_issue_dropped": ("train_04_all_types", m_truth_drop),
    "truth_evidence_truncated": ("train_04_all_types", m_truth_bad_evidence),
    "truth_ghost_evidence": ("train_04_all_types", m_truth_ghost_id),
    "fx_key_dropped": ("train_04_all_types", m_fx_key_dropped),
    "fx_key_wrong_date": ("train_04_all_types", m_fx_key_wrong_date),
    "dup_ledger_row_dropped": ("train_04_all_types", m_dup_row_dropped),
    "alloc_invoice_doc_dropped": ("train_04_all_types", m_alloc_invoice_doc_dropped),
    "ledger_before_unbalanced": ("train_04_all_types", m_ledger_before_unbalanced),
}


def main():
    tmp = tempfile.mkdtemp(prefix="auditgym_mut_")
    ok = True
    try:
        names = ["untouched"] + list(MUTANTS)
        for name in names:
            ep = "train_04_all_types" if name == "untouched" else MUTANTS[name][0]
            mname = f"mut_{name}"
            pub = os.path.join(tmp, "episodes", "train", mname)
            hid = os.path.join(tmp, "grader_data", "train", mname)
            shutil.copytree(os.path.join(REPO, "episodes", "train", ep), pub)
            shutil.copytree(os.path.join(REPO, "grader_data", "train", ep), hid)
            if name != "untouched":
                MUTANTS[name][1](pub, hid)
        out = subprocess.run([sys.executable, VERIFY, "--root", tmp], capture_output=True, text=True).stdout
        for name in names:
            line = next((l for l in out.splitlines() if l.startswith(f"train/mut_{name} ")), "")
            res = line.split()[-1] if line else "MISSING"
            want = "PASS" if name == "untouched" else "FAIL"
            good = res == want
            ok &= good
            print(f"{'ok ' if good else 'BAD'} {name:<28} expected {want} got {res}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
