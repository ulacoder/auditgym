"""Детерминированная генерация файлов конфигураций сплитов (train/dev/holdout_example)."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Z = {"duplicate_invoice": 0, "payment_allocation": 0, "fx_conversion": 0, "period_cutoff": 0}
ZI = {"fx_invoice_with_partial_payment": 0, "partial_fx_payment_at_boundary": 0}


def E(**kw):
    return {**Z, **kw}


def I(**kw):
    return {**ZI, **kw}


def ep(name, seed, errors=None, interactions=None, **extra):
    return {"name": name, "seed": seed, "config": {"errors": errors or E(), "interactions": interactions or I(), **extra}}


train = [
    ep("train_00_clean", 1001),
    ep("train_01_clean_shuffled", 1002, order_seed=11),
    ep("train_02_dup_alloc", 1003, E(duplicate_invoice=1, payment_allocation=1)),
    ep("train_03_fx_cutoff", 1004, E(fx_conversion=2, period_cutoff=1)),
    ep("train_04_all_types", 1005, E(duplicate_invoice=1, payment_allocation=1, fx_conversion=1, period_cutoff=1)),
    ep("train_05_all_types_x2", 1006, E(duplicate_invoice=2, payment_allocation=2, fx_conversion=2, period_cutoff=2)),
    ep("train_06_interaction_fx_partial", 1007, E(duplicate_invoice=1), I(fx_invoice_with_partial_payment=1)),
    ep("train_07_interaction_boundary", 1008, E(period_cutoff=1), I(partial_fx_payment_at_boundary=1)),
    ep("train_08_both_interactions", 1009, E(duplicate_invoice=1, payment_allocation=1, fx_conversion=1, period_cutoff=1),
       I(fx_invoice_with_partial_payment=1, partial_fx_payment_at_boundary=1)),
    ep("train_09_large_ledger", 1010, E(duplicate_invoice=1, payment_allocation=2, fx_conversion=1, period_cutoff=1),
       n_expenses=[700, 760], order_seed=19),
]
dev = [
    ep("dev_00_clean", 2001, id_salt="dev"),
    ep("dev_01_clean_eur", 2002, base_currency="EUR", timezone="-05:00"),
    ep("dev_02_dup_only", 2003, E(duplicate_invoice=2)),
    ep("dev_03_alloc_only", 2004, E(payment_allocation=3)),
    ep("dev_04_fx_only", 2005, E(fx_conversion=3)),
    ep("dev_05_cutoff_only", 2006, E(period_cutoff=2), timezone="+09:00"),
    ep("dev_06_mixed", 2007, E(duplicate_invoice=1, payment_allocation=2, fx_conversion=1, period_cutoff=1), order_seed=5),
    ep("dev_07_interaction_fx_partial", 2008, E(fx_conversion=1), I(fx_invoice_with_partial_payment=2)),
    ep("dev_08_interaction_boundary", 2009, E(payment_allocation=1), I(partial_fx_payment_at_boundary=2)),
    ep("dev_09_max_issues", 2010, E(duplicate_invoice=2, payment_allocation=2, fx_conversion=2, period_cutoff=2),
       I(fx_invoice_with_partial_payment=1, partial_fx_payment_at_boundary=1), id_salt="dev9"),
]
R = random.Random(9000)
hold = []
for k in range(5):
    hold.append(ep(f"hold_{k:02d}_clean", 9001 + k, order_seed=R.randint(0, 999)))
for k in range(5, 10):
    ix = I(fx_invoice_with_partial_payment=R.randint(0, 1))
    if not ix["fx_invoice_with_partial_payment"]:
        ix["partial_fx_payment_at_boundary"] = 1
    else:
        ix["partial_fx_payment_at_boundary"] = R.randint(0, 1)
    hold.append(ep(f"hold_{k:02d}_interaction", 9001 + k,
                   E(**{t: R.randint(0, 2) for t in Z}), ix, order_seed=R.randint(0, 999)))
for k in range(10, 20):
    errs = E(**{t: R.randint(0, 2) for t in Z})
    if not any(errs.values()):
        errs["fx_conversion"] = 1
    hold.append(ep(f"hold_{k:02d}_mixed", 9001 + k, errs, id_salt=f"h{k}"))

for name, eps, ver in (("train", train, "train-v1"), ("dev", dev, "dev-v1"), ("holdout_example", hold, "holdout-example-v1")):
    (ROOT / "configs" / f"{name}.json").write_text(json.dumps(
        {"config_version": ver, "env_version": "1.0.0",
         "note": "holdout_example imitates the jury's closed set (5 clean, 5 interacting); real hidden seeds are not here"
         if name == "holdout_example" else f"open {name} split",
         "episodes": eps}, indent=2, ensure_ascii=False) + "\n")
print("ok")
