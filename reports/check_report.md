# Check report

Overall: **PASS** (20.53 s)

## Validation

- train: 10/10
- dev: 10/10
- holdout_example: 20/20

## Baselines (holdout_example, 20 episodes)

| policy | episodes | P | R | F1 (mean) | F1 (micro) | return | steps | time, s | invalid |
|---|---|---|---|---|---|---|---|---|---|
| random_explorer | 20 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 25.0 | 0.018 | 0 |
| naive_reconciler | 20 | 0.124 | 0.471 | 0.191 | 0.215 | -0.638 | 14.7 | 0.021 | 0 |
| flag_all | 20 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 3.0 | 0.016 | 0 |
| oracle | 20 | 0.750 | 0.750 | 1.000 | 1.000 | +1.000 | 1.0 | 0.013 | 0 |
| scripted_auditor | 20 | 0.750 | 0.750 | 1.000 | 1.000 | +0.973 | 14.7 | 0.024 | 0 |

## Failure reasons (unmatched findings)

- random_explorer: {'clean_record': 17, 'distractor:recurring_amount': 6, 'distractor:split_payment': 1, 'distractor:tz_boundary_invoice': 1, 'distractor:legit_partial_payment': 1}
- naive_reconciler: {'distractor:recurring_amount': 113, 'distractor:unreferenced_payment': 76, 'distractor:tz_boundary_invoice': 52, 'clean_record': 22, 'real_issue_wrong_details': 4, 'distractor:split_payment': 3, 'distractor:legit_partial_payment': 1}
- flag_all: {'clean_record': 330, 'distractor:split_payment': 88, 'distractor:recurring_amount': 83, 'distractor:legit_partial_payment': 42, 'distractor:tz_boundary_invoice': 27, 'distractor:multi_invoice_payment': 13, 'distractor:next_period_unposted_invoice': 9, 'real_issue_wrong_details': 8}
- oracle: {}
- scripted_auditor: {}

## Per error type (found/total, FP)

- random_explorer: duplicate_invoice 0/18 FP=9, fx_conversion 0/19 FP=8, payment_allocation 0/18 FP=7, period_cutoff 0/14 FP=2
- naive_reconciler: duplicate_invoice 18/18 FP=105, fx_conversion 5/19 FP=40, payment_allocation 14/18 FP=80, period_cutoff 4/14 FP=46
- flag_all: duplicate_invoice 0/18 FP=140, fx_conversion 0/19 FP=140, payment_allocation 0/18 FP=200, period_cutoff 0/14 FP=120
- oracle: duplicate_invoice 18/18 FP=0, fx_conversion 19/19 FP=0, payment_allocation 18/18 FP=0, period_cutoff 14/14 FP=0
- scripted_auditor: duplicate_invoice 18/18 FP=0, fx_conversion 19/19 FP=0, payment_allocation 18/18 FP=0, period_cutoff 14/14 FP=0

## Paired checks

20/20 passed
