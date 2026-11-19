# Check report

Overall: **PASS** (14.03 s)

## Validation

- train: 10/10
- dev: 10/10
- holdout_example: 20/20

## Baselines (holdout_example, 20 episodes)

| policy | episodes | P | R | F1 (mean) | F1 (micro) | return | steps | time, s | invalid |
|---|---|---|---|---|---|---|---|---|---|
| random_explorer | 20 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 23.9 | 0.032 | 0 |
| naive_reconciler | 20 | 0.115 | 0.461 | 0.181 | 0.207 | -0.658 | 14.9 | 0.031 | 0 |
| flag_all | 20 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 3.0 | 0.024 | 0 |
| oracle | 20 | 0.750 | 0.750 | 1.000 | 1.000 | +1.000 | 1.0 | 0.024 | 0 |
| scripted_auditor | 20 | 0.750 | 0.750 | 1.000 | 1.000 | +0.972 | 14.9 | 0.035 | 0 |

## Failure reasons (unmatched findings)

- random_explorer: {'clean_record': 13, 'distractor:recurring_amount': 4, 'distractor:legit_partial_payment': 2, 'distractor:tz_boundary_invoice': 2, 'distractor:split_payment': 1, 'distractor:next_period_unposted_invoice': 1}
- naive_reconciler: {'distractor:recurring_amount': 113, 'distractor:unreferenced_payment': 76, 'distractor:tz_boundary_invoice': 52, 'clean_record': 27, 'real_issue_wrong_details': 5, 'distractor:split_payment': 3, 'distractor:legit_partial_payment': 1}
- flag_all: {'clean_record': 336, 'distractor:recurring_amount': 86, 'distractor:split_payment': 84, 'distractor:legit_partial_payment': 39, 'distractor:tz_boundary_invoice': 27, 'distractor:multi_invoice_payment': 12, 'distractor:next_period_unposted_invoice': 8, 'real_issue_wrong_details': 8}
- oracle: {}
- scripted_auditor: {}

## Per error type (found/total, FP)

- random_explorer: duplicate_invoice 0/18 FP=6, fx_conversion 0/19 FP=5, payment_allocation 0/18 FP=3, period_cutoff 0/14 FP=9
- naive_reconciler: duplicate_invoice 18/18 FP=105, fx_conversion 5/19 FP=45, payment_allocation 13/18 FP=81, period_cutoff 4/14 FP=46
- flag_all: duplicate_invoice 0/18 FP=140, fx_conversion 0/19 FP=140, payment_allocation 0/18 FP=200, period_cutoff 0/14 FP=120
- oracle: duplicate_invoice 18/18 FP=0, fx_conversion 19/19 FP=0, payment_allocation 18/18 FP=0, period_cutoff 14/14 FP=0
- scripted_auditor: duplicate_invoice 18/18 FP=0, fx_conversion 19/19 FP=0, payment_allocation 18/18 FP=0, period_cutoff 14/14 FP=0

## Paired checks

20/20 passed
