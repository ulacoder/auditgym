# Jury compliance audit (FinTech case, sections 03-07, 09, 10, 12)

Auditor stance: strict. Evidence = file:function or command output. Verified on 2026-11-19 with
Python 3.13 and Python 3.11 (`uv venv -p 3.11`: 74 tests pass, fingerprint matches, `check` PASS in about 12 s).
Docker build could NOT be verified (registry blocked in audit sandbox).

Legend: MET / PARTIAL / NOT MET / AMBIG.

## 03 Episode and data format

| requirement | verdict | evidence |
|---|---|---|
| 1 company, 1 period; 200-2000 rows, 20-200 docs, 0-12 issues | MET | `validate.validate_episode` size_in_range; `config.MAX_ISSUES=12`; README: 280-1820 rows, 94-180 docs |
| manifest: schema_version, episode_id, base_currency, period_start/end, timezone, counts, rounding | MET | `generator.generate` manifest; `episodes/*/manifest.json` |
| timezone format | AMBIG | offset string `"+03:00"`, not IANA name; fine without DST, but undocumented as a choice |
| ISO-8601 with offset | MET | `validate.ISO_OFFSET` |
| money = integer minor units + exponent | MET | `manifest.currency_exponents` |
| FX = decimal strings | MET | `fx.csv`, `validate.DEC` |
| no cross-currency comparison | MET | `matcher.compare` currency check; ledger is base-only |
| ledger.csv / invoices.jsonl / payments.jsonl / fx.csv exact columns | MET | `test_public_records_have_only_schema_fields`; `cmd_export` |
| policy.json: rate date, period boundary, rounding | MET | `generator.build_policy` |
| missing refs = explicit null | PARTIAL | JSON/API: `null`. CSV: empty field (`cmd_export` writes `""`). A CSV reader gets `""`, not null. Declared only in `manifest.null_encoding` |
| truth.json: issue_id, type, affected_ids, expected_amount_minor, currency, allowed evidence sets | MET | `grader_data/*/truth.json` (`evidence_sets`, + `interaction`). No schema_version in truth |
| provenance.json: generation history and injected changes | MET/PARTIAL | seed, config, changes with rows_before/rows_after, distractors. The full pre-error books are NOT exported, so the jury cannot check "equalities before the error" from files alone |
| agent sees neither the grader files nor the seed | PARTIAL | `server.RemoteEnv.reset` sends the seed from the agent-side process. `harness.evaluate` calls `generate()` and keeps `truths` in the same process as the policy |
| grader runs outside the agent process | PARTIAL | grading runs in the `serve` subprocess, but truth is also built in the parent (policy) process for the oracle and for stats |
| no truth leak via observations/paths/errors/debug | NOT MET (latent) | Demonstrated: with the default `id_salt=""`, the seed is recovered from any public invoice ID in 0.01 s (`sha256(f"{s}||invoice|inv_0001")`). Error documents get the highest canonical counters (hold_09: affected invoices = counters 61..67 of 67). `generate(seed, {})` then diffs out every error. fx.csv is also a pure function of the seed. Tests check strings only (`test_no_truth_in_observations`) |

## 04 Observations, actions, termination

| requirement | verdict | evidence |
|---|---|---|
| reset(seed, config) -> (obs, info) | MET | `env.AuditEnv.reset`. No Gymnasium `reset(seed=None, options=...)` alias, and `seed=None` gives TypeError |
| first obs: task, manifest, short policy, budget | MET | `_start`: task, manifest, policy, tools, page_limit, steps_remaining |
| step -> 5-tuple, JSON-compatible | MET | `test_step_returns_five_tuple` |
| reproducibility, reset isolation | MET | `test_isolation`; paired checks 20/20 |
| list_records(table,cursor,limit) / get_record(table,id) / get_document(doc_id) / calculator(expression) / submit(findings) | MET | `_dispatch`. Action envelope is `{"tool","args"}` (not documented as JSON Schema). get_record(invoices, doc_id) -> unknown_id (only invoice_id accepted) |
| page <= 50, stable cursors | MET | cursor `table@offset`, forgeable but harmless |
| calculator arithmetic only | MET | probes: `__import__` rejected, `9**9**9` rejected, `1e5` rejected, `2**13` rejected (exponent cap 12) |
| unknown id/action -> bounded error, consumes step | MET | probe: unknown_tool / invalid_action / unknown_id each consumed a step |
| obs contains only requested data + result/error + remaining steps | MET | `_obs` |
| 100 steps incl. submit; submit -> terminated; step/time limit -> truncated + empty answer | MET | tests `test_submit_as_100th_step_is_terminated`, `test_time_limit_truncates`. Time limit is checked only at step entry, so a stalled `act()` is not interrupted |
| zero reward during exploration; eval mode hides reward/diagnostics | MET | `_reward_info`; `test_eval_mode_hides_reward_and_diagnostics` |

## 05 Findings and reward

| requirement | verdict | evidence |
|---|---|---|
| findings.json structure | MET | `matcher.validate_submission`. Only the object form is accepted. A bare list gets -1 |
| explanation = short justification | MET | length <= 1000, ignored by reward (`test_explanation_text_does_not_affect_reward`) |
| limits 30 / 10 / 1000 | MET + extra | also `MAX_AFFECTED=10` (not in case) -> -1. Exceeding 30 findings -> -1 (AMBIG vs truncation) |
| invalid structure -> -1 + grader-log error | MET | `grade()` returns `valid:false,error`; server writes it to `--grader-log` |
| 1-to-1 matching, TP rule, FP/FN, duplicates | MET | Kuhn matching in `match`; tests |
| generator provides sufficient evidence | PARTIAL | duplicate_invoice sets = 2 docs only: they do not prove that the copy was POSTED, and the base-currency amount is not visible (0/30 sets contain it). fx_conversion sets = doc + AR/revenue row; the fx rate rows are not addressable as evidence at all (0/43 contain the amount). Validator relies on `analysis.audit` reading ALL tables, and it shares `money.py` with the generator (not fully independent) |
| publish normalization/tolerance/ambiguity code | MET | `matcher.py` docstring and code; `policy.json.tolerance` |
| P/R/F1 edge rules | MET | `metrics`, `test_f1_edge_cases` |
| reward = clip(2F1-1-0.002*inspection_steps) | MET | `terminal_reward`. inspection_steps = all non-submit steps, including errored ones (documented) |
| 5 required test properties + guessed IDs/evidence/currency/resubmit/bad input/timeout | MET | `tests/test_reward.py`, `tests/test_contract.py` |

## 06 Hidden check / jury policies

| requirement | verdict | evidence |
|---|---|---|
| jury can plug in its own 4 policies | PARTIAL | `Policy.reset/act` exists but is undocumented in README. `evaluate --policies` accepts only names from `make_policy` (no `module:Class`). The jury would have to edit our package |
| oracle gets truth only on grader side | PARTIAL | `Oracle(truth_provider)` is fed from `generate()` inside the client harness. No server-side oracle op and no standalone `grade findings.json` command |
| new configs/hidden seeds | PARTIAL | `evaluate --split /path/hidden.json` works (undocumented). `export --splits /path/hidden.json` CRASHES (`NotADirectoryError`). A relative path gives KeyError |
| jury transforms (order/ID) on files | NOT MET | no loader from `episodes/` + `grader_data/` dirs (`AuditEnv` only regenerates from the seed). Transforms exist only via config `id_salt`/`order_seed` |

## 07 / 09 Deliverables

| item | verdict | evidence |
|---|---|---|
| generator: 4 types, 2 interactions, clean, distractors | MET | `generator.py`, `test_generator.py` |
| reset/step env, section-04 actions, 100 steps | MET | as above |
| matcher + reward + attack tests | MET | `test_reward.py` |
| trajectories without truth | MET | 679 rows, 10 train episodes x 4 policies, required fields + `policy`. No truth markers found. Byte-identical on regeneration |
| offline, no paid API/GPU | MET | stdlib only |
| container instructions + single check command | MET (Docker unverified) | `Dockerfile` (base tag not pinned by digest), `python -m auditgym check` PASS |
| source + license + pinned deps | MET | MIT, `requirements.txt` pinned (no hashes) |
| README | MET | |
| matcher/reward, schemas, policies | PARTIAL | schemas exist only as a Python dict (`validate.SCHEMA`, tables only). No JSON Schema for findings/action/observation/manifest/truth/provenance |
| 10 train + 10 dev | MET | `episodes/`, `grader_data/`, export reproduces byte-identically |
| tests | MET | 74 pass on 3.11 and 3.13 |
| 4 baseline adapters | PARTIAL | 4 own policies, not adapters for organizer policies (see 06) |
| design note | MET | `DESIGN.md` |
| demo | MET | `python -m auditgym demo` == `reports/demo_output.txt` |

## 12

| item | verdict | evidence |
|---|---|---|
| fictional data, no cards/PII | MET/AMBIG | names come from `NAMES_A x NAMES_B` (e.g. "Altyn", "Arman Motors") and may coincide with real KZ firms. No `synthetic: true` marker in manifest |
| offline | MET | |
| not an audit opinion; no unproven training claims | MET | README "Честные ограничения" |

## Score estimate (/100)

- Contract & isolation: 20/25. Contract is solid. Risks: seed-recovery leak, seed and truth held in the agent-side process, missing Gymnasium kwargs, extra `MAX_AFFECTED` rule.
- Data & truth: 19/25. Equalities and reproduction are validated. Risks: evidence sufficiency (duplicate posting, FX rates), no pre-error books in grader files, shared `money.py`.
- Reward robustness: 22/25. Strong matcher and tests. Risks: bare-list submission and >30 findings both give -1, which may differ from the jury's expected semantics.
- Reproducibility & generalization: 11/15. Own paired checks pass. The jury's file-level transforms cannot be served.
- Engineering: 8/10. One command 2, config versions 2, trajectory logs 2, failure analysis 1 (99% of reasons are `no_issue_with_these_ids`, only for the naive baseline), limitations 1-2 (omit leak and evidence caveats).
- Total: about 80 (range 72-88). The leak, if probed, can block grading.

## Top 10 improvements (points per hour)

1. Fix `export --splits <path>` and document `--split <path>` for hidden configs (0.25 h).
2. Add a documented `Policy` protocol, `evaluate --policy pkg.mod:Class [--policy-kwargs JSON]`, and `docs/JURY.md` (1 h).
3. Grader-side oracle: `serve --episodes hidden.json` with `{"op":"reset","index":k}` (seed never leaves the grader), a `--allow-oracle` op that builds oracle findings inside the server, and `python -m auditgym grade --truth truth.json --findings findings.json --inspection-steps N` (1.5 h).
4. Leak fix: mix a grader secret (`AUDITGYM_SECRET` or `config.secret`, kept out of the manifest) into `sub_rng`, `IdMapper` and `episode_id`. Permute canonical counters before assigning IDs. Add a test that the seed cannot be recovered from public IDs. Document high-entropy seeds (1 h).
5. Add `schemas/*.schema.json` (findings, action, observation, manifest, truth, provenance, 4 record types) and a test that `validate.SCHEMA` and `validate_submission` agree with them (1.5 h).
6. Export pre-error books (`grader_data/*/ledger_before.csv`) and add `python -m auditgym verify <episode_dir> <grader_dir>` that works from files only (1 h).
7. Evidence sufficiency: add the copy's AR/revenue row to duplicate sets, add an FX key `FX:date:SRC:TGT` to FX sets (or document fx.csv as implicit reference data), and add a validator check that recomputes the amount from the evidence set plus fx/policy only. Regenerate artifacts and FINGERPRINT (2 h).
8. `AuditEnv.from_dir(episode_dir, grader_dir)` and `serve --episode-dir` so the jury's own permutations and renamings can be graded (1.5 h).
9. Contract leniency and documentation: Gymnasium `reset(seed=None, options={"config":...})`, accept `get_record(invoices, doc_id)`, drop or document `MAX_AFFECTED`, list every invalid-structure rule in README (0.75 h).
10. Measured failure analysis: attribute naive/flag-all FPs to distractor kinds via provenance, report generator retry rates and step/time distributions, and extend the limitations section with the leak and evidence caveats (1.5 h).
