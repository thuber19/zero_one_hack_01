# ASSUMPTIONS — read before trusting any number or pushing a submission

These are the things we **assume** about the (not-yet-distributed) official eval
files and scorer. They follow `generation_rules.md` / `submission/SUBMISSION.md`,
but **we have NOT seen the real files**. Each is marked so we don't ship a guess
by accident. **When the organizers' `eval_input_*.csv` + `eval_metrics.py`
arrive, verify every item below against them.**

| # | Assumption | Source | Where it lives | Risk if wrong |
|---|---|---|---|---|
| **A1** | `eval_input_valid.csv` columns = `EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE`; `PARTIAL_SEQUENCE` is **pipe(`|`)-separated** steps | generation_rules.md §5.1 | `inference._parse*` / `evaluate` | parser misreads → wrong/empty predictions |
| **A2** | `eval_input_anomaly.csv` columns = `EXAMPLE_ID, FAMILY, SEQUENCE`; `SEQUENCE` pipe-separated | §5.1 | inference parsers | same |
| **A3** | `FAMILY` ∈ {MOSFET, IGBT, IC}. The hidden 4th family is NOT in `FAMILY_TOKENS`; we map it to `[UNK]` instead of crashing. Physics is family-agnostic. | inferred | `tokenizer.encode_sequence` | *was a crash; now handled* — if a 4th token were expected, model loses the family hint (physics still fine) |
| **A4** | Task-2 `PREDICTED_SEQUENCE` = **completion only** (steps AFTER the cut), not repeating the partial | §5.3 | `generate_task2_submission` | doubled/garbage edit-distance |
| **A5** | Task-3 `SCORE` = **P(valid)** ∈ [0,1] (higher = more valid) | §5.3 | `detect_anomaly` | inverted AUC |
| **A6** | Submission filenames = `nextstep.csv`, `completion.csv`, `anomaly.csv` | SUBMISSION.md | `generate_all_submissions` | rejected/misparsed upload |
| **A7** | CSVs are comma-delimited, UTF-8 (BOM-tolerant) | standard | all parsers | parse error |
| **A8** | **Block-level Accuracy** = position-wise match on the *category* ("block") sequence (consecutive duplicates collapsed). This is **OUR PROXY** — the official definition is unknown. | NOT specified | `evaluate._block_seq` | our internal Block number ≠ official; **submission CSVs unaffected** |
| **A9** | Our self-eval set (100/family × {0.6,0.8} valid + all-10-rule anomaly set) *mirrors* the official distribution. It is **synthetic, not the official set**. | §5.1 (sizes) | `create_self_eval_set` | self-eval is an *estimate*; only the official run is truth |
| **A10** | The 10 rules + windows in `physics/process_knowledge.py` exactly match `generate_sequences.validate_sequence`. **VERIFIED**, not assumed: `exhaustive_test.py` 42/42 **and** `differential_fuzz.py` = **0 disagreements over 8500+ random in-vocab mutations, all 10 rules** (binary verdict AND rule set). | generation_rules.md §3 | physics engine | n/a (verified) |
| **A11** | The grader labels **in-vocabulary** sequences with logic equivalent to `validate_sequence`; the hidden 4th family **mostly reuses the shared vocabulary** (README: "step names are mostly shared"). Therefore `validate_sequence_combined` routes **all-in-vocab → exact reference checker** (grader-equivalent, proven) and **any-novel-token → category engine** (generalises per §"...regardless of whether individual steps appear in the vocabulary"). | README §"Family as context"; generation_rules.md §3 intro | `validate_sequence_combined` + `physics/known_vocab.py` | if the 4th family used *entirely novel* vocab AND the grader were vocab-locked, our category routing would over-accept some novel-vocab violations (precision/recall trade documented in `real_family_benchmark.py`) |

## What is NOT an assumption (verified facts)
- Rule logic vs the provided reference checker: **bit-for-bit on 3000 sequences** (`exhaustive_test.py`) **and 8500+ random mutations across all 10 rules** (`differential_fuzz.py`, 0 disagreements).
- Submission formats: produced AND machine-checked against §5.3 by `validate_submission.py` (header, IDs, rank count, special-token leak, SCORE range, rule validity, no partial-repeat).
- End-to-end runs: `make_sample_eval.py` → `inference.py` → `validate_submission.py` → `self_score.py` runs clean on a held-out sample (T1 Top-5 1.000, T3 F1 1.000, RuleAttr 18/18).
- No crashes on malformed input: `robustness_test.py` (empty / over-length / unknown family / novel vocab / blank steps) PASS.
- The model is real (trained transformer + RF), open-stack, no API. Only pickle load is the RF you trained (see `random_forest.load` security note).

## The honest boundary
- We can benchmark **relatively** (config A vs B, model vs model+physics, ID vs OOD)
  and we are **certain on known-family rule logic** (verified).
- We are **NOT certain** of (a) the official Block-level definition, (b) the exact
  eval-file schema/casing, (c) the actual 4th family. Those resolve only with the
  organizers' files + `eval_metrics.py`. Until then, treat absolute self-eval
  numbers as estimates, not the official score.
