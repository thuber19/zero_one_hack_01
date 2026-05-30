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
| **A10** | The 10 rules + windows in `physics/process_knowledge.py` exactly match `generate_sequences.validate_sequence`. **This one is VERIFIED**, not assumed: `exhaustive_test.py` = 42/42 on all 3000 provided sequences. | generation_rules.md §3 | physics engine | n/a (verified) |

## What is NOT an assumption (verified facts)
- Rule logic vs the provided reference checker: **bit-for-bit on 3000 sequences** (`exhaustive_test.py`).
- Submission formats: produced and inspected to match §5.3.
- The model is real (trained transformer + RF), open-stack, no API.

## The honest boundary
- We can benchmark **relatively** (config A vs B, model vs model+physics, ID vs OOD)
  and we are **certain on known-family rule logic** (verified).
- We are **NOT certain** of (a) the official Block-level definition, (b) the exact
  eval-file schema/casing, (c) the actual 4th family. Those resolve only with the
  organizers' files + `eval_metrics.py`. Until then, treat absolute self-eval
  numbers as estimates, not the official score.
