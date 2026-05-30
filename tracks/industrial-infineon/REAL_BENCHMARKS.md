# Real Benchmarks (official `eval_metrics.py`, held-out split)

These are **real** numbers — produced by the organizers' official scorer
(`data/eval_metrics.py`) on a **disjoint held-out split** (no self-scoring, no
leakage). They are the honest "where we are" snapshot for a **small** model.

## Setup (fully reproducible)
```bash
# 1. data: provided variants split 900 train / 100 eval per family (disjoint)
python src/generate_data.py --extra-data 400 --output-dir outputs_bench --seed 1
# 2. train a TINY transformer with the model-internalization levers
OUTPUT_DIR=outputs_bench python src/train.py --arch transformer --model-size tiny \
    --epochs 10 --aux-category --unk-dropout 0.15 --no-rf --device cpu
# 3. inference -> submissions
python src/inference.py --output-dir outputs_bench --eval-dir outputs_bench --model-size tiny
# 4. OFFICIAL scoring
python data/eval_metrics.py --task next-step  --ground-truth outputs_bench/eval_set_valid.csv     --predictions outputs_bench/submissions/nextstep.csv
python data/eval_metrics.py --task completion --ground-truth outputs_bench/eval_set_valid.csv     --predictions outputs_bench/submissions/completion.csv
python data/eval_metrics.py --task anomaly    --ground-truth outputs_bench/eval_set_forbidden.csv --predictions outputs_bench/submissions/anomaly.csv \
                            --valid-supplement outputs_bench/eval_set_valid_supplement.csv
```
Model: transformer **tiny** (583,570 params), **10 epochs**, CPU, **no RF**,
`--aux-category` + `--unk-dropout 0.15`. Best val-acc 0.8045.

## Results

### Task 1 — Next-step prediction (600 held-out examples)
| Metric | Value |
|---|---|
| Top-1 | **0.690** |
| Top-3 | 0.998 |
| Top-5 | **1.000** |
| MRR | 0.8435 |

Per family: MOSFET Top-1 0.725 / IGBT 0.680 / IC 0.665. By cut: 60% Top-1 0.760, 80% Top-1 0.620.

### Task 2 — Sequence completion (600 held-out examples)
| Metric | Value |
|---|---|
| Normalized Edit Distance (lower=better) | **0.226** |
| Exact Match | 0.002 |
| Token Accuracy | 0.441 |
| Block-level Accuracy | **0.692** |

Per family NED: MOSFET 0.159 / IGBT 0.236 / IC 0.284.

### Task 3 — Anomaly detection (450 held-out: 150 invalid / 300 valid)
| Metric | Value |
|---|---|
| Accuracy | **1.000** |
| Precision / Recall / F1 (invalid) | 1.000 / 1.000 / 1.000 |
| ROC-AUC | 1.000 |
| Rule-Attribution Accuracy | 1.000 |
Confusion: TP=150, FP=0, FN=0, TN=300.

### Model's own functional understanding (aux head, no physics) — `category_eval.py`
| Metric | Value | Note |
|---|---|---|
| Random baseline | 0.056 | 18 categories |
| ID next-category acc | **0.944** | strong in-distribution functional learning |
| OOD next-category acc | 0.371 | unseen families — **6.6× random** (partial transfer) |
| OOD lexical (name) acc | 0.146 | category ≫ name ⇒ learned function, not just names |

## Honest reading of these numbers
- **Task 1 / Task 2 are genuine model skill** (tiny, 10-epoch, no-RF). Top-5 1.000
  and MRR 0.84 are strong because the vocabulary is small and the physics reranker
  guarantees legal candidates; Top-1 0.690 is the real per-step difficulty. Block
  accuracy 0.69 means completions are structurally ~70% right but rarely exact
  (expected when completing 20–60 steps). **These will rise with a larger model /
  more epochs / RF on GPU.**
- **Task 3 = 1.000 is the rule ENGINE, not the model**, and it is on a **narrow**
  held-out forbidden set: `generate_data.inject_violation` produced **only
  `RULE_DEP_NO_CLEAN`** (150/150). So this confirms the engine is exact on the
  rule that was injected, in-vocab — it is **not** evidence across all 10 rules.
  The all-10-rule equivalence is proven separately by `differential_fuzz.py`
  (0 disagreements, all rules), and broad detection by `exhaustive_test.py`.
- The **model-only** functional numbers (ID 0.944, OOD 0.371) are the honest
  measure of what the *network* learned — it generalizes function partially, far
  above chance, even at this tiny scale.

## Bottom line
On the official scorer, this **tiny** model already gets **Top-5 1.000 / MRR 0.84
(T1)**, **NED 0.23 / Block 0.69 (T2)**, and **F1 1.000 (T3, single-rule in-vocab,
engine-driven)**. This is the floor; the hand-off recipe (`HANDOFF_TRAINING.md`:
larger model + more epochs + RF + GRPO on Leonardo) is what lifts the model-driven
tasks. Task-3 breadth across all 10 rules is established by the test suite, not by
this single-rule held-out set.
