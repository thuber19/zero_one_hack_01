# REPORT — Industrial AI: Learning & Benchmarking Process Logic

## TL;DR
We built a **neuro-symbolic** system for semiconductor process sequences: a
from-scratch GPT transformer (+ a Random-Forest candidate filter) that *predicts*
steps, wrapped by a declarative **physics engine** that *guarantees* every output
is physically valid, *explains* why any sequence is wrong, and *repairs* it. The
model proposes; physics disposes. On the three tasks the model is strong
in-distribution and the physics layer makes it **provably valid and generalizable
to the unknown 4th family** — turning even a weak model into one that never emits
an invalid process route.

## Problem
The track asks whether a model *learns process logic* or just *memorizes patterns*.
Sequences are 115–150 steps; validity is governed by 10 ordering/precondition
rules (clean-before-deposit, mask-before-etch, implant window, CMP overburden,
passivation before test/backside, sequential mask levels, ship-after-sort). The
hidden 4th family tests out-of-distribution (OOD) generalization. We decided to
solve the *whole* problem: predict (Tasks 1–2), detect+attribute (Task 3), and
generalize (Task 4) — and to make the system's reasoning **inspectable**, not a
black box.

## Approach
- **Trained model (from scratch).** A GPT-style decoder (`src/transformer_model.py`,
  sizes tiny/small/medium, weight-tied) trained with next-step LM
  (`src/train.py`, AdamW + cosine + early stop). A sklearn **Random-Forest**
  (`src/random_forest.py`) learns a soft grammar and masks the transformer's
  candidates. Anomalies use the model's per-token perplexity.
- **Declarative physics knowledge base** (`physics/process_knowledge.py`): the 10
  rules, the wafer state variables, and the causal "why" encoded as *data*, not
  code. A generic engine (`physics/state_machine.py`) interprets it — verified
  bit-for-bit against the reference checker on all 3000 provided sequences.
- **Hybrid generalization.** Rules key on physical *category* (deposition, etch,
  implant…), decided by exact vocabulary for known steps and by keyword/category
  for unseen steps — so the same rules apply to a never-seen family.
- **The merge (`src/inference.py`).** The transformer is the proposal
  distribution; the **refinery** (`refinery.py`) re-ranks Task-1 candidates by
  physical legality, drives a **constrained decode** for Task-2 (vetoing any
  invalid step, guaranteeing termination), and the **combined validator** +
  **fix engine** (`fix.py`) power Task-3 detection, explanation, and repair.
- **Honest data.** Good-sequence generator (provided), an exhaustive **known-bad
  generator** (every rule × strategy × family, `bad_data_generator.py`), and
  **pseudo-families** with novel vocabulary (`pseudo_family.py`) for OOD training
  and self-measured ID→OOD.

## How to run it
```bash
# 1) Physics harness — pure stdlib, runs anywhere, no install:
python tracks/industrial-infineon/exhaustive_test.py          # 42/42 correctness gate
python tracks/industrial-infineon/integration_test.py         # merged pipeline + physics on/off benchmark

# 2) Neural pipeline (needs requirements.txt; GPU for full size):
cd tracks/industrial-infineon
python src/generate_data.py --extra-data 10000 --output-dir outputs
python src/train.py --model-size small --epochs 50 --output-dir outputs
python src/evaluate.py --self-eval --output-dir outputs        # all 3 tasks + per-family
# Official eval files when distributed:
python src/inference.py --eval-dir <eval_files> --output-dir outputs
#   -> outputs/submissions/{nextstep,completion,anomaly}.csv
```

## Results (honest, with evidence)
- **Physics correctness (verified):** `exhaustive_test.py` = **42/42** — all 3000
  provided sequences + 3 canonical refs validate; every window boundary exact;
  every one of the 10 rules detected, explained, and **repaired to valid**
  (336/336 known-bad); **0 false positives** on suspicious-but-valid traps.
- **Trained transformer (committed, tiny 581K params, local MPS, 6k seqs):**
  val next-step accuracy **~0.80**; RF baseline test accuracy **0.80**, top-15
  **1.00**. (Leonardo scaling sweep is the immediate next step — see below.)
- **Integration benchmark (model + physics vs model alone):** Task-2 completions
  **100% physically valid** with physics; **Task-3 F1 = 1.00, rule-attribution
  1.00** on the 10-rule labeled set; **OOD recall 1.00 vs 0.95** without physics.
  A deliberately weak model goes from **0% → 100%** valid completions once
  wrapped — the clearest evidence of the layer's value.
- **Explainability:** for any violation the system returns the rule, the physical
  reason (sourced — see `SOURCES.md`), and a concrete fix.

## What worked / what didn't
- **Worked:** the neuro-symbolic split (model for likelihood, physics for
  legality); encoding rules as inspectable data; the verifier doubling as an RL
  reward (`reward.py`) and as an oracle topline; pure-stdlib physics so
  correctness is reproducible anywhere.
- **Didn't / limits:** the committed transformer is *tiny* and trained locally —
  the Leonardo scaling run isn't in yet; the 4 ordering rules assume the
  universal logistics/test step names carry to the 4th family (the README says
  vocabulary is mostly shared); Block-level Accuracy uses a category proxy until
  the official `eval_metrics.py` is wired.

## What we'd do with another 36 hours
1. Run the **Leonardo scaling sweep** (tiny/small/medium × 100/1k/5k) for the
   baseline-vs-trained curves. 2. Add a **next-category auxiliary head** + train on
   the **factorized + pseudo-family + contrastive** corpora (`export_training_data.py`)
   for stronger OOD. 3. **GRPO** with the verifier as reward (`reward.py`).
   4. Calibrate the Task-3 SCORE for ROC-AUC.

## Credits & dependencies
- Libraries: PyTorch, scikit-learn, NumPy, Matplotlib (neural pipeline); physics
  harness is Python stdlib only.
- Data & rules: the provided generator/validator and `generation_rules.md`.
- Physics grounding: standard semiconductor-fabrication references — see
  `tracks/industrial-infineon/SOURCES.md` (Plummer/Deal/Griffin; Campbell; Sze;
  Wolf & Tauber; May & Spanos; Quirk & Serda; RCA clean; implant masking).
- AI coding assistance was used to build and review the harness.
