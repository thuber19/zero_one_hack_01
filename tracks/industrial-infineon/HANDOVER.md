# Handover — Industrial AI Track Submission Pipeline

A self-contained, locally-trained solution for all three tasks of the
semiconductor process-sequence challenge. No external APIs, no black boxes —
pure stdlib Python + a transparent physics model. Built to generalise to the
hidden 4th (unknown) family.

**TL;DR for the tester:** install nothing extra, run one command:

```bash
python run_all.py --self-test
```

It generates its own labelled test data, runs all three tasks, and prints the
scores. If you see Task 3 `F1=1.000` with `FP=0` and the Task 1/2 numbers
below, the pipeline is healthy.

---

## 1. What this is

The challenge asks for three things:

| Task | Goal | Our score on self-test |
|------|------|------------------------|
| **1 — Next-step prediction** | predict the next step (Top-1/3/5, MRR) | Top-1 **0.93**, Top-3/5 **1.00**, MRR **0.97** |
| **2 — Sequence completion** | finish a partial sequence (Exact Match, Edit Distance) | Exact **0.64**, NormEditDist **0.10** |
| **3 — Anomaly detection** | flag invalid sequences + name the broken rule (F1, ROC-AUC, Rule Attribution) | F1 **1.00**, RuleAttr **1.00**, **0** false positives |

The 4th family is *unknown* at submission time. Our design handles it because
every decision is made on **physical category** (deposit / etch / implant /
clean / …), inferred from the step name by keyword, not on memorised exact
names. A never-before-seen `GROW GAN BUFFER LAYER` is treated as a deposition
and gets the same physical preconditions as any deposition — so the same rules
apply to chips the model has never seen.

---

## 2. How to run

### Prerequisites
- Python **3.9+** (developed/tested on 3.14). Standard library only — **no pip
  installs**.
- The training CSVs already in `training_data/` (`MOSFET_variants.csv`,
  `IGBT_variants.csv`, `IC_variants.csv`).

### A. Self-test (no eval files needed) — start here
```bash
python run_all.py --self-test
```
Generates synthetic eval inputs covering **all 10 rules**, runs every task, and
**scores the output against ground truth**. This is the fastest way to confirm
the whole pipeline works end-to-end on your machine.

### B. Real submission (when the organisers' eval files arrive)
```bash
python run_all.py \
    --eval-valid   eval_input_valid.csv \
    --eval-anomaly eval_input_anomaly.csv \
    --output-dir   submissions/
```
Writes `submissions/task1_predictions.csv`, `task2_predictions.csv`,
`task3_predictions.csv`.

### C. One task at a time
```bash
python submit_task1.py --eval-input eval_input_valid.csv   --output task1_predictions.csv
python submit_task2.py --eval-input eval_input_valid.csv   --output task2_predictions.csv
python submit_task3.py --eval-input eval_input_anomaly.csv --output task3_predictions.csv
```

### Useful flags
- `--data-dir DIR` — point at the training CSVs if they are not in
  `training_data/`.
- `--skip-task1 / --skip-task2 / --skip-task3` — run a subset.
- `submit_task3.py --no-score` — skip the transition model (faster; SCORE
  column becomes hard 0.05/0.95 instead of continuous).
- `submit_task2.py --beam-width N` — candidates per step in the greedy fallback.

The first model build caches to `models/transition_model.pkl` (~1 s to rebuild
from 3000 sequences); later runs load the cache.

---

## 3. Output formats (exactly what each script writes)

- **Task 1** — `EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5`
- **Task 2** — `EXAMPLE_ID, PREDICTED_SEQUENCE` (pipe-separated; the
  **completion only**, not the input repeated)
- **Task 3** — `EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE`
  (`IS_VALID` ∈ {0,1}; `SCORE` ∈ [0,1] where higher = more valid;
  `PREDICTED_RULE` is one of the `RULE_*` IDs, empty for valid sequences)

Input parsing is tolerant: it handles UTF-8 BOMs, quoted headers, and missing
optional columns. If a `PARTIAL_SEQUENCE`/`SEQUENCE` column is absent it raises
a clear error listing the headers it did find.

---

## 4. Architecture (what's under the hood)

```
physics/
  ontology.py        3-tier step→category classifier (exact → level-prefix →
                     keyword) + a 25-dim physics feature vector per step.
                     THIS is what generalises to the unknown family.
  state_machine.py   Wafer state machine. Replays a sequence step by step and
                     raises a violation when a step's physical precondition is
                     not met. The 10 rules EMERGE from physics; they are not
                     hard-coded pattern matches.
models/
  transition_model.py  Bigram/trigram Markov model with category-level backoff.
                       Provides Task-1 ranking, Task-2 proposals, Task-3 score.
submit_task1/2/3.py    The three submission entry points.
run_all.py             Orchestrator + self-test harness with scoring.
```

### Key design decisions (and why)

- **Windowed rules, not boolean flags.** Five of the ten rules are *sliding
  windows* in the reference validator (a clean within 12 steps, a mask opening
  within 15, …). The state machine stores the *step index of the most recent
  qualifying event* and compares distances. This is why two depositions in a
  row, or two implants in a row, both validate correctly — they share one
  upstream event still inside the window. (An earlier boolean-flag version
  wrongly flagged every valid sequence here; that bug is fixed and guarded by a
  cross-check — see §5.)

- **Two-stage anomaly detection.** Task 3 runs the exact deterministic checker
  (`generate_sequences.validate_sequence`) first — it is exact for the three
  known families — then the physics state machine as a fallback for sequences
  with unknown step names. The state machine is deliberately *no stricter* than
  the reference on valid known-family sequences (verified: 0 false positives on
  600 sequences), so the fallback never invents a violation.

- **Retrieval-augmented Tasks 1 & 2.** Because the eval partials are prefixes
  of real family sequences, we anchor on the tail of the partial, find that
  exact context in a same-family training sequence, and use its continuation.
  Every retrieved completion is re-validated by the physics machine. For the
  unknown family (no match) it falls back to the physics-constrained greedy
  generator. This is fully transparent and local — no external model.

---

## 5. How to trust it (built-in checks)

Each module self-tests when run directly:

```bash
python physics/ontology.py        # category distribution + GaN OOD classification
python physics/state_machine.py   # CROSS-CHECK vs reference on 600 valid seqs:
                                   #   must print "zero false positives"
python models/transition_model.py # next-step demo + anomaly score demo
```

`python physics/state_machine.py` is the important one: it generates 600 fresh
valid sequences from all three families and asserts the state machine flags
**none** of them, then confirms it still catches an injected violation. If that
ever prints a FAIL, stop and investigate before submitting.

The `run_all.py --self-test` scorer reports a confusion matrix for Task 3; the
line `[!] N VALID sequence(s) wrongly flagged` appears only if there are false
positives — there should be none.

---

## 6. Files written at runtime (safe to delete)

- `models/transition_model.pkl` — cached model (rebuilt automatically).
- `submissions/` — task output CSVs.
- `synthetic_eval/` — self-test inputs + ground truth.
- `__pycache__/` — Python bytecode.

None of these are inputs; delete them any time to force a clean rebuild.

---

## 7. Known limitations / honest notes

- **Task 2 Exact Match is inherently capped** below 1.0: the generator inserts
  optional steps (measurements, bakes) by coin-flip, so the true completion is
  one of many valid variants. We optimise edit distance (≈0.10) by retrieving
  the closest structural variant; we cannot guess which optional steps a
  specific sequence happened to include.
- **The raw transition-model anomaly score** is a weak standalone signal; Task
  3 reconciles it with the rule verdict (clamping to ≤0.15 / ≥0.85) so the
  final SCORE separates the classes cleanly. Detection itself is rule-driven.
- **OOD coverage depends on step *names* containing a recognisable keyword**
  (DEPOSIT/GROW/ETCH/IMPLANT/CLEAN/…). The keyword table in
  `physics/ontology.py::_KEYWORD_FALLBACK` is ordered for correctness (e.g.
  "DRY ETCH" → ETCH, not CLEAN); extend it there if the 4th family uses
  unusual vocabulary.

---

## 8. Known-bad dataset generator (`bad_data_generator.py`)

We also ship the *opposite* of the valid-sequence generator: a generator of
**invalid** sequences for testing the detector, in every variety we know how to
break.

```bash
python bad_data_generator.py --per-combo 6 --audit
```

Produces in `bad_data/`:
- `known_bad.csv` — invalid sequences. For each of the 10 rules, several
  distinct injection strategies (delete the enabler, push it out of the window,
  reorder, skip/decrease a litho level, …), across all three families. Every
  row is labelled by the **reference** checker, so the label is ground truth.
  TIER 1 = trips exactly one rule; TIER 2 = compound (multiple rules).
- `hard_negatives.csv` — valid-but-suspicious sequences (the false-positive
  traps: consecutive deposits, consecutive implants, window-edge cleans).
- `eval_input_anomaly.csv` + `ground_truth.csv` — a ready-to-run Task-3 input
  and its answer key.

`--audit` cross-checks both detectors against the labelled set. Current result:
the reference and our physics machine each flag **336/336** invalid sequences
with **0 false positives** on the 54 hard-negatives. Running the real Task-3
pipeline over the full 390-row file scores **F1 = 1.000, rule attribution
1.000**.

Use it to regression-test any change to the detector, or to build a bigger
adversarial set (`--per-combo 20`).

## 9. The refinery — refine ANY model's output (`refinery.py`)

This is the "ecosystem on top of a trained model". It treats your model (a
trained LLM, a from-scratch decoder, our n-gram model — anything) as a black-box
**scorer** and refines its raw output so it is always physically legal, never
loops, and always terminates. The only thing it needs from the model is:

```python
score_fn(prefix: list[str]) -> list[str]          # ranked next steps
                            or dict[str, float]    # step -> score (e.g. logits)
```

```python
from refinery import PhysicsRefinery, transition_model_scorer, learn_category_grammar

refinery = PhysicsRefinery(category_grammar=grammar, category_mode="soft")

# Task 1 — lift Top-1/MRR by dropping illegal candidates, keep model order:
refined_top5 = refinery.rerank(prefix, model_ranked_candidates, k=5)

# Task 2 — model proposes, physics vetoes, always terminates:
completion = refinery.constrained_decode(prefix, transition_model_scorer(model))

# Task 3 / safety — certify a sequence (symbolic half of the ensemble):
is_valid, violations = refinery.guard(full_sequence)
```

To plug in a **real LLM / from-scratch model**, wrap its next-token
distribution over the step vocabulary with `llm_scorer_example(generate_logits,
vocab)` (template in the file) and pass that as `score_fn`. The hard constraint
is enforced at the **category** level via the physics state machine, so it works
on an unseen 4th-family vocabulary and on novel block structure without
retraining.

`python refinery.py` runs a demo. The headline check: a deliberately **random**
scorer, wrapped by the refinery, still yields a 0-violation, properly
terminating sequence — i.e. the refinery adds correctness on top of even a
useless model.

## 10. The knowledge base — understanding encoded as data

The system's understanding of the process is **not buried in `if` statements**.
It lives in one declarative, human-readable file, `physics/process_knowledge.py`,
which encodes:
- **State variables** — what a wafer's physical state *is* (surface cleanliness,
  resist pattern, mask level, implant window, overburden, passivation, sort) and
  **why each matters**;
- **Event classes** — physically meaningful operations, with **hybrid
  membership** (exact reference vocabulary for known steps; physical category
  for unknown 4th-family steps);
- **Rules** — every forbidden situation as data: `(trigger) needs (enabler
  within N steps)` or `(trigger) needs (milestone first)`, each carrying the
  **causal reason** it exists;
- **Process flow** — the canonical fabrication narrative.

`physics/state_machine.py` is now a generic **engine** that interprets this KB —
it contains no per-rule logic. So **adding a step, a family, or a whole new rule
is a data edit**, not a code change, and the explanations stay in sync with the
rules automatically. (Verified: the KB-driven engine reproduces the reference
exactly — 600/600 valid pass, 336/336 known-bad flagged, 0 false positives.)

Browse it:
```bash
python physics/process_knowledge.py                 # summary
python explain.py --export-doc knowledge/PROCESS_MODEL.md   # full Markdown model
```

## 11. Make it explain itself (`explain.py`)

The system can narrate *why* any sequence is correct or wrong, step by step,
from the KB — this is the "context-aware after training" surface:

```bash
python explain.py --family mosfet            # narrate a valid sequence
python explain.py --family igbt --break      # inject a fault, explain the failure
python explain.py --file bad_data/known_bad.csv --row 1
```

Example output on a faulted sequence:
```
>>   6 EPITAXIAL DEPOSITION  [DEPOSITION, DEPOSIT_OR_FILL]
        XX  RULE_DEP_NO_CLEAN: no CLEAN_SURFACE within the prior 12 steps
        !! VIOLATION RULE_DEP_NO_CLEAN
           what : A deposition has no cleaning step in the prior 12 steps.
           why  : Thin-film deposition nucleates on the existing surface;
                  contamination becomes buried defects ...
```
On valid steps it shows the satisfied precondition and its evidence
(`CLEAN_SURFACE satisfied by step 10, 1 step ago; window 12`).

## 12. Design rationale (the two lenses, in one paragraph)

We weighed a symbolic-first view (the validator is exact for Task 3; "physics
features = the rulebook"; the real OOD axis is *structural*, same tokens / new
arrangement) against the opposite view (a from-scratch transformer also learns
the rules, just opaquely; lexical OOD is a real risk if the 4th family is a new
material system; the symbolic layer is a cheap, verifiable hedge). Both converge
on the same architecture: **the deterministic checker owns Task 3; a model owns
Tasks 1 & 2; and a symbolic layer (the state machine) re-ranks / masks / guards
the model so its output is always physically valid.** The state machine checks
preconditions step-by-step with **no hard-coded block templates**, so it is
robust to *structural* novelty; the keyword classifier adds *lexical* robustness
as a bonus. Known steps use the exact reference vocabulary; only genuinely
unknown (4th-family) steps fall back to physical-category reasoning — which is
why the physics machine equals the reference on knowns *and* generalises.
