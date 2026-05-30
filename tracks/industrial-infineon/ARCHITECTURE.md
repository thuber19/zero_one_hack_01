# Architecture — how the two layers fit together

This track contains **two cohesive layers**, both kept and both working:

```
                ┌─────────────────────────────────────────────────────────┐
                │  LAYER A — learned models  (team / main: solution/procseq)│
                │  • Llama-style decoder  → Task 1 next-step, Task 2 complete│
                │  • DeBERTa encoder      → Task 3 anomaly + span            │
                │  • trained from scratch via pixi -e procseq (jobs/run.sh)  │
                └───────────────┬─────────────────────────────────────────-─┘
                                │  proposes steps / sequences / scores
                                ▼
                ┌─────────────────────────────────────────────────────────┐
                │  LAYER B — physics + training injections  (this fork)     │
                │  model-agnostic: validates / repairs / explains ANY output│
                │  • deterministic rule engine (physics/) — grader-equivalent│
                │  • Task-2 physics-vetoed beam decode (refinery.beam_decode)│
                │  • explicit VALID/INVALID/INSUFFICIENT_INFORMATION verdict │
                │  • training injections that make the MODEL internalize it: │
                │    aux-category head · UNK-dropout · synonym-collapse ·     │
                │    pseudo-family OOD (generate_data --ood) · GRPO reward    │
                │  • honesty/benchmarks: oracle_ceiling · differential_fuzz · │
                │    ood_benchmark · robustness_test · category_eval          │
                └─────────────────────────────────────────────────────────-─┘
```

**Principle: model proposes, physics disposes.** Layer A is the team's neural
pipeline (unchanged, on `main`). Layer B is this fork's value-add — it is
*model-agnostic*, so it composes with Layer A's decoder/encoder outputs **or**
with the simpler transformer in `src/`. Nothing in Layer B replaces Layer A.

### The seam is real and tested (not just claimed)
- **A → B (categories):** `solution/procseq/external.py` imports
  `physics.ontology.classify_step` + `STEP_CATEGORY` (with a safe identity
  fallback). Verified live: `CATEGORIZER_AVAILABLE=True`, procseq's 200-step vocab
  == our `known_vocab`, procseq's `RULE_IDS` == our 10 rules.
- **B over A (refine outputs):** `physics_postprocess.py` takes **any** model's
  `submissions/` (procseq's or ours) + the eval inputs and emits
  physics-refined CSVs — Task-1 re-ranked legal-first, Task-2 repaired to
  guaranteed-valid, Task-3 re-decided by the verified engine. Tested end-to-end on
  a synthetic procseq-style submission (invalid completion → repaired valid;
  illegal Top-1 → legal; anomaly → exact rule).
  Run: `python physics_postprocess.py --submission-dir <out>/submissions --eval-dir data`

## Which is "the submission"?
The **team's `solution/procseq`** models are the primary learned artifact (run on
Leonardo via `jobs/run.sh`, options 7/8). This fork **adds** the verification +
training-injection layer on top — the rule engine guarantees every emitted route
is physically valid, the injections raise the model's intrinsic understanding,
and `CEILING_ANALYSIS.md` explains why we optimise T2/T4 + physics rather than the
saturated next-token metric.

## How to run (this fork's layer)
```bash
cd tracks/industrial-infineon

# 1) Physics + correctness (pure stdlib, no install, runs anywhere):
python exhaustive_test.py            # 42/42 rule engine vs reference grader
python differential_fuzz.py --n 8000 # engine == grader, all 10 rules (incl. casing)
python oracle_ceiling.py             # the ~0.82 next-step Bayes ceiling, model-free
python ood_benchmark.py              # Task-4 OOD (token-transfer), 0-FP + 1-novel-token
python robustness_test.py            # no crash on empty/novel/over-long/malformed

# 2) Train OUR transformer with the injections (needs requirements.txt; GPU for full size):
python src/generate_data.py --extra-data 5000 --ood 3000 --output-dir outputs_run
OUTPUT_DIR=outputs_run python src/train.py --arch transformer --model-size medium \
    --epochs 100 --aux-category --unk-dropout 0.15 --device cuda
OUTPUT_DIR=outputs_run python src/train_grpo.py --init-from outputs_run/best_model.pt \
    --data-dir outputs_run --device cuda
python src/inference.py --output-dir outputs_run --eval-dir data    # -> submissions/
python data/eval_metrics.py --task next-step --ground-truth outputs_run/eval_set_valid.csv \
    --predictions outputs_run/submissions/nextstep.csv               # OFFICIAL scoring
# Leonardo (full run, all injections): sbatch jobs/leonardo/train.slurm
```

## How to run (the team's procseq layer)
```bash
# On Leonardo (see solution/README.md + jobs/run.sh options 7/8):
bash jobs/run.sh      # menu -> procseq decoder (7) / encoder (8), pixi -e procseq
```

## Clean-checkout smoke (no GPU, ~1 min) — proves the layer runs from a fresh clone
```bash
cd tracks/industrial-infineon
python exhaustive_test.py && python differential_fuzz.py --n 2000 && python oracle_ceiling.py
# expect: 42/42 · 0 disagreements · Top-1 ceiling ~0.82
```

## Key docs
- `CEILING_ANALYSIS.md` — why next-step caps at ~0.82 (and what we optimise instead).
- `HANDOFF_TRAINING.md` — the injection levers + the Leonardo recipe.
- `EVIDENCE_AUDIT.md` / `AUDIT_AND_PLAN.md` — the hostile, evidence-first reviews.
- `jobs/leonardo/LEONARDO.md` — secure connect + this fork's training job.
- `solution/README.md` — the team's procseq models (Layer A).
