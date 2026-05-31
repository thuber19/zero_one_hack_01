# TBD — Industrial AI (Infineon): Learning & Benchmarking Process Logic

## Team

- **Fathy Shalaby** — neural pipeline (procseq): from-scratch decoder + encoder, training on Leonardo
- **Mina Mikail** — physics / verification layer, hybrid inference, evaluation harness
- **Tobias Huber** — data pipeline, infrastructure & Leonardo orchestration
- **Khaled El Yamany** — interactive dashboard, results visualization, and demo

**Track:** Industrial AI (Infineon) — process-sequence learning & benchmarking
**Team name:** TBD

---

## TL;DR

We built **procseq** — two from-scratch neural models that *learn* semiconductor
manufacturing process logic — wrapped in a symbolic **physics verification layer**
(*"the model proposes, physics disposes"*). A Llama-style **decoder** handles
next-step prediction and sequence completion; a DeBERTa-style **encoder** with a
supervised-contrastive objective handles anomaly detection. On held-out evaluation
the submitted hybrid reaches **Top-1 0.94 / Top-5 1.00** next-step accuracy
(category-level **1.00**), **0.94 block-accuracy completions that are 100% rule-valid**,
and a physics-hybrid anomaly detector at **F1 1.00 with 0.97 rule-attribution** — while
we report openly that the *learned* anomaly encoder alone is ≈ chance.

---

## Problem

Semiconductor fabrication follows strict process routes where **order is logic**:
you must clean a surface before depositing on it, pattern (litho) before you etch,
open an implant window before implanting, and test before you ship. The track asks
whether a model can learn this *hidden logic* — not memorize sequences — across
three product families (**MOSFET, IGBT, IC**) and generalize to an unseen **4th
family**, on three tasks:

1. **Next-step prediction** — given a partial route, predict the next step (Top-5).
2. **Sequence completion** — finish a partially-executed route.
3. **Anomaly detection** — flag routes that violate the 10 process-logic rules, and
   attribute the broken rule.

Our angle: **don't fine-tune a generic LLM** — the vocabulary is ~200 atomic process
steps, so we train *purpose-built* small models from scratch and pair them with a
symbolic layer that guarantees physical validity rather than hoping the model learns it.

---

## Approach

- **From-scratch decoder (Tasks 1 & 2).** A small Llama-style causal LM over a
  ~200-token atomic-step vocabulary. Short, strictly-ordered sequences don't need a
  50K-token pretrained model; a purpose-built one trains fast and predicts the next
  step / completes the route.
- **From-scratch encoder + supervised-contrastive (Task 3).** A DeBERTa-style encoder
  trained with a contrastive objective on hard-negative "twins" (valid routes vs.
  minimally-broken ones) to *learn* what an anomaly looks like — a neural result, not
  a hand-written rule.
- **Physics hybrid — "model proposes, physics disposes."** The learned models are
  wrapped by a deterministic rule engine + refinery (`refinery.PhysicsRefinery`,
  `physics.ontology`): Task 1 is re-ranked legal-first, Task 2 uses physics
  beam-search + repair so **every completion is rule-valid**, and Task 3 takes the
  rule engine's exact verdict + the encoder's continuous confidence score. The
  symbolic layer is the **verification companion** — it makes outputs guaranteed-valid
  and the submission honest.
- **Where it runs.** Trained on the **Leonardo** A100 cluster (pixi `-e procseq`,
  `torch==2.3.1` CUDA-12); inference + scoring run anywhere.

---

## How to run it

Full setup, Leonardo runbook, and architecture are in
[`tracks/industrial-infineon/solution/README.md`](tracks/industrial-infineon/solution/README.md).

```bash
git clone https://github.com/thuber19/zero_one_hack_01.git
cd zero_one_hack_01
pip install -r requirements.txt          # procseq deps (torch, transformers, accelerate, …)

cd tracks/industrial-infineon/solution
make smoke                               # CPU, ~30s: unit tests + tiny train + infer + score → "SMOKE OK"

# Full pipeline (GPU; Leonardo): train both models, infer all 3 tasks, score
python -m procseq.run_all --config configs/leonardo_decoder.yaml
# inference-only on existing checkpoints:
python -m procseq.run_all --config configs/leonardo_decoder.yaml --skip-train
```

**Needs:** a CUDA GPU for full training (we used Leonardo A100s). No API keys, no
external datasets — the synthetic data is generated in-repo from the organizer's
`generate_sequences.py`. Trained checkpoints (~210 MB) exceed GitHub's 100 MB
file limit and are therefore **not committed**; they are reproduced by the command
above, and the produced **submission CSVs, scores, and training logs are committed**
under [`tracks/industrial-infineon/solution/artifacts/`](tracks/industrial-infineon/solution/artifacts/).

---

## Results

Scores from the official `data/eval_metrics.py` on a **held-out self-eval split**
(600 valid routes, 987 anomaly routes). The submitted CSVs are the same models'
predictions on the **organizer eval inputs** (`*_real`), in `solution/artifacts/`.
Raw scores: [`solution/artifacts/metrics.json`](tracks/industrial-infineon/solution/artifacts/metrics.json).

### Task 1 — Next-step prediction
| Variant | Top-1 | Top-3 | Top-5 | MRR | Category Top-1 |
|---|---|---|---|---|---|
| Decoder alone | 0.877 | 1.000 | 1.000 | 0.938 | 0.990 |
| **+ physics legal-first rerank (submitted)** | **0.937** | **1.000** | **1.000** | **0.968** | **0.998** |

**Baseline:** random next-step over ~200 tokens ≈ 0.005 Top-1; a frequency/bigram
baseline is far below. Physics reranking lifts Top-1 by +0.06 (it floats legal
candidates up). The **0.998 category accuracy** shows the model learns the *operation*
(deposit / etch / clean…), not just the surface token.

### Task 2 — Sequence completion
| Metric | Score |
|---|---|
| **Block-level accuracy** | **0.937** |
| **Logic validity (rule-valid completions)** | **1.000** |
| Category-token accuracy | 0.834 |
| Token accuracy | 0.708 |
| Exact-match | 0.283 |

Physics beam-search + repair guarantees **100% rule-valid** completions; block
accuracy 0.94 shows they're structurally right. Exact full-sequence match (0.28) is
lower because many valid completions differ only by interchangeable steps.

### Task 3 — Anomaly detection (honest split)
| Variant | Binary acc | Precision | Recall | F1 | Rule attribution | ROC-AUC |
|---|---|---|---|---|---|---|
| Learned encoder (alone) | 0.608 | 0.000 | 0.000 | 0.000 | 0.000 | 0.487 |
| **Physics hybrid (submitted)** | **1.000** | **1.000** | **1.000** | **1.000** | **0.972** | 0.49 † |

The **submitted Task-3 is the physics hybrid** — rule-engine verdict + encoder
confidence score. The rule engine delivers a perfect in-distribution verdict
(precision/recall/F1 = 1.0) and **0.972 rule-attribution**. The *learned* encoder alone
is **degenerate** (it predicts everything valid → F1 0, AUC 0.49 ≈ chance), which we
report openly. † ROC-AUC is computed from the encoder's *uncalibrated* SCORE, so it sits
at chance even though the binary verdict is perfect — calibrating that score (see next
steps) is exactly what would lift it.

### Per-family breakdown (MOSFET / IGBT / IC)
Same scorers, split by product family — these rows aggregate (n-weighted) back to the
headline numbers above *exactly* (verified). Full output:
[`solution/artifacts/metrics_per_family.json`](tracks/industrial-infineon/solution/artifacts/metrics_per_family.json)
+ [`per_family_scores.txt`](tracks/industrial-infineon/solution/artifacts/per_family_scores.txt);
reproduce with `python solution/per_family_eval.py --run <run-dir>`.

| Family | T1 Top-1 | T1 MRR | T2 Block | T2 Token | T2 Exact | T3 F1 | T3 Rule-attr |
|---|---|---|---|---|---|---|---|
| MOSFET | 0.945 | 0.973 | 0.962 | 0.765 | 0.365 | 1.000 | 1.000 |
| IGBT | 0.955 | 0.978 | 0.951 | 0.786 | 0.365 | 1.000 | 1.000 |
| IC | 0.910 | 0.955 | 0.899 | 0.572 | 0.120 | 1.000 | 0.915 |

**MOSFET and IGBT are close and strong; IC is the weakest learned family** (T1 0.910,
T2 token 0.572, exact 0.120) — IC routes are the most structurally distinct (shortest,
fewest repeated litho cycles), so the decoder has the least pattern to lean on. Task-3
F1 is 1.0 for all three (the rule engine is family-agnostic); rule-attribution is
perfect for MOSFET/IGBT and 0.915 for IC.

---

## What worked

- **The decoder genuinely learned the logic.** Top-5 1.00, Top-1 0.94 (hybrid), and
  **0.998 category accuracy** are strong from-scratch results — it predicts the right
  *operation*, not just a memorized token.
- **Guaranteed-valid completions.** Physics-vetoed beam search + repair → **100%** of
  Task-2 completions satisfy all 10 rules, with 0.94 block accuracy.
- **A clean neuro-symbolic seam.** The learned models and the rule engine compose
  through one interface (`infer_hybrid` / `infer_anomaly_hybrid`), so we get learned
  flexibility *and* deterministic safety without retraining.

## What didn't work

- **The learned anomaly encoder.** ≈ chance — it collapses to predicting "valid" for
  everything (F1 0, AUC 0.49); supervised-contrastive on hard-negative twins did not, at
  this scale/step budget, beat the rule engine. We ship the hybrid and report the
  encoder's weakness rather than hide it.
- **Exact-match completion** is low (0.28) — expected given interchangeable steps, but
  it means we optimize block/validity rather than exact-match.
- **OOD on the hidden 4th family is unmeasured** at submission time (see below) — the
  rule engine's in-distribution near-perfection will *not* transfer to keyword-free
  novel vocabulary.

## What you'd do with another 36 hours

Targeted at our actual gaps, in priority order:

- **Measure the OOD cliff — the one number we're missing.** Task 4 is graded post-submission
  on a hidden 4th family, and our hybrid's near-perfect Task-3 verdict is in-distribution
  *by construction* — it won't hold there. First action: run `procseq/ood_novel.py` on our
  trained checkpoints across rename-fraction 0.0 → 0.5 → 1.0 to turn the current "pending"
  into a real degradation curve, so we know exactly where the rules stop helping.
- **Make the learned encoder actually contribute to Task 3 (our weakest result, AUC 0.49).**
  Right now the rule engine carries the verdict and the model adds nothing to the scored
  anomaly output. Two concrete lifts: (a) **calibrate** the encoder's probability
  (temperature scaling on a held-out split) so the hybrid's `SCORE` column is a true
  P(valid) — this raises the submitted `anomaly.csv` ROC-AUC with *zero* retraining; and
  (b) retrain it with **harder contrastive negatives** (multi-rule and near-window breaks)
  to push AUC above chance *where the rules can't fire* — i.e., on OOD, the only place its
  value is real.
- **Add the ontology input channel — highest-leverage fix for the OOD gap.** We already
  show the decoder learns the *operation* (0.998 next-category) but it tokenizes raw step
  *names*, so a novel family is all-`[UNK]`. Feeding each step's physical category as a
  second input embedding lets the models read — and the encoder flag — a family they never
  tokenized, directly improving Task-1/2 and anomaly on the hidden 4th family.
- **Lift IC, the weakest family.** The per-family breakdown (now committed) shows IC
  trailing on every learned metric — T1 0.910 vs IGBT 0.955, T2 token 0.572 vs 0.786,
  exact 0.120 vs 0.365. IC routes are the most structurally distinct, so it's the clearest
  target for more IC-weighted training data or the ontology channel above.
- **Chart the scaling curve + ship the weights.** Train `small`/`base`/`large` on identical
  data for an accuracy-vs-compute curve (the track's stretch goal; we have a single `base`,
  16k-step run), and host the ~210 MB checkpoints externally (GitHub blocks LFS on a fork)
  so the trained weights are downloadable, not only reproducible.

---

## Track-specific deliverables (Industrial AI)

- [x] Eval submission files in `tracks/industrial-infineon/solution/artifacts/`:
  `nextstep.csv` (600), `completion.csv` (600), `anomaly.csv` (987) — predictions on
  the organizer eval inputs (best variant per task; raw per-variant CSVs in `artifacts/raw/`).
- [x] Training logs / loss curves: `artifacts/tb_logs/{decoder,encoder}/`
  (view with `tensorboard --logdir artifacts/tb_logs`).
- [x] Checkpoints: decoder + encoder in `tracks/industrial-infineon/models/`
  (also on Dropbox via `download_models.sh`); reproducible via `run_all`.
- [x] Scores from `eval_metrics.py` on all three tasks: `artifacts/metrics.json`
  (decoder/encoder) + `artifacts/metrics_hybrid.json` (submitted hybrid).
- [x] Per-family breakdown (MOSFET / IGBT / IC, all 3 tasks):
  `artifacts/metrics_per_family.json` + `per_family_scores.txt`
  (reproduce: `python solution/per_family_eval.py --run <run-dir>`).
- [ ] **Demo video (≤2 min): pending** — shows baseline vs. hybrid on identical inputs.

---

## Credits & dependencies

- **Open-source libraries:** PyTorch 2.3.1 (CUDA 12), Transformers ≥4.44, Accelerate
  ≥0.33, Tokenizers ≥0.19, TensorBoard, NumPy, scikit-learn, Matplotlib, PyYAML.
- **Pre-trained models:** **none** — both models trained from scratch (architectures
  only: Llama-style decoder, DeBERTa-style encoder).
- **External APIs:** none.
- **AI coding assistants used during the hackathon:** Claude Code.
- **Datasets:** organizer-provided synthetic process sequences
  (`data/generate_sequences.py` + the MOSFET/IGBT/IC variant CSVs), generated in-repo.
- **Compute:** CINECA **Leonardo** A100 cluster.

---

## A note on honesty

- **Reported numbers are held-out self-eval, not the organizer's hidden test.** The
  submitted CSVs are our models' predictions on the organizer eval inputs; we cannot
  see those labels, so the tables above are our best honest estimate from a held-out split.
- **Task 3's near-perfect accuracy is the rule engine, not the model.** The hybrid's
  in-distribution score is strong **by construction** (the checker shares rule
  definitions with the data generator) and will **not** transfer to a novel 4th family;
  the learned encoder alone is ≈ chance (AUC 0.49, collapses to predicting "valid").
- **Nothing in the pipeline is mocked or hardcoded** beyond the deterministic rule
  engine (intentional and exact). Training, inference, and scoring are real.

---

*Submitted by team TBD for Zero One Hack_01, May 2026.*
