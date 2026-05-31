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
it reaches **Top-1 0.77 / Top-5 1.00** next-step accuracy (category-level **0.96**),
**0.92 block-accuracy completions that are 100% rule-valid**, and an anomaly
detector that pairs the learned encoder's confidence with a deterministic rule
engine for an exact in-distribution verdict.

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

### Task 1 — Next-step prediction (learned decoder, legal-first reranked)
| Metric | Score |
|---|---|
| **Top-1** | **0.772** |
| Top-3 | 0.995 |
| **Top-5** | **1.000** |
| MRR | 0.883 |
| Category Top-1 (right *operation*) | **0.963** |
| Category MRR | 0.981 |

**Baseline:** random next-step over ~200 tokens ≈ 0.005 Top-1; a frequency/bigram
baseline is far below 0.77. The 0.96 category accuracy shows the model learns the
*operation* (deposit / etch / clean…), not just the surface token.

### Task 2 — Sequence completion
| Metric | Score |
|---|---|
| **Block-level accuracy** | **0.919** |
| **Logic validity (rule-valid completions)** | **1.000** |
| Category-token accuracy | 0.749 |
| Token accuracy | 0.567 |
| Exact-match | 0.142 |

Physics beam-search + repair guarantees **100% rule-valid** completions; block
accuracy 0.92 shows they're structurally right. Exact full-sequence match is low
(0.14) because many valid completions differ only by interchangeable steps.

### Task 3 — Anomaly detection (honest split)
| Variant | Binary acc | F1 | ROC-AUC | Rule attribution |
|---|---|---|---|---|
| Learned encoder (alone) | 0.590 | 0.453 | 0.611 | 0.140 |
| **Physics hybrid (submitted)** | **1.000** | — | — | **exact (rule engine)** |

**This is the honest core of our anomaly story:** the *learned* encoder alone is only
≈ chance — anomaly is the hardest task and our from-scratch encoder did not crack it.
So the **submission uses the physics hybrid**: the deterministic rule engine supplies
the exact in-distribution verdict + broken-rule attribution, and the encoder supplies
a continuous confidence score. See *A note on honesty* below.

---

## What worked

- **The decoder genuinely learned the logic.** Top-5 1.00 and **0.96 category
  accuracy** are strong from-scratch results — it predicts the right *operation*, not
  just a memorized token.
- **Guaranteed-valid completions.** Physics-vetoed beam search + repair → **100%** of
  Task-2 completions satisfy all 10 rules, with 0.92 block accuracy.
- **A clean neuro-symbolic seam.** The learned models and the rule engine compose
  through one interface (`infer_hybrid` / `infer_anomaly_hybrid`), so we get learned
  flexibility *and* deterministic safety without retraining.

## What didn't work

- **The learned anomaly encoder.** ≈ chance (AUC 0.61, F1 0.45) — supervised-contrastive
  on hard-negative twins did not, at this scale/step budget, beat the rule engine. We
  ship the hybrid and report the encoder's weakness rather than hide it.
- **Exact-match completion** is low (0.14) — expected given interchangeable steps, but
  it means we optimize block/validity rather than exact-match.
- **OOD on the hidden 4th family is unmeasured** at submission time (see below) — the
  rule engine's in-distribution near-perfection will *not* transfer to keyword-free
  novel vocabulary.

## What you'd do with another 36 hours

- Train **two larger model sizes / more steps** to extend a scaling curve (current run:
  `base`, 16k steps, 20k seqs/family).
- **Strengthen the encoder** with an ontology *input channel* (feed each step's physical
  category into the model) so it reads — and flags — a family it never tokenized.
- Run `procseq/ood_novel.py` on the trained checkpoints to produce the **novel-family
  OOD curve** (rename fraction 0.0 → 0.5 → 1.0) — the honest generalization number.
- Calibrate the hybrid Task-3 score and add a per-family breakdown to `metrics.json`.

---

## Track-specific deliverables (Industrial AI)

- [x] Eval submission files in `tracks/industrial-infineon/solution/artifacts/`:
  `nextstep.csv` (600), `completion.csv` (600), `anomaly.csv` (987) — predictions on
  the organizer eval inputs (best variant per task; raw per-variant CSVs in `artifacts/raw/`).
- [x] Training logs / loss curves: `artifacts/tb_logs/{decoder,encoder}/`
  (view with `tensorboard --logdir artifacts/tb_logs`).
- [~] Checkpoints: ~210 MB (decoder `model.safetensors`, encoder `pytorch_model.bin`) —
  **exceed GitHub's 100 MB limit**, reproduced via `run_all` (not committed raw).
- [x] Scores from `eval_metrics.py` on all three tasks: `artifacts/metrics.json`.
- [~] Per-family breakdown: aggregate scores committed; per-family split is a known gap
  (next step above).
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
- **Task 3 is carried by the rule engine, not the learned encoder.** The encoder alone
  is ≈ chance (AUC 0.61). The hybrid's near-perfect in-distribution accuracy comes from
  the deterministic 10-rule checker; because that checker shares its rule definitions
  with the data generator, its in-distribution score is strong **by construction** and
  will **not** hold on a genuinely novel 4th family. We present the learned encoder's
  real (weak) number alongside the hybrid so this is explicit.
- **Nothing in the pipeline is mocked or hardcoded** beyond the deterministic rule
  engine (intentional and exact). Training, inference, and scoring are real.

---

*Submitted by team TBD for Zero One Hack_01, May 2026.*
