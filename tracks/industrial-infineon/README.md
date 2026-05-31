# Industrial AI (Infineon) — Learning and Benchmarking Process Logic

## Quick Start (Reproduce Submissions)

```bash
# 1. Clone
git clone https://github.com/thuber19/zero_one_hack_01.git
cd zero_one_hack_01/tracks/industrial-infineon

# 2. Install dependencies (pick one)
pip install -r ../../requirements.txt          # option A: pip
# OR
pixi install                                   # option B: pixi (used on Leonardo)

# 3. Download model weights (~110MB from Dropbox)
bash download_models.sh

# 4. Run inference (produces all 3 submission files + physics hybrid)
cd solution
PYTHONPATH=$PWD python3 -m procseq.run_all --config configs/inference.yaml --skip-train
```

Output in `submissions/`:
- `nextstep.csv` (Task 1 — physics-reranked hybrid)
- `completion.csv` (Task 2 — grammar-constrained completion)
- `anomaly.csv` (Task 3 — physics-hybrid anomaly detection)

The eval input files are at `data/eval_input_valid.csv` and `data/eval_input_anomaly.csv`.
To run on different eval files, replace those files before running.

## Model Weights

Weights are hosted on Dropbox (too large for GitHub):
```bash
bash download_models.sh
```
This downloads decoder (~105MB) and encoder (~25MB) into `models/`.

## Architecture

Two from-scratch neural models wrapped in a physics verification layer:

- **Decoder** (Llama-style, 27M params): Next-step prediction (Task 1) + sequence completion (Task 2) with grammar-constrained decoding
- **Encoder** (DeBERTa-style): Anomaly detection (Task 3) with contrastive learning
- **Physics refinery**: Rule engine that guarantees every emitted route is physically valid ("model proposes, physics disposes")

## Results (official scorer, held-out self-eval — run `procseq_base_d20000_s16001_seed11101`)

Source of truth: [`solution/artifacts/metrics.json`](solution/artifacts/metrics.json)
(decoder/encoder alone) + [`solution/artifacts/metrics_hybrid.json`](solution/artifacts/metrics_hybrid.json)
(submitted hybrid). Full breakdown + honesty notes in the root [`REPORT.md`](../../REPORT.md).

| Task | Metric | Score |
|---|---|---|
| **1. Next-step** (hybrid) | Top-1 Accuracy | **93.7%** |
| | Top-3 / Top-5 | 100% / 100% |
| | MRR | 0.968 |
| | Next-operation (category) | **99.8%** |
| **2. Completion** | Block-level Accuracy | **93.7%** |
| | Logic Validity (rule-valid) | **100%** |
| | Token Accuracy | 70.8% |
| | Exact Match | 28.3% |
| **3. Anomaly** (physics hybrid) | F1 | **1.00** |
| | Rule attribution | **0.97** |
| | *Learned encoder alone* | *F1 0.0, AUC 0.49 (≈ chance — honest)* |

> Task 3's verdict is the deterministic rule engine (exact in-distribution by
> construction); the learned encoder alone is ≈ chance — we report both. See `REPORT.md`.

### Per-family (MOSFET / IGBT / IC)

| Family | T1 Top-1 | T2 Block | T2 Token | T3 F1 | T3 Rule-attr |
|---|---|---|---|---|---|
| MOSFET | 0.945 | 0.962 | 0.765 | 1.000 | 1.000 |
| IGBT | 0.955 | 0.951 | 0.786 | 1.000 | 1.000 |
| IC | 0.910 | 0.899 | 0.572 | 1.000 | 0.915 |

IC is the weakest learned family (most structurally distinct). Full output:
[`solution/artifacts/metrics_per_family.json`](solution/artifacts/metrics_per_family.json)
+ [`per_family_scores.txt`](solution/artifacts/per_family_scores.txt)
(reproduce: `python solution/per_family_eval.py --run <run-dir>`).

## Training on Leonardo

```bash
# Install pixi (one time)
curl -fsSL https://pixi.sh/install.sh | bash

# Setup environment
cd tracks/industrial-infineon
pixi install

# Launch training (interactive menu)
bash jobs/run.sh
# Option 3: Train procseq (decoder + encoder, all 3 tasks)
# Recommended: base model, 20000 sequences/family, 16000-20000 steps, batch 64
```

Training takes ~30 min on a single A100 GPU. Produces submission files automatically.

## Project Structure

```
tracks/industrial-infineon/
  data/                    # Training data, eval files, official scorer
    eval_input_valid.csv   # Official eval input (Tasks 1+2)
    eval_input_anomaly.csv # Official eval input (Task 3)
    eval_metrics.py        # Official scoring script
    generate_sequences.py  # Sequence generator (process grammar)
    generation_rules.md    # Grammar documentation + 10 forbidden rules
  solution/                # Procseq pipeline (decoder + encoder)
    procseq/               # Model code, training, inference
    configs/               # Training/inference configs
  src/                     # LSTM/Transformer pipeline (alternative approach)
  physics/                 # Physics rule engine
  refinery.py              # Physics-constrained decoding
  models/                  # Model checkpoints (download with download_models.sh)
  submissions/             # Final submission CSV files
  jobs/run.sh              # Leonardo job launcher
```

## Submission Files

In `submissions/`:
- `nextstep.csv` — Task 1: EXAMPLE_ID, RANK_1..5
- `completion.csv` — Task 2: EXAMPLE_ID, PREDICTED_SEQUENCE
- `anomaly.csv` — Task 3: EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE

## Credits

- **Infrastructure**: Leonardo supercomputer (CINECA), NVIDIA A100 GPUs
- **Libraries**: PyTorch, HuggingFace Transformers, scikit-learn
- **Team**: Tobias Huber, Mina Mikail, Khaled El Yamany, Fathy Shalaby
- **AI Tools**: Claude Code (Anthropic)
