# Implementation Plan: LSTM Baseline for Fab-Process Sequence Modeling

**Branch**: `khaled_experiments` | **Date**: 2026-05-30 | **Spec**: [specs/005-lstm-baseline/spec.md](./spec.md)

**Input**: Feature specification from `/specs/005-lstm-baseline/spec.md`

---

## Summary

Train a vanilla unidirectional LSTM (2–4 layers, hidden 256–512) on the same shared vocabulary, splits, and tokenizer produced by Spec 001. The model performs next-step token prediction (autoregressive, causal) and is evaluated through the identical `src/eval/sequence_metrics.py` harness used by Specs 001 and 002, producing a `metrics_lstm.json` and a merged `comparison_report.json` consumable by the Spec 004 GUI leaderboard. The LSTM is the honest baseline; it establishes the minimum bar the Transformer and BERT must meaningfully beat.

---

## Technical Context

**Language/Version**: Python 3.11

**Primary Dependencies**:
- PyTorch 2.x (CUDA 12.2 build)
- `torch.cuda.amp` for optional AMP (fp16/bf16)
- PyYAML (config loading)
- NumPy
- Standard library `hashlib`, `json`, `logging`

**Storage**:
- Vocab/tokenizer: `$WORK/checkpoints/001-gpt-fab/tokenizer.json` (read-only; owned by Spec 001)
- Splits: `$WORK/data/fab_sequences/splits.json` (read-only; owned by Spec 001)
- LSTM checkpoints: `$WORK/checkpoints/005-lstm-baseline/`
- Metrics output: `$WORK/checkpoints/005-lstm-baseline/metrics_lstm.json`
- Comparison report: `$WORK/reports/comparison.json`
- Training data CSVs: `tracks/industrial-infineon/training_data/`

**Testing**: pytest (unit tests for model shapes, tokenizer load, metrics schema)

**Target Platform**: Leonardo HPC — CINECA Booster partition; single A100 64 GB GPU (single-GPU mode primary; 4-GPU DDP optional)

**Performance Goals**:
- Full training run ≤ 2 hours wall-clock on 1× A100 under `boost_usr_prod`
- Debug smoke run ≤ 30 minutes on `boost_qos_dbg`
- Inference: next-step logits in < 10 ms per step on GPU

**Constraints**:
- Login node: no training, no GPU, no loops > 10 min CPU time
- Must not re-tokenize or re-split data; load Spec 001's artifacts exactly
- Fail-fast if `tokenizer.json` or `splits.json` are missing at job start
- Checkpoints must not exhaust `$WORK` quota; keep only 3 recent + best

**Scale/Scope**:
- ~1,500–3,000 unique sequences across 3 product families (IC / IGBT / MOSFET)
- Vocabulary ≈ 207 tokens (inherited from Spec 001)
- Sequence length: up to 256 tokens (matching Spec 001 `max_len`)

---

## Constitution Check

*GATE: Must pass before Phase 0. Re-check after Phase 1.*

| Principle | Check | Status |
|-----------|-------|--------|
| **Honest evaluation** | Evaluation uses the SAME `src/eval/sequence_metrics.py` module as Spec 001 and 002 — not a custom LSTM scorer. Same test split, same metrics (top-1, top-5, perplexity, memorization probe). Apples-to-apples. | PASS |
| **Reproducibility** | Fixed seeds (`PYTHONHASHSEED`, `torch.manual_seed`, `numpy.random.seed`, `torch.cuda.manual_seed_all`). `cudnn.deterministic=True`, `cudnn.benchmark=False`. Seed embedded in `metrics_lstm.json`. | PASS |
| **No test-set leakage** | Splits loaded from `$WORK/data/fab_sequences/splits.json` — never recreated. Training script asserts zero overlap between train and test sequence IDs at startup. | PASS |
| **Shared vocab contract** | SHA-256 of `tokenizer.json` and `splits.json` recorded in `metrics_lstm.json`. If Spec 001 retrains and vocab changes, hashes diverge — flagged in comparison report as "vocab drift detected". | PASS |
| **Complexity justified** | LSTM is explicitly simpler than the 85M-param Transformer. Justified as honest baseline, not gold-plated. | PASS |

---

## Architecture Decision

### Why 2–4 Layer LSTM, Hidden 256–512

The Transformer baseline (Spec 001) is 85M params, 12 layers, d_model=512. The LSTM must be **deliberately simpler** — its role is to show what a strong recurrent baseline achieves, not to match the Transformer's capacity.

| Config | Params | Est. training time (1× A100, 20 epochs) | Fits in 2h? |
|--------|--------|-----------------------------------------|-------------|
| 2-layer, hidden=256 | ~3M | ~20 min | Yes |
| 2-layer, hidden=512 | ~8M | ~35 min | Yes |
| 4-layer, hidden=512 | ~16M | ~60 min | Yes |
| 4-layer, hidden=1024 | ~55M | ~100 min | Marginal |

**Default**: 2-layer LSTM, hidden=512. Gives strong representational capacity for sequences ≤ 256 tokens while completing well within the 2h budget. 4-layer variant available via config flag for stretch experimentation.

**Architecture pipeline**:
```
input_ids  →  Embedding(vocab_size=207, embed_dim=128)
           →  Dropout(p=0.1)
           →  LSTM(input_size=128, hidden_size=512, num_layers=2,
                   batch_first=True, dropout=0.1, bidirectional=False)
           →  Dropout(p=0.1)
           →  Linear(512 → vocab_size=207)   # next-step logit projection
```

- **Embedding dim**: 128 (smaller than Transformer's 512 d_model; appropriate for vocab_size=207)
- **Unidirectional**: mandatory for autoregressive next-step prediction (causal)
- **Dropout**: 0.1 between layers and after the final LSTM output (inter-layer dropout handled natively by PyTorch LSTM when `num_layers > 1`; additional dropout applied to final output before projection)
- **Tied embeddings**: output projection weights tied to input embedding matrix (reduces params, standard practice for language models over small vocab)
- **Stretch P2 — classification head**: when `--task classification` is passed, a mean-pooling layer over LSTM hidden states feeds a separate `Linear(512 → num_yield_buckets)` head; gated by config flag `task: classification` in YAML; the LM head is frozen during classification fine-tuning

---

## Tokenizer Reuse (Fail-Fast Contract)

The LSTM uses Spec 001's tokenizer artifact verbatim. No re-tokenization, no re-fitting.

**Startup sequence** (implemented in `src/train/train_lstm.py`):

```python
TOKENIZER_PATH = Path(os.environ["WORK"]) / "checkpoints/001-gpt-fab/tokenizer.json"
SPLITS_PATH    = Path(os.environ["WORK"]) / "data/fab_sequences/splits.json"

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        f"Spec 001 tokenizer not found at {TOKENIZER_PATH}. "
        "Run Spec 001 training first, or copy tokenizer.json to that path."
    )
if not SPLITS_PATH.exists():
    raise FileNotFoundError(
        f"Spec 001 splits not found at {SPLITS_PATH}. "
        "Run Spec 001 data pipeline first."
    )

tokenizer_sha = sha256_file(TOKENIZER_PATH)
splits_sha    = sha256_file(SPLITS_PATH)
logger.info(f"tokenizer_sha={tokenizer_sha}  splits_sha={splits_sha}")
```

These SHA-256 hashes are embedded in `metrics_lstm.json` for drift detection.

---

## Data Pipeline

Reuse `src/data/sequences.py` (the `FabSequenceDataset` class shared with Spec 001) without modification. The dataset class:
- Loads tokenized sequences from the training CSVs under `tracks/industrial-infineon/training_data/`
- Applies the Spec 001 tokenizer (loaded from `tokenizer.json`) for token-ID mapping
- Returns `(input_ids, target_ids)` pairs for next-step LM (input_ids shifted right by 1 = target_ids)
- Reads train/val/test membership from `splits.json`
- Pads sequences to `max_len=256` with `[PAD]` (id=0); attention masks exclude pad positions

**DataLoader settings** (configurable via YAML):

```yaml
batch_size: 128          # per GPU; effective batch = 128 (single GPU default)
num_workers: 4
pin_memory: true
shuffle: true            # train only; val/test shuffle=false
```

Batch size 64–256 is the target range. 128 is the default; reduce to 64 if GPU memory is tight with 4-layer variant.

---

## Training Loop

**Optimizer**: AdamW — `lr=1e-3`, `betas=(0.9, 0.999)`, `weight_decay=0.01`, `eps=1e-8`

**LR Schedule**: Cosine decay with linear warmup
- Warmup: 5% of total steps
- Decay: cosine from `lr=1e-3` to `lr_min=1e-4`
- Implemented via `torch.optim.lr_scheduler.CosineAnnealingLR` with warmup wrapper

**Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` — applied before every optimizer step; `max_norm` configurable in YAML

**AMP (optional)**:
```python
if config.amp:
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        logits = model(input_ids)
        loss   = criterion(logits.view(-1, vocab_size), targets.view(-1))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(...)
    scaler.step(optimizer)
    scaler.update()
else:
    # full fp32; log warning if config.amp was True but hardware doesn't support it
    loss.backward()
    clip_grad_norm_(...)
    optimizer.step()
```
AMP off by default; `amp: false` in YAML.

**Epochs**: 20 max, or early stopping at patience=3 on validation loss

**Checkpoint cadence**:
- Save every 2 epochs to `$WORK/checkpoints/005-lstm-baseline/checkpoint_epoch{E:03d}.pt`
- Save best model as `checkpoint_best.pt` on every validation improvement
- Retain only 3 most recent epoch checkpoints + `checkpoint_best.pt`

**Checkpoint content**: model state dict, optimizer state dict, scheduler state dict, epoch, global step, best val loss, config dict, `tokenizer_sha`, `splits_sha`, seed

**Auto-resume**: if `$WORK/checkpoints/005-lstm-baseline/checkpoint_best.pt` exists and `--resume` flag set, load and continue

**Seed setup** (at job start, before any data loading):
```python
SEED = config.seed  # default 42
random.seed(SEED)
numpy.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)
```

---

## Eval Harness — Shared Module Contract

The LSTM evaluation MUST import the SAME module as Specs 001 and 002:

```python
from src.eval.sequence_metrics import EvalHarness

harness = EvalHarness(tokenizer=tokenizer, device=device)
metrics = harness.evaluate(model=model, dataloader=test_loader)
```

The `EvalHarness` class (owned by Spec 001) computes:

| Metric | Description |
|--------|-------------|
| `top1_accuracy` | Fraction of positions where argmax prediction = ground truth |
| `top5_accuracy` | Fraction of positions where ground truth in top-5 predictions |
| `perplexity` | `exp(mean cross-entropy loss)` over all non-PAD tokens |
| `probe_score` | Memorization probe: `mean_score_ratio = score(original) / score(perturbed)` |

If `src/eval/sequence_metrics.py` does not exist (Spec 001 not yet implemented), LSTM training is blocked — dependency is explicit.

**Output written to `$WORK/checkpoints/005-lstm-baseline/metrics_lstm.json`**:

```json
{
  "model": "lstm-baseline",
  "date": "2026-05-30T...",
  "top1_accuracy": 0.0,
  "top5_accuracy": 0.0,
  "perplexity": 0.0,
  "probe_score": 0.0,
  "tokenizer_sha": "<sha256>",
  "split_sha": "<sha256>",
  "seed": 42,
  "config": { ... full config dict ... }
}
```

---

## Comparison Report

`generate_comparison_report.py` aggregates metrics from all trained models into a single JSON consumed by the Spec 004 GUI leaderboard.

**Output path**: `$WORK/reports/comparison.json`

**Schema** (one entry per row, matching Spec 004 contract):

```json
{
  "models": [
    {
      "model": "lstm-baseline",
      "metric": "top1_accuracy",
      "value": 0.74,
      "ci_low": 0.72,
      "ci_high": 0.76
    },
    {
      "model": "lstm-baseline",
      "metric": "top5_accuracy",
      "value": 0.91,
      "ci_low": 0.89,
      "ci_high": 0.93
    },
    {
      "model": "transformer",
      "metric": "top1_accuracy",
      "value": null,
      "ci_low": null,
      "ci_high": null
    }
  ]
}
```

Row schema: `{model: str, metric: str, value: float|null, ci_low: float|null, ci_high: float|null}`

- CI bounds: bootstrapped 95% CI over the test set (1,000 bootstrap samples); falls back to `null` if fewer than 100 test samples available
- If a model's metrics file does not exist, all its metric rows have `value: null` (partial report is valid)
- The script reads from `$WORK/checkpoints/{model}/metrics_{model}.json` for each known model slug

---

## SLURM Scripts

### `scripts/slurm/lstm_debug.sh` — fast smoke test

```bash
#!/bin/bash
#SBATCH --job-name=lstm_debug
#SBATCH --account=<your_account>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/lstm_debug_%j.out
#SBATCH --error=logs/lstm_debug_%j.err

module load cuda/12.2
source $WORK/envs/fab_lstm/bin/activate

export PYTHONHASHSEED=42

python src/train/train_lstm.py \
  --config configs/lstm_baseline.yaml \
  --max_epochs 1 \
  --batch_size 32 \
  --debug
```

### `scripts/slurm/lstm_train.sh` — production run

```bash
#!/bin/bash
#SBATCH --job-name=lstm_train
#SBATCH --account=<your_account>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/lstm_train_%j.out
#SBATCH --error=logs/lstm_train_%j.err

module load cuda/12.2
source $WORK/envs/fab_lstm/bin/activate

export PYTHONHASHSEED=42

# Stage data to node-local SSD for faster I/O
cp -r $WORK/data/fab_sequences/ $TMPDIR/fab_sequences/

python src/train/train_lstm.py \
  --config configs/lstm_baseline.yaml \
  --data_dir $TMPDIR/fab_sequences \
  --output_dir $WORK/checkpoints/005-lstm-baseline

python generate_comparison_report.py \
  --output $WORK/reports/comparison.json
```

Single-GPU (`--ntasks-per-node=1`, `--gres=gpu:1`) is sufficient for the 2h budget with the 2-layer hidden=512 default config. 4-GPU DDP is available by changing those SLURM flags and adding `torchrun --nproc_per_node=4` prefix; single-GPU mode must work for debugging without `torchrun`.

---

## Project Structure Additions

### Documentation (this feature)

```text
specs/005-lstm-baseline/
├── spec.md          # Feature specification
├── plan.md          # This file
└── tasks.md         # Generated by /speckit-tasks (next step)
```

### Source Code

```text
src/
├── model/
│   └── lstm.py                    # NEW — LSTMModel class
├── train/
│   └── train_lstm.py              # NEW — training entry point
├── data/
│   └── sequences.py               # REUSED — FabSequenceDataset (owned by Spec 001)
└── eval/
    └── sequence_metrics.py        # REUSED — EvalHarness (owned by Spec 001)

configs/
└── lstm_baseline.yaml             # NEW — all hyperparameters

scripts/slurm/
├── lstm_debug.sh                  # NEW — boost_qos_dbg, 30 min, 1 GPU
└── lstm_train.sh                  # NEW — boost_usr_prod, 2 h, 1 GPU

generate_comparison_report.py      # NEW — aggregates metrics_*.json → comparison.json
```

**Structure Decision**: Single-project layout, extending Spec 001's `src/` tree. The LSTM model is a new module under `src/model/`; all data and eval code is shared without copying.

---

## Complexity Tracking

No constitution violations. The LSTM is simpler than the Transformer baseline by design — fewer params, shorter training time, no DDP required. The added complexity of the comparison report generator is justified as the direct interface contract with Spec 004.

---

## Phase Plan

### Phase 0 — Verify Tokenizer Artifact Exists (< 5 min, login node)

**Goal**: Confirm prerequisites from Spec 001 are in place before submitting any job.

**Steps**:
1. SSH to Leonardo login node
2. Verify `$WORK/checkpoints/001-gpt-fab/tokenizer.json` exists; print its SHA-256
3. Verify `$WORK/data/fab_sequences/splits.json` exists; check it has `train`, `val`, `test` keys
4. Verify `src/eval/sequence_metrics.py` exists in the repo (Spec 001 dependency)
5. Verify `src/data/sequences.py` exists and imports cleanly (quick `python -c "from src.data.sequences import FabSequenceDataset"`)
6. Create output directories: `$WORK/checkpoints/005-lstm-baseline/` and `$WORK/reports/`

**Go/No-Go**: All checks must pass before proceeding. If `tokenizer.json` is missing, block on Spec 001 completion.

---

### Phase 1 — Single-GPU Smoke on Debug QOS (≤ 30 min)

**Goal**: Confirm model trains, loss decreases, checkpoints are written, and metrics file is produced — before committing to the 2h production run.

**Steps**:
1. Write `src/model/lstm.py` with `LSTMModel` class (embedding → LSTM → linear projection)
2. Write `src/train/train_lstm.py` (SHA check, data loading, training loop, checkpoint saving)
3. Write `configs/lstm_baseline.yaml` with debug overrides
4. Write `scripts/slurm/lstm_debug.sh`
5. Submit: `sbatch scripts/slurm/lstm_debug.sh`
6. Monitor: `squeue -u $USER`; tail `logs/lstm_debug_*.out`

**Validation**:
- Loss decreases monotonically over the 1 debug epoch
- Checkpoint written to `$WORK/checkpoints/005-lstm-baseline/`
- `metrics_lstm.json` written with all required fields (values may be poor — 1 epoch only)
- No CUDA OOM; no import errors

---

### Phase 2 — Full Training Run (≤ 2 h wall-clock)

**Goal**: Train LSTM to convergence on full dataset; produce final checkpoint and metrics.

**Steps**:
1. Update `configs/lstm_baseline.yaml` to production settings (20 epochs, batch=128, hidden=512, 2 layers)
2. Submit: `sbatch scripts/slurm/lstm_train.sh`
3. Monitor job; inspect validation loss in log every ~10 min
4. On completion, verify `metrics_lstm.json` is written with non-null values for all metric keys

**Validation**:
- Training completes within 2h walltime (`sacct -j <jobid> --format=Elapsed`)
- `metrics_lstm.json` contains `top1_accuracy`, `top5_accuracy`, `perplexity`, `probe_score`, `tokenizer_sha`, `split_sha`, `seed`, `config`
- Checkpoint exists: `$WORK/checkpoints/005-lstm-baseline/checkpoint_best.pt`

---

### Phase 3 — Generate Comparison Report

**Goal**: Produce `comparison.json` consumable by the Spec 004 GUI.

**Steps**:
1. Write `generate_comparison_report.py` (reads `metrics_lstm.json`, `metrics_transformer.json` if present, computes bootstrap CIs, writes `comparison.json`)
2. Run: `python generate_comparison_report.py --output $WORK/reports/comparison.json`
3. Validate schema matches Spec 004 contract
4. Commit `metrics_lstm.json` reference copy and `comparison.json` to repo

**Validation**:
- `comparison.json` parses as valid JSON
- Each entry has `{model, metric, value, ci_low, ci_high}` keys
- LSTM rows have non-null values; Transformer rows have null values if not yet trained (partial report accepted)
- Spec 004 GUI can render the leaderboard without modification

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Vocab drift** — Spec 001 retrains and produces a new `tokenizer.json` | Medium | High — metrics not comparable | SHA-256 check at job start; flag in comparison report if SHA differs from reference; re-train LSTM from scratch if vocab changes |
| **Vanishing gradients on 100-step sequences** | Medium | Medium — slow convergence, poor accuracy | Gradient clipping at `max_norm=1.0` (default); monitor `grad_norm` in training log; reduce LR if norm is consistently near clip value |
| **OOM in 4-layer variant** | Low | Low — debug QOS catches it | Phase 1 smoke with debug batch=32 before committing to 4-layer config; fallback to 2-layer is the default |
| **`src/eval/sequence_metrics.py` not yet written** | Medium | High — blocks eval entirely | Phase 0 check; if missing, write a minimal stub that computes top-1 and perplexity only, then align with Spec 001 before final report |
| **Class imbalance in yield bucket (P2 stretch)** | High | Low (P2 only) | Log class distribution at classification head startup; apply `weight=` to `CrossEntropyLoss` based on inverse class frequency; support stratified sampling via config flag |
| **Non-determinism across runs** | Low | Low — tolerance is ±0.005 | `cudnn.deterministic=True`; document that CUDA atomic operations may cause minor variation; accepted within ±0.005 tolerance per SC-004 |
| **`$WORK` quota exhaustion** | Low | Medium — training stalls | Retain only 3 epoch checkpoints + best; check `cindata` before submitting production run |

---

## Stretch Goal — P2: Yield Bucket Classification Head

Gated by `task: classification` in `configs/lstm_baseline.yaml` (default: `task: lm`).

**Architecture addition** in `src/model/lstm.py`:
```python
if self.task == "classification":
    self.pool = MeanPooling()          # mean over non-PAD LSTM outputs
    self.cls_head = nn.Linear(hidden_size, num_yield_buckets)
```

**Training**:
- Freeze LM head; fine-tune classification head on top of a pre-trained LM checkpoint
- Loss: `CrossEntropyLoss(weight=class_weights)` where `class_weights` is inverse frequency
- Metrics added to `metrics_lstm.json`: `classification.macro_f1`, `classification.per_class`

**SLURM**: re-use `lstm_train.sh` with `--task classification --pretrained $WORK/checkpoints/005-lstm-baseline/checkpoint_best.pt`

Not on the critical path; implement only after P1 metrics are confirmed.
