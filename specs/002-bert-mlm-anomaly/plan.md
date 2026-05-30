# Implementation Plan: BERT-style MLM for Process-Step Anomaly Detection

**Branch**: `khaled_experiments` | **Date**: 2026-05-30 | **Spec**: `specs/002-bert-mlm-anomaly/spec.md`

**Input**: Feature specification from `/specs/002-bert-mlm-anomaly/spec.md`

---

## Summary

Train a 6-layer, 8-head, d_model=256 encoder-only Transformer with BERT-style masked language modelling on Infineon fab sequences (IC / IGBT / MOSFET). At inference, pseudo-perplexity scoring (mask each token once, measure cross-entropy) yields a per-step anomaly score; a calibrated p95/p99 threshold converts that score into a binary sequence-level flag. The tokenizer vocabulary is shared with Spec 001 (loaded from `$WORK/artifacts/001/vocab.json`), ensuring identical token-to-id mappings across both models.

---

## Technical Context

**Language/Version**: Python 3.11

**Primary Dependencies**:
- PyTorch 2.2.x (bf16 AMP, DDP via torchrun)
- HuggingFace Transformers 4.40.x (BertConfig + BertForMaskedLM or custom encoder)
- HuggingFace Datasets / pandas for data loading
- scikit-learn 1.4.x (logistic regression probe, ROC-AUC, F1)
- matplotlib 3.8.x (ROC curve PNG)
- PyYAML 6.0 (config file)
- tqdm, jsonlines

**Storage**:
- Training data: `$WORK/data/fab_sequences/` (staged from `tracks/industrial-infineon/training_data/`)
- Splits: `$WORK/data/fab_sequences/splits.json` (shared with Spec 001 — same file, same sequence-level 80/10/10 split)
- Checkpoints: `$WORK/checkpoints/002/`
- Tokenizer artifact (primary): `$WORK/artifacts/001/vocab.json` (emitted by Spec 001)
- Tokenizer artifact (fallback): `$WORK/artifacts/002/vocab.json` (built by this spec if 001 artifact absent)
- Results: `results/002/eval_report.json`, `results/002/roc_curve.png`

**Testing**: pytest; unit tests run on login node (CPU, small synthetic sequences); integration tests submitted as `boost_qos_dbg` jobs

**Target Platform**: CINECA Leonardo Booster — NVIDIA A100 64 GB SXM4, CUDA 12.2, NVLink 3.0; login node for development/testing only

**Performance Goals**:
- Training: complete within 4 h on 1 node × 4 A100 (boost_usr_prod)
- Inference: single 100-step sequence scored in < 5 s on CPU (100 forward passes, pseudo-perplexity)
- Recalibration: new threshold.json from 50 sequences in < 2 min on CPU

**Constraints**:
- bf16 throughout (fall back to fp16 AMP with logged warning if BF16 unavailable)
- Login node: zero GPU usage, zero heavy preprocessing; only environment setup, file ops, job submission
- `$WORK` quota: checkpoint retention policy limits to 3 rolling epoch checkpoints + best_model symlink
- Lustre discipline: no `ls -l`/`find`/`du` on large directories; use `lfs find`; aggregate small files

**Scale/Scope**: ~tens of thousands of unique sequences across three variant CSVs; single-node DDP; one engineer end-to-end

---

## Constitution Check

*GATE: Must pass before implementation begins. Re-check after architecture finalisation.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **Honest evaluation** | PASS | Anomaly injection follows `generation_rules.md` exactly; synthetic anomalies are rule-violating, not random noise; p95/p99 thresholds are derived from clean held-out val, not train; test set is never touched during threshold calibration |
| **Reproducibility** | PASS | Global seed 42 for Python/NumPy/PyTorch; seed logged in threshold.json; splits.json shared with Spec 001 (created once, committed); environment.yaml with pinned versions; SLURM scripts checked in |
| **HPC discipline** | PASS | No GPU workloads on login node; all training via SLURM; data staged to `$TMPDIR` at job start; checkpoint rolling deletion to stay under quota; no `touch` to extend SCRATCH timestamps |
| **Leakage prevention** | PASS | Splits at sequence level, stratified by variant; calibration runs on val only; test set used only for final eval_report.json; same splits.json as Spec 001 enforces shared holdout |
| **Tokenizer contract** | PASS | vocab.json loaded from Spec 001 artifact path; compatibility verified by `scripts/verify_tokenizer_compat.py`; failure mode is explicit error, not silent fallback to a different mapping |
| **Simplicity** | PASS | 6L/256D encoder chosen over BERT-base — smaller, faster, sufficient for 100-step sequences; single YAML config for sweeps; no external serving infrastructure |
| **Modularity** | PASS | Shared eval utilities planned under `src/eval/shared.py` for reuse by Specs 001 and 005; anomaly report format (JSON/CSV) is GUI-ready without tight coupling |

---

## Architecture

### Encoder Specification

| Hyperparameter | Value | Justification |
|---|---|---|
| `n_layers` | 6 | 100-step sequences have moderate complexity; 12L BERT-base is 3× over-parameterised; 6L trains ~2× faster on same hardware |
| `d_model` | 256 | Matches Spec 001 tokenizer embedding dim for potential weight sharing; fits comfortably at bs=64×4 GPUs with bf16 |
| `n_heads` | 8 | d_head = 32; standard for this model size |
| `d_ff` | 1024 | 4× d_model; standard |
| `max_position` | 100 | Exactly covers the fixed sequence length (padded to 100 tokens) |
| `dropout` | 0.1 | Regularisation for small dataset |
| `vocab_size` | ~212 | ~200 step-type tokens + 5 special tokens ([MASK]=200, [PAD]=201, [CLS]=202, [SEP]=203, [UNK]=204) + variant tokens [IC], [IGBT], [MOSFET] already in Spec 001 vocab; exact count determined by loaded vocab.json |
| `positional_embeddings` | Learned | Fab step position carries domain-specific meaning; sinusoidal would impose equidistance bias inappropriate for process flows |

**Parameter count estimate**:
- Embedding table: 212 × 256 ≈ 54K params
- Positional embeddings: 100 × 256 ≈ 26K params
- Each encoder layer: self-attention (4 × 256 × 256) + FFN (256 × 1024 + 1024 × 256) ≈ 786K params
- 6 layers: ≈ 4.7M params
- MLM head (linear + bias): 256 × 212 ≈ 54K params
- **Total: ~5M parameters**

At bf16: ~10 MB per model replica. With AdamW optimizer states (2× params), activations, and batch: trivially fits all 4 A100s. The 4 h walltime constraint is driven by dataset iteration count, not memory.

**Config file**: `configs/002_mlm.yaml` — all hyperparameters defined here; no magic numbers in code.

### Tokenizer Compatibility Contract with Spec 001

Spec 001 emits its vocab to `$WORK/artifacts/001/vocab.json` with the following schema:
- Keys: token string (e.g., `"THERMAL OXIDATION"`, `"[MASK]"`)
- Values: integer token ID
- Special tokens at high IDs: [PAD], [BOS], [EOS], [UNK], variant tokens ([IC], [IGBT], [MOSFET])

Spec 002 loading logic (`src/tokenizer.py`):
```
1. If $WORK/artifacts/001/vocab.json exists → load it directly; verify [MASK], [PAD], [CLS], [SEP] are present
2. If [CLS] or [SEP] missing from Spec 001 vocab → extend with two new IDs appended at the end (do not renumber existing tokens)
3. If $WORK/artifacts/001/vocab.json absent → build vocab from scratch from training CSVs, save to $WORK/artifacts/002/vocab.json, then symlink ← $WORK/artifacts/001/vocab.json
```

**Spec 001 special-token alignment note**: Spec 001 uses [BOS]/[EOS]/[PAD]/[UNK] plus variant tokens; Spec 002 needs [CLS]/[SEP]/[MASK] additionally. The extension strategy (append new IDs without renumbering) guarantees all existing Spec 001 token IDs remain stable. The compatibility script checks that every token in Spec 001 vocab appears at the same ID in the Spec 002 vocab.

---

## Data Pipeline

### Source Files

All CSVs in `tracks/industrial-infineon/training_data/` (staged to `$WORK/data/fab_sequences/` at job start):
- `syntheticIC.csv`, `syntheticIGBT.csv`, `synthetic_mosfet.csv` — canonical reference sequences → held out in test set (as per Spec 001 FR-008)
- `IC_variants.csv`, `IGBT_variants.csv`, `MOSFET_variants.csv` — variant sequences → train/val/test source

### Split Strategy

- Shared `splits.json` with Spec 001: sequence-level 80/10/10 split, stratified by product family, created once and committed to repo
- The three synthetic*.csv reference sequences are forced into the test set
- Spec 002 reads `splits.json` directly; it does NOT recreate splits
- Assertion at startup: no sequence ID appears in both train and test sets (hard failure if violated)

### Tokenisation

Each sequence is tokenised as: `[CLS] step_1 step_2 ... step_N [SEP]` padded to 100 tokens with `[PAD]`.
- Padding positions are excluded from MLM loss (attention mask = 0 for PAD)
- [CLS], [SEP], [PAD] are never selected as masking targets (FR-008)
- Parameter tokens: for this spec, step-type tokens only (same as Spec 001 vocab); parameter sub-token masking (FR-007 `--masking-strategy param`) is reserved for later extension once parameter tokenisation is agreed across specs

### Data Paths on Leonardo

```
$WORK/data/fab_sequences/          # staged raw CSVs
$WORK/data/fab_sequences/splits.json  # shared with Spec 001
$TMPDIR/fab_sequences/             # per-job node-local copy (staged in SLURM prolog)
$WORK/artifacts/001/vocab.json     # primary tokenizer artifact
$WORK/artifacts/002/vocab.json     # fallback if 001 absent
$WORK/checkpoints/002/             # checkpoints + threshold.json
results/002/                       # eval outputs (committed to repo)
```

---

## Masking Strategy

**Default (FR-005)**: BERT-style random token masking
- 15% of non-[CLS]/[SEP]/[PAD] tokens selected uniformly at random
- Of those selected: 80% replaced with [MASK], 10% replaced with a random non-special token, 10% left unchanged
- Rationale: standard BERT masking is well-studied; for 100-step sequences with ~85 non-special tokens, 15% ≈ 12-13 tokens masked per sequence — sufficient signal without destroying too much context

**Span masking (`--masking-strategy span`)**: spans of 1–5 consecutive tokens, total budget ≈15% of non-special tokens; spans cannot cross [CLS]/[SEP] boundaries. Chosen upper bound of 5 because common process sub-flows (e.g., clean → deposit → pattern → etch) span 3–5 steps; span masking forces the model to learn sub-sequence coherence.

**Parameter masking (`--masking-strategy param`)**: reserved for future extension; stubs in config but raises NotImplementedError in this version with a clear message, since parameter sub-tokenisation is not yet defined in the shared vocab.

**Short-sequence fallback**: if sequence length (excluding [CLS]/[SEP]/[PAD]) < 10, span masking falls back to single-token masking. This is logged per FR edge case. Configurable via `min_seq_len_for_span` in `configs/002_mlm.yaml` (default: 10).

**All-masked guard**: if the masking pipeline accidentally selects all tokens (edge case bug), the dataset collator detects `sum(mask_labels != -100) == len(input_ids)` and returns a sentinel batch with `loss = NaN` flag, logs a WARNING, and skips the batch rather than crashing.

---

## Training Loop

### Optimizer and Schedule

- **AdamW**: lr=1e-4 (peak), β1=0.9, β2=0.999, weight_decay=0.01, eps=1e-8
  - No weight decay on embedding layers and LayerNorm parameters (separate parameter groups)
- **LR schedule**: linear warmup for 10% of total training steps → cosine decay to lr_min=1e-5
- **Gradient clipping**: max_norm=1.0 before each optimizer step

### Batch Size and DDP

- Per-device batch size: 64 sequences × 4 GPUs = **effective batch size 256**
- Gradient accumulation: 1 (no accumulation needed at this model size; increase to 2 if OOM occurs)
- DDP via `torchrun --nproc_per_node=4`; NCCL backend; fall back to gloo with logged warning if NCCL unavailable
- bf16 via `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`; no loss scaling needed for bf16
- fp16 AMP fallback: `torch.cuda.amp.GradScaler()` activated automatically if A100 bf16 support not detected

### Checkpoint Cadence

- Save every 500 steps: `$WORK/checkpoints/002/checkpoint_step{S:07d}.pt`
- Save at each epoch end: `$WORK/checkpoints/002/checkpoint_epoch{E:03d}.pt`
- Best checkpoint by validation masked-token accuracy: symlinked to `$WORK/checkpoints/002/best_model.pt`
- Retention: keep 3 most recent epoch checkpoints + best_model symlink; delete older epoch checkpoints automatically to respect `$WORK` quota
- Checkpoint content: model state dict, optimizer state dict, scheduler state dict, epoch, global step, best val accuracy, config dict, tokenizer vocab path, seed

### Epochs and Early Stopping

- Max 20 epochs; early stop if validation masked-token accuracy has not improved for 3 consecutive epochs
- Validation every epoch (not every N steps, to keep the logic simple in a hackathon context)

---

## Anomaly Scoring at Inference

### Pseudo-Perplexity Procedure (FR-012)

For a sequence of N non-padding tokens:
1. For each position i in [0, N):
   a. Replace token at position i with [MASK]
   b. Run one forward pass
   c. Record cross-entropy loss at position i: `loss_i = CE(logits[i], true_token_i)`
   d. Restore token i
2. Per-step raw anomaly score: `raw_score_i = loss_i`
3. Per-step z-score: `z_i = (loss_i - mean(losses)) / (std(losses) + 1e-8)`
4. Sequence-level score (max): `seq_score_max = max(raw_scores)` — used for threshold comparison
5. Sequence-level score (mean): `seq_score_mean = mean(raw_scores)` — also reported
6. OOD score: `seq_score_mean` compared against `ood_p99` threshold

This requires N forward passes per sequence. For N=100 on CPU, each pass takes ~50 ms → ~5 s total, meeting SC-006.

**Optimisation**: batch the 100 masked versions into a single forward pass (batch_size=100) by constructing a 100×100 matrix where row i has [MASK] at position i. This reduces 100 sequential passes to 1 batched pass, improving CPU inference to ~500 ms. Implement as `infer.py --batch-scoring` (default on); `--sequential-scoring` available for debugging.

### Output Format (AnomalyReport)

```json
{
  "sequence_id": "...",
  "per_step_raw_loss": [0.12, 0.08, ..., 2.41],
  "per_step_zscore": [-0.3, -0.5, ..., 3.1],
  "seq_score_max": 2.41,
  "seq_score_mean": 0.31,
  "is_anomalous": true,
  "is_ood": false,
  "anomalous_steps": [98],
  "threshold_used": {"p95_loss": 1.20, "p99_loss": 1.85, "ood_p99": 0.85},
  "warnings": []
}
```

### Calibration Script (`scripts/calibrate_threshold.py`)

- Runs pseudo-perplexity scoring on every non-masked position of the clean validation split
- No masking during calibration: uses a full-sequence pass with all tokens unmasked, measuring reconstruction loss by masking each token individually (same pseudo-perplexity procedure)
- Collects distribution of per-step losses and sequence-level mean losses
- Outputs `$WORK/checkpoints/002/threshold.json`:

```json
{
  "p95_loss": 1.20,
  "p99_loss": 1.85,
  "ood_p99": 0.85,
  "calibration_n": 1240,
  "calibration_date": "2026-05-30",
  "variant_tag": "all_variants",
  "seed": 42
}
```

### Recalibration Script (`scripts/recalibrate_threshold.py`)

- Accepts `--data <csv>` and `--checkpoint <ckpt>` arguments
- Warns if reference CSV has < 10 sequences but continues (does not abort)
- Writes `threshold.json` with updated p95/p99/ood_p99 and `variant_tag` set to the source CSV filename stem
- Runs on CPU in < 2 min for 50 sequences (SC-007)

---

## Evaluation

### Synthetic Anomaly Injection

Following `generation_rules.md` exactly:
- **Type A — Rule-violating reorderings**: swap two steps that have a defined ordering constraint (e.g., move a deposition step before the required clean step)
- **Type B — Illegal step substitutions**: replace a step with one incompatible with the current product variant (e.g., insert an IGBT-only step into an IC sequence)

Injection is implemented in `src/eval/anomaly_injector.py` which parses `generation_rules.md` directly (not hardcoded rules) so the evaluation automatically stays consistent as rules evolve.

### Metrics (FR-018)

| Metric | Computation |
|--------|-------------|
| Precision | TP / (TP + FP) at p95 threshold |
| Recall | TP / (TP + FN) at p95 threshold |
| F1 | Harmonic mean of precision and recall |
| ROC-AUC | Sweep threshold over all unique per-step losses; area under curve |
| Per-variant breakdown | Above metrics split by IC, IGBT, MOSFET |

Targets: ROC-AUC ≥ 0.80, Precision ≥ 0.70, Recall ≥ 0.65 (SC-003, SC-004).

### MLM Embedding Probe (FR-019)

- Extract [CLS] token embeddings from the trained encoder for all val-set sequences
- Train a logistic regression (scikit-learn, max_iter=1000, C=1.0) to predict step type from per-step last-hidden-state embeddings
- Report accuracy; target > 80% (significantly above chance of ~1/200 = 0.5%)
- Included in `eval_report.json` under `"probe_accuracy"`

### Shared Eval Module

`src/eval/shared.py` provides:
- `compute_roc_auc(scores, labels)` — shared with Spec 005 LSTM baseline
- `compute_precision_recall_f1(scores, labels, threshold)` — shared
- `inject_anomalies(sequences, rules_path, anomaly_types)` — shared
- `plot_roc_curve(fpr, tpr, auc, output_path)` — shared

Specs 001 and 005 import from this module; changes here affect all three specs and must be coordinated.

### Output Files

- `results/002/eval_report.json`: all metrics, per-variant breakdowns, probe accuracy, leakage checks, calibration metadata, checkpoint path, git commit hash, timestamp
- `results/002/roc_curve.png`: ROC curve for three variants + overall, saved as 300 DPI PNG

---

## SLURM Scripts

### Debug Script (`scripts/slurm/bert_debug.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=bert_mlm_debug
#SBATCH --account=<YOUR_ACCOUNT>          # replace with cineca account code
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg              # 30-min debug QOS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=logs/bert_debug_%j.out
#SBATCH --error=logs/bert_debug_%j.err

module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0   # verify exact name with: module avail openmpi

conda activate $WORK/envs/fab_bert

# Stage data to node-local SSD to avoid repeated Lustre access
cp -r $WORK/data/fab_sequences/ $TMPDIR/fab_sequences/

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

torchrun \
  --nproc_per_node=4 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  src/train_mlm.py \
    --config configs/002_mlm.yaml \
    --data-dir $TMPDIR/fab_sequences \
    --splits $WORK/data/fab_sequences/splits.json \
    --vocab $WORK/artifacts/001/vocab.json \
    --output-dir $WORK/checkpoints/002 \
    --max-epochs 2 \
    --debug          # reduces dataset to first 200 sequences for fast smoke test
```

### Production Script (`scripts/slurm/bert_train.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=bert_mlm_train
#SBATCH --account=<YOUR_ACCOUNT>          # replace with cineca account code
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_usr_prod             # production QOS; change to boost_qos_dbg for debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/bert_train_%j.out
#SBATCH --error=logs/bert_train_%j.err

module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0   # verify exact name with: module avail openmpi

conda activate $WORK/envs/fab_bert

# Stage data to node-local SSD
cp -r $WORK/data/fab_sequences/ $TMPDIR/fab_sequences/

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

torchrun \
  --nproc_per_node=4 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  src/train_mlm.py \
    --config configs/002_mlm.yaml \
    --data-dir $TMPDIR/fab_sequences \
    --splits $WORK/data/fab_sequences/splits.json \
    --vocab $WORK/artifacts/001/vocab.json \
    --output-dir $WORK/checkpoints/002 \
    --seed 42

# After training: run calibration and evaluation on rank 0 only
if [ "$SLURM_PROCID" -eq 0 ]; then
  python scripts/calibrate_threshold.py \
    --checkpoint $WORK/checkpoints/002/best_model.pt \
    --splits $WORK/data/fab_sequences/splits.json \
    --data-dir $TMPDIR/fab_sequences \
    --output $WORK/checkpoints/002/threshold.json

  python src/eval_mlm.py \
    --checkpoint $WORK/checkpoints/002/best_model.pt \
    --threshold $WORK/checkpoints/002/threshold.json \
    --splits $WORK/data/fab_sequences/splits.json \
    --data-dir $TMPDIR/fab_sequences \
    --rules tracks/industrial-infineon/generation_rules.md \
    --output-dir results/002/
fi
```

---

## Project Structure

### Documentation (this feature)

```text
specs/002-bert-mlm-anomaly/
├── spec.md          # Feature specification
└── plan.md          # This file
```

### Source Code Additions (relative to Spec 001 layout)

```text
configs/
└── 002_mlm.yaml              # All MLM hyperparameters; no magic numbers in code

src/
├── tokenizer.py              # Shared tokenizer loader (reads vocab.json; extends with [CLS]/[SEP]/[MASK] if needed)
├── train_mlm.py              # DDP training entry point for Spec 002
├── infer.py                  # Pseudo-perplexity inference; batch-scoring mode by default
├── validate.py               # Validation loop; reports masked-token accuracy
└── eval/
    ├── shared.py             # ROC-AUC, P/R/F1, anomaly injection, ROC plot — reused by Specs 001, 002, 005
    ├── anomaly_injector.py   # Parses generation_rules.md; implements Type A + Type B injections
    └── probe.py              # Logistic regression probe on per-step embeddings (FR-019)

scripts/
├── calibrate_threshold.py    # Derive p95/p99/ood_p99 from clean val set; write threshold.json
├── recalibrate_threshold.py  # Update threshold for new variant from small reference CSV
├── verify_tokenizer_compat.py # Assert Spec 001 and Spec 002 vocabs are identical (shared IDs)
└── slurm/
    ├── bert_debug.sh         # boost_qos_dbg, 30 min, 2 epochs, --debug flag
    └── bert_train.sh         # boost_usr_prod, 4 h, full training + calibration + eval

results/
└── 002/
    ├── eval_report.json      # All metrics; committed to repo after run
    └── roc_curve.png         # ROC curve; committed to repo after run

tests/
├── unit/
│   ├── test_masking.py       # Verify masking rates, short-sequence fallback, all-masked guard
│   ├── test_tokenizer.py     # Verify vocab loading, [CLS]/[SEP] extension, compat check
│   └── test_anomaly_score.py # Verify per-step loss, z-score, ties-at-threshold, OOD flag
└── integration/
    └── test_pipeline_smoke.py # End-to-end: tiny dataset, 1 epoch, CPU only — runs on login node
```

**Structure Decision**: Single-project layout extending Spec 001's existing `src/` tree. No new top-level package — Spec 002 adds modules to the same `src/` directory to maximise code reuse and avoid import path complexity during the hackathon.

---

## Phase Plan

### Phase 0 — Environment and Data (Day 1, login node, ~1 h)

1. SSH into Leonardo login node; verify `$WORK` and `$FAST` mounts
2. Create conda env: `conda create --prefix $WORK/envs/fab_bert python=3.11`; install PyTorch 2.2 + transformers 4.40
3. Stage CSVs: `rsync -av tracks/industrial-infineon/training_data/ $WORK/data/fab_sequences/`
4. Run `python scripts/verify_tokenizer_compat.py` — expect error ("Spec 001 vocab not found") until Spec 001 runs first; or build vocab from scratch with `python src/tokenizer.py --build --data-dir $WORK/data/fab_sequences/`
5. Verify splits.json exists (created by Spec 001 pipeline) or generate it here using the same 80/10/10 stratified logic

### Phase 1 — Debug Run (Day 1 evening, ~30 min job + 30 min analysis)

1. Submit `sbatch scripts/slurm/bert_debug.sh`
2. Monitor: `squeue -u $USER`; tail `logs/bert_debug_<jobid>.out`
3. Verify: job exits code 0; checkpoint written to `$WORK/checkpoints/002/`; no OOM; masked-token loss decreasing across 2 epochs
4. Fix any wiring issues before production run

### Phase 2 — Production Training (Day 1 night → Day 2 morning, 4 h job)

1. Submit `sbatch scripts/slurm/bert_train.sh`
2. Calibration and eval run automatically in SLURM prolog after training (rank 0 only)
3. Pull `results/002/eval_report.json` and `results/002/roc_curve.png` to local machine for review
4. Check SC-003 (ROC-AUC ≥ 0.80) and SC-004 (P ≥ 0.70, R ≥ 0.65); if not met, see Risk section

### Phase 3 — Integration and Verification (Day 2, ~1 h)

1. Run `python scripts/verify_tokenizer_compat.py` with both Spec 001 and Spec 002 artifacts present; confirm exit 0
2. Run `python infer.py` on a clean and an anomalous test sequence; confirm JSON output and flags
3. Run `python scripts/recalibrate_threshold.py` on a held-out IGBT sub-variant CSV; confirm updated threshold.json
4. Run `pytest tests/unit/ tests/integration/` on login node (CPU, small synthetic data)
5. Commit `results/002/eval_report.json`, `results/002/roc_curve.png`, and `$WORK/checkpoints/002/threshold.json` path to submission notes

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Spec 001 vocab artifact not ready when Spec 002 training begins | Medium | Spec 002 builds its own vocab from the same CSVs (FR-002 fallback); compat script verifies post-hoc that both vocabs agree |
| ROC-AUC < 0.80 on synthetic anomaly set | Medium | Two levers: (a) switch to span masking (`--masking-strategy span`) which better captures contiguous-step anomalies; (b) increase epochs from 20 to 30 within the 4 h budget (5M param model is fast) |
| pseudo-perplexity inference too slow for demo | Low | Batch scoring mode (100 masked sequences in one forward pass) brings CPU inference to ~500 ms; GPU inference < 50 ms |
| Dataset too small for anomaly detection generalisation | Low | Dataset has ~tens of thousands of sequences across three variant files; at 5M params this is a high data-to-param ratio; augmentation via `generate_sequences.py` (Spec 001 tool) available if needed |
| SLURM job hits 4 h walltime before calibration runs | Low | Calibration is < 5 min; with 20 epochs on 5M params / 4 GPUs, training should finish in ~1 h leaving ample margin |
| `generation_rules.md` ambiguous for anomaly injection | Low | Anomaly injector parses the rules file directly; if a rule is ambiguous, the injector raises an explicit ValueError with the offending rule line rather than silently injecting a non-anomalous "anomaly" |

---

## Open Questions

None that block implementation — all design decisions are resolved in this plan. Decisions that were made unilaterally:

1. **Vocab extension strategy** (append [CLS]/[SEP] at end of Spec 001 vocab rather than inserting mid-range) is the safest approach since it preserves all existing Spec 001 token IDs. If Spec 001 is updated to include [CLS]/[SEP] natively, the extension code becomes a no-op.

2. **Batch pseudo-perplexity** (construct 100-row batched input rather than 100 sequential passes) is the default inference mode. This is non-obvious but well-established in the anomaly detection literature and necessary to meet the < 5 s CPU requirement.

3. **Shared `splits.json` with Spec 001** is assumed to already exist or be co-created. If Spec 001 has not yet been run, Spec 002's Phase 0 creates splits.json using the same 80/10/10 stratified logic and commits it, so Spec 001 can pick it up without conflict.

---

## Complexity Tracking

No constitution violations requiring justification. The 5M-parameter encoder is well within the simplest viable architecture for this dataset size and task.
