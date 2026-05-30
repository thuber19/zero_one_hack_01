# Feature Specification: BERT-style MLM for Process-Step Anomaly Detection

**Feature Branch**: `002-bert-mlm-anomaly`

**Created**: 2026-05-30

**Status**: Draft

**Input**: User description: "Encoder-only BERT MLM trained on 100-step Infineon fab sequences; per-token reconstruction loss is the anomaly score."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - End-to-End MLM Training with Calibrated Threshold (Priority: P1)

A process engineer submits a SLURM job on Leonardo Booster that trains a BERT-style MLM on the Infineon fab-sequence dataset and produces (a) a checkpoint with validation perplexity logged and (b) a JSON file containing a calibrated anomaly threshold derived from the clean held-out set — all within 4 hours.

**Why this priority**: Without a trained checkpoint and a threshold, no inference or evaluation is possible. Everything downstream depends on this.

**Independent Test**: Run `sbatch scripts/train_mlm.sh` on `boost_usr_prod`; confirm job completes under 4 h; verify `$WORK/checkpoints/002/best_model.pt` and `$WORK/checkpoints/002/threshold.json` exist; validate log shows falling validation loss and non-trivial mask-prediction accuracy (>random baseline).

**Acceptance Scenarios**:

1. **Given** the dataset CSVs are present in `tracks/industrial-infineon/training_data/` and a conda env is activated, **When** the SLURM script is submitted with `boost_usr_prod` QOS on 1–4 A100 nodes, **Then** the job finishes within 4 hours, writes a checkpoint and a `threshold.json` with `p95_loss` and `p99_loss` keys.
2. **Given** the training run completes, **When** `validate.py --split val` is executed, **Then** masked-token prediction accuracy exceeds a random-guess baseline (1/|vocab|) by at least 20 percentage points absolute.
3. **Given** a debug test on `boost_qos_dbg` with reduced epochs and batch size, **When** the job runs for ≤30 min, **Then** the script exits with code 0 and the checkpoint is written — confirming the full pipeline is wired correctly before a production run.

---

### User Story 2 - Per-Step Anomaly Scoring at Inference (Priority: P1)

Given a process sequence (up to 100 steps), the inference script returns a per-step anomaly score (reconstruction loss) and a boolean sequence-level flag indicating whether the sequence likely contains an anomalous step.

**Why this priority**: This is the core deliverable; the anomaly detector has no value unless it can score novel sequences at inference time.

**Independent Test**: Run `python infer.py --sequence <path>` with a known clean sequence and a known anomalous sequence; confirm per-step scores are returned as a JSON/CSV; confirm the sequence-level flag is `False` for the clean input and `True` for the anomalous input.

**Acceptance Scenarios**:

1. **Given** a 100-step clean IC process sequence and a trained checkpoint, **When** `infer.py` is called, **Then** it returns a list of 100 per-step loss values, all below `threshold.json["p95_loss"]`, and the sequence flag is `False`.
2. **Given** a sequence with one rule-violating step substitution (per `generation_rules.md`), **When** `infer.py` is called, **Then** the anomalous step has the highest per-step score, and the sequence flag is `True`.
3. **Given** a sequence shorter than 100 steps, **When** `infer.py` is called, **Then** padding is handled transparently and scores are returned only for non-padding positions.

---

### User Story 3 - Tokenizer Compatibility with Spec 001 (Priority: P2)

The MLM tokenizer shares its vocabulary artifact with the autoregressive Transformer from Spec 001. A verification script confirms the vocabularies are identical (same token-to-id mapping) and that both models load the same `vocab.json` without modification.

**Why this priority**: A shared vocab reduces divergence in downstream tooling, avoids duplicate preprocessing, and enables a unified GUI signal. It is not a blocker for training but important for integration.

**Independent Test**: Run `python scripts/verify_tokenizer_compat.py`; it exits 0 and prints `Tokenizers compatible: True` plus the shared vocab size.

**Acceptance Scenarios**:

1. **Given** the Spec 001 tokenizer artifact is present at `$WORK/artifacts/001/vocab.json`, **When** `verify_tokenizer_compat.py` is executed, **Then** it loads both vocabularies, asserts identical token-to-id mappings, and prints the shared vocab size (expected: ~200–500 tokens for step-type + parameter tokens).
2. **Given** the Spec 001 artifact is absent, **When** `verify_tokenizer_compat.py` is executed, **Then** it exits with a clear error message (`Spec 001 vocab not found at <path>`) rather than a cryptic traceback.

---

### User Story 4 - Threshold Recalibration for New Variants (Priority: P3)

A process engineer runs a recalibration script against a small clean reference set from a new product variant (e.g., a previously unseen IGBT sub-variant), producing an updated `threshold.json` without retraining the model.

**Why this priority**: Fab engineers introduce new process variants over time; a fixed threshold from training may not generalise, but recalibration is much cheaper than retraining.

**Independent Test**: Provide a CSV of 50 clean sequences from a held-out IGBT variant; run `python scripts/recalibrate_threshold.py --data <csv> --checkpoint <ckpt>`; confirm a new `threshold.json` is written with updated p95/p99 values.

**Acceptance Scenarios**:

1. **Given** a checkpoint and a clean reference CSV of ≥10 sequences, **When** `recalibrate_threshold.py` is executed, **Then** it writes a `threshold.json` with `p95_loss`, `p99_loss`, and a `variant_tag` field reflecting the source CSV name.
2. **Given** a reference CSV with fewer than 10 sequences, **When** the script is run, **Then** it warns the user that the calibration set is small (n < 10) but still produces an output rather than aborting.

---

### Edge Cases

- **All-masked sequence**: If every step token in a sequence is masked (e.g., due to a bug in the masking pipeline), the model cannot use any context. The scoring pipeline MUST detect this and return a sentinel value (e.g., `loss = NaN`) rather than crashing, and log a warning.
- **Variant not seen during training**: A sequence from a product variant absent from the training set may yield uniformly high losses. The inference script MUST not assume the highest-loss step is the "root cause" without also reporting that the variant is OOD (out-of-distribution). An OOD flag based on global sequence-level perplexity > `threshold.json["ood_p99"]` SHOULD be included.
- **Ties at threshold**: When multiple steps share the exact same per-step loss and it equals the threshold, all must be flagged — not just the first. The sequence-level flag is `True` if any step's loss ≥ threshold.
- **Very short sequences (< 5 steps)**: Span masking of length 3 on a 4-step sequence leaves too little context. The masking strategy MUST fall back to single-token masking for sequences shorter than a minimum configurable length (default: 10 steps), and this MUST be logged.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Data & Tokenizer

- **FR-001**: The system MUST load sequences from `tracks/industrial-infineon/training_data/syntheticIC.csv`, `syntheticIGBT.csv`, and `synthetic_mosfet.csv`, plus supporting CSVs (`*_variants.csv`, `*_Longdescr.csv`, `*_longdescription_parameters.csv`).
- **FR-002**: The tokenizer MUST be reused from Spec 001 if the artifact exists at `$WORK/artifacts/001/vocab.json`; otherwise it MUST be built from the training corpus and saved to `$WORK/artifacts/002/vocab.json`, then a symlink or copy placed at `$WORK/artifacts/001/vocab.json` if absent.
- **FR-003**: The tokenizer vocab schema MUST include: step-type tokens (one per unique process step name), parameter-value tokens (discretised bins or raw strings), and special tokens `[MASK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[UNK]`.
- **FR-004**: Sequences MUST be truncated or padded to a fixed length of 100 tokens. Padding positions MUST be excluded from the MLM loss computation.

#### Masking Strategy

- **FR-005**: The default masking strategy MUST implement BERT-style random masking: 15% of non-padding step tokens selected at random; of those, 80% replaced with `[MASK]`, 10% replaced with a random token, 10% left unchanged.
- **FR-006**: A span-masking variant (spans of 1–5 consecutive tokens, total budget ≈15%) MUST be available as an optional training flag (`--masking-strategy span`), since contiguous-step anomalies are common in process flows.
- **FR-007**: Parameter masking (masking the parameter sub-tokens of a step while preserving the step-type token) MUST be available as `--masking-strategy param`, enabling the model to learn parameter-level expectations.
- **FR-008**: `[CLS]`, `[SEP]`, and `[PAD]` tokens MUST never be selected as masking targets.

#### Encoder Architecture

- **FR-009**: The default encoder architecture MUST be: 6 transformer layers, 8 attention heads, `d_model=256`, feed-forward dimension 1024, dropout 0.1. Rationale: a 100-step fab sequence has moderate complexity; a full BERT-base (12L/768D) is over-parameterised and would exceed the 4 h training budget on 1 A100; 6L/256D provides sufficient capacity while training ~3× faster.
- **FR-010**: Positional embeddings MUST be learned (not sinusoidal) to reflect that step position in a fab flow carries domain-specific meaning.
- **FR-011**: The architecture MUST be configurable via a single YAML config file (`configs/002_mlm.yaml`) so layer count, heads, and `d_model` can be swept without code changes.

#### Anomaly Scoring

- **FR-012**: At inference, the system MUST compute per-token cross-entropy loss for every non-padding position in the sequence by masking each token individually and measuring the model's reconstruction loss (pseudo-perplexity approach: each token is masked in turn; this is not the training procedure but the inference-time scoring procedure).
- **FR-013**: Per-step anomaly score MUST be normalised by sequence mean and standard deviation of reconstruction losses (z-score) to produce a distribution-agnostic score alongside the raw loss.
- **FR-014**: Sequence-level anomaly score MUST be `max(per_step_losses)` and also `mean(per_step_losses)` — both reported; the threshold comparison MUST use `max` by default.
- **FR-015**: Calibration MUST be performed on the clean validation split: compute reconstruction loss for every non-masked position, collect the distribution, and set `p95_loss` and `p99_loss` as calibration thresholds. An `ood_p99` threshold based on mean sequence-level loss MUST also be computed.
- **FR-016**: The calibration output MUST be stored as `threshold.json` with keys: `p95_loss`, `p99_loss`, `ood_p99`, `calibration_n` (number of sequences used), `calibration_date`, `variant_tag`.

#### Evaluation

- **FR-017**: The evaluation script MUST inject synthetic anomalies using the rules in `generation_rules.md`: (a) rule-violating step reorderings and (b) illegal step substitutions (replacing a valid step with one incompatible with the current process variant).
- **FR-018**: Evaluation MUST report: precision, recall, F1 at the p95 threshold; ROC-AUC over a sweep of thresholds; and per-variant breakdown (IC, IGBT, MOSFET).
- **FR-019**: A probe evaluation MUST be included: train a logistic regression on per-step MLM embeddings to predict step type from context; accuracy significantly above chance (>80%) indicates the model has learned step-type structure, not just frequencies.
- **FR-020**: The evaluation script MUST output a `results/002/eval_report.json` with all metrics and a `results/002/roc_curve.png`.

#### Training

- **FR-021**: Optimizer MUST be AdamW with weight decay 0.01, β1=0.9, β2=0.999. Learning rate: linear warmup for 10% of steps, then cosine decay to 0. Peak LR: 1e-4 (configurable).
- **FR-022**: Training MUST use mixed precision (BF16 on A100) via PyTorch AMP or native BF16. DDP MUST be used when `--nodes > 1` or when 4 GPUs are available on a single node.
- **FR-023**: Checkpoints MUST be saved every 500 steps and at each epoch end. The best checkpoint by validation masked-token accuracy MUST be symlinked to `$WORK/checkpoints/002/best_model.pt`.
- **FR-024**: A SLURM script `scripts/002_train_mlm.sbatch` MUST be provided, configured for 1 node, 4× A100, `boost_usr_prod`, 4 h walltime, with comments indicating where to change for `boost_qos_dbg` (30 min, debug).

#### Reproducibility

- **FR-025**: A global random seed (default: 42) MUST be set for Python, NumPy, and PyTorch at the start of training. Seed MUST be logged and included in `threshold.json`.
- **FR-026**: A `requirements.txt` or `environment.yaml` MUST be provided listing all dependencies with pinned versions.
- **FR-027**: The SLURM script MUST document the exact `module load` commands used (CUDA version, OpenMPI version) at the time of the hackathon run.

### Key Entities

- **ProcessSequence**: A list of up to 100 process steps, each step having a `step_type` (string) and zero or more named parameters with discrete or continuous values. Maps to one row in the CSV after tokenisation.
- **TokenizedSequence**: A fixed-length integer tensor of length 100, with padding, `[CLS]`, and `[SEP]` bookends, derived from a ProcessSequence.
- **MLMCheckpoint**: PyTorch state dict + config YAML + tokenizer vocab, bundled together as a versioned artifact under `$WORK/checkpoints/002/`.
- **AnomalyReport**: Per-step loss array + z-scores + sequence-level scores + boolean flag, serialised as JSON or CSV for downstream consumption (e.g., GUI).
- **CalibrationThreshold**: JSON artifact with statistical thresholds derived from the clean validation set.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Training completes within 4 hours on 1 node (4× A100) under `boost_usr_prod` QOS.
- **SC-002**: Masked-token prediction accuracy on the validation set exceeds random baseline by ≥20 percentage points absolute.
- **SC-003**: ROC-AUC ≥ 0.80 on the synthetic-anomaly evaluation set (rule-violating reorderings + illegal substitutions).
- **SC-004**: Precision ≥ 0.70 and Recall ≥ 0.65 at the p95 calibrated threshold on the synthetic-anomaly set.
- **SC-005**: Tokenizer compatibility verification script exits 0 when Spec 001 vocab artifact is present.
- **SC-006**: Inference on a single 100-step sequence completes in < 5 seconds on CPU (pseudo-perplexity scoring via 100 forward passes).
- **SC-007**: Recalibration script produces a new `threshold.json` in < 2 minutes on CPU given a reference set of 50 sequences.

---

## Assumptions

- The Infineon dataset CSVs (`syntheticIC.csv`, `syntheticIGBT.csv`, `synthetic_mosfet.csv`) are already present in `tracks/industrial-infineon/training_data/` on Leonardo `$WORK` and locally for development.
- Spec 001 tokenizer artifact may or may not exist at the time Spec 002 training begins; the pipeline handles both cases gracefully (see FR-002).
- Sequences are at most 100 steps long as stated in the brief; sequences longer than 100 steps are truncated with a warning (none are expected in the current dataset).
- The `generation_rules.md` file is the authoritative source of truth for valid step orderings and substitutions; the evaluation anomaly injection directly parses and violates these rules.
- GUI integration is out of scope for this feature; the output format (JSON/CSV anomaly report) is designed for easy ingestion but the GUI itself is not built here.
- Autoregressive Transformer (Spec 001), LSTM baseline, and constrained decoding are out of scope.
- Mixed-precision training assumes A100 GPUs with BF16 support; the code MUST fall back to FP16 AMP if BF16 is unavailable, with a logged warning.
- A single `boost_usr_prod` 4-hour job is sufficient for the expected dataset size (~tens of thousands of sequences); if the dataset is substantially larger, batch size or epoch count should be reduced rather than extending walltime beyond the hackathon budget.
