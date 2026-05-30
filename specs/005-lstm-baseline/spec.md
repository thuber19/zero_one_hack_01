# Feature Specification: LSTM Baseline for Fab-Process Sequence Modeling

**Feature Branch**: `005-lstm-baseline`

**Created**: 2026-05-30

**Status**: Draft

**Input**: User description: "Vanilla LSTM next-step classifier over the 100-step Infineon fab pipeline. The honest baseline that the Transformer (Spec 001) and BERT MLM (Spec 002) must meaningfully beat. Required by the project constitution's honest-evaluation principle."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Engineer Trains LSTM and Produces Comparable Metrics Report (Priority: P1)

An engineer wants to train the LSTM baseline using the same vocabulary, splits, and tokenizer shared with Specs 001 and 002, then produce a metrics report (next-step top-1/top-5 accuracy, perplexity, real-logic probe) that is directly comparable to the other models.

**Why this priority**: This is the core deliverable. Without an apples-to-apples baseline, the project constitution's honest-evaluation principle cannot be satisfied, and the Transformer/BERT results have no meaningful reference point.

**Independent Test**: Can be fully tested by running `train_lstm.py --config configs/lstm_baseline.yaml` on the shared splits and verifying the output `metrics_lstm.json` contains top-1 accuracy, top-5 accuracy, perplexity, and probe score.

**Acceptance Scenarios**:

1. **Given** the shared vocab file and split files exist under `tracks/industrial-infineon/training_data/`, **When** the engineer runs the LSTM training script with default config, **Then** training completes in under 2 hours on a single A100 and writes `metrics_lstm.json` to `$WORK/checkpoints/005-lstm-baseline/`.
2. **Given** `metrics_lstm.json` exists, **When** it is loaded by the evaluation harness module shared with Spec 001 and 002, **Then** all metric keys match the schema expected by the GUI (Spec 004) and the comparison report.
3. **Given** a completed training run, **When** the model checkpoint is reloaded and evaluated, **Then** perplexity and top-1/top-5 accuracy values are reproducible within floating-point tolerance across two identical runs with the same seed.

---

### User Story 2 - Comparison Report Consumable by GUI (Priority: P1)

A developer integrating the GUI (Spec 004) needs a standard comparison report (JSON or CSV) that lists LSTM, Transformer, and BERT side-by-side on the same metrics so the GUI can render a leaderboard table without custom parsing per model.

**Why this priority**: The comparison report is a direct interface contract with Spec 004. Without it, the GUI cannot display a side-by-side leaderboard, which is the primary visualization deliverable for the hackathon demo.

**Independent Test**: Can be fully tested by running `generate_comparison_report.py` after all three models have produced their metrics files, verifying the output JSON/CSV matches the schema documented in this spec.

**Acceptance Scenarios**:

1. **Given** `metrics_lstm.json`, `metrics_transformer.json`, and `metrics_bert.json` exist under `$WORK/checkpoints/`, **When** `generate_comparison_report.py` is run, **Then** it produces `comparison_report.json` with a `models` array where each entry has `model_name`, `top1_accuracy`, `top5_accuracy`, `perplexity`, and `probe_score`.
2. **Given** only `metrics_lstm.json` exists (other models not yet trained), **When** `generate_comparison_report.py` is run, **Then** it still produces a valid partial report with LSTM results and null placeholders for missing models.

---

### User Story 3 - Sequence Classification Head for Yield Bucket (Priority: P2)

An engineer wants to optionally attach a classification head to the LSTM encoder to predict yield bucket (high/low/reject) from the full sequence, as a stretch goal that reuses the trained LSTM backbone.

**Why this priority**: Secondary value — enriches the model's utility for the Infineon use case but does not block the baseline comparison or constitution compliance.

**Independent Test**: Can be fully tested independently by enabling `--task classification` in the training script and verifying that a classification report (per-class F1, macro F1) is appended to `metrics_lstm.json`.

**Acceptance Scenarios**:

1. **Given** the LSTM is trained for next-step prediction, **When** the engineer runs fine-tuning with `--task classification --head yield_bucket`, **Then** training completes and the metrics file includes `classification.macro_f1` and `classification.per_class`.
2. **Given** class imbalance in the yield bucket labels, **When** the classification head trains, **Then** the training script logs per-class counts and applies weighted loss or oversampling (configurable via config YAML).

---

### User Story 4 - Reproducibility via Single SLURM Script and Seed (Priority: P3)

A reviewer wants to reproduce the LSTM baseline results from scratch using only the repository and a single SLURM job script, with all random seeds fixed.

**Why this priority**: Required by the project constitution's reproducibility principle and the hackathon judging rubric, but lowest urgency since reproducibility can be verified after training results are confirmed.

**Independent Test**: Can be fully tested by cloning the repo into a fresh `$WORK` directory, running `sbatch slurm/train_lstm.sbatch`, and diffing the resulting `metrics_lstm.json` against the reference copy committed to the repo.

**Acceptance Scenarios**:

1. **Given** a clean environment with only the repo and `$WORK/envs/`, **When** `sbatch slurm/train_lstm.sbatch` is submitted, **Then** the job runs to completion and produces `metrics_lstm.json` matching the reference within ±0.005 on all scalar metrics.
2. **Given** the SLURM script, **When** it is inspected, **Then** it sets `PYTHONHASHSEED`, PyTorch seed, NumPy seed, and `cudnn.deterministic=True`, and all are documented in the config YAML.

---

### Edge Cases

- **Vocab/split drift from Spec 001**: If the shared tokenizer or split files are updated after LSTM training, the metrics are no longer comparable. The training script MUST record the SHA of the tokenizer and split files in `metrics_lstm.json`.
- **Unknown tokens at inference**: If a fab-step token unseen during training appears at inference time, the model MUST handle it with an `<UNK>` fallback without crashing.
- **Vanishing gradients on long sequences**: For sequences approaching 100 steps, gradient norms can collapse. Gradient clipping (`max_norm` configurable, default 1.0) MUST be applied.
- **Class imbalance in yield bucket**: Yield categories are likely imbalanced. Classification head training MUST log class distribution and support weighted cross-entropy or stratified sampling.
- **Checkpoint directory missing**: If `$WORK/checkpoints/005-lstm-baseline/` does not exist at job start, the script MUST create it rather than crash.
- **Mixed precision on A100**: Automatic Mixed Precision (AMP) with `torch.cuda.amp` is optional but MUST degrade gracefully if disabled (full fp32 fallback with a logged warning).

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The LSTM training pipeline MUST load vocabulary and train/val/test splits from the same files used by Specs 001 and 002, located under `tracks/industrial-infineon/training_data/`. The exact file paths MUST be documented in the config YAML and the README for this feature.
- **FR-002**: The model architecture MUST follow the sequence: embedding layer → multi-layer LSTM (unidirectional, autoregressive) → linear projection head for next-step token prediction. Hidden size, number of layers, and dropout rate MUST be configurable via YAML.
- **FR-003**: The training loop MUST support the primary task of next-step prediction (causal language modeling over fab-step tokens). A secondary classification head for yield bucket prediction is optional (stretch goal, enabled by `--task classification`).
- **FR-004**: The optimizer, learning rate, batch size, LR schedule (e.g., cosine decay or reduce-on-plateau), and early stopping patience MUST all be configurable parameters. Defaults MUST be documented.
- **FR-005**: Gradient clipping MUST be applied during training with a configurable `max_norm` (default 1.0) to mitigate vanishing/exploding gradients on long sequences.
- **FR-006**: The evaluation harness MUST be the same shared module used by Specs 001 and 002. The module name and import path MUST be stated in this spec: `tracks/industrial-infineon/eval/eval_harness.py`. Metrics computed: next-step top-1 accuracy, top-5 accuracy, perplexity, and real-logic-vs-memorization probe score.
- **FR-007**: After evaluation, the pipeline MUST write `metrics_lstm.json` to `$WORK/checkpoints/005-lstm-baseline/`. The JSON schema MUST include: `model`, `date`, `top1_accuracy`, `top5_accuracy`, `perplexity`, `probe_score`, `tokenizer_sha`, `split_sha`, `seed`, `config`.
- **FR-008**: A comparison report generator (`generate_comparison_report.py`) MUST produce a `comparison_report.json` compatible with the Spec 004 GUI schema. The schema: `{"models": [{"model_name": str, "top1_accuracy": float, "top5_accuracy": float, "perplexity": float, "probe_score": float}]}`.
- **FR-009**: A SLURM job script (`slurm/train_lstm.sbatch`) MUST be provided that targets `boost_usr_prod` (or `boost_qos_dbg` for debug runs), requests 1 node / 4 GPUs / 4 tasks / 8 CPUs per task, and sets all random seeds for reproducibility.
- **FR-010**: The training script MUST record and log the SHA-256 hash of the tokenizer vocab file and the split manifest file at job start, and embed them in `metrics_lstm.json` to detect vocab/split drift.

### Key Entities

- **LSTMModel**: The core model class. Key attributes: `vocab_size`, `embed_dim`, `hidden_size`, `num_layers`, `dropout`, `num_classes` (for classification head). Unidirectional by default; bidirectional variant may be added for classification only, with justification in config comments.
- **FabSequenceDataset**: Dataset class shared with or mirroring Specs 001/002. Loads tokenized sequences from `training_data/`, returns (input_ids, target_ids) for next-step task or (input_ids, yield_label) for classification.
- **EvalHarness**: Shared evaluation module at `tracks/industrial-infineon/eval/eval_harness.py`. Computes top-1/top-5 accuracy, perplexity, and real-logic probe. Called identically by Specs 001, 002, and 005.
- **MetricsReport**: JSON artifact written to `$WORK` after evaluation. Serves as the interchange format between training runs and the comparison report generator.
- **ComparisonReport**: JSON artifact consumed by the GUI (Spec 004). Aggregates MetricsReport entries from all models into a single leaderboard array.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: LSTM training on the full dataset completes in under 2 hours wall-clock time on a single A100 GPU under `boost_usr_prod` or `boost_qos_dbg`.
- **SC-002**: `metrics_lstm.json` is produced with all required fields (`top1_accuracy`, `top5_accuracy`, `perplexity`, `probe_score`, `tokenizer_sha`, `split_sha`, `seed`, `config`) after a successful training run.
- **SC-003**: The baseline numbers are documented and committed to the repository so that Specs 001 and 002 results can be compared against them quantitatively (not just qualitatively).
- **SC-004**: Running the SLURM script twice with the same seed produces `metrics_lstm.json` files whose scalar metrics differ by no more than ±0.005 (floating-point and CUDA non-determinism tolerance).
- **SC-005**: `comparison_report.json` produced by `generate_comparison_report.py` is accepted without modification by the Spec 004 GUI rendering code.
- **SC-006**: An engineer unfamiliar with the codebase can reproduce the full training run by following the instructions in `specs/005-lstm-baseline/README.md` (or inline SLURM script comments) within 30 minutes of setup time.

---

## Assumptions

- The shared vocabulary file and train/val/test split files are stable at the time of LSTM training. If they change, metrics MUST be regenerated for all three models simultaneously to preserve comparability.
- The LSTM is unidirectional (autoregressive, left-to-right) for the next-step prediction task, consistent with Spec 001's Transformer. A bidirectional LSTM may be added as a separate variant only for the classification head (stretch goal), and its metrics MUST NOT be mixed with the autoregressive next-step results.
- Mixed precision (AMP) is optional and off by default; the target A100 hardware supports it but determinism is prioritized for the baseline.
- The `tracks/industrial-infineon/eval/eval_harness.py` shared module already exists or will be created as part of Spec 001 before this spec is implemented. If it does not exist, LSTM implementation is blocked on Spec 001.
- Compute budget is a single A100 node for up to 2 hours. Multi-GPU data-parallel training within the node is supported (4 GPUs via `torch.nn.parallel.DistributedDataParallel`) but single-GPU mode MUST also work for debugging.
- The yield bucket classification label is present in the dataset and its distribution is known before training begins. If labels are absent, the classification head (P2) is deferred without blocking P1.
- Out of scope: GUI rendering (Spec 004), constrained decoding, anomaly detection, hyperparameter search beyond manual tuning.
