# Feature Specification: Transformer/GPT-Style Fab-Process Sequence Model

**Feature Branch**: `001-gpt-fab-sequence-model`

**Created**: 2026-05-30

**Status**: Draft

**Input**: User description: "Decoder-only autoregressive Transformer (~50M–300M params) that models 100-step semiconductor fab process sequences. Predicts next process step + parameters, conditioned on product variant (IC / IGBT / MOSFET). Must learn real process logic, not memorize. Train tonight on Leonardo HPC, single node, 6–8 h."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - End-to-End Training on Leonardo HPC (Priority: P1)

A process engineer SSH-es into the Leonardo login node, stages data to `$WORK`, submits a single SLURM script, and 6–8 hours later retrieves a checkpoint plus an eval report showing next-step accuracy and perplexity by product family.

**Why this priority**: Without a trained checkpoint, all downstream work (Spec 003 constrained decoding, Spec 004 GUI, Spec 005 LSTM baseline comparison) is blocked. This is the critical path for tonight's hackathon deadline.

**Independent Test**: Can be tested end-to-end by running `sbatch scripts/train_gpt_fab.sh` on Leonardo and verifying that (a) the job completes within 8 hours without OOM, (b) a checkpoint appears in `$WORK/checkpoints/001-gpt-fab/`, and (c) `eval_report.json` is written with top-1 accuracy and perplexity figures.

**Acceptance Scenarios**:

1. **Given** the SLURM script and data staged at `$WORK/data/fab_sequences/`, **When** `sbatch scripts/train_gpt_fab.sh` is submitted on Leonardo, **Then** the job runs to completion within the 8-hour walltime without OOM errors and writes a final checkpoint to `$WORK/checkpoints/001-gpt-fab/checkpoint_final.pt`.
2. **Given** training completes, **When** the eval script runs on the held-out test split, **Then** `eval_report.json` is written containing `top1_accuracy`, `top5_accuracy`, `perplexity`, and per-variant breakdowns for IC, IGBT, and MOSFET.
3. **Given** a partially-completed run (e.g., job preempted), **When** `sbatch scripts/train_gpt_fab.sh` is resubmitted with `RESUME=1`, **Then** training resumes from the latest checkpoint without loss spikes and produces an equivalent final checkpoint.

---

### User Story 2 - Calibrated Next-Step Probability Output for Spec 003 (Priority: P1)

Another engineer consuming this model (for Spec 003 constrained decoding) can load the checkpoint, pass a partial sequence, and receive a calibrated probability distribution over all vocab tokens for the next step, suitable for beam search with rule-augmented filtering.

**Why this priority**: The output format and calibration quality directly determine whether Spec 003 can be built on top of this model. Poor calibration breaks downstream rule-augmented generation. Both specs are in-flight simultaneously, so the interface must be stable.

**Independent Test**: Can be tested by calling `model.next_step_logits(partial_sequence, variant_token)` in isolation and verifying that (a) the returned tensor has shape `[vocab_size]`, (b) softmax probabilities sum to 1.0, and (c) top-1 prediction matches the expected next step on 10 reference prefixes from the canonical sequences.

**Acceptance Scenarios**:

1. **Given** a loaded checkpoint and a partial sequence prefix (pipe-separated steps), **When** `model.next_step_logits(steps, variant)` is called, **Then** a float32 tensor of shape `[vocab_size]` is returned within 50 ms per call (CPU inference).
2. **Given** a partial IC sequence ending after `DEVELOP PHOTORESIST`, **When** the model produces top-5 predictions, **Then** all 5 candidates are valid step names from the known vocabulary (no UNK tokens), and at least 3 of 5 are process-logically plausible follow-up steps per `generation_rules.md`.
3. **Given** the variant token is missing from the input, **When** `model.next_step_logits()` is called, **Then** the function raises a `ValueError` with a clear message rather than silently producing incorrect predictions.

---

### User Story 3 - Reproducible Training from Seed + Env Spec (Priority: P2)

Any team member can reproduce training results from scratch by following the README: cloning the repo, creating the conda environment from `environment.yml`, staging data, and submitting the SLURM script with the same seed — and obtaining a checkpoint whose eval metrics are within ±2% of the reference run.

**Why this priority**: Hackathon judging criteria explicitly require reproducibility. A run that only one person can reproduce is a liability on demo day and blocks honest evaluation.

**Independent Test**: Can be tested independently by a second team member following the README steps on a fresh login session, submitting the training job with `SEED=42`, and comparing the resulting `eval_report.json` against the reference report.

**Acceptance Scenarios**:

1. **Given** `environment.yml` and the SLURM script with `SEED=42`, **When** a team member runs a fresh training job, **Then** the resulting top-1 next-step accuracy is within ±2 percentage points of the reference checkpoint's accuracy on the same test split.
2. **Given** `environment.yml`, **When** `conda env create -f environment.yml` is run on Leonardo's login node, **Then** the environment is created without dependency conflicts and `python -c "import torch; print(torch.cuda.is_available())"` returns `True` inside a job.

---

### User Story 4 - Variant-Conditioned Generation Without Retraining (Priority: P3)

A process engineer can generate a plausible 100-step fab sequence for a specific product variant (IC, IGBT, or MOSFET) from a single trained checkpoint, by providing only the variant token and a `[BOS]` token as the initial sequence.

**Why this priority**: Variant conditioning is a key differentiator from a generic LM. It enables the model to be used for process design exploration without per-variant fine-tuning. P3 because the core training (P1) must exist first, and the conditioning is implemented via a single prepended token — low incremental effort once training is done.

**Independent Test**: Can be tested independently by running `python generate.py --variant IC --checkpoint $WORK/checkpoints/001-gpt-fab/checkpoint_final.pt --num_samples 5` and inspecting whether generated sequences start with `RECEIVE WAFER LOT`, pass `validate_sequence()` with zero violations, and have `WAFER SORT TEST` before `SHIP LOT`.

**Acceptance Scenarios**:

1. **Given** a trained checkpoint and `--variant MOSFET`, **When** `generate.py` samples 10 sequences at temperature 0.8, **Then** at least 7/10 sequences pass `validate_sequence()` with zero violations.
2. **Given** `--variant IC` vs `--variant IGBT`, **When** 20 sequences are generated from each variant, **Then** IC sequences include `DEPOSIT PAD OXIDE` and `GRINDING WAFER BACKSIDE` steps while IGBT sequences include `IMPLANT P BODY` — confirming variant-specific logic is captured.

---

### Edge Cases

- **Sequence longer than `max_len`**: At inference, truncate from the left (drop oldest steps), emit a warning to stderr, and continue. Do not crash. Log a counter of truncated sequences in the eval report.
- **Unknown step token at inference**: Map to `[UNK]` token. If the prediction output is `[UNK]`, re-rank to the highest-probability known step. Log occurrences.
- **OOM on long sequences**: If a single sequence exceeds GPU memory during training, skip that batch with a warning and log the sequence ID. Use gradient checkpointing by default to reduce activation memory. If OOM occurs more than 5% of batches, halt and recommend reducing `max_len` or `d_model`.
- **Checkpoint corruption / resume failure**: On startup, validate checkpoint integrity by loading state dicts and checking for NaN in parameters. If corrupt, fall back to the second-latest checkpoint. Raise an explicit error if no valid checkpoint exists.
- **Variant token missing at inference**: Raise `ValueError("variant token required; pass --variant IC|IGBT|MOSFET")` immediately. Do not default silently to a fallback variant.
- **Data split leakage**: If sequence IDs appear in both train and test splits, the data pipeline must raise an assertion error at startup. Split is performed at the sequence level, never at the step level.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Tokenization

- **FR-001**: The tokenizer MUST use a flat vocabulary of all unique step name strings found across `syntheticIC.csv`, `syntheticIGBT.csv`, `synthetic_mosfet.csv`, `IC_variants.csv`, `IGBT_variants.csv`, and `MOSFET_variants.csv`. Estimated unique step count: ~200 distinct step strings (based on generation_rules.md step vocabulary).
- **FR-002**: The vocabulary MUST include the following special tokens: `[PAD]` (id=0), `[BOS]` (id=1), `[EOS]` (id=2), `[UNK]` (id=3), `[IC]` (id=4), `[IGBT]` (id=5), `[MOSFET]` (id=6). Total estimated vocab size: ~207 tokens.
- **FR-003**: Each training sequence MUST be represented as: `[VARIANT_TOKEN] [BOS] step_1 step_2 ... step_N [EOS]`, where `[VARIANT_TOKEN]` is one of `[IC]`, `[IGBT]`, `[MOSFET]` prepended as the first token to condition the model on product family throughout the full sequence.
- **FR-004**: The tokenizer MUST handle step name synonyms (e.g., `STRIP PHOTORESIST` vs `STRIP RESIST`, `RCA CLEAN 1` vs `WET CLEAN RCA1`) by mapping synonyms to a canonical token ID, ensuring the vocabulary is not bloated with duplicates.
- **FR-005**: The tokenizer MUST be saved as a JSON file (`tokenizer.json`) alongside checkpoints so inference can be performed without the training code.

#### Sequence Handling

- **FR-006**: `max_len` MUST be set to **256 tokens** (covering the longest reference sequence of 151 steps + variant token + BOS + EOS + buffer). Sequences shorter than `max_len` MUST be right-padded with `[PAD]` tokens; attention masks MUST exclude pad positions.
- **FR-007**: The dataloader MUST use **sequence packing** (bin-packing multiple short sequences into a single `max_len` context, separated by `[EOS][BOS]` boundaries) to maximize GPU utilization. Sequences that individually exceed `max_len` MUST be truncated from the left with a warning.
- **FR-008**: Train/val/test splits MUST be performed at the sequence level (not step level) with ratio 80/10/10, stratified by product family. Sequence IDs from `*_variants.csv` files MUST be used as the split key. The three reference sequences (`synthetic*.csv`) MUST be held out in the test set.

#### Model Architecture

- **FR-009**: The model MUST be a **decoder-only causal Transformer** with the following architecture justified against the 4×A100 64 GB memory budget and 6–8 h walltime:

  | Hyperparameter | Value | Justification |
  |---|---|---|
  | `n_layers` | 12 | Balances expressiveness vs. training speed |
  | `d_model` | 512 | Fits 4×A100 comfortably at bs=256; ~85M params |
  | `n_heads` | 8 | d_head = 64; standard ratio |
  | `d_ff` | 2048 | 4× d_model; standard |
  | `max_len` | 256 | Covers longest sequence + special tokens |
  | `dropout` | 0.1 | Regularization for small dataset |
  | **Total params** | ~85M | Well within 50M–300M target range |

  At bf16 precision: ~85M × 2 bytes = ~170 MB per model copy. With optimizer states (AdamW: 2× param size), activations, and batch: comfortably fits across 4×A100 with DDP.

- **FR-010**: The model MUST use **learned positional embeddings** (not sinusoidal), with max position 256.
- **FR-011**: The model MUST use **Pre-LayerNorm** (LayerNorm before attention and FFN sub-layers) for training stability.
- **FR-012**: The model MUST use **causal attention masking** (upper-triangular mask) so step `t` can only attend to steps 0..t.
- **FR-013**: The variant conditioning token `[IC]`/`[IGBT]`/`[MOSFET]` at position 0 provides global context; no additional cross-attention or FiLM conditioning is required (the autoregressive mechanism propagates variant context naturally).

#### Training Schedule

- **FR-014**: Optimizer MUST be **AdamW** with `lr=3e-4`, `betas=(0.9, 0.95)`, `weight_decay=0.1`, `eps=1e-8`. No weight decay on embedding or LayerNorm parameters.
- **FR-015**: LR schedule MUST be **cosine decay with linear warmup**: warmup for 5% of total steps, then cosine decay to `lr_min=3e-5`.
- **FR-016**: Effective batch size MUST be **512 sequences** achieved via `per_device_batch_size=128` × `gradient_accumulation_steps=1` × `4 GPUs` = 512. Adjust `gradient_accumulation_steps` if OOM occurs.
- **FR-017**: Training MUST use **bf16 mixed precision** via `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`. Loss scaling is not required for bf16.
- **FR-018**: Training MUST use **single-node DDP** via `torch.distributed.launch` or `torchrun`. The SLURM script MUST set `--ntasks-per-node=4`, `--cpus-per-task=8`, `--gres=gpu:4`.
- **FR-019**: **Gradient clipping** MUST be applied with `max_norm=1.0` before each optimizer step.
- **FR-020**: Training MUST run for **20 epochs** or until the validation loss has not improved for 3 consecutive epochs (early stopping), whichever comes first. An epoch is defined as one full pass over the training set.
- **FR-021**: **Gradient checkpointing** MUST be enabled by default to reduce activation memory, at the cost of ~30% compute overhead. This ensures 256-token sequences fit with large batch sizes.

#### Checkpoint Strategy

- **FR-022**: Checkpoints MUST be saved to `$WORK/checkpoints/001-gpt-fab/` with the naming scheme `checkpoint_epoch{E:03d}_step{S:07d}.pt`. The checkpoint file MUST contain: model state dict, optimizer state dict, scheduler state dict, epoch, global step, best val loss, and tokenizer vocab.
- **FR-023**: Checkpoints MUST be saved every **2 epochs** and on every validation improvement (best model). Only the **3 most recent epoch checkpoints** are retained (older ones deleted) plus the `checkpoint_best.pt` symlink. This prevents quota exhaustion.
- **FR-024**: On startup, the training script MUST auto-detect the latest valid checkpoint in the output directory and resume from it if `--resume` flag is set (or `RESUME=1` env var is set in the SLURM script). Checkpoint integrity is validated by checking for NaN in all parameter tensors before resuming.

#### Evaluation Metrics

- **FR-025**: The eval script MUST compute and report the following metrics on the test split, broken down by family (IC / IGBT / MOSFET) and overall:

  | Metric | Description |
  |---|---|
  | `top1_accuracy` | Fraction of positions where predicted top-1 step = ground truth |
  | `top5_accuracy` | Fraction of positions where ground truth in top-5 predictions |
  | `perplexity` | `exp(mean cross-entropy loss)` over all non-PAD tokens |
  | `mrr` | Mean Reciprocal Rank of the ground truth step in the predicted distribution |

- **FR-026**: The eval script MUST include a **leakage/memorization probe**:
  - **Held-out variant probe**: Generate 50 sequences per family using only the variant token + BOS as the prompt. Run `validate_sequence()` on each. Report `valid_fraction` per family. A model that has merely memorized will still produce high `valid_fraction`, but perplexity on perturbed sequences (see next) will be low.
  - **Perturbed-sequence scoring**: Take 100 test sequences, inject a random RULE_DEP_NO_CLEAN violation (swap a cleaning step before a deposition step), and score both the original and perturbed sequence. Report `mean_score_ratio = score(original) / score(perturbed)`. A well-generalized model should have `mean_score_ratio > 5.0`; a memorizing model will score both similarly.
  - **Held-out variants OOD test**: Reserve one sub-variant group from each family (defined by a unique combination of optional steps from Section 4 of `generation_rules.md`) entirely from training. Report the performance drop on these OOD examples vs. in-distribution examples.

- **FR-027**: All metrics MUST be written to `$WORK/checkpoints/001-gpt-fab/eval_report.json` with a timestamp, checkpoint path, and git commit hash for reproducibility.

#### Honest Evaluation Guardrails

- **FR-028**: Train/val/test splits MUST be created once, saved as `$WORK/data/fab_sequences/splits.json`, and NEVER recreated. All runs (including reruns and ablations) MUST use the same splits file. The splits file MUST be committed to the repository.
- **FR-029**: Hyperparameter tuning (if any) MUST use only the **validation set**. Test set MUST be touched only for the final reported numbers. This is enforced by the eval script refusing to run on test if the training log shows `val_loss` was consulted more than once during a single run (i.e., early stopping triggers count as one look).
- **FR-030**: The eval report MUST include a `"leakage_checks"` section documenting: (a) that no test sequence IDs appear in training, (b) the split was stratified by variant, (c) the perturbed-sequence `mean_score_ratio`.

#### Reproducibility

- **FR-031**: A global seed MUST be set at the start of training via `torch.manual_seed(SEED)`, `numpy.random.seed(SEED)`, `random.seed(SEED)`, and `torch.cuda.manual_seed_all(SEED)`. Default `SEED=42`.
- **FR-032**: `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` MUST be set. Note: DDP with NCCL may introduce minor non-determinism across runs; this is acceptable and MUST be documented in the README.
- **FR-033**: The conda environment MUST be pinned in `environment.yml` with exact package versions. The SLURM script MUST activate this environment. The environment MUST be created in `$WORK/envs/fab_gpt` (not `$HOME`) per the HPC storage policy.
- **FR-034**: The SLURM script MUST be self-contained and submitted as `scripts/train_gpt_fab.sh`. It MUST set: `--partition=boost_usr_prod`, `--qos=boost_usr_prod`, `--nodes=1`, `--ntasks-per-node=4`, `--cpus-per-task=8`, `--gres=gpu:4`, `--time=08:00:00`. The script MUST also include a `boost_qos_dbg` variant (`scripts/train_gpt_fab_debug.sh`) for 30-minute prototype runs.

### Key Entities *(feature involves data)*

- **ProcessStep**: A single semiconductor fab operation, represented as an uppercase string token (e.g., `"THERMAL OXIDATION"`). Has a canonical token ID in the vocabulary. May have synonyms that map to the same canonical ID.
- **FabSequence**: An ordered list of 80–180 ProcessSteps representing a complete wafer lot journey from `RECEIVE WAFER LOT` to `SHIP LOT`. Has a product family label (IC / IGBT / MOSFET) and a unique `SEQUENCE_ID`.
- **ProductVariant**: One of three semiconductor device families (IC, IGBT, MOSFET). Determines which FAMILY_SPECIFIC_PREP block and which set of mandatory/optional steps apply. Represented as a special conditioning token prepended to each sequence.
- **Checkpoint**: A serialized model state (weights, optimizer state, scheduler state, metadata). Stored in `$WORK/checkpoints/001-gpt-fab/`. The canonical artifact for downstream use.
- **Tokenizer**: Maps step strings to integer IDs and back. Stored as `tokenizer.json`. Handles synonym normalization. Required for both training and inference.
- **Split**: A partition of FabSequences into train/val/test sets. Stored in `$WORK/data/fab_sequences/splits.json` and committed to the repo. Created once, never recreated.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Training job completes within the 8-hour walltime on a single Leonardo Booster node (`boost_usr_prod`) without OOM errors.
- **SC-002**: Final model achieves **top-1 next-step accuracy ≥ 80%** on the held-out test split (overall, across all families). This is the go/no-go threshold for comparing against the LSTM baseline (Spec 005).
- **SC-003**: Final model achieves **top-5 next-step accuracy ≥ 95%** on the held-out test split. This indicates the model's distribution is tight enough for beam-search-based constrained decoding (Spec 003).
- **SC-004**: Final model achieves **sequence perplexity ≤ 5.0** on the held-out test split. Higher perplexity indicates the model has not learned the underlying grammar.
- **SC-005**: **Leakage/memorization probe**: `mean_score_ratio` (original sequence score vs. perturbed sequence score) MUST be **≥ 5.0**. Ratios below this suggest the model is scoring based on surface statistics rather than process logic.
- **SC-006**: **Variant-conditioned generation**: At least **70%** of auto-regressively generated sequences (from variant token + BOS prompt only) pass `validate_sequence()` with zero violations, for each of the three product families.
- **SC-007**: The training job is fully reproducible: a second run with `SEED=42` and the same data splits produces top-1 accuracy within **±2 percentage points** of the reference run.
- **SC-008**: Go/No-Go before demo: If SC-002 (≥ 80% top-1) or SC-005 (score ratio ≥ 5.0) is not met, the team must flag this in the submission honest-evaluation section and pivot to either (a) longer training or (b) architecture change before Sunday 10:00.

---

## Assumptions

- The variants CSV files (`IC_variants.csv`, `IGBT_variants.csv`, `MOSFET_variants.csv`) contain sufficient diversity (~1,000 unique sequences per family, estimated from line counts ~115K/147K/125K lines at ~100–150 steps/sequence) for training a well-generalizing model. If fewer unique sequences exist, data augmentation via `generate_sequences.py` with varied seeds MUST be used to reach ≥ 500 unique sequences per family.
- Step name strings are treated as atomic tokens. No sub-word tokenization (BPE, WordPiece) is applied. Estimated vocabulary size of ~200 unique steps plus 7 special tokens is small enough that a learned embedding table is trivially sized.
- The Leonardo Booster node delivers ≈ 312 TFLOPS/A100 at bf16. At 85M parameters and batch size 512 (sequence length 256), estimated throughput is ~2,000 sequences/second, completing ~50K sequences × 20 epochs = 1M sequence-passes in ~8.3 minutes — the bottleneck is I/O and the number of unique sequences, not compute. Adjust to 20 epochs if dataset is small; reduce to 5 epochs if >100K unique sequences per family.
- The `generate_sequences.py` script and `validate_sequence()` function are correct implementations of the grammar in `generation_rules.md`. No changes to these files are in scope for this feature.
- NCCL is available on Leonardo for DDP communication. If not, fall back to `gloo` backend.
- `$WORK` filesystem is accessible from all Booster nodes with adequate bandwidth for checkpoint I/O. Hot training data is staged to `$TMPDIR` at job start to avoid repeated Lustre access.
- Spec 003 (constrained decoding) will consume the `next_step_logits()` API defined here. The interface is frozen with this spec and must not change without a new spec version.
- The LSTM baseline (Spec 005) is out of scope and will be built independently. The comparison metric is next-step top-1 accuracy on the same test split.
- Mobile/web GUI (Spec 004) is out of scope for this feature.
- BERT-style masked language modeling (Spec 002) is a separate model and out of scope.
- Rule-augmented constrained decoding (Spec 003) is out of scope; this model produces raw probability distributions, not rule-validated outputs.

---

## Out of Scope

- GUI or interactive interface (Spec 004)
- Rule-augmented / constrained decoding (Spec 003)
- BERT/MLM training objective (Spec 002)
- LSTM baseline model (Spec 005)
- Multi-node distributed training (only single-node DDP required within walltime)
- Inference serving / REST API
- Parameter enrichment from `*_Longdescr.csv` or `*_longdescription_parameters.csv` (step-level parameter prediction is a future extension; this spec predicts step names only)
- Hyperparameter sweep / NAS
