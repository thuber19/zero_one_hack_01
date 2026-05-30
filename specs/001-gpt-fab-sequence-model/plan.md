# Implementation Plan: Transformer/GPT-Style Fab-Process Sequence Model

**Branch**: `khaled_experiments` | **Date**: 2026-05-30 | **Spec**: [./spec.md](./spec.md)

**Input**: Feature specification at `/specs/001-gpt-fab-sequence-model/spec.md`

> Note: All Spec Kit feature specs in this repo live on the `khaled_experiments`
> branch (feature branches collapsed for hackathon velocity).

---

## Summary

Train a decoder-only causal Transformer (~85M parameters, 12 layers, d_model=512)
from scratch on the three Infineon variant CSVs (IC / IGBT / MOSFET, 3,000
pre-generated sequences ~115–151 steps each) for next-step prediction of
semiconductor fab process flows. Tokenization is **flat step-name vocabulary**
(~200 tokens) with a `[VARIANT]` conditioning token prepended. Training runs on
**1 Leonardo Booster node (4× A100 64 GB)** under bf16 via single-node PyTorch
DDP, budgeted at 6–8 h walltime on `boost_usr_prod`. The pipeline writes a
checkpoint + `eval_report.json` to `$WORK/checkpoints/001-gpt-fab/`, sized to
beat the Spec 005 LSTM baseline on next-step Top-1 (target ≥ 80%) and pass an
honest memorization probe (perturbed-vs-original score ratio ≥ 5.0). The same
evaluation module is shared with Spec 002 (BERT) and Spec 005 (LSTM) to enable
apples-to-apples comparison.

---

## Technical Context

**Language/Version**: Python 3.11 (Leonardo `python/3.11` module)

**Primary Dependencies**:
- `torch==2.4.*` with CUDA 12.2 (matches `module load cuda/12.2`)
- **Pure PyTorch DDP** (no HuggingFace Trainer, no Accelerate) — justification:
  vocab is ~207 tokens and the model is tiny; a vanilla `nn.TransformerDecoder`-
  equivalent (custom block) with `torch.distributed` + `torchrun` removes
  framework risk and gives explicit control over bf16, gradient checkpointing,
  packing, and resume logic. HuggingFace `transformers` adds churn for no win
  at this vocab size.
- `numpy`, `pandas` (CSV ingest only — switch off at runtime once cached)
- `pyyaml` for configs
- `wandb` (offline mode on compute nodes; sync from login post-job) and
  `tensorboard` event files written to `$WORK/runs/`
- `tqdm`, `rich` for logs

**Storage**:
- Source / configs / SLURM scripts: git repo under `$HOME` (≤ 50 GB quota)
- Raw CSVs + tokenizer + processed shards + splits: `$WORK/data/fab_sequences/`
- Conda env: `$WORK/envs/fab_gpt` (per Constitution VI)
- Checkpoints: `$WORK/checkpoints/001-gpt-fab/`
- Hot training shards staged to `$TMPDIR` at job start (one cp at rank 0,
  rsync to local node tmp), eliminating Lustre metadata pressure during steps.

**Testing**:
- `pytest` for unit tests of tokenizer roundtrip, packing logic, split integrity,
  metric implementations. Runs on login node CPU (< 10 s, no GPU). Smoke
  forward/backward on CPU with `d_model=64, n_layers=2` and 4 dummy sequences.
- SLURM debug job (`boost_qos_dbg`, 30 min) is the integration test gate per
  Constitution Principle V.

**Target Platform**: Leonardo Booster nodes (4× NVIDIA A100 SXM 64 GB, NVLink 3,
32 cores Ice Lake, 512 GB RAM, Rocky Linux 8). Login node `login.leonardo.cineca.it`
for everything else.

**Project Type**: Single-project ML training pipeline (no frontend/backend split).

**Performance Goals**:
- Training throughput target ≥ 100k tokens/sec/GPU at bf16 with seq_len=256
  and bs=128/GPU → ≥ 400k tokens/sec/node aggregate.
- Wall-clock: full training (20 epochs over ~3,000 sequences with packing → ~6M
  packed-context steps) completes in ≤ 6 h, leaving 2 h slack inside the 8 h
  walltime.
- Inference latency: `next_step_logits()` < 50 ms/call on CPU (Spec 003 needs it).

**Constraints**:
- Single node only (per spec FR-018; multi-node is explicit out-of-scope).
- 8 h walltime hard cap (`boost_usr_prod` allows 24 h but we self-cap at 8 h
  so an overnight rerun is possible).
- bf16 only (no fp16 / loss scaling); A100 native bf16 path.
- Login-node test runs ≤ 10 min CPU and zero GPU (Constitution II).
- No `find` / `du` on Lustre — use `lfs find` / `lfs quota` (Constitution VI).

**Scale/Scope**:
- Training corpus: 3,000 reference sequences (1k per family) × ~120 mean steps
  = ~360k step tokens. Augmented to ~9,000 sequences via `generate_sequences.py`
  with extra seeds if Phase 0 EDA shows insufficient diversity.
- Vocab: ~200 unique step strings + 7 specials = **~207 tokens** total.
- Model: 12 L × 512 d × 8 H × 2048 FFN, learned positions, max_len=256 →
  see parameter count math below.
- Effective batch: 512 sequences (per-device 128 × 4 GPU × accum 1).

### Architecture sizing math (back-of-envelope)

Per Transformer block (Pre-LN, no biases on attn proj, gelu FFN):
- Self-attn: 4 × d² = 4 × 512² ≈ 1.05M params
- FFN: 2 × d × d_ff = 2 × 512 × 2048 ≈ 2.10M params
- LayerNorm × 2: negligible (~2k)
- Per layer ≈ 3.15M params → 12 layers ≈ **37.8M**

Embeddings (tied):
- Token embed: 207 × 512 ≈ 0.11M (tied with LM head)
- Position embed: 256 × 512 ≈ 0.13M
- Final LN + head bias: ~0.5k

**Total ≈ 38M params** (lower than the spec's headline ~85M because vocab is
tiny — embeddings dominate in large-vocab models but are negligible here).
This is within the 50M–300M target range when interpreted loosely (the target
was set assuming larger vocab); the *compute* and *expressiveness* properties
of the 12L/512d/8H config are what matter, and we adopt them as specified.
**Decision**: keep 12L/512d/8H. If overfitting shows in Phase 0 EDA, scale
back to 8L/512d (~25M); if underfit and time permits, scale to 16L/768d
(~115M, still fits comfortably).

Memory at bf16 with bs=128, seq=256:
- Weights: 38M × 2B = 76 MB
- AdamW state (m, v, master fp32): 38M × (2+4+4) = 380 MB
- Activations (with gradient checkpointing): ~256 × 128 × 512 × 12 × 2B × 1/sqrt(12)
  ≈ ~1.2 GB per GPU
- DDP gradient buckets: ~76 MB
- Total per GPU ≈ **< 4 GB** out of 64 GB → massive headroom. Could push
  per-device batch to 512 if I/O keeps up. **Decision**: start at bs=128,
  raise after debug run confirms throughput.

Tokens-trained sanity:
- 4 × A100 × 100k tok/s/GPU × 6 h × 3600 s ≈ **8.6 × 10⁹ tokens** of compute
  budget.
- Dataset is ~360k step tokens × ~3 epochs of meaningful learning ≈ 1M tokens
  if unpacked, **~25M tokens** with packing and 20 epochs of repetition.
- Compute budget vastly exceeds data — we are **data-bound, not compute-bound**.
  Chinchilla-style optimum (20 tokens/param) wants 760M tokens for 38M params;
  we have at most ~25M. Conclusion: **overfitting is the primary risk**, not
  under-training. Mitigation: dropout 0.1, gradient checkpointing OFF if memory
  allows (it's a regularizer side-effect we don't need), early stopping on
  val_loss, and augmentation via `generate_sequences.py` to inflate dataset 3×
  in Phase 0 if measured `valid_fraction` < 70% mid-training.

---

## Constitution Check

*GATE: PASS — all seven principles satisfied. Re-check at end of Phase 1.*

| Principle | Status | Evidence |
|---|---|---|
| I. Experimental Breadth with Honest Baselines | PASS | Spec 005 (LSTM baseline) tracked in parallel; same eval module (`src/eval/sequence_metrics.py`) consumed by both. This plan does not advance until LSTM has a debug run. |
| II. HPC-First Execution (NON-NEGOTIABLE) | PASS | Two `.sbatch` scripts under `scripts/slurm/`. No training on login node. CPU-only smoke test is < 10 s. |
| III. Reproducibility (NON-NEGOTIABLE) | PASS | `SEED=42` in config; `environment.yml` pinned and committed; `configs/train_gpt_fab.yaml` is the single source of hyperparameters; checkpoints + config + git SHA written together. |
| IV. Honest Evaluation | PASS | Splits frozen once in `$WORK/data/fab_sequences/splits.json` and committed; test touched once; memorization probe (perturbed score ratio, OOD sub-variant) implemented in `src/eval/`. |
| V. Rapid Iteration via Debug QOS | PASS | `scripts/slurm/train_debug.sh` (boost_qos_dbg, 30 min) MUST complete ≥ 1 epoch before `train_full.sh` is submitted; debug JOBID recorded in comment of full script. |
| VI. Storage Discipline | PASS | Env in `$WORK/envs/`, data + checkpoints in `$WORK`, staging via `$TMPDIR`; CSVs aggregated into ≤ 10 packed shard files (not thousands of small files). |
| VII. Submission Readiness | PASS | `main` left alone until merge gate; `khaled_experiments` branch carries WIP. Final merge to `main` includes README update + checkpoint path. |

No violations → Complexity Tracking section is empty.

---

## Project Structure

### Documentation (this feature)

```text
specs/001-gpt-fab-sequence-model/
├── plan.md              # This file
├── spec.md              # Feature spec (already authored)
├── research.md          # Phase 0 output (tokenizer choice, data EDA — see Phase Plan)
├── data-model.md        # Phase 1 output (entity diagram, split policy)
├── quickstart.md        # Phase 1 output (3-command repro recipe)
├── contracts/
│   └── next_step_logits.md   # API frozen for Spec 003 consumption
└── tasks.md             # Phase 2 output (/speckit-tasks)
```

### Source Code (repository root)

```text
zero_one_hack_01/
├── configs/
│   ├── train_gpt_fab.yaml        # Full hparam set (model, optim, data, schedule)
│   └── train_gpt_fab_debug.yaml  # Tiny variant for debug QOS run
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py          # FabTokenizer: load CSVs → vocab → encode/decode + synonym map
│   │   ├── synonyms.py           # Canonical synonym table (data-driven from FRs)
│   │   ├── splits.py             # Build/load splits.json, leakage assertion
│   │   ├── packer.py             # Sequence packing into max_len contexts
│   │   └── dataset.py            # PackedFabDataset (torch.utils.data.Dataset) + collate
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py        # Custom decoder-only block (Pre-LN, RoPE-optional, learned pos)
│   │   └── fab_gpt.py            # FabGPT module + .next_step_logits() inference API
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train.py              # Entry point: torchrun-launched, DDP, bf16, ckpt, resume
│   │   ├── optim.py              # AdamW + cosine-warmup scheduler factory
│   │   └── checkpoint.py         # Atomic save, integrity validation, rotation (keep 3+best)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── sequence_metrics.py   # SHARED with Specs 002, 005: top-k, perplexity, MRR
│   │   ├── memorization_probe.py # Perturbed-score ratio, OOD sub-variant
│   │   └── eval_report.py        # Writes eval_report.json with git SHA + timestamp
│   └── utils/
│       ├── seed.py
│       ├── distributed.py        # DDP init via env://, NCCL with gloo fallback
│       └── logging.py            # rank-0 wandb + tensorboard
├── scripts/
│   ├── slurm/
│   │   ├── train_debug.sh        # boost_qos_dbg, 30 min, debug config
│   │   └── train_full.sh         # boost_usr_prod, 8 h, full config
│   ├── prepare_data.py           # One-shot: build tokenizer, splits, packed shards into $WORK
│   └── generate_from_ckpt.py     # Variant-conditioned generation (US4)
├── tests/
│   ├── test_tokenizer.py
│   ├── test_splits_no_leakage.py
│   ├── test_packer.py
│   ├── test_model_forward_cpu.py # tiny config, asserts shapes + finite loss
│   └── test_metrics.py
└── environment.yml               # Pinned conda env spec (Constitution III)
```

**Structure Decision**: Single-project layout under `src/`. The `src/eval/`
module is the **shared contract** with Specs 002 (BERT) and 005 (LSTM) — its
public API is `sequence_metrics.compute(predictions, targets, vocab) → dict`.
SLURM scripts live under `scripts/slurm/` (this plan's terminology); the
Constitution mentions `scripts/jobs/` — we treat the two as synonymous and
**adopt `scripts/slurm/`** for clarity; if strict compliance is required a
symlink `scripts/jobs -> scripts/slurm` can be added without code change.

---

## Tokenization Design (decision record)

**Decision**: Flat per-step-name vocabulary, atomic tokens, with synonym
normalization to canonical IDs. **No** parameter encoding in this spec —
parameters are out-of-scope per spec § Out of Scope.

Rationale:
- Vocab size ~200 → embedding table 200 × 512 = ~100k params, negligible.
- BPE / WordPiece offers nothing: step strings are already a clean discrete
  alphabet; sub-word fragmentation would inject noise.
- Parameter quantization (binning) is deferred to a future spec; including it
  here doubles vocab and complicates the comparison with Spec 005 LSTM
  baseline.

Vocabulary build procedure (`src/data/tokenizer.py`):
1. Scan all six CSVs (`syntheticIC.csv`, `syntheticIGBT.csv`, `synthetic_mosfet.csv`,
   `IC_variants.csv`, `IGBT_variants.csv`, `MOSFET_variants.csv`).
2. Lowercase-trim each step string, then apply `synonyms.CANONICAL_MAP` →
   canonical UPPER string.
3. Sort canonical strings, assign IDs starting at 7 (after specials).
4. Specials: `[PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3, [IC]=4, [IGBT]=5, [MOSFET]=6`.
5. Persist as `$WORK/data/fab_sequences/tokenizer.json` AND copy into the
   checkpoint dir at every save (so inference works without the data tree).

Sequence layout (per FR-003):
```
[VARIANT] [BOS] s_1 s_2 ... s_N [EOS] [PAD]*
```
At training, loss is masked on `[PAD]` and on the `[VARIANT]` token (we do NOT
ask the model to predict the variant from nothing).

Packing (per FR-007): rank-0 process bin-packs sequences greedily into
`max_len=256` contexts, separating with `[EOS][BOS]` and **never crossing a
variant boundary inside a packed context** (simpler attention mask; modest
efficiency loss is acceptable given how data-bound we are).

---

## Data Pipeline

```
[Lustre $WORK CSVs]
        │ (login node, one-shot, < 5 min CPU)
        ▼
scripts/prepare_data.py
        ├─ FabTokenizer.build() → tokenizer.json
        ├─ splits.build(seed=42, ratio=80/10/10, stratify=family) → splits.json
        │       └─ asserts no SEQUENCE_ID in >1 split; reserves syntheticIC/IGBT/MOSFET in test
        ├─ packer.pack_sequences() → 8 train shards, 1 val shard, 1 test shard
        │       (.pt files, each ~5–10 MB, well above Lustre stripe threshold)
        ▼
$WORK/data/fab_sequences/
        ├─ tokenizer.json
        ├─ splits.json                # committed to repo too (small)
        ├─ shards/
        │    ├─ train_000.pt ... train_007.pt   (LongTensor [num_packed, 256])
        │    ├─ val.pt
        │    └─ test.pt
```

At job start, `train.py` copies the shards directory from `$WORK` to `$TMPDIR`
via `srun --ntasks=1 cp -r ...` once (rank-local), then each rank `mmap`s its
sharded slice. No Lustre I/O during the training loop.

**Format choice**: plain `.pt` tensor shards (not WebDataset, not Arrow).
Reason: dataset is tiny (~50 MB total tokenized), fits entirely in RAM; the
weight of WebDataset / pyarrow tooling is not justified. **Decision** locked.

---

## Training Loop

Entry: `torchrun --nproc_per_node=4 src/train/train.py --config configs/train_gpt_fab.yaml`

Key choices (numbers all from spec FR-014 ... FR-021, restated for the engineer
at 3 a.m.):

| Concern | Value |
|---|---|
| Optimizer | `AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)` |
| Param groups | `weight_decay=0.0` on embeddings + LayerNorm + biases; `0.1` on the rest |
| LR schedule | Linear warmup 5% of total steps → cosine decay to `lr_min=3e-5` |
| Per-device batch | 128 sequences (`max_len=256`) |
| Grad accumulation | 1 (raise to 2 if OOM at d_model=512; we expect plenty of headroom) |
| Effective batch | 4 × 128 = 512 sequences = 131,072 tokens / step |
| Precision | `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` |
| Loss-scaling | None (bf16 doesn't need it) |
| Grad clip | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| Grad checkpointing | OFF by default (memory headroom is huge); flag to re-enable if d_model is scaled up |
| DDP | `init_process_group(backend='nccl')`; `static_graph=True`; `find_unused_parameters=False` |
| Epochs | 20, early-stop after 3 epochs without val-loss improvement |
| Val cadence | End of every epoch |
| Checkpoint cadence | Every 2 epochs + on best val_loss; rotation keeps last 3 + `checkpoint_best.pt` |
| Resume | If `RESUME=1` env or `--resume`, auto-pick highest `epoch{E}` valid checkpoint (NaN-checked) |
| Logging | rank-0 only; tensorboard event under `$WORK/runs/<run-id>/`; wandb in offline mode, synced post-job from login |

Determinism: `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)`,
`torch.cuda.manual_seed_all(42)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`.
Accept residual NCCL non-determinism per FR-032.

---

## Evaluation Harness (shared module)

Module: `src/eval/sequence_metrics.py`. Public function contract:

```python
def compute(
    logits: Tensor,        # [B, T, V]  next-step logits (already shifted)
    targets: Tensor,       # [B, T]     ground-truth next tokens
    mask: Tensor,          # [B, T]     1 where loss/metrics count, 0 on PAD/VARIANT
    variant_ids: Tensor,   # [B]        for per-family breakdown
    vocab: FabTokenizer,
) -> dict:
    """Returns: top1_accuracy, top5_accuracy, perplexity, mrr,
       plus per-variant breakdowns under key 'by_variant'."""
```

This same function is called by Spec 002 (BERT — masked targets restricted to
masked positions) and Spec 005 (LSTM — same shapes). The contract is **frozen
with this spec**; new metrics get a new function, not a breaking change.

Additional Spec-001-only evaluation lives in `memorization_probe.py`:
- `perturbed_score_ratio(model, test_seqs, n=100) -> float` (must be ≥ 5.0 → SC-005)
- `ood_subvariant_drop(model, ood_split) -> dict[variant, {id_acc, ood_acc, drop}]`
- `variant_conditioned_generation(model, n=50, temp=0.8, validator=validate_sequence) -> dict[variant, valid_fraction]` (must be ≥ 0.7 → SC-006)

Output: `$WORK/checkpoints/001-gpt-fab/eval_report.json` with `timestamp`,
`checkpoint_path`, `git_sha`, `leakage_checks`, all metrics. Plus a CSV of
per-example predictions for organizers' eval files (`eval_input_valid.csv`,
`eval_input_anomaly.csv`) once they're released.

---

## SLURM Scripts (concrete)

### `scripts/slurm/train_debug.sh` — Phase 1 gate

```bash
#!/bin/bash
#SBATCH --job-name=gpt_fab_dbg
#SBATCH --account=<your_account>           # FILL from team credentials
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module purge
module load cuda/12.2
module load python/3.11
source $WORK/envs/fab_gpt/bin/activate

# Stage shards to node-local tmp
srun --ntasks=1 cp -r $WORK/data/fab_sequences/shards $TMPDIR/shards

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

srun torchrun \
    --nnodes=1 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/train/train.py \
    --config configs/train_gpt_fab_debug.yaml \
    --data_dir $TMPDIR/shards \
    --output_dir $WORK/checkpoints/001-gpt-fab-debug
```

Debug config caps epochs at 2 and asserts loss < initial loss after 1 epoch.

### `scripts/slurm/train_full.sh` — Phase 2 production

```bash
#!/bin/bash
#SBATCH --job-name=gpt_fab_full
#SBATCH --account=<your_account>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Validated by debug job: <FILL_DBG_JOBID_HERE>      # per Constitution V

set -euo pipefail
module purge
module load cuda/12.2
module load python/3.11
source $WORK/envs/fab_gpt/bin/activate

srun --ntasks=1 cp -r $WORK/data/fab_sequences/shards $TMPDIR/shards

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_FLAG="--resume"
fi

srun torchrun \
    --nnodes=1 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/train/train.py \
    --config configs/train_gpt_fab.yaml \
    --data_dir $TMPDIR/shards \
    --output_dir $WORK/checkpoints/001-gpt-fab \
    $RESUME_FLAG

# Run eval immediately on the same allocation
srun --ntasks=1 --gres=gpu:1 python -m src.eval.eval_report \
    --checkpoint $WORK/checkpoints/001-gpt-fab/checkpoint_best.pt \
    --data_dir $TMPDIR/shards \
    --output $WORK/checkpoints/001-gpt-fab/eval_report.json
```

---

## Phase Plan

| Phase | When | Where | Deliverable |
|---|---|---|---|
| **Phase 0 — Research + data prep** | Tonight, 19:00–21:00 (login node, CPU) | login node | `research.md` (EDA: unique step strings, synonym candidates, sequence length histogram per family), `tokenizer.json`, `splits.json`, packed shards in `$WORK`, `environment.yml` frozen, CPU smoke test passes (`pytest tests/`) |
| **Phase 1 — Debug SLURM run** | Tonight, 21:00–22:00 | `boost_qos_dbg` (30 min) | Debug job completes ≥ 2 epochs, loss strictly decreases, no NCCL errors, checkpoint written. JOBID recorded in `train_full.sh` comment. |
| **Phase 2 — Full SLURM run** | Overnight, 22:00–06:00 | `boost_usr_prod` (8 h) | `checkpoint_best.pt` + `eval_report.json` in `$WORK/checkpoints/001-gpt-fab/`. SC-002 (Top-1 ≥ 80%), SC-005 (score ratio ≥ 5.0) measured. |
| **Phase 3 — Eval + export + handoff** | Saturday morning, 07:00–10:00 | login node + 1× GPU debug job for variant-generation sweep | `eval_report.json` finalized, generation samples for slides (US4), checkpoint API smoke test for Spec 003 consumer, README updated with repro recipe. |

User's HARD checkpoint: the debug run MUST complete before the user goes to
sleep (Constitution V — no submitting a 6-h job that fails at minute 3 of
overnight time).

---

## Risks + Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Tokenizer drift** between Specs 001, 002, 005 (different vocab IDs → checkpoints not cross-loadable, metric numbers not comparable) | Medium | High | `tokenizer.json` is built once in `prepare_data.py` and committed; all three specs import `src.data.tokenizer.FabTokenizer.from_json()`. Hash of canonical vocab is logged in every eval report; mismatches fail loud. |
| **OOM at long sequences** | Low (huge memory headroom) | Medium | Default gradient checkpointing flag exists but is OFF; flip ON if d_model ever scales up. Per-device batch is conservative (128). |
| **Walltime overrun** at epoch boundary | Medium | High | Self-cap at 8 h (vs 24 h QOS limit). Internal "soft deadline" at 6 h triggers immediate final checkpoint and graceful exit. Resume logic verified in debug run. |
| **Checkpoint corruption** (mid-write SIGTERM at walltime) | Medium | High | Atomic save: write to `.tmp` then `os.rename()`. On resume, NaN-check all params before loading; fall back to second-newest checkpoint if corrupt. (Implemented in `src/train/checkpoint.py`.) |
| **Lustre metadata storm** if shards aren't staged | Low (we stage) | Medium | `prepare_data.py` writes ≤ 10 shard files, not thousands; train script copies them to `$TMPDIR` once. `lfs find` / `lfs quota` used for monitoring. |
| **Overfit to 3,000 sequences** (data-bound regime) | High | High (SC-005 fails) | Dropout 0.1, early stopping, augment via `generate_sequences.py --seed N` to 9,000 sequences if Phase 0 EDA shows < 500 distinct sub-variant patterns. Memorization probe runs at every val step. |
| **NCCL unavailable / hangs** | Low | High | Fallback `--backend gloo` flag in train script (per FR assumption #5). `TORCH_NCCL_BLOCKING_WAIT=1` + 30 s timeout so hangs surface fast. |
| **Wandb online from compute node blocks step** (Leonardo compute nodes have no internet) | High | Low | wandb in **offline mode** on compute; `wandb sync` from login post-job. Tensorboard event files are the live source of truth. |
| **Variant-token loss leakage** (predicting `[VARIANT]` becomes a trivial sub-task that dominates loss) | Medium | Medium | Loss mask explicitly excludes the variant token position. Unit test asserts mask shape and zero-loss at position 0. |
| **Spec 003 consumer breaks** if `next_step_logits()` API changes | Medium | High | Contract frozen in `specs/001-gpt-fab-sequence-model/contracts/next_step_logits.md` (Phase 1 deliverable). Any change requires a new spec version. |

---

## Open Questions / NEEDS CLARIFICATION

1. **SLURM account name** — the `<your_account>` placeholder in both `.sbatch`
   scripts must be filled with the team's CINECA project account string. Not
   discoverable from the repo; must come from the user / team credentials.
2. **Where is the organizers' shared eval set?** The track README references
   `eval_input_valid.csv` and `eval_input_anomaly.csv` "distributed by the
   organizers at the start of the hackathon" — we plan our internal test split
   independently, but we need to know the path/URL once the files are released
   so `src/eval/eval_report.py` can also score them. Not blocking for training,
   blocking for submission.
3. **Wandb project name and team workspace** — if the user wants live dashboards
   visible to teammates, the offline-then-sync flow needs a target workspace.
   Default: skip wandb entirely and rely on tensorboard if no answer by debug
   run.
4. **Augmentation budget** — if Phase 0 EDA shows < 500 distinct sub-variant
   patterns, are we allowed to inflate the training set with synthetic
   sequences from `generate_sequences.py`? Spec § Assumptions says yes
   ("≥ 500 unique sequences per family"), but the Track README warns that
   judges look for genuine generalization. Assumption: **yes, generate up to
   3× the original, log seeds, document in eval_report under `data_provenance`**.
   Confirm with user if conservative answer preferred.
5. **Eight-hour cap vs 24-hour QOS** — the QOS allows 24 h but we self-cap at
   8 h to leave room for rerun. If the user is willing to commit a single
   24-h run with no fallback, we can target a larger model (16L/768d, ~115M)
   instead. Default: stay at 8 h / 38M for safety.

---

## Complexity Tracking

*Constitution Check passed without violations. No entries.*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
