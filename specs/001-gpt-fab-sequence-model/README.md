# Spec 001 — Transformer / GPT-style Fab-Sequence Model

Implementation of the decoder-only Transformer for next-step prediction over the Infineon 100-step chip-fab pipeline.

## Layout

```
configs/train_gpt_fab.yaml      # all hyperparameters
src/
  data/
    tokenizer.py                # FabTokenizer (vocab build + save/load)
    sequences.py                # CSV → per-SEQUENCE_ID step lists + splits
    dataset.py                  # PackedShardDataset + loss_mask
  model/
    fab_gpt.py                  # FabGPT + frozen next_step_logits() API for Spec 003
  train/
    train.py                    # DDP training loop (bf16, cosine+warmup, atomic ckpt, resume)
  eval/
    sequence_metrics.py         # SHARED contract for Specs 001/002/005
    memorization_probe.py       # perturbed-sequence score ratio
scripts/
  prepare_data.py               # one-shot data prep (CSV → tokenizer + splits + shards)
  slurm/train_full.sh           # 4×A100, 8h, boost_usr_prod
```

## Run on Leonardo

```bash
# 1. Provision (one-time)
bash scripts/provision.sh
source ~/zero_one_env/bin/activate
pip install torch==2.4.* numpy pandas pyyaml

# 2. Sanity smoke (login node, < 30s, CPU)
python -c "from src.model.fab_gpt import FabGPT, FabGPTConfig; \
  m = FabGPT(FabGPTConfig(vocab_size=207, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_len=64)); \
  import torch; print(m(torch.zeros(2,16,dtype=torch.long)).shape)"

# 3. Prepare data once (login node, ~1 min CPU)
python scripts/prepare_data.py --config configs/train_gpt_fab.yaml --work_dir $WORK

# 4. Submit full training (8 h, 4× A100)
export SLURM_ACCOUNT=<your_cineca_account>     # <— FILL THIS IN
sbatch --account=$SLURM_ACCOUNT scripts/slurm/train_full.sh

# 5. Watch
squeue -u $USER
tail -f logs/gpt_fab_full_*.out
```

## Artifacts

- `$WORK/data/fab_sequences/{tokenizer.json,splits.json,test_sequences.json,shards/}`
- `$WORK/checkpoints/001-gpt-fab/{checkpoint_best.pt,checkpoint_final.pt,checkpoint_epochNNN.pt,metrics.json,eval_report.json,tokenizer.json}`

## Resume

```bash
RESUME=1 sbatch --account=$SLURM_ACCOUNT scripts/slurm/train_full.sh
```

## Honest-eval signals in `eval_report.json`

- `test_metrics.top1_accuracy` — target ≥ 0.80 per spec SC-002
- `memorization_probe.ratio` — target ≥ 5.0 per spec SC-005
- `metrics_history` — per-epoch train/val loss + accuracy
- `vocab_hash` — frozen across Specs 001/002/005 for apples-to-apples comparison
