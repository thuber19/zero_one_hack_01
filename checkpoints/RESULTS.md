# Checkpoint Results Summary

All models trained on the Infineon semiconductor process-flow dataset (vocab_size=208, vocab_hash=`02f28c1d8b87371e`).
Train / val / test split: 120k / 15k / 15k sequences.

---

## 001-gpt-fab — GPT (autoregressive)

| Parameter | Value |
|-----------|-------|
| Params | 38.0 M |
| Epochs trained | 15 |
| Best val loss | 0.3172 (epoch 13) |
| **Test top-1 accuracy** | **81.4%** |
| Test top-5 accuracy | 99.99% |
| Test MRR | 90.4% |
| Test perplexity | 1.373 |
| Memorization probe ratio | 36.1× (true NLL 0.36 vs perturbed 12.99) |
| Training time | ~0.07 h |

**Per-variant top-1 accuracy (test):**

| Variant | Top-1 Acc | N tokens |
|---------|-----------|---------|
| 4 | 78.8% | 585 989 |
| 5 | 82.4% | 750 261 |
| 6 | 82.7% | 636 429 |

**Training curve (val loss):**

| Epoch | Train loss | Val loss | Top-1 acc |
|-------|-----------|---------|-----------|
| 0 | 1.332 | 0.328 | 81.1% |
| 1 | 0.326 | 0.321 | 81.3% |
| 5 | 0.319 | 0.318 | 81.4% |
| 10 | 0.318 | 0.318 | 81.4% |
| 13 | 0.317 | **0.317** | 81.4% |
| 14 | 0.317 | 0.317 | 81.4% |

**Notes:**
- Train ≈ val loss throughout — no overfitting.
- Model plateaued after epoch 1; remaining epochs shaved the 4th decimal place only.
- High memorization probe ratio (36×) is desirable for anomaly detection — the model strongly rejects perturbed sequences.

---

## 002 — BERT MLM (masked language model)

| Parameter | Value |
|-----------|-------|
| Params | 4.87 M |
| Epochs trained | 10 (early stop, patience=3) |
| Best val MLM loss | 0.158 (epoch 9) |
| **Best val masked accuracy** | **89.4%** (epoch 7) |
| Val masked tokens per epoch | 210 000 |
| Training time | ~7 min (1 GPU, 3/4 processes failed on port 29500) |

**Training curve:**

| Epoch | Train loss | Val MLM loss | Val masked acc |
|-------|-----------|------------|----------------|
| 0 | 3.882 | 2.388 | 38.5% |
| 1 | 1.777 | 0.945 | 71.4% |
| 2 | 0.577 | 0.239 | 87.8% |
| 5 | 0.186 | 0.165 | 89.3% |
| 7 | 0.173 | 0.159 | **89.4%** |
| 9 | 0.169 | **0.158** | 89.2% |
| 10 | 0.166 | 0.159 | 89.1% ← early stop |

**Notes:**
- Not directly comparable to GPT/LSTM (different task: fill-in-the-blank vs next-token prediction).
- Val loss tracks train loss closely — no overfitting.
- Run used only 1 of 4 GPUs due to a port binding error (`MASTER_PORT=29500` already in use). Fix for future runs: set `--master_port` to a free port in the SLURM script.

---

## 005-lstm-baseline — LSTM (autoregressive)

| Parameter | Value |
|-----------|-------|
| Params | 3.55 M |
| Architecture | 2-layer LSTM, hidden=512, embed=128, dropout=0.1 |
| Epochs trained | 9 (early stop, patience=3) |
| Best val loss | 0.319 (epoch 3) |
| **Test top-1 accuracy** | **81.3%** |
| Test top-5 accuracy | 99.99% |
| Test MRR | 90.4% |
| Test perplexity | 1.375 |
| Memorization probe ratio | 24.8× |
| Training time | ~0.056 h |

**Per-variant top-1 accuracy (test):**

| Variant | Top-1 Acc | N tokens |
|---------|-----------|---------|
| 4 | 78.7% | 585 989 |
| 5 | 82.3% | 750 261 |
| 6 | 82.7% | 636 429 |

**Training curve:**

| Epoch | Train loss | Val loss | Top-1 acc |
|-------|-----------|---------|-----------|
| 0 | 1.282 | 0.325 | 81.3% |
| 3 | 0.322 | **0.319** | 81.4% |
| 5 | 0.320 | 0.318 | 81.4% |
| 8 | 0.320 | 0.318 | 81.4% |

**Notes:**
- Val loss occasionally below train loss — expected with dropout.
- No overfitting; model plateaued early just like GPT.

---

## Cross-model Comparison

| Model | Params | Top-1 acc | MRR | Perplexity | Probe ratio |
|-------|--------|-----------|-----|-----------|-------------|
| GPT-fab (001) | 38.0 M | **81.4%** | **90.4%** | **1.373** | **36.1×** |
| LSTM baseline (005) | 3.55 M | 81.3% | 90.4% | 1.375 | 24.8× |
| BERT MLM (002) | 4.87 M | — (MLM task) | — | — | — |

**Key finding:** The 38 M parameter GPT and 3.55 M parameter LSTM reach identical next-token prediction performance (~81.4% top-1, MRR 90.4%). The GPT's higher memorization probe ratio (36× vs 25×) makes it the stronger anomaly detector. Variant 4 is consistently harder to predict than variants 5 and 6 across all models.
