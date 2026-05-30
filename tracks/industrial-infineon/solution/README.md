# procseq — Infineon Process-Logic Pipeline

A from-scratch sequence-modelling pipeline that learns the hidden logic of
semiconductor manufacturing process sequences (Infineon Track, Zero-One Hack 2026).

---

## Overview

The pipeline solves three tasks from the Infineon briefing:

| Task | Objective | Model |
|------|-----------|-------|
| **Task 1** | Predict the next process step given a partial sequence | Decoder (CLM) |
| **Task 2** | Complete a sequence from any cut point | Decoder (CLM) |
| **Task 3** | Flag anomalous sequences and localise the anomaly | DeBERTa Encoder |

Two models are trained end-to-end on synthetically generated process sequences:

- **From-scratch GPT-style Decoder** (Llama architecture, 3 sizes: tiny / base / large).
  Trained with causal language modelling on token-level process steps.
- **DeBERTa-style Encoder** (3 sizes: tiny / base / large).
  Fine-tuned for sequence-level anomaly classification + span localisation.

### Why from-scratch?

Process sequences consist of a domain-specific vocabulary of ~50 step tokens
with rigid grammar rules not present in any public pre-trained checkpoint.
A pre-trained LLM would require substantial fine-tuning just to unlearn its
prior; training from scratch on the grammar-generated data converges faster,
gives full control over the tokenizer vocabulary, and produces a model whose
learned representations are directly interpretable by the logic probe.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python >= 3.10.  GPU is optional (CUDA or Apple MPS auto-detected).
For Leonardo runs, use the cluster venv procedure in the **Leonardo Runbook**
section below.

---

## Quickstart — smoke test (CPU, ~30 s)

```bash
make smoke
```

This runs unit tests, generates tiny synthetic data, trains both models for
5 steps, runs inference, and scores all three tasks.  Expected final line:
`SMOKE OK`.

To run only the unit tests:

```bash
make test
```

---

## Full Pipeline Commands

### 1. Generate training data

```bash
make data
# equivalent:
python -m procseq.build_data --n-per-family 2000 --seed 42
```

Writes `artifacts/sequences_*.jsonl` for each grammar family.

### 2. Train the decoder (Task 1 + 2)

```bash
# Single-GPU / local
python -m procseq.train_decoder --config configs/leonardo_decoder.yaml

# Multi-GPU with Accelerate (recommended on Leonardo)
accelerate launch -m procseq.train_decoder --config configs/leonardo_decoder.yaml
```

TensorBoard logs land in `runs/<run_name>/`.

### 3. Train the encoder (Task 3)

```bash
# Single-GPU / local
python -m procseq.train_encoder --config configs/leonardo_decoder.yaml   # reuses same config

# Multi-GPU
accelerate launch -m procseq.train_encoder --config configs/leonardo_decoder.yaml
```

### 4. Run inference (all three tasks)

```bash
python -m procseq.infer --all --config configs/leonardo_decoder.yaml
```

Writes `artifacts/submission_task1.csv`, `artifacts/submission_task2.csv`,
and `artifacts/submission_task3.csv`.

### 5. Evaluate

```bash
python -m procseq.run_eval --config configs/leonardo_decoder.yaml
# or: make eval
```

Prints per-family and aggregate scores for all three tasks.

### 6. Interactive demo

```bash
python -m procseq.demo --config configs/leonardo_decoder.yaml
# or: make demo
```

Side-by-side comparison: rule-oracle baseline vs. trained model prediction.

### 7. Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Browse training curves, per-family metrics, and anomaly heat-maps interactively.

### 8. Leave-one-family-out OOD probe (Task 4 self-simulation)

```bash
python -m procseq.ood_probe --holdout IC --data-per-family 2000 --max-steps 3000
```

Trains with one grammar family held out and reports ID vs OOD Top-1 delta.

### 9. Scaling sweep

```bash
python -m procseq.sweep_run --config configs/leonardo_decoder.yaml
```

Runs tiny / base / large decoder variants and logs results to `artifacts/sweep/`.

---

## Leonardo Runbook (CINECA HPC)

### Prerequisites

1. Log in: `ssh <username>@login.leonardo.cineca.it`
2. Clone/copy the repo to `$WORK/procseq/`.

### One-time venv setup

```bash
module load profile/deeplrn
module load python cuda          # adjust to available module names
cd $WORK/procseq/tracks/industrial-infineon/solution
python -m venv $HOME/procseq-venv
source $HOME/procseq-venv/bin/activate
pip install -r requirements.txt
```

### Edit SLURM scripts

Open `slurm/train_decoder.sbatch` and replace the placeholder tokens:

```
#SBATCH --account=__ACCOUNT__    →  your CINECA project account code
#SBATCH --qos=__QOS__            →  your QOS (e.g. normal, boost_qos_dbg)
module load python cuda          →  adjust to the actual module names on Leonardo
```

Set the environment variables expected by the scripts:

```bash
export VENV=$HOME/procseq-venv
export SOLUTION_DIR=$WORK/procseq/tracks/industrial-infineon/solution
```

### Submit jobs

```bash
cd $WORK/procseq/tracks/industrial-infineon/solution

# Decoder training (4 × A100 Booster, ~2 h)
sbatch slurm/train_decoder.sbatch

# Scaling sweep (tiny / base / large in sequence)
sbatch slurm/scaling_sweep.sbatch   # if present, otherwise submit manually per size
```

Monitor:

```bash
squeue -u $USER
tail -f logs/dec_<JOBID>.out
```

Checkpoints land in `artifacts/decoder_base/` (or the path set in the config).

---

## Architecture Summary

### Decoder (Tasks 1 & 2)

- Llama-style causal transformer, 3 sizes (tiny: 6 M params, base: ~85 M, large: ~300 M).
- Custom domain tokenizer with a ~60-token vocabulary covering all grammar step tokens
  plus special tokens (`<BOS>`, `<EOS>`, `<PAD>`, family markers).
- Trained with cross-entropy on full sequences; inference uses temperature-1 greedy
  decoding with grammar-constrained beam search as an optional postprocessing step.

### Encoder (Task 3)

- DeBERTa-v3-style encoder, same 3 sizes.
- Sequence-level head: binary classifier (normal / anomalous).
- Token-level head: BIO span tagger to localise the anomalous step(s).
- Joint training with weighted cross-entropy (anomaly class upweighted ×3).

### Why from-scratch (repeated for emphasis)

No public checkpoint has seen semiconductor process-step sequences.
The grammar vocabulary is closed (~50 domain tokens), so a pre-trained tokenizer
would fragment them into meaningless sub-word pieces.
Training from scratch takes < 2 h on 4 × A100 and gives a clean,
interpretable baseline whose learned features the logic probe can directly analyse.

---

## Honest Evaluation Framing

All reported numbers are compared against the following ladder:

| Reference | What it measures |
|-----------|-----------------|
| **n-gram floor** | Dumb Markov baseline (bigram / trigram). Sets the floor. |
| **Perplexity baseline** | Uniform-random next-step picker calibrated to vocab size. |
| **Rule-oracle ceiling** | Deterministic grammar oracle. Unreachable upper bound. |
| **Logic probe** | Linear probe on decoder hidden states predicting grammar-family label. Measures how much structure the model internalised. |
| **Leave-one-family-out OOD** | Train on 2 families, test on the 3rd. Measures out-of-distribution generalisation. |

A result is only "good" if it beats the n-gram floor by a meaningful margin
and the gap to the oracle is honestly reported.

---

## Results

*Fill in after real training runs on Leonardo.*

| Metric | n-gram floor | Our model | Oracle ceiling |
|--------|-------------|-----------|----------------|
| Task 1 Top-1 accuracy | — | — | — |
| Task 1 Top-5 accuracy | — | — | — |
| Task 2 BLEU-4 | — | — | — |
| Task 2 exact-match | — | — | — |
| Task 3 F1 (anomaly) | — | — | — |
| Task 3 span IoU | — | — | — |
| Logic probe accuracy | — | — | n/a |
| OOD delta (avg) | — | — | n/a |

---

## Project Layout

```
solution/
  procseq/            Python package (all pipeline modules)
    grammar.py        Synthetic grammar + FAMILIES / RULE_IDS
    tokenizer.py      Domain tokenizer (encode_sequence / decode_to_steps)
    data.py           Data utilities + UCBS scaling
    datasets.py       PyTorch Dataset wrappers
    models/           Decoder + Encoder model definitions
    train_decoder.py  CLM training entry-point
    train_encoder.py  Encoder fine-tuning entry-point
    infer.py          Decoder inference (Task 1 + 2)
    infer_anomaly.py  Encoder inference (Task 3)
    run_eval.py       End-to-end evaluation driver
    eval_metrics.py   All task metrics + logic probe
    baselines.py      n-gram and perplexity baselines
    demo.py           Before/after demo script
    ood_probe.py      Leave-one-family-out OOD probe
    sweep_run.py      Scaling sweep runner
  configs/            YAML configs (smoke, leonardo_decoder, ...)
  slurm/              SLURM batch scripts for Leonardo
  dashboard/          Streamlit dashboard app
  artifacts/          Generated data, checkpoints, CSVs (git-ignored)
  tests/              pytest suite
  Makefile            Convenience targets
  requirements.txt    Python dependencies
```

---

## License

MIT — see `LICENSE` at the repository root.
