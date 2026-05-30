# REPORT — Industrial AI: Learning and Benchmarking Process Logic

## TL;DR

We built a hybrid neuro-symbolic system that learns semiconductor process sequences by combining a neural sequence model (LSTM/Transformer) with a Random Forest candidate filter and physics-based validation. The system predicts next process steps with 67% Top-1 / 100% Top-5 accuracy, completes partial sequences with 70% block-level accuracy, and detects process rule violations with 100% precision — all trained from scratch on synthetic data using the Leonardo supercomputer.

## Problem

Semiconductor manufacturing follows complex process routes where order matters: you must clean before depositing, pattern before etching, test before shipping. The challenge is whether a model can learn this underlying process logic — not just memorize sequences, but generalize to unseen variations and detect violations.

We address three tasks:
1. **Next-step prediction**: Given a partial process, predict what comes next
2. **Sequence completion**: Complete a partially executed process route
3. **Anomaly detection**: Identify process rule violations in a sequence

## Approach

### Architecture: Hybrid RF + Neural Model

```
Sequence so far → [Random Forest] → candidate set (top-15)
                → [LSTM/Transformer] → ranked predictions
                → [Physics validator] → guaranteed valid output
```

- **Random Forest**: Trained on n-gram features (current step, 3 previous steps, family, litho level, block position, sequence progress). Produces a candidate set of ~15 plausible next steps with 99.9% recall — the correct answer is almost always in the set.
- **Neural model**: Small LSTM (1.1M params) or Transformer (6.4M params) trained from scratch with a custom 205-token vocabulary where each process step is a single token. Ranks candidates using full sequence context via next-token prediction (causal LM objective).
- **Physics validator**: Rule-based system implementing the 10 process logic constraints. Used for anomaly detection and optionally for constrained decoding during sequence completion.

### Key Technical Decisions

- **From-scratch training** over fine-tuning a pretrained LLM: the vocabulary is only ~205 tokens, making a pretrained 50K-token model wasteful. A purpose-built model trains in minutes.
- **LSTM over Transformer**: Both architectures converge to identical accuracy (~80.9%), but the LSTM trains 12x faster (2s/epoch vs 24s) with 6x fewer parameters. Process sequences are relatively short (~130 steps) and strictly ordered — LSTM's sequential inductive bias is well-suited.
- **RF masking at inference**: Rather than having the neural model choose from 205 tokens, the RF narrows to ~15 candidates. This eliminates structurally impossible transitions and boosts Top-3/5 accuracy to near-perfect.
- **Separate anomaly detection**: Task 3 uses the rule validator directly (the 10 forbidden patterns from the process grammar), achieving 100% accuracy. The neural model's per-token loss serves as an additional soft signal.

### Training Setup

- **Data**: Up to 60K synthetic sequences (20K per product family: MOSFET, IGBT, IC) generated from the provided process grammar with validated rule compliance
- **Infrastructure**: Leonardo supercomputer, NVIDIA A100 GPUs, pixi environment management
- **Training**: AdamW optimizer, cosine annealing LR schedule, early stopping (patience=20)
- **Evaluation**: 10% held-out split, scored with the official `eval_metrics.py` scorer

## How to Run

```bash
# On Leonardo:
cd ~/process-sequence-model

# Install environment
~/.pixi/bin/pixi install

# Launch interactive training job
bash jobs/run.sh
# Select architecture, data size, epochs, batch size, RF on/off, physics on/off

# Or run inference on official eval files directly
pixi run python3 src/inference.py \
    --model-dir $SCRATCH/runs/<run_name> \
    --eval-valid data/eval_input_valid.csv \
    --eval-anomaly data/eval_input_anomaly.csv \
    --out-dir submissions/
```

## Results

### Self-Evaluation (official scorer, held-out split)

| Task | Metric | Random Baseline | Frequency Baseline | Our Model (LSTM+RF) |
|---|---|---|---|---|
| Next-step | Top-1 Accuracy | ~0.5% | ~35% | **66.9%** |
| Next-step | Top-3 Accuracy | ~1.5% | ~50% | **99.2%** |
| Next-step | Top-5 Accuracy | ~2.5% | ~60% | **100%** |
| Next-step | MRR | ~0.02 | ~0.42 | **0.83** |
| Completion | Token Accuracy | ~5% | ~20% | **43.5%** |
| Completion | Block-level Accuracy | ~15% | ~35% | **70.0%** |
| Completion | Edit Distance | ~0.95 | ~0.65 | **0.23** |
| Anomaly | F1 | 0% | 0% | **100%** |

### Per-Family Breakdown

| Family | Top-1 | Top-3 | MRR |
|---|---|---|---|
| MOSFET | 70.8% | 99.4% | 0.85 |
| IGBT | 69.2% | 100% | 0.85 |
| IC | 60.6% | 98.2% | 0.80 |

### Scaling Analysis

| Model | Parameters | Val Accuracy | Time/Epoch |
|---|---|---|---|
| LSTM tiny | 200K | 80.8% | 1.8s |
| LSTM small | 1.1M | 80.9% | 2.0s |
| LSTM medium | 6.4M | 80.9% | 11s |
| Transformer small | 6.4M | 81.6% | 14s |
| Transformer medium | 33.8M | 80.9% | 350s |

All architectures converge to ~80.9% validation accuracy regardless of size, indicating the ceiling is set by inherent task ambiguity (synonym choices, optional steps) rather than model capacity.

## What Worked

- **RF + neural hybrid**: The Random Forest's 99.9% top-15 recall means the neural model almost never has to rank among impossible candidates. This is the single biggest contributor to our Top-3/5 scores.
- **Custom tokenizer**: One token per process step eliminates subword noise and gives clean attention patterns.
- **Small models, fast iteration**: The LSTM small model trains in ~2 minutes on an A100, enabling dozens of experiments.
- **Physics validator for Task 3**: Using the actual process rules for anomaly detection is both honest and effective.

## What Didn't Work

- **Scaling up**: More parameters, more data, and more epochs all plateau at ~80.9% accuracy. The bottleneck is synonym ambiguity, not model capacity.
- **Block-based hard masking**: Filtering predictions by process block structure was too aggressive — many steps appear in multiple blocks (e.g., litho steps in via/metal blocks). Dropped Top-1 from 67% to 54%.
- **Canonicalization**: Merging synonyms improved self-eval numbers but would fail on the organizer's eval set which uses original names.

## What We'd Do With Another 36 Hours

- **Synonym-aware evaluation**: Count predictions as correct if they're valid synonyms of the true answer
- **Category auxiliary head**: Predict the step's functional category (CLEAN, DEPOSIT, ETCH, etc.) alongside the step itself — transfers better to unseen families
- **Contrastive learning for anomaly detection**: Train the model to distinguish valid from invalid sequences in embedding space, rather than relying solely on the rule validator
- **OOD generalization**: Test on the unknown 4th product family using category-level predictions and physics constraints

## Credits & Dependencies

- **Infrastructure**: Leonardo supercomputer (CINECA), NVIDIA A100 GPUs, AI Factory Austria
- **Libraries**: PyTorch 2.3.1, scikit-learn, numpy, matplotlib
- **Data**: Synthetic sequences from provided grammar (`generate_sequences.py`)
- **Physics layer**: Process knowledge base and state machine (Mina's contribution)
- **Environment**: pixi package manager
- **AI tools**: Claude Code (Anthropic) for code generation and debugging

---

## Appendix — Bayes ceiling & training injections (this fork's contributions)

### Why next-step accuracy plateaus at ~81%
Measured model-independently (`oracle_ceiling.py`, full write-up in
`CEILING_ANALYSIS.md`): the generator makes random *valid* choices (documented
synonyms + optional steps) at ~half of all positions, so the best predictor that
can exist caps at **Top-1 ≈ 0.82 / Top-5 = 1.000** (0% duplicate sequences → no
leakage). Models converge to ~98% of that ceiling, so next-token Top-1 is
*saturated*; we optimize the non-saturated signals — **Task-2 completion, Task-4
OOD, the physics layer** — and report Top-5/MRR for Task-1.

### How the model internalizes the rules (injections, in `jobs/leonardo/train.slurm`)
- **Next-category aux head** (`--aux-category`) — learn step *function* (transfers OOD).
- **UNK-dropout** (`--unk-dropout`) — rely on context, not memorized names.
- **Synonym-collapse loss** (`--synonym-collapse`, `physics/synonyms.py`) — credit
  any synonym of the gold step; removes coin-flip entropy (outputs stay exact).
- **Pseudo-family OOD augmentation** (`generate_data.py --ood`, novelty spectrum).
- **GRPO with the physics verifier as reward** (`src/train_grpo.py`).
- **Task-2 physics-vetoed beam search** (`refinery.beam_decode`) — −36% edit distance vs greedy.

Reproducible correctness: `exhaustive_test.py` 42/42 · `differential_fuzz.py`
0 disagreements (incl. casing) · `ood_benchmark.py` · `robustness_test.py`.
