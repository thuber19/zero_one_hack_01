# Hand-off: train a model that INTERNALIZES the process rules

This is the short list for the team. Everything here is wired and **smoke-verified
on CPU** (tiny model, a few steps) — the real runs go on Leonardo (GPU). The goal:
the *model itself* learns the constraints, with the deterministic physics engine
as the reward signal and the always-on safety net.

## The 4 levers you can use (and what each does FOR THE MODEL)

| Lever | File / flag | What it teaches the model | Cost |
|---|---|---|---|
| **Integrated data (real + pseudo-families)** | `src/generate_integrated_data.py` | the 3 families' grammar **+** OOD generalization (pseudo-families span a *novelty spectrum*: mostly-shared → heavily-novel vocabulary, matching the README's hidden 4th family) | cheap |
| **Aux category head** | `src/train.py --aux-category` | predict each step's physical **function** (deposition/etch/…), not just its name → transfers to unseen families | ~free |
| **UNK-dropout** | `src/train.py --unk-dropout 0.15` | rely on **context**, not memorized tokens → robust to novel vocabulary | ~free |
| **GRPO (verifier reward)** | `src/train_grpo.py` | **prefer physically-legal continuations** — the physics verifier (`reward.py`) rewards valid sequences, pushing the rules into the weights | medium (RL) |

## The recipe (run on Leonardo, in order)

```bash
cd tracks/industrial-infineon

# 1) Data: real (all variation axes) + pseudo-families (novelty spectrum)
python src/generate_integrated_data.py --extra-data 30000 --ood 8000 --output-dir outputs_run

# 2) SFT: next-step + aux category head + UNK-dropout  (the model learns patterns + function)
OUTPUT_DIR=outputs_run python src/train.py --model-size medium --epochs 60 \
    --aux-category --unk-dropout 0.15 --device cuda

# 3) GRPO: make the model PREFER valid continuations (verifier reward)  <-- the new internalization step
OUTPUT_DIR=outputs_run python src/train_grpo.py --init-from outputs_run/best_transformer.pt \
    --data-dir outputs_run --steps 3000 --group-size 8 --device cuda

# 4) Evaluate the MODEL's own understanding (not just the engine)
OUTPUT_DIR=outputs_run python src/evaluate.py --self-eval --output-dir outputs_run --model-size medium
python category_eval.py --model-dir outputs_run --model-size medium   # ID vs OOD functional accuracy
python ood_benchmark.py                                               # token + structure OOD (no model needed)
```

Final model for submission: use `outputs_run/grpo_transformer.pt` (or
`best_transformer.pt` if GRPO didn't help on your metrics — compare with
`benchmark_models.py`).

## Always keep the safety net (do NOT remove)
At inference, the model **proposes** and the physics **disposes** — this is what
guarantees correctness regardless of how well training went:
- `src/inference.py` already wires it: Task-1 rerank, Task-2 constrained decode
  (guaranteed-valid completions), Task-3 = `validate_sequence_combined`.
- It is **proven equivalent to the organizers' reference checker** on in-vocab
  inputs (`differential_fuzz.py`, 0 disagreements incl. case/whitespace) and
  returns **INSUFFICIENT_INFORMATION** instead of guessing on opaque OOD tokens.

## How to measure that the MODEL actually learned it (not just the engine)
- `category_eval.py` — does the model predict the next step's *function*? (ID vs OOD)
- `src/evaluate.py --self-eval` — prints a **model-only** Task-3 ROC-AUC (no physics) so you can see the network's *unaided* anomaly skill improve as you scale/GRPO.
- `ood_benchmark.py` — token + structure transfer (engine-level, verifiable ground truth).

## What's verified vs what needs your GPU run
- **Verified now (CPU smoke / full):** all 4 levers run; GRPO loop works and the
  verifier reward flows; engine is grader-equivalent; OOD benchmark ~1.0;
  exhaustive 42/42; submission format clean.
- **Needs your run:** full-size SFT + GRPO at scale (tiny CPU models are
  undertrained — that's the whole point of handing off to the cluster).

## One honest note
The deterministic engine is the reliable part and should stay. GRPO + aux head +
UNK-dropout are what make the *model* internalize the rules — expect the model's
*standalone* OOD/anomaly numbers to rise with scale, but the engine remains the
correctness guarantee. See `EVIDENCE_AUDIT_R2.md` for the proof-of-correctness and
`AUDIT_AND_PLAN.md` for the full picture.
