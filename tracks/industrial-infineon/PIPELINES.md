# Two pipelines, one track — comparison & how we finalize into one

We currently have **two working pipelines** for the Infineon process-logic track,
both submittable, both producing the same 3 CSVs (`next-step`, `completion`,
`anomaly`). This doc explains each, where they differ, and the plan to converge
on a single submission.

- **Pipeline A — "physics" (the team's, `src/` + `physics/`)** — *model proposes, physics disposes.*
- **Pipeline B — "procseq" (Claude's, `solution/procseq/`)** — *two specialized neural models that learn the logic.*

Both run on Leonardo via pixi (A: default env, B: `procseq` env), both on `torch==2.3.1` (CUDA 12).

---

## 1. At a glance

| Aspect | **A — physics (`src/`)** | **B — procseq (`solution/`)** |
|---|---|---|
| Core idea | One generative model + a **symbolic physics layer** that vetoes/repairs/ranks | **Two specialized neural models** + a thin grammar safety-net |
| Generative model | Transformer **or** LSTM (custom), `transformer_model.py` / `lstm_model.py` | From-scratch **Llama-style decoder** (HF) |
| Task 1 (next-step) | model probs → **RF candidate mask** → **physics rerank** (legal-first) | decoder top-5, optional grammar-veto re-rank |
| Task 2 (completion) | **physics beam search** with rule veto + loop detection + repair → *guaranteed valid* | decoder greedy + **`validate_sequence` veto** → valid, but no beam/repair |
| Task 3 (anomaly) | **physics rule engine** (deterministic; ≈ the grader) → near-perfect in-distribution | from-scratch **DeBERTa encoder** + **supervised-contrastive** (learned) |
| Extra training | **Random Forest** candidate filter; **GRPO** (RL) `train_grpo.py` | none (CLM + classification only) |
| Categorization | **core**: `physics/ontology.py` drives OOD + constrained decode | **reused as a diagnostic only** (we import their `classify_step`) |
| Tokenizer | `src/tokenizer.py` | `procseq/tokenizer.py` (atomic-step WordLevel) |
| Data | `src/generate_data.py` → train + eval split | `procseq/build_data.py` (wraps `generate_sequences.py`) |
| Training UX | epoch-based, CosineAnnealingLR, early stopping, val_acc | step-based (Accelerate), CosineAnnealingLR, live logging, periodic val |
| Eval | official `data/eval_metrics.py` | own `procseq/eval_metrics.py` (+ can use official) |
| Launcher | `jobs/run.sh` options 1–6 | `jobs/run.sh` options 7–8 |
| Maturity | **mature, audited** (`AUDIT.md`, fuzz/robustness tests) | newer, unit-tested + smoke |

---

## 2. What each is strong at

**A — physics** wins on raw submission score for the parts the grader rewards most:
- **Task 2:** physics-vetoed **beam search + repair** → completions are *always*
  rule-valid and closer in edit distance. This is hard to beat.
- **Task 3:** the deterministic rule engine is essentially the grader's own checker
  → near-perfect F1 + exact rule attribution **in-distribution**.
- Guarantees: it *cannot* emit an illegal step; well audited; won't crash.
- Risk: the physics layer is hand-built; its OOD behavior on the hidden 4th family
  rests on the **category** fallback (a heuristic), and Task-1/2 lexical OOD is hard
  for any model.

**B — procseq** wins on the "did the model actually *learn* the logic?" story:
- **Task 3:** a **learned** DeBERTa classifier + **contrastive** objective (hard-negative
  twins) — an honest "neural understanding" result and an **OOD hedge** that doesn't
  depend on hand-written rules.
- Clean **ablations**: grammar-veto on/off, canonicalization on/off, category-level
  metrics, leave-one-family-out OOD probe — strong material for the "honest evaluation"
  judging criterion.
- Two specialized models (decoder + encoder) instead of one + symbolic glue.
- Risk: Task-2 decoding is simpler (greedy veto, no beam/repair) → likely higher edit
  distance than A; Task-3 learned classifier is unlikely to beat the rule engine
  in-distribution.

---

## 3. Where they already agree (no work needed)

- Same **data** (`data/`, `generate_sequences.py` grammar, official eval files).
- Same **submission format** (the 3 CSV schemas from §5.3).
- Same **official scorer** (`data/eval_metrics.py`).
- Same **cluster setup** (pixi, torch 2.3.1, `jobs/run.sh`, SLURM account/reservation).
- B already **imports A's `physics.ontology.classify_step`** — the categorization is shared.

---

## 4. The decision: how we finalize into one

The two are **complementary, not competing**. Recommended convergence:

> **Submit Pipeline A as the primary scoring pipeline, and keep Pipeline B as the
> "learned-models + honest-evaluation" companion — then submit the best CSV per task.**

Concretely, the **single final submission** = best-of-both, per task:

| Task | Use for submission | Why |
|---|---|---|
| **1 — next-step** | whichever model scores higher on `data/eval_metrics.py` (A's transformer+RF+rerank vs B's decoder) — **run both, pick the winner** | both are transformers; decide by measured Top-1/MRR |
| **2 — completion** | **A** (physics beam + repair) | guaranteed-valid + lower edit distance is very hard to beat |
| **3 — anomaly** | **A** (rule engine) as primary; report **B** (DeBERTa+contrastive) as the *learned* result + OOD hedge | rules ≈ grader in-distribution; B carries the honesty/OOD story |

And in the **write-up / slides**, B provides the comparison that makes the submission
credible: "the rule engine guarantees validity; our learned models reach X% of that
*without* the rules, and generalize to the hidden family at Y% (category-level Z%)."

### Alternative (if we want ONE codebase, not two)
Merge B's two learned models *into* A as additional backends behind the same
`inference.py`/`refinery`, and delete the duplicated tokenizer/data/eval in B. More
work, only worth it if we have time after a correct submission exists.

---

## 5. What we need to do (action checklist)

**Now (correctness + numbers):**
- [ ] Get **both** pipelines producing real submissions on Leonardo (A: options 1–6; B: options 7–8) — B's GPU fix is in (`torch==2.3.1`).
- [ ] Score **both** on the held-out eval with `data/eval_metrics.py` and fill a results table (Top-1/3/5, MRR, EM, NormEdit, F1, ROC-AUC, rule-attr).
- [ ] Pick the **per-task winner** (table in §4) for the official submission CSVs.

**Convergence:**
- [ ] Agree the **primary = A**, **companion = B** (or commit to the single-codebase merge).
- [ ] Make B import A's tokenizer/data where it reduces duplication (optional).
- [ ] One **`make submit`-style** command per pipeline that writes the 3 final CSVs.

**Story / submission:**
- [ ] Run B's ablations (canon on/off, contrastive on/off, leave-one-family-out OOD).
- [ ] Slides: A = "guaranteed-valid physics"; B = "learned + honest eval + OOD".
- [ ] One repo README pointing at both, with the results table and the per-task choice.

---

## 6. File map (so nobody steps on each other)

```
tracks/industrial-infineon/
├── src/         physics/   jobs/run.sh(1-6)   ← Pipeline A (team)
├── solution/    procseq/   jobs/run.sh(7-8)   ← Pipeline B (procseq)
├── data/        generate_sequences.py, eval_input_*.csv, eval_metrics.py  ← shared
└── PIPELINES.md  (this file)
```
A and B live in **disjoint directories**; the only shared code B touches is read-only
(`data/`, `physics/ontology.py`). Nothing in A depends on B.
