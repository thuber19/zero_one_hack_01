# The submission is procseq — physics is its verification companion

> **DECISION (final).** The **submission is Pipeline B — `solution/procseq`** (two
> from-scratch neural models that *learn* the process logic). **Pipeline A —
> `src/` + `physics/`** is kept only as the **verification companion**: procseq
> *imports* its rule engine + refinery (`physics.ontology`, `refinery.PhysicsRefinery`)
> for the Task-2/3 hybrids, and its deterministic checker is the honest yardstick
> we measure the learned models against. A is **not** submitted on its own.

This doc compares the two and records why procseq is the deliverable. Both run on
Leonardo via pixi (procseq: `-e procseq` env), both on `torch==2.3.1` (CUDA 12).

- **Pipeline B — "procseq" (`solution/procseq/`) — THE SUBMISSION** — two specialized neural models that learn the logic.
- **Pipeline A — "physics" (`src/` + `physics/`) — verification companion** — *model proposes, physics disposes*; its rule engine is the yardstick + the symbolic safety net procseq reuses.

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

**procseq (B) is the submission; physics (A) is the verification companion.** The
two are complementary, and procseq deliberately *reuses* A's symbolic layer rather
than competing with it:

| Task | What procseq submits | How physics (companion) is used |
|---|---|---|
| **1 — next-step** | procseq **decoder** top-5 (`infer_hybrid`: legal-first rerank) | A's `refinery.rerank` vetoes illegal candidates |
| **2 — completion** | procseq **decoder** + **physics beam-decode + repair** (`infer_hybrid`) → *guaranteed rule-valid* | A's `refinery.beam_decode` is the safety net procseq drives |
| **3 — anomaly** | procseq **encoder** (DeBERTa + contrastive) gives the *learned* verdict + continuous score (`infer_anomaly_hybrid`); rule engine supplies the exact label/attribution | A's rule engine = the verdict procseq's score is calibrated against + the honesty yardstick |

In the **write-up / slides**, physics provides the credibility yardstick:
"the rule engine guarantees validity; our **learned** models reach X% of that
*without* relying on the rules, and generalize to the hidden 4th family at Y%
(category-level Z%)." The submitted artifact is procseq; physics proves it honest.

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

---

## 7. Combining the two: which of A's harnesses B should reuse

A red-team audit (see `solution/WHAT_IS_PROCSEQ.md` §10) showed that **A's evaluation
harnesses are circular** (the rule checker grades itself) — so we should NOT reuse A's
*self-consistency* tests (`exhaustive_test`, `differential_fuzz`, `make_bad_testset`)
as evidence of quality. But several of A's harnesses are genuinely useful to B:

| A's harness | Reuse in B? | Why / how |
|---|---|---|
| **`pseudo_family.py`** (4th-family generator) | ✅ **done** | B's new `procseq/ood_novel.py` reuses `pseudo_sequence` + `inject_violation` for a *real* novel-vocabulary OOD test of the learned models (the auditor's keyword-free attack, owned). |
| **`data/eval_metrics.py`** (official scorer) | ✅ **adopt as single source of truth** | Score *both* pipelines with this one file; keep `procseq/eval_metrics.py` only as a fast dev convenience. Kills the "duplicate scorers may diverge" risk. |
| **`robustness_test.py`** (malformed-input fuzz) | ✅ adapt | Fire empty / over-length / unknown-family / novel-vocab inputs through B's `infer.py` and assert no crash + well-formed CSV. |
| `physics/refinery.py` (beam + repair decode) | ✅ **BUILT** (`procseq/infer_hybrid.py`) | B's learned decoder proposes, A's refinery disposes: Task 1 = legal-first rerank, Task 2 = physics beam-decode → completions are *both* learned **and** guaranteed rule-valid (verified: 0 violations even on an untrained model). Writes `submission_task{1,2}_hybrid{,_real}.csv`. |
| `src/random_forest.py` (candidate filter) | ⚠️ deferred | RF is bound to A's **own tokenizer indices** + `block_classifier`, and the trained `random_forest.pkl` only exists *after* A's run — so it can only be layered on B at the *name* level once that artifact exists. Small marginal gain on top of decoder+physics-legality; do it only if Task-1 numbers ask for it. |
| `exhaustive_test` / `differential_fuzz` | ❌ skip | Circular — they prove a function equals itself, not skill. |

### The "one pipeline" options (pick one)

- **Option 1 — Two pipelines, one submission (recommended, low risk).** Keep A and B
  separate; score both with the official `data/eval_metrics.py`; submit the
  per-task winner (§4). B's role: the honest "did the model learn it + where it breaks"
  evidence (`ood_novel`). *This is the fastest credible path.*
- **Option 2 — Hybrid model (higher ceiling, more work).** Make B's learned decoder the
  *generator* and feed its candidates into A's `refinery` beam/repair for Task 2, and
  report B's encoder **and** A's rule engine for Task 3 side by side. One `inference`
  entry point, two model backends. Do this only after Option 1 produces real numbers.
- **Option 3 — Full merge into A.** Register B's decoder/encoder as backends inside
  A's `inference.py`; delete B's duplicate tokenizer/metrics. Most work; only if there's
  time after a correct submission exists.

### Concrete next steps for convergence
- [ ] Run **both** pipelines on Leonardo; score **both** with `data/eval_metrics.py`.
- [ ] Run `procseq/ood_novel.py` on B's trained models → the honest OOD curve.
- [ ] Decide Option 1 vs 2 based on the numbers (don't merge before measuring).
- [ ] One results table + one `data/eval_metrics.py` for both → the submission story.
