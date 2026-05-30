# Training Recipe, OOD Strategy & Pitch

How the harness encodes process understanding into the two Leonardo models,
what secondary systems make it survive an unknown family, and how to show the
whole thing to judges.

---

## 0. The shape of the team's product

```
                         ┌─────────────────────────────────────────┐
   training_data/  ──►   │  HARNESS (this repo, Mina)               │
   (grammar, CSVs)       │   • process_knowledge.py  (the KB)       │
                         │   • step_semantics.py     (CSV physics)  │
                         │   • generate_sequences / bad_data_gen    │
                         │   • export_training_data.py  ────────────┼──► corpora
                         │   • refinery.py  (guard/rerank/decode)   │
                         │   • run_all.py / eval_metrics  (scoring) │
                         │   • explain.py   (why)                   │
                         └───────────────┬─────────────────────────┘
                                         │ training corpora
                  ┌──────────────────────┴───────────────────────┐
                  ▼                                               ▼
        Leonardo model A (from-scratch)              Leonardo model B (fine-tune)
                  │                                               │
                  └───────────────► refinery wraps both ◄─────────┘
                          (always-valid output, even on the 4th family)
```

The two models do Tasks 1/2/3. The harness **feeds** them (data), **guards**
them (refinery), **scores** them (eval), **explains** them (explain.py), and
**bounds** them (symbolic oracle = topline). That division is the product.

---

## 1. Encoding the knowledge INTO the models

Run the exporter (scale `--n-valid` / `--n-bad` up on Leonardo CPU nodes):

```bash
python export_training_data.py --n-valid 5000 --n-bad 20000 --out-dir training_export
```

### Model A — from-scratch sequence model (OOD-robust)
- **Data:** `lm_plain.txt` (one sequence/line, `[FAMILY]` token, steps `|`-joined)
  and `lm_factorized.jsonl` (parallel `steps` / `categories` / `roles`).
- **Tokenisation:** one step string = one token (vocab ~120). Add `[FAMILY]`,
  BOS, EOS.
- **Objective:** next-step language modelling. Add **an auxiliary "next-category"
  head** trained on the `categories` stream (small extra loss). This is the
  single highest-leverage OOD move: a 4th-family step *name* is unseen, but its
  *category* is not — the category head keeps predicting sensibly when the token
  head is lost.
- **Scaling sweep (rubric stretch goal):** train at ≥2 sizes × {100, 1k, 5k+}
  sequences; plot metric-vs-data and metric-vs-params. The exporter makes any
  data volume trivially available.

### Model B — fine-tune LLM (physics-aware, the "understanding")
- **SFT data:** `instruct_nextstep.jsonl`, `instruct_completion.jsonl`,
  `instruct_anomaly.jsonl` (instruction / input / output).
- **The key file is `instruct_anomaly.jsonl`:** the target is not just
  `INVALID` — it is `INVALID. Rule: RULE_X. <physical reason> (step: <what it
  does>)`. Training on this teaches the model to *reason about why*, which is
  exactly what the track is testing (logic vs memorisation).
- **Knowledge injection:** do a short continued-pretraining pass on
  `knowledge_cards.jsonl` (each step's description, real fab parameters,
  category, and preconditions+why), or load it into a retrieval store / system
  prompt. This is where "how electronics are made — in general and specifically
  these" enters the model.
- **Base model:** any open model (no API). Keep it small enough to train/serve
  in the quota; the knowledge cards do the heavy lifting, not raw scale.

### What "fully and thoroughly encoded" means here
The model learns the process three ways at once: **token order** (sequences),
**function** (category/role streams), and **causality** (anomaly-with-why +
knowledge cards). The harness then *guarantees* correctness at inference even
where the model is unsure (next section).

---

## 2. Secondary systems for new families (beyond categories)

Teaching categories is necessary but **not sufficient**. Stack these on top —
they are cheap, transparent, and compose:

1. **Physics guard (already built — `refinery.guard` / `constrained_decode`).**
   Hard constraint at the *category* level: the model proposes, the physics
   state machine vetoes any step that would violate a rule. Works on unknown
   vocabulary because rules name event *classes*, not steps. This alone makes
   **Task 3 ~100%** and makes Tasks 1/2 output **always physically valid**.
2. **Next-category auxiliary head (Model A).** Transfers structure when names
   don't. Use its distribution to break ties among physics-legal candidates.
3. **Category-grammar prior (`refinery.learn_category_grammar`).** A soft signal:
   demote category transitions never seen in training. Soft, not hard, so it
   doesn't over-constrain a genuinely new structure.
4. **Abstention / confidence.** When the model's top candidates are all
   physics-illegal (or it hits many unknown tokens), fall back to the
   category-head prediction filtered by the physics mask. Never emit an invalid
   step; degrade gracefully instead of hallucinating.
5. **The symbolic oracle as a safety net for Task 3** regardless of the model:
   `validate_by_state_machine` is the primary anomaly verdict; the model's score
   is reported alongside (and used for AUC). The model can only *help*, never
   *break*, Task 3.

Net effect: on the unknown 4th family, even if both models degrade, the guarded
output is still valid and Task 3 stays near-perfect — the "performance drop
ID→OOD" the organizers measure is bounded by design.

---

## 3. Evaluation, oracle, and CI (already built)

- **`run_all.py`** produces all three submission files and self-scores with the
  full official metric set (Top-1/3/5, MRR; Exact, NormEditDist, TokenAcc,
  BlockAcc; Acc/Prec/Rec/F1/ROC-AUC/RuleAttribution). Swap in the organizers'
  `eval_metrics.py` when it arrives — formats already match the spec.
- **Symbolic oracle = topline.** Report *model* vs *model+refinery* vs *oracle*.
  The gap between the model alone and the oracle is the **memorisation-vs-logic
  measurement** the track is literally about — a headline result, not an excuse.
- **`exhaustive_test.py`** is the CI gate (36/36 on real data; run it after any
  change). **`bad_data_generator.py --count N`** builds arbitrarily large,
  verified, labelled adversarial sets for robustness testing.

---

## 4. Demo (before/after, live, OOD)

A tight 4-beat demo that maps to the rubric:

1. **Baseline vs trained (next-step & completion).** Same partial sequence;
   show the untrained model's implausible guess vs the trained model's
   process-correct prediction. (`explain.py` narrates why the trained one is
   right.)
2. **Live anomaly + explanation.** Paste a sequence with an injected fault;
   the system flags it AND explains *why it is physically impossible* (from the
   KB). This is the "understanding" money shot.
3. **Unknown-family live test.** Feed a made-up family (e.g. GaN steps the
   models never saw). Show the guarded output stays valid and the violations are
   still caught — generalisation, on stage.
4. **Scaling curve.** One plot: metric vs training-data size (100/1k/5k) and vs
   model size. Directly answers the Level-3 stretch goal.

Visuals to prepare: loss curves, metric-over-training, baseline-vs-trained bar
charts, the ID→OOD drop, and the model-vs-oracle gap.

---

## 5. Pitch — mapped to the judging criteria

> **"We didn't just train a model to copy process sequences — we built a system
> that understands how chips are made, can prove why a process is wrong, and
> keeps working on chips it has never seen."**

| Judging criterion | What we show |
|---|---|
| Technical depth & traceable decisions | Neuro-symbolic stack: declarative KB + interpreter + two trained models + guard. Every decision cites a physical reason. |
| Training & benchmark setup on real infra | Two models on Leonardo (from-scratch + fine-tune), scaling sweep, reproducible export→train→eval pipeline. |
| Reproducibility & clarity of eval | One-command `run_all.py`, full official metrics, `exhaustive_test.py` CI, verified data generators. |
| Baseline vs trained vs scaling | Before/after demo + scaling curves + the model-vs-oracle logic gap. |
| Demo, viz, presentation | Live anomaly explanation + live unknown-family test + clean plots. |
| Generalisation (Task 4) | Category-factorised training + physics guard → bounded ID→OOD drop, demonstrated live. |
| European AI sovereignty | 100% open, local, stdlib stack; no API anywhere; own data generation; trained on EU HPC. |

**One-line positioning:** *a process-logic copilot for semiconductor fabs that
learns the process, guards it with physics, and explains every verdict — open,
reproducible, and sovereign.*

---

## 6. Where work runs

| Work | Where | Why |
|---|---|---|
| Train Model A & B, scaling sweep, eval intervals | **Leonardo GPU** | the graded deliverable |
| Generate millions of sequences; build big bad sets; export corpora | **Leonardo CPU** | embarrassingly parallel, large |
| KB, refinery, explain, exhaustive_test, scoring | **laptop / Leonardo CPU** | runs in seconds, no GPU; ship beside training as the eval/guard layer |
