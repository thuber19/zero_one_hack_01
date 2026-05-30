# What is "procseq"? — a plain-English explanation of what you built

You were lost in the weeds. Here is the whole thing, top to bottom, in human terms.

---

## 0. The challenge (what Infineon actually asked)

A chip is made by running a wafer through a long, ordered list of factory steps
(clean → oxidize → pattern → etch → implant → … → test → ship). For 3 product
families (MOSFET, IGBT, IC), each "recipe" is ~115–150 steps. The order matters:
some steps are illegal unless an earlier step happened (you can't etch without
first patterning a mask, can't deposit on a dirty surface, can't ship before testing).

They ask 3 things, and a 4th hidden one:
1. **Next-step prediction** — given a partial recipe, what's the next step?
2. **Sequence completion** — given the first 60%/80%, predict the rest.
3. **Anomaly detection** — given a full recipe, is it valid? If not, which rule broke?
4. **(hidden) OOD** — does it generalize to a 4th, unseen family?

The deep question: **does a model truly learn the process *logic*, or just memorize?**

---

## 1. What procseq IS (the elevator pitch)

**procseq is a complete, from-scratch machine-learning pipeline that treats a chip
recipe like a sentence and learns the "grammar" of chip-making.** It trains two
small neural networks — one that *writes* recipes (for tasks 1 & 2) and one that
*judges* them (for task 3) — plus all the plumbing to generate data, train on the
Leonardo supercomputer, run the models, and score them honestly.

"procseq" = **proc**ess **seq**uences. It's the name of the Python package
(`solution/procseq/`). That's all the word means.

---

## 2. The 3 big ideas (and why)

**Idea 1 — treat each factory step as a single "word."**
Normal language tokenizers would chop `"DEPOSIT NITRIDE FILM"` into pieces. That's
nonsense here. So we built a **custom tokenizer** where each whole step =
**one token** (a vocabulary of ~210 "words"). The model reads/writes recipes the
way you read/write sentences. *(file: `tokenizer.py`, `vocab.py`)*

**Idea 2 — two specialized models instead of one do-everything model.**
- A **decoder** (a small GPT/Llama-style network) to *generate* the next step or
  finish a recipe — that's tasks 1 & 2. *(`models/decoder.py`)*
- An **encoder** (a DeBERTa-style network) to *classify* a whole recipe as
  valid/invalid and say which of 10 rules broke — that's task 3. *(`models/encoder.py`)*

We trained both **from scratch** (no pretrained internet model) because the
vocabulary is tiny and domain-specific — pretrained English knowledge is useless
here, and from-scratch is faster and is the whole "European AI sovereignty / own
stack" point of the hackathon.

**Idea 3 — measure honestly, don't just claim it works.**
We built our own scoring + baselines so we can prove the model learned *logic*, not
just patterns (more in §6).

---

## 3. How it works end-to-end (follow the data)

```
generate_sequences.py (the official grammar that makes valid recipes)
        │
        ▼
 build_data.py ──> makes lots of training recipes + a held-out test set
        │           and builds the tokenizer
        ▼
 train_decoder.py ──> trains the "writer" model   (tasks 1 & 2)
 train_encoder.py ──> trains the "judge"  model   (task 3)
        │
        ▼
 infer.py / infer_anomaly.py ──> run the models, write the 3 submission CSVs
        │                         (also `--real` = on the organizers' actual eval files)
        ▼
 run_eval.py + eval_metrics.py ──> score everything (accuracy, F1, etc.)
        │
        ▼
 demo.py ──> before/after examples + plots for the slides
```

One command, `make smoke`, runs this whole chain locally in under a minute on a
tiny config to prove it all connects. On Leonardo it runs full-size on an A100.

---

## 4. The two models, explained simply

**The decoder ("the writer") — tasks 1 & 2.**
It reads the recipe so far and predicts the next step, over and over, like
autocomplete. For task 1 we take its top-5 guesses. For task 2 we let it keep going
until it writes "SHIP LOT". We added a safety net called **grammar-constrained
decoding**: at each step we check the candidate against the *official rule checker*
(`validate_sequence`) and skip any step that would break a rule — so the recipes it
writes are actually legal. *(`infer.py`)*

**The encoder ("the judge") — task 3.**
It reads a whole recipe at once and outputs (a) is this valid? and (b) which of the
10 rules is violated. We added a **contrastive** training trick: for every valid
recipe we also make a "twin" that's broken in exactly one place, and teach the model
to push those apart in its internal representation — so it learns *why* something is
invalid, not just memorize. *(`models/encoder.py`, `contrastive.py`, `anomaly_inject.py`)*

---

## 5. Every file, one line each

**The data layer**
- `grammar.py` — bridge to the official recipe generator + rule checker (`validate_sequence`).
- `vocab.py` — the list of ~210 step "words".
- `tokenizer.py` — turns recipes into numbers (one token per step) and back.
- `data.py` — loads/generates recipes, splits train/val/test, balances by length (UCBS).
- `anomaly_inject.py` — breaks a valid recipe in exactly one of 10 ways (to train the judge).
- `anomaly_data.py` — builds the balanced valid/invalid training set for the encoder.
- `make_eval.py` — builds our own held-out eval files (mirrors of the organizers' format).
- `build_data.py` — the "make all the data + tokenizer" driver.
- `canon.py` / `external.py` — optional synonym-collapsing + reuse of the team's
  category map (`classify_step`) for diagnostics.

**The models + training**
- `models/decoder.py` — the writer network (Llama-style).
- `models/encoder.py` — the judge network (DeBERTa-style) + contrastive head.
- `datasets.py` — feeds recipes to the models in batches.
- `train_decoder.py` — trains the writer (with live logging + a val accuracy readout).
- `train_encoder.py` — trains the judge (binary + 10-rule + contrastive losses).
- `contrastive.py` — the "push valid/invalid apart" loss (SupCon).
- `config.py` — reads the YAML settings.

**Running + scoring**
- `infer.py` — produces task 1 & 2 submission CSVs (with the grammar safety-net).
- `infer_anomaly.py` — produces the task 3 submission CSV.
- `run_eval.py` — orchestrates scoring of all 3 tasks.
- `eval_metrics.py` — the actual metric math (accuracy, MRR, F1, ROC-AUC, edit distance…).
- `baselines.py` — dumb baselines (n-gram, perplexity, rule-oracle) to compare against.

**Experiments + demo**
- `ood_probe.py` — train on 2 families, test on the 3rd → fakes the hidden Task 4.
- `sweep_run.py` — train many sizes × data amounts → the "scaling" experiment.
- `demo.py` — before/after examples + plots.
- `dashboard/app.py` — optional Streamlit results dashboard.

**Infra**
- `configs/` — YAML settings (`smoke.yaml` = tiny/local, `leonardo_*.yaml` = full).
- `slurm/` — Leonardo job scripts.
- `Makefile` — shortcuts (`make smoke`, `make data`, etc.).
- `tests/` — 29 unit tests proving each piece works.

---

## 6. The clever bits (your talking points for the write-up)

These are the things that make procseq more than "we trained a model":

1. **Atomic-step tokenizer** — one token per process step (no subword fragmentation).
2. **Grammar-constrained decoding** — completions are *guaranteed rule-valid* because
   we veto illegal next-steps using the official checker.
3. **Contrastive anomaly detection** — uses free "hard-negative twins" (a valid recipe
   vs. the same recipe broken once) so the judge learns the *reason* for invalidity.
4. **The logic probe** — we run the model's *generated* recipes back through the rule
   checker and report what % are valid. This separates "learned the logic" from
   "memorized prefixes" — the heart of the hackathon question.
5. **Category-level metrics (diagnostic)** — measures whether the model got the right
   *kind* of step (clean/etch/deposit…) even when the exact word is wrong — the clearest
   "it understands structure" signal, especially out-of-distribution.
6. **OOD probe + scaling sweep** — leave-one-family-out generalization, and accuracy
   vs. model size / data volume.
7. **Honest baselines** — an n-gram model (the "memorization floor") and the rule-engine
   (the "perfect-rules ceiling") so every number has context.

---

## 7. How to run it

**Locally (sanity check, < 1 min):**
```bash
cd solution
make smoke          # runs the whole pipeline tiny → proves it connects
```

**On Leonardo (the real thing):** pick option **7** (writer) or **8** (judge) in
`jobs/run.sh`. It trains on an A100 and writes the real submissions to
`$SCRATCH/runs/procseq_*/submission_task*_real.csv`. (See `LEONARDO.md`.)

---

## 8. Honest limits (say these out loud — judges reward it)

- The smoke numbers are bad **on purpose** (5-step untrained toy models). Real numbers
  come from the Leonardo run.
- For raw score, the team's physics/rule pipeline likely beats procseq on tasks 2 & 3
  (rules guarantee validity). procseq's value is the **learned + honest-evaluation +
  OOD** story. See `../PIPELINES.md` for how the two fit together.
- Canonicalization is off by default (the grader scores exact step names).

---

## TL;DR

You built **procseq**: a from-scratch, two-model ML pipeline (a "writer" decoder for
next-step/completion + a "judge" encoder for anomaly detection) on a custom
step-level tokenizer, with grammar-constrained decoding, contrastive learning, an
honest self-scoring harness, OOD + scaling experiments, and full Leonardo/pixi
plumbing. It answers the hackathon's core question — *did the model learn the
process logic?* — with measurements, not hand-waving.
