# Design Spec: Infineon Process-Logic Pipeline (`procseq`)

**Date:** 2026-05-29
**Track:** Industrial AI — Learning & Benchmarking Process Logic (Infineon)
**Status:** Approved — ready for implementation planning

---

## 1. Purpose & Scope

Build a complete, reproducible, end-to-end pipeline that learns semiconductor
process-flow logic from synthetic step sequences and is benchmarked on the three
hackathon submission tasks, plus self-simulated OOD generalization. The pipeline
must run unchanged as a tiny **local smoke test** (CPU/MPS on a laptop) and at
full scale on the **Leonardo (CINECA) GPU cluster** via SLURM + Accelerate +
DeepSpeed.

This is "the whole nine": all 3 submission tasks, a scaling experiment (model
size × data volume), a before/after demonstrator, and a self-scoring evaluation
harness.

### Confirmed decisions

- **Two models, both trained from scratch** on the ~120-step domain vocabulary:
  - **Decoder** (Llama-style causal LM) for Task 1 (next-step) and Task 2 (completion).
  - **Encoder** (DeBERTa-v2-style, disentangled attention) for Task 3 (anomaly + rule attribution).
  - Rationale: the domain vocab is ~150 atomic tokens; pretrained NL weights bring
    large embedding tables and subword fragmentation for near-zero benefit. From-scratch
    is faster to iterate, enables clean scaling experiments, and maximizes the
    "European AI sovereignty / own stack" narrative.
- **Stack:** HuggingFace Transformers (model definitions) + Accelerate + DeepSpeed
  (ZeRO-2) + SLURM launcher.
- **Leonardo assumptions:** templated SLURM scripts with sensible CINECA Leonardo
  Booster defaults (`boost_usr_prod` partition, 4×A100-64GB/node, `module load`,
  `srun accelerate launch`); account / quota / storage paths are fill-in variables.
- **Anomaly submission** = the trained DeBERTa classifier. The rule-checker
  (`validate_sequence`) is reported only as an honest **oracle ceiling**, never
  submitted. A perplexity-threshold detector is reported as a **lower-bound baseline**.
- **Experiment tracking:** TensorBoard by default (self-hosted, no external API);
  W&B in offline mode optional.
- **Location:** `tracks/industrial-infineon/solution/`.

### Out of scope (for this build)

- DeepSpeed-Ulysses sequence parallelism — token sequences are ~150 long, so
  ZeRO-2 data parallelism suffices. We will state this explicitly rather than
  cargo-cult it.
- Fine-tuning large pretrained LLMs (Qwen/Mistral). Left as a possible future
  variant; the from-scratch models are the primary submission.
- Real Leonardo account credentials (filled in at the cluster).

---

## 2. Background (from the track materials)

- Three product families: **MOSFET** (~126 steps), **IGBT** (~151), **IC** (~107).
- Sequences are long-format CSV (`SEQUENCE_ID, STEP`), one step per row; always
  start with `RECEIVE WAFER LOT`, end with `SHIP LOT`.
- The track ships `generate_sequences.py` with:
  - `generate_sequence(family, rng)` / `generate_dataset(...)` — grammar-driven valid generation.
  - `validate_sequence(steps) -> list[Violation]` — implements all 10 forbidden-pattern rules.
  - `read_csv_sequences(path) -> dict[id, list[step]]`, `write_csv(path, sequences)`.
- `generation_rules.md` is authoritative: vocab (~120 steps, 12 categories), block
  grammar per family, the 10 rules, variation axes, eval protocol, submission formats.
- **Not in repo** (we must build): `eval_metrics.py` (organizers distribute their
  own; ours is for self-scoring), the organizer eval input files
  (`eval_input_valid.csv`, `eval_input_anomaly.csv` — distributed at event start),
  and all model/training/inference code.

### The 10 rules (Task-3 attribution labels)

`RULE_DEP_NO_CLEAN`, `RULE_METAL_ETCH_NO_LITHO`, `RULE_ETCH_NO_MASK`,
`RULE_LITHO_LEVEL_SKIP`, `RULE_IMPLANT_NO_MASK`, `RULE_CMP_NO_DEP`,
`RULE_PAD_OPEN_BEFORE_DEP`, `RULE_TEST_BEFORE_PASSIVATION`,
`RULE_SHIP_BEFORE_TEST`, `RULE_BACKSIDE_BEFORE_PASSIVATION`.

### Eval protocol shapes we must match

- `eval_input_valid.csv`: `EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE`
  (pipe-separated steps), 600 rows = 100 seqs × 3 families × {0.6, 0.8}.
- `eval_input_anomaly.csv`: `EXAMPLE_ID, FAMILY, SEQUENCE` (pipe-separated), 987 rows
  = 600 valid + 387 with injected violations, shuffled/unlabeled.
- Submission formats:
  - Task 1: `EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5`
  - Task 2: `EXAMPLE_ID, PREDICTED_SEQUENCE` (suffix only — steps **after** the cut)
  - Task 3: `EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE`

---

## 3. Repository Layout

```
tracks/industrial-infineon/solution/
├── procseq/                       # python package
│   ├── __init__.py
│   ├── vocab.py                   # canonical step list + step<->token mapping
│   ├── tokenizer.py               # atomic-step WordLevel tokenizer (build/load)
│   ├── data.py                    # scaling, UCBS, splits, eval-mirror builders
│   ├── anomaly_inject.py          # inject each of the 10 rule violations + label
│   ├── models/
│   │   ├── __init__.py
│   │   ├── decoder.py             # build_decoder(size) -> HF LlamaForCausalLM
│   │   └── encoder.py             # build_encoder(size) -> DeBERTa + 2 heads
│   ├── train_decoder.py           # Accelerate entrypoint (CLM)
│   ├── train_encoder.py           # Accelerate entrypoint (binary + rule heads)
│   ├── infer.py                   # produce the 3 submission CSVs
│   ├── eval_metrics.py            # self-scoring harness (all task metrics + logic probe)
│   ├── baselines.py               # n-gram/Markov, perplexity, rule-oracle
│   └── demo.py                    # before/after side-by-side + plots
├── configs/
│   ├── smoke.yaml                 # tiny local config (CPU/MPS)
│   ├── leonardo_decoder.yaml
│   ├── leonardo_encoder.yaml
│   ├── scaling_grid.yaml          # size × data-volume sweep definition
│   ├── ds_zero2.json              # DeepSpeed ZeRO-2 config
│   └── accelerate.yaml            # Accelerate config (filled by SLURM env)
├── slurm/
│   ├── train_decoder.sbatch
│   ├── train_encoder.sbatch
│   └── scaling_sweep.sbatch       # array job over the grid
├── dashboard/
│   └── app.py                     # optional Streamlit (README bonus)
├── artifacts/                     # checkpoints, predictions, plots, metrics.json (gitignored)
├── tests/                         # unit tests for tokenizer, metrics, injection
├── Makefile
├── requirements.txt
└── README.md
```

The package imports the existing `generate_sequences.py` (grammar, `validate_sequence`,
`read_csv_sequences`, `write_csv`) rather than reimplementing it. Import path is made
robust (the track folder isn't a package), e.g. via a thin local re-export module.

---

## 4. Components

### 4.1 Vocabulary & Tokenizer (`vocab.py`, `tokenizer.py`)

- **Canonical step set:** derived from `generation_rules.md` Section 1 plus every
  distinct `STEP` observed in the provided CSVs (union, to catch synonyms). Each step
  is one atomic unit.
- **Atomic-step encoding:** map internal spaces to underscores
  (`"RECEIVE WAFER LOT"` → `RECEIVE_WAFER_LOT`); join steps with spaces; use a
  `tokenizers` `WordLevel` model + `Whitespace` pre-tokenizer → **one token per step**.
  This eliminates the subword fragmentation the brief warns about.
- **Special tokens:** `[PAD] [UNK] [BOS] [EOS]`, family tokens
  `[FAM_MOSFET] [FAM_IGBT] [FAM_IC]`, and `[CLS] [SEP] [MASK]` (encoder).
- Wrapped as `PreTrainedTokenizerFast`. Deterministic; vocab serialized to
  `artifacts/tokenizer/`. Round-trip tested (encode→decode identity on real sequences).
- **`ALIGN MASK LEVEL N` / `EXPOSE LITHO LEVEL N`:** level numbers vary. Decision:
  treat each `... LEVEL {1..6}` as its own token (small, bounded set), preserving the
  level signal the model needs for `RULE_LITHO_LEVEL_SKIP`. The vocab builder
  enumerates levels 1–8 to be safe.

### 4.2 Data Layer (`data.py`, `anomaly_inject.py`)

- **Scaling:** `build_dataset(family, n, seed)` wraps `generate_dataset` to emit N
  valid sequences/family (default 5,000), de-duplicated against each other and against
  the provided 1,000.
- **UCBS (Uniform Case-Based Sampling):** bucket sequences by length; sampler draws
  uniformly across buckets so batch composition isn't length-biased.
- **Splits:** deterministic train / val / held-out test by `SEQUENCE_ID` hash — no
  sequence appears in two splits.
- **Internal eval mirrors** (built from the held-out test split, matching organizer schemas):
  - `eval_input_valid.csv` + `eval_valid_groundtruth.csv`: 100 held-out seqs/family,
    truncated at 0.6 and 0.8 → 600 partials, with full sequences kept for scoring.
  - `eval_input_anomaly.csv` + `eval_anomaly_labels.csv`: 600 valid + 387 injected
    (≈39 per rule across the 10 rules), shuffled; labels (`IS_VALID`, `RULE`) held aside.
- **Anomaly injection (`anomaly_inject.py`):** for each of the 10 rules, a function
  that takes a valid sequence and minimally perturbs it to trigger exactly that rule
  (e.g., delete the clean step preceding a deposition → `RULE_DEP_NO_CLEAN`; swap
  `ALIGN MASK LEVEL 2/3` order → `RULE_LITHO_LEVEL_SKIP`; move `SHIP LOT` before
  `WAFER SORT TEST` → `RULE_SHIP_BEFORE_TEST`). Every injected sequence is verified
  with `validate_sequence()` to confirm it triggers the intended rule (and ideally
  only that rule). This same labeled set trains the encoder.
- **OOD probe:** leave-one-family-out splits (train on 2 families, evaluate on the
  3rd) to self-simulate the hidden Task-4 family and report ID→OOD drop ourselves.

### 4.3 Decoder Model — Tasks 1 & 2 (`models/decoder.py`, `train_decoder.py`)

- From-scratch **`LlamaForCausalLM`** built from a small `LlamaConfig` + our tokenizer
  (RoPE, causal attention, RMSNorm, SwiGLU — HF defaults).
- **Input format:** `[BOS] [FAM_x] step_1 step_2 … step_N [EOS]`. Family conditioning
  is the prefix token; loss is standard next-token CLM (optionally masked so the
  `[FAM_x]` prefix isn't predicted).
- **Scaling grid** (Level-3 stretch):
  - sizes: `tiny` (~1M), `small` (~5M), `base` (~25M), `large` (~85M) params
    (vary `hidden_size`, `num_layers`, `num_heads`);
  - data volumes: {100, 1k, 5k, 20k} sequences/family.
- **Max length** ~200 tokens (covers ~151-step IGBT + specials).
- Trainer: shared Accelerate loop, AdamW + cosine schedule, bf16 on A100, gradient
  checkpointing optional, periodic in-training eval (Task-1 Top-1 on val).

### 4.4 Encoder Model — Task 3 (`models/encoder.py`, `train_encoder.py`)

- From-scratch **DeBERTa-v2-style** encoder (`DebertaV2Config`, disentangled
  content+position attention) + pooled `[CLS]`.
- **Two heads:**
  - binary valid/invalid → single logit, `BCEWithLogitsLoss` (numerically stable, per brief);
  - 10-way rule attribution → multi-label `BCEWithLogitsLoss` (a sequence can break >1 rule).
- **Input:** `[CLS] [FAM_x] full-sequence [SEP]`, max length ~200.
- **Training data:** balanced valid + injected-anomaly set from §4.2; bf16; class
  weighting if needed.
- `SCORE` output = sigmoid of the (negated) invalid logit, i.e. P(valid) in [0,1] for AUC.

### 4.5 Training Infrastructure (`configs/`, `slurm/`)

- **Accelerate + DeepSpeed ZeRO-2.** Both training entrypoints are config-driven
  (YAML), resume-safe (checkpoint every N steps), fixed seeds.
- **DeepSpeed config** `ds_zero2.json`: ZeRO stage 2, bf16, gradient accumulation.
- **SLURM templates** (Leonardo Booster): `#SBATCH` for `boost_usr_prod`, `--gres=gpu:4`,
  `module load` placeholders, `srun accelerate launch`. Variables `ACCOUNT`, `QOS`,
  `TIME`, `DATA_DIR`, `OUT_DIR` at top of each script.
- `scaling_sweep.sbatch`: SLURM **array job** that launches the §4.3 grid in parallel,
  one config per array index.
- **Tracking:** TensorBoard writer (loss, lr, eval metrics over steps); W&B offline optional.

### 4.6 Evaluation Harness (`eval_metrics.py`)

Self-contained (stdlib + numpy). Reads predictions + ground truth, emits
`artifacts/metrics.json` with **per-family and per-truncation breakdowns**.

- **Task 1:** Top-1 / Top-3 / Top-5 accuracy, MRR.
- **Task 2:** Exact Match rate, Normalized Edit Distance (`1 - lev/max(|A|,|B|)`),
  token-level accuracy, block-level accuracy (cluster alignment of step groups).
- **Task 3:** Binary accuracy, precision, recall, F1, confusion matrix, ROC-AUC,
  rule-attribution accuracy (among detected violations).
- **OOD:** ID→OOD performance drop Δ per primary metric (using the leave-one-family-out probe).
- **Process-logic probe (differentiator):** run the decoder's generated completions
  through `validate_sequence()`; report the fraction that are rule-valid. Separates
  "learned logic" from "memorized prefixes" beyond Exact Match.
- CLI compatible with the documented `--task {nextstep,completion,anomaly}
  --ground-truth … --predictions …` shape so it mirrors the organizer script.

### 4.7 Baselines (`baselines.py`)

- **n-gram / Markov next-step** model (statistical floor) — beating it shows the model
  learned more than co-occurrence.
- **Perplexity-threshold anomaly detector** using the decoder's sequence NLL
  (lower-bound for Task 3).
- **Rule-checker oracle** (`validate_sequence`) — labeled clearly as the ceiling /
  answer key, **never submitted**.

### 4.8 Inference → Submission (`infer.py`)

Reads organizer (or mirror) `eval_input_*.csv` and emits exact submission formats:
- Task 1: top-5 ranked next steps per partial (vocab-constrained logits ranking).
- Task 2: autoregressive generation from the partial until `[EOS]`/max-len; output
  **suffix only**, pipe-joined.
- Task 3: encoder → `IS_VALID`, `SCORE`, `PREDICTED_RULE`.
One command per task; submission day is `make submit`.

### 4.9 Demonstrator (`demo.py`, `dashboard/app.py`)

- Side-by-side **baseline vs trained** on identical prompts: next-step (top-5),
  completion (full suffix), anomaly verdict + attributed rule. Saved as text transcripts.
- Plots (matplotlib → PNG): training loss curves, metric bars across families,
  confusion matrix, **scaling curves** (metric vs params, metric vs data volume),
  ID vs OOD comparison.
- Optional **Streamlit dashboard** surfacing the above (README bonus).

### 4.10 Orchestration & Reproducibility (`Makefile`, `README.md`, `tests/`)

- `Makefile` targets: `smoke` (full pipeline, tiny config, <~1 min, CPU/MPS),
  `data`, `train-decoder`, `train-encoder`, `eval`, `submit`, `demo`, `test`.
- Pinned `requirements.txt`; all randomness seeded; `artifacts/` gitignored.
- README documents local smoke run + exact Leonardo launch (module load, sbatch).
- `tests/`: tokenizer round-trip, metric correctness on tiny hand-built cases,
  each anomaly injector triggers its intended rule (asserted via `validate_sequence`).

---

## 5. Data Flow

```
generate_sequences.py (grammar)                generation_rules.md (rules)
        │ generate_dataset / validate_sequence          │
        ▼                                               ▼
data.py ── scale + dedup ── UCBS ── splits ──► train / val / test
        │                                       │
        │ (held-out test)                       ▼
        ├──► eval mirrors (valid + anomaly via anomaly_inject.py, labeled)
        │
        ▼
tokenizer.py (atomic-step WordLevel)
        │
   ┌────┴─────────────────────────┐
   ▼                              ▼
train_decoder.py (CLM)        train_encoder.py (binary + rule heads)
   │  Accelerate+DeepSpeed        │  (+ SLURM on Leonardo)
   ▼                              ▼
decoder ckpt                  encoder ckpt
   │                              │
   ▼                              ▼
infer.py ──► Task1 CSV, Task2 CSV, Task3 CSV
   │                              │
   ▼                              ▼
eval_metrics.py ──► metrics.json (per family / truncation / OOD / logic-probe)
   │
   ▼
demo.py + dashboard ──► transcripts, PNG plots, Streamlit
```

---

## 6. Error Handling & Edge Cases

- **Unknown steps** at inference → `[UNK]`; the eval harness counts `[UNK]` outputs as
  wrong and logs their frequency (signals OOD vocab).
- **Decoder runs past max length / never emits `[EOS]`** → hard cap at
  `max_full_len`, truncate, count as non-exact-match.
- **Empty / malformed eval rows** → skipped with a logged warning; never crash a run.
- **Injection that accidentally triggers multiple rules** → keep only if the intended
  rule fires; multi-rule cases are allowed but labeled with all fired rules (the
  rule head is multi-label).
- **Import of `generate_sequences.py`** from the non-package track dir → resolved via a
  thin re-export shim with an explicit `sys.path` insert, tested in CI/smoke.
- **bf16 unsupported locally** (Mac MPS/CPU) → smoke config falls back to fp32.

---

## 7. Testing Strategy

- **Unit:** tokenizer round-trip; metric functions on tiny hand-computed fixtures
  (MRR, edit distance, F1, ROC-AUC); each of the 10 anomaly injectors asserts its rule
  fires via `validate_sequence`.
- **Smoke (integration):** `make smoke` runs data→tokenizer→both trainings (1–2 steps,
  tiny models)→infer→eval→demo on CPU/MPS, asserting all artifacts are produced and
  the submission CSVs have the exact required columns.
- **Verification before completion:** no "done" claim without showing `make smoke`
  output and `metrics.json`.

---

## 8. Build Phasing

1. **Foundation:** `vocab.py`, `tokenizer.py`, `data.py`, `anomaly_inject.py`,
   eval mirrors, `eval_metrics.py`, `baselines.py`, tests, `make smoke` (data + metrics
   + baselines green). Gives an immediate statistical before/after signal.
2. **Decoder:** `models/decoder.py`, `train_decoder.py`, `infer.py` Tasks 1+2,
   Accelerate config. Smoke train + self-score.
3. **Encoder:** `models/encoder.py`, `train_encoder.py`, `infer.py` Task 3.
   Smoke train + self-score; compare vs perplexity baseline + oracle.
4. **Scale:** `configs/scaling_grid.yaml`, SLURM templates, DeepSpeed config,
   `scaling_sweep.sbatch`; Leonardo run instructions in README.
5. **Demonstrate:** `demo.py`, plots, optional Streamlit dashboard, final README +
   submission docs.

Each phase is independently runnable and verified by `make smoke`.

---

## 9. Success Criteria

- `make smoke` runs the entire pipeline locally end-to-end and produces valid-format
  submission CSVs + `metrics.json`.
- Trained decoder beats the n-gram baseline on Task-1 Top-1 and produces rule-valid
  completions at a markedly higher rate than the baseline (logic probe).
- Trained encoder beats the perplexity baseline on Task-3 F1 and approaches the oracle
  ceiling, with non-trivial rule-attribution accuracy.
- A scaling story (metric vs params, metric vs data volume) is plotted.
- An ID→OOD drop is measured via leave-one-family-out.
- SLURM scripts launch the same code on Leonardo with only fill-in variables changed.
- Before/after demonstrator shows qualitatively sensible improvements.
