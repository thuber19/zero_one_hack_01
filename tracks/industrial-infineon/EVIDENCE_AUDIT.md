# Evidence-First Re-Audit

Method: every row shows the source `file:line` and/or the exact command output it
is based on. Status words (VERIFIED / PARTIAL / UNVERIFIED / MISSING /
PROVISIONAL) are assigned **after** the evidence, never before. The intent is
adversarial — to find where the system is wrong. Findings that contradict the
earlier optimistic audit are called out explicitly.

Commands referenced below were run in this session against the working tree at
the time of writing; re-run them to reproduce.

---

## Area 1 — Lithography Skip Rule

### Implementation locations (evidence)
- Deterministic engine: `physics/state_machine.py:149-162` (skip/decrease check via
  `state.last_aligned_level`) and `:200-203` (level update); helper
  `_extract_litho_level` `:105`.
- Reference/grader: `training_data/generate_sequences.py:216-248` — builds
  `align_steps` only for tokens that **`startswith("ALIGN MASK LEVEL ")` AND the
  suffix `.isdigit()`** (`:221-224`), case-SENSITIVE.

### Per-path table
| Path | File · function | Rule present | Tested | Evidence |
|---|---|---|---|---|
| Reference impl | generate_sequences.py:216-248 `validate_sequence` | yes | provided data | grader logic itself |
| Deterministic engine | state_machine.py:149-162 `apply_step` | yes | differential_fuzz, exhaustive | flags skip/decrease (below) |
| Combined router | state_machine.py:216-252 `validate_sequence_combined` | yes (delegates) | differential | flags skip (below) |
| Heuristic / OOD path | same engine (category) | yes **iff** ALIGN keyword present | ood_benchmark | flags skip with novel materials |
| Repair | fix.py:150-159 `_fix_one` | yes | exhaustive | renumbers → valid (prior run) |
| Synthetic / benchmark gen | bad_data_generator (injects), pseudo_family (preserves) | yes | exhaustive [7] | prior runs |
| spec_strict (advisory) | physics/spec_strict.py litho block | completion only | selftest | `SPEC_LITHO_NO_DEVELOP` |

### Bypass / adversarial attempts (engine vs reference, this session)
```
two-digit 1->10     engine=[LITHO_SKIP] ref=[LITHO_SKIP]      agree
lowercase 1->3      engine=[LITHO_SKIP] ref=[]                <-- DISAGREE
nondigit level      engine=[LITHO_SKIP] ref=[LITHO_SKIP]      agree
trailing text       engine=[]           ref=[]               agree (BOTH miss)
skip 2->5 (clean)   engine=[LITHO_SKIP] ref=[LITHO_SKIP]      agree
```
Through the **scored router** on lowercase input:
```
combined router : [RULE_LITHO_LEVEL_SKIP]
reference/grader: VALID
=> the scored path DIVERGES from the grader on lowercase input
```
`differential_fuzz.py` contains **no** case mutation (grep for lower/upper/case →
none), so this divergence was never exercised by the equivalence proof.

### Conclusions (after evidence)
- The claim "enforced on every path with **no bypass**" is **FALSE as stated.**
  Two concrete defects:
  1. **Case divergence (PARTIAL):** engine upper-cases (`state_machine.py:149`),
     reference is case-sensitive (`generate_sequences.py:223`). On non-canonical
     casing the scored router **over-flags** vs the grader. Impact depends on the
     real eval's casing, which is **UNVERIFIED** (provided data is uppercase).
  2. **"Trailing text" variant:** both engine and reference silently ignore
     `ALIGN MASK LEVEL 1 (rework)` (neither extracts a level) → a litho skip
     hidden behind decorated names is **missed by both**. Consistent with the
     grader, but a real-world blind spot.
- For **canonical-cased, in-vocabulary** sequences the engine matches the
  reference on the litho rule (differential_fuzz, 0 disagreements — see Area 4).
  Status: **PARTIAL** (verified for canonical case; diverges otherwise; OOD relies
  on the shared `ALIGN MASK LEVEL` keyword, which is UNVERIFIED for a renamed 4th family).

---

## Area 2 — Rule Coverage Matrix

Rule definitions: `physics/process_knowledge.py:283-357` (9 windowed+ordering) and
`:373` (litho). Reference: `generate_sequences.py` `validate_sequence`. Test:
`differential_fuzz.py` (8,500+ mutations; reproducible — Area 4) and
`exhaustive_test.py` (42/42, prior runs).

| Rule ID | Source (rules.md) | Impl location | Test | Status |
|---|---|---|---|---|
| RULE_DEP_NO_CLEAN | §3 L470 | process_knowledge.py:283 | differential+exhaustive | VERIFIED (canonical, in-vocab) · PARTIAL (OOD opaque verb, Area 7) |
| RULE_METAL_ETCH_NO_LITHO | §3 L484 | :292 | differential+exhaustive | VERIFIED (canonical, in-vocab) |
| RULE_ETCH_NO_MASK | §3 L495 | :299 | differential+exhaustive | VERIFIED (canonical, in-vocab) |
| RULE_IMPLANT_NO_MASK | §3 L517 | :306 | differential+exhaustive | VERIFIED (canonical, in-vocab) |
| RULE_CMP_NO_DEP | §3 L528 | :313 | differential+exhaustive | VERIFIED (canonical, in-vocab) |
| RULE_PAD_OPEN_BEFORE_DEP | §3 L539 | :322 | differential+exhaustive | VERIFIED (canonical, in-vocab) |
| RULE_TEST_BEFORE_PASSIVATION | §3 L548 | :336 | differential+exhaustive | VERIFIED (canonical, in-vocab) |
| RULE_SHIP_BEFORE_TEST | §3 L559 | :347 | differential+exhaustive | VERIFIED (canonical, in-vocab) |
| RULE_BACKSIDE_BEFORE_PASSIVATION | §3 L568 | :357 | differential+exhaustive | VERIFIED (canonical, in-vocab) |
| RULE_LITHO_LEVEL_SKIP | §3 L508 | :373 + state_machine.py:149 | differential+exhaustive | PARTIAL (case divergence, Area 1) |

**Scope caveat that applies to the whole matrix:** "VERIFIED" means *the engine
matches the reference checker on the canonical-cased, in-vocabulary mutation
space tested by differential_fuzz*. It does **NOT** mean: (a) verified against the
organizers' real eval labels (those files are undistributed — UNVERIFIED); (b)
verified across casing/whitespace variants (UNVERIFIED, Area 1); (c) verified on
novel-vocabulary 4th-family steps (PARTIAL, Area 7).

---

## Area 3 — Assumption Audit

| Assumption | Enters at | Doc support? | Used in code | Risk | Action |
|---|---|---|---|---|---|
| Eval schema = EXAMPLE_ID,FAMILY,(PARTIAL_)SEQUENCE,… | inference._read_eval | YES (rules.md §5.1) | submission readers | Med | loud schema raise added; **ASSUMPTION** until real file seen |
| SCORE = P(valid), higher=valid | inference.detect_anomaly:289-294 | partial (§5.3) | Task-3 SCORE | Med | **ASSUMPTION**; trivially flippable |
| Block-accuracy = category position match | evaluate compute_completion_metrics | NO | internal metric only | Low | **ASSUMPTION**; not in submission |
| 4th family "mostly shares vocabulary" | known_vocab routing (state_machine.py:240) | partial (README L163, one sentence) | routing decision | Med | **ASSUMPTION**; OOD benchmark quantifies for token-swap only |
| Eval tokens are UPPERCASE / canonical case | engine upper-cases; reference does not | NO (casing unspecified) | litho + keyword matching | **Med–High** | **ASSUMPTION** — newly surfaced (Area 1); unhandled |
| Window sizes 12/15/6 | process_knowledge.py:283-313 | YES (rules.md §3 + ref code) | windowed rules | Low | sourced, not assumed |
| OOD keyword fallbacks (PROBE/PAD/BACKSIDE) | process_knowledge.py:181-198 | derived, not documented | OOD ordering rules | Low | **ASSUMPTION** (heuristic); category-gated |

Newly surfaced and **not previously listed**: the **casing assumption**. It is
undocumented and currently causes the Area-1 divergence.

---

## Area 4 — Benchmark Audit

| Benchmark | Size | Generator | Seed | Reproducible? | Output | Evidence |
|---|---|---|---|---|---|---|
| differential_fuzz | 8000 (+500 orig) | mutate provided variants | CLI `--seed` | **YES** | stdout | same seed → identical: `2000 (valid=1754,invalid=246) BINARY=0` twice this session |
| exhaustive_test | 3003 + boundaries | provided + bad_data_generator | fixed | YES (deterministic) | stdout | prior `42/42` (not re-run this turn → treat as prior-run) |
| ood_benchmark | 600 × 7 f-levels | category-preserving renamer | CLI `--seed` | YES | stdout | prior `f=0.30 Acc 1.000`; see Area 5 caveat |
| real_family_benchmark | 8 families | hand-authored flows | fixed | YES | stdout | author-graded (self-disclosed); prior `F1 0.978` |
| category_eval (model) | M3 ckpt | held-out + real families | fixed | only if ckpt present | outputs_M3/category_eval.json | requires `outputs_M3` (gitignored) → **UNVERIFIED from clean checkout** |
| self_score / make_sample_eval | 36+36 | sampled provided | CLI `--seed` | YES | stdout | reference-labelled (in-vocab) |

Notes: differential_fuzz reproducibility is **VERIFIED** (shown). exhaustive_test
`42/42` is from earlier this session, **not re-run in this evidence pass** — mark
as prior-run, re-runnable. category_eval needs a checkpoint dir that is gitignored,
so from a fresh clone it is **UNVERIFIED** until a model is trained.

---

## Area 5 — OOD Evaluation Audit

### What `ood_benchmark.py` actually does (evidence: the renamer)
- It renames only **device-material words** (`_MATERIALS` set) and **keeps the
  functional verb and the full step ORDER/STRUCTURE** (`make_ood_family`,
  `_novelize`). Shared structural vocabulary (litho/clean/test/passivation) is
  **left intact**.
- Ground truth is **inherited** from the reference verdict on the *original*
  (known-named) sequence.

### Training vs evaluation universe / leakage
- "Training universe" (for the **rules**, which is what's evaluated): the 10 rules
  + category ontology. The category of a renamed step is recovered from the
  **preserved verb** (e.g. `DEPOSIT XENULON` → DEPOSIT). So the evaluation universe
  shares the **category-bearing verb** with the training universe.
- **Leakage analysis:** because the verb is preserved, `classify_step` recovers the
  category almost trivially → the benchmark primarily tests **token (material)
  substitution**, NOT **novel structure** and NOT **novel category-verbs**.

### Conclusion (after evidence)
- The "≈1.000 transfer" number is **PROVISIONAL** and scoped: it measures
  robustness to **renamed materials with preserved verbs and structure**. It does
  **NOT** establish transfer to **novel structure** or **opaque/novel verbs** —
  the latter is exactly where Area 7 shows a failure. The OOD definition in use is
  "token substitution within shared structure," which is narrower than "unseen
  family." Re-label prior "Family-Transfer 1.000" as **PROVISIONAL (token-transfer
  only)**.

---

## Area 6 — Silent-Failure Inventory (from repo grep this session)

| File · line | Handler | Failure mode | User-visible? | Risk |
|---|---|---|---|---|
| state_machine.py:247 | `except Exception` (known_vocab import) | routing degrades to engine | YES (`_warn_once` stderr) | Low (now loud) |
| state_machine.py:266 | `except ImportError` (reference) | uses engine instead of grader | YES (`_warn_once`) | Low (now loud) |
| process_knowledge.py:62 | `except Exception` (HAVE_REFERENCE import) | classify falls to category-only for KNOWN steps | **NO** | **Med** — silent behavior change if reference import fails |
| process_knowledge.py:563 | `except Exception` (to_mermaid/causal) | diagram only | NO | Low (non-core) |
| step_semantics.py:173-174,214-215 | `except Exception: return ""` | empty description/params | partially (demo text) | Low (not validation) |
| step_semantics.py:240,262 | `except Exception` | synonym map degrade | NO | Low |
| inference.py:75 | `except Exception` (RF load) | runs without RF mask | NO | **Med** — silent capability loss |
| refinery.py:63, fix.py:33 | `except Exception: pass` | stdout reconfigure | NO | None (encoding) |
| advisory.py:89 | `except Exception` (LLM backend) | returns error string | YES (in output) | None (labelled) |
| benchmark_models.py:20, demo.py:22, make_logic_diagram.py:22 | `except Exception: pass` | stdout reconfigure | NO | None (encoding) |

**Silent items worth action (not previously flagged):** `process_knowledge.py:62`
(reference-import guard → silent classify behavior change) and `inference.py:75`
(silent RF-absence). Neither emits a warning. Everything else is either now loud or
benign encoding setup.

---

## Area 7 — Unknown-Token Handling (execution evidence, this session)

```
one novel inert (GLORP) before known deposition:
   classify=[('PRE CLEAN WAFER',CLEAN),('GLORP WAFER',UNKNOWN),('DEPOSIT POLYSILICON',DEPOSIT)]
   verdict =[RULE_SHIP_BEFORE_TEST]                 # deposition has a clean -> ok
two fully-novel tokens:
   classify=[('GLORP STEP',UNKNOWN),('FLEEB LAYER',UNKNOWN)]
   verdict =[RULE_SHIP_BEFORE_TEST]                 # inert
novel deposition WITH verb cue, no clean:
   classify=[('GROW ZEPHYR LAYER',DEPOSIT)]
   verdict =[RULE_DEP_NO_CLEAN, RULE_SHIP_BEFORE_TEST]   # CAUGHT
opaque novel deposition (NO verb cue), no clean:
   classify=[('ZEPHYR FORMATION',UNKNOWN)]
   verdict =[RULE_SHIP_BEFORE_TEST]                 # <-- BYPASS: DEP_NO_CLEAN MISSED
malformed empty token:
   classify=[('DEPOSIT POLYSILICON',DEPOSIT)]
   verdict =[RULE_DEP_NO_CLEAN, RULE_SHIP_BEFORE_TEST]   # no crash
```

### Conclusions
- **No crash** on any variant (incl. empty token). ✓
- **Confirmed bypass (correctness gap):** an **opaque** novel deposition
  (`ZEPHYR FORMATION`, no DEPOSIT/GROW/SPUTTER… cue) classifies UNKNOWN →
  `RULE_DEP_NO_CLEAN` is **silently missed**. This is a real false-negative on a
  4th family that names a deposition opaquely.
- **No explicit uncertainty state:** every output is binary valid/invalid (+
  advisory). There is **no** `UNKNOWN / UNSUPPORTED / INSUFFICIENT_INFORMATION`
  verdict. So the opaque-deposition case above is reported as essentially valid,
  with no low-confidence signal. The previous audit's "robustness" claim is
  **PARTIAL**: robust to inert/verb-cued novelty, **not** to opaque novelty, and
  it cannot *say* when it doesn't know.

---

## Area 8 — Dashboard Feature Separation (evidence: advisory.py)

- Production/deterministic: `deterministic_report` (advisory.py:42-63) = scored
  verdict + engine-verified repair + `spec_strict` warnings. No model. ✓
- Experimental/LLM: `advisory(..., llm, include_llm)` (advisory.py:78-101) — LLM
  called **only** if a backend is explicitly passed; output wrapped with banner
  "EXPERIMENTAL — non-deterministic … NOT used for any scored decision"
  (advisory.py:93-97). ✓
- **Gap:** the labels in code are "EXPERIMENTAL / NON-DETERMINISTIC", not the exact
  triplet the prompt requests ("EXPERIMENTAL / UNVERIFIED / ADVISORY ONLY"). Status:
  **PARTIAL** — separation is correct; the literal label string differs. Trivial fix.

---

## Area 9 — Risk Register (rebuilt from the above evidence)

| # | Risk | Likelihood | Impact | Detectability | Mitigation | Priority |
|---|---|---|---|---|---|---|
| R1 | Scored path diverges from grader on non-canonical **casing**/whitespace (Area 1) | Unknown (eval casing UNVERIFIED) | High (wrong labels) | Low (silent; differential never tested it) | normalize casing at ingestion to match grader; add case mutations to differential_fuzz | **P0** |
| R2 | **Opaque** novel deposition/clean bypasses windowed rules (Area 7) | Med on a real 4th family | Med–High (missed anomaly) | Low (silent, no uncertainty state) | explicit verdict-enum incl. INSUFFICIENT_INFORMATION when a rule-position token is UNKNOWN | **P0** |
| R3 | Eval schema / SCORE polarity ≠ assumed (Area 3) | Med | High | Med (schema now raises) | one-switch parser; confirm on real file | P1 |
| R4 | OOD number is token-transfer only, mis-read as family transfer (Area 5) | High (already happened in prior audit) | Med (overconfidence) | — | re-label PROVISIONAL; add structure-mutation OOD benchmark | P1 |
| R5 | `process_knowledge.py:62` / `inference.py:75` silent degradations (Area 6) | Low | Med | Low (silent) | add warnings | P2 |
| R6 | category_eval not reproducible from clean checkout (Area 4) | High | Low | — | commit a tiny ckpt or document | P2 |
| R7 | LLM advisory mistaken for truth (Area 8) | Low | High | High (banner) | tighten label triplet | P2 |
| R8 | Model undertrained (tiny/CPU) | High | Med | High | Leonardo run (external) | P1(ext) |

**P0 is NOT empty.** Two correctness risks (R1 casing divergence, R2 opaque-token
bypass) are open, each with low detectability.

---

## Area 10 — Roadmap (evidence-driven)

- **P0 (correctness):**
  1. **Casing/whitespace normalization** so the scored path matches the grader.
     Evidence: Area 1 lowercase DISAGREE. Location: `state_machine.py:149` vs
     `generate_sequences.py:223`. Failure mode: over-flag valid sequences.
  2. **Explicit verdict-state enum** (`INSUFFICIENT_INFORMATION` when a
     rule-relevant token is UNKNOWN). Evidence: Area 7 opaque-deposition bypass.
     Location: `state_machine.apply_step` / `validate_sequence_combined`. Failure
     mode: silent false-negative on OOD.
- **P1 (benchmark confidence):**
  3. Add **case + whitespace mutations** to `differential_fuzz.py` (Area 1/4).
  4. Add a **structure-mutation** OOD benchmark (re-order/insert blocks), not just
     token substitution (Area 5). Re-label existing OOD result PROVISIONAL.
  5. Confirm eval schema + SCORE polarity against the real file when distributed (Area 3).
- **P2 (robustness):**
  6. Warn on `process_knowledge.py:62` / `inference.py:75` silent degradations (Area 6).
  7. Tighten LLM advisory label to EXPERIMENTAL/UNVERIFIED/ADVISORY-ONLY (Area 8).
  8. Make category_eval reproducible from a clean checkout (Area 4/6).
- **P3 (optional):** ontology verb-coverage expansion (gated by differential=0),
  per-step hybrid routing.

---

## Final Classification

### VERIFIED FACTS (evidence shown above)
- The 10 rule definitions exist at `process_knowledge.py:283-357,373`; the litho
  engine logic at `state_machine.py:149-162`.
- On **canonical-cased, in-vocabulary** mutations, the engine and the reference
  agree (differential_fuzz, 0 disagreements; **reproducible** — same seed → identical twice).
- The engine does **not crash** on one/two-novel, unknown, empty, or malformed tokens (Area 7).
- A **verb-cued** novel deposition without a clean is correctly flagged (Area 7).
- Dashboard separates deterministic core from an off-by-default LLM layer (advisory.py:78-101).

### ASSUMPTIONS (not proven from docs)
- Eval input is uppercase/canonical-cased (undocumented; drives the Area-1 divergence).
- SCORE = P(valid); Block-accuracy definition; "4th family mostly shares vocabulary."

### HYPOTHESES (plausible, untested)
- The real eval is uppercase (so R1 may not bite) — untestable until files arrive.
- OOD token-transfer ≈ structure-transfer — **contradicted** by Area 7 for opaque verbs.

### FUTURE WORK / UNVERIFIED
- Equivalence to the organizers' **real** `eval_metrics.py` labels (files undistributed).
- Casing/whitespace equivalence (no test exists — must add).
- category_eval numbers from a clean checkout (checkpoint gitignored).
- Structure-level OOD transfer (no benchmark exists yet).

### Headline correction to the prior audit
"Enforced on every path with no bypass" and "≈1.000 family transfer" were
**overstated**. Evidence shows: (1) a real engine↔grader **case divergence** on the
litho rule, untested by the equivalence proof; (2) an **opaque-token deposition
bypass** of the windowed rules with **no uncertainty state**; (3) the OOD number is
**token-substitution-only**, not structure transfer. P0 is **not** empty.
