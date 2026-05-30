# Audit, Hardening & Implementation Plan

A consolidated, traceable audit against the **original hackathon repository as the
sole source of truth** (`README.md`, `Track_industrial_en.md`,
`training_data/generation_rules.md`, `training_data/generate_sequences.py`), plus
a prioritized plan to maximize correctness, traceability, robustness, and minimum
hallucination risk.

**Status legend:** ✅ implemented + verified · ⚠️ partial / residual gap · ⬜ planned · 🧪 experimental (never affects scored output)

**Guiding invariants (enforced throughout):** deterministic validation is the
sole source of truth; no rule is invented; every rule traces to a source section;
no silent failure; no silent low-confidence success; unknowns become explicit
states, never fabricated outputs.

---

## Phase 1 — Rule Coverage Traceability Matrix

### 1a. The 10 scored forbidden patterns (the grader's rules)

Source of truth: `generation_rules.md §3` and the reference implementation
`generate_sequences.validate_sequence`. "Implemented" = in the scored engine
(`physics/state_machine.py` + KB `physics/process_knowledge.py`). "Verified" =
proven equivalent to the reference by `differential_fuzz.py` (8,500+ mutations, 0
disagreements, all 10 rules exercised) and `exhaustive_test.py` (42/42).

| Rule ID | Source | Exact rule (as graded) | Implemented | Test | Gap |
|---|---|---|---|---|---|
| RULE_DEP_NO_CLEAN | §3 L470, ref L36-80,167 | deposition needs a clean within prior 12 | ✅ | differential+exhaustive | none (in-vocab) |
| RULE_METAL_ETCH_NO_LITHO | §3 L484, ref | metal etch needs EXPOSE+DEVELOP within 15 | ✅ | differential+exhaustive | none |
| RULE_ETCH_NO_MASK | §3 L495, ref | patterned etch needs DEVELOP within 12 (spacer excluded) | ✅ | differential+exhaustive | none |
| RULE_LITHO_LEVEL_SKIP | §3 L508, ref L216-248 | consecutive ALIGN levels non-decreasing, no gap >1 | ✅ | differential+exhaustive+litho multipath | see 1c |
| RULE_IMPLANT_NO_MASK | §3 L517, ref L250-265 | implant needs oxide-etch/DEVELOP within 15 | ✅ | differential+exhaustive | none |
| RULE_CMP_NO_DEP | §3 L528, ref | CMP needs a deposition/fill within 6 | ✅ | differential+exhaustive | none |
| RULE_PAD_OPEN_BEFORE_DEP | §3 L539, ref | pad-open needs DEPOSIT+CURE PASSIVATION earlier | ✅ | differential+exhaustive | none |
| RULE_TEST_BEFORE_PASSIVATION | §3 L548, ref | electrical tests after CURE PASSIVATION | ✅ | differential+exhaustive | none |
| RULE_SHIP_BEFORE_TEST | §3 L559, ref | SHIP LOT after WAFER SORT TEST | ✅ | differential+exhaustive | none |
| RULE_BACKSIDE_BEFORE_PASSIVATION | §3 L568, ref | backside metal after CURE PASSIVATION | ✅ | differential+exhaustive | none |

**Conflict check:** the 10 rules are independent local/ordering constraints; no
two conflict (verified — `differential_fuzz` never produces a rule pair that
cannot co-hold on a valid sequence). No rule is partially implemented or
mis-interpreted relative to the reference (bit-for-bit on the fuzz set).

### 1b. Documented grammar constraints the GRADER does NOT score (→ advisory)

Source: `generation_rules.md §2` (process grammar) + §4 "Fixed (not variable)".
The reference checker does not enforce these; they are encoded **deterministically**
in `physics/spec_strict.py` as **advisory** warnings (never change a scored label).

| Constraint | Source | Implemented (advisory) | Test |
|---|---|---|---|
| Litho-cycle completion (develop per level; first level = 1) | §3 L509-511 prose, §2.3 | ✅ spec_strict `SPEC_LITHO_NO_DEVELOP/FIRST_NOT_1` | spec_strict selftest |
| Mandatory post-etch STRIP + CLEAN | §2.3 L337-338 | ✅ `SPEC_ETCH_NO_STRIP/CLEAN` | 0 physical FP / 3000 valid |
| Implant → activation anneal | §2.3 L341 | ✅ `SPEC_IMPLANT_NO_ANNEAL` | 0 physical FP / 3000 valid |
| Block macro-order (ILD→via→metal→passivation→backside→ship) | §2.2 L224-231 | ✅ `SPEC_BLOCK_ORDER` | 0 physical FP / 3000 valid |
| Start RECEIVE WAFER LOT / end SHIP LOT | Key Concepts, §2.3 | ✅ `SPEC_BAD_START/END` | — |
| TEST_SUITE internal order | §2.3 L441-449, §4 Fixed | ✅ `SPEC_TEST_ORDER` (convention tier) | surfaced 16.7% of provided "valid" data |

> **Finding (traceable, not invented):** ~16.7% of the provided *valid* dataset
> violates the documented "Fixed TEST_SUITE order" — the grader ignores it. We
> surface it as a `convention` advisory, never as a scored failure.

### 1c. Lithography-skip rule — "enforced everywhere, no bypass" (special attention)

Verified on **every path** (single reproducible script):

| Path | Skip (1→3) caught | Decrease (3→2) caught |
|---|---|---|
| Deterministic engine (`validate_by_state_machine`) | ✅ | ✅ |
| Reference checker (`validate_sequence`) | ✅ | ✅ |
| Combined router (`validate_sequence_combined`) | ✅ | ✅ |
| Heuristic/OOD path (novel tokens around shared litho vocab) | ✅ | ✅ |
| Repair (`fix.repair`) | ✅ renumbers to sequential → valid | ✅ |
| Synthetic/benchmark generation (`bad_data_generator`, `pseudo_family`) | ✅ injects/preserves | ✅ |
| Spec-strict advisory (incomplete cycle, numerically-ok) | ✅ `SPEC_LITHO_NO_DEVELOP` | — |

**No bypass path exists** for in-vocab sequences (the grader's definition). The
only residual is a 4th family that renames the *litho keywords themselves*
(`ALIGN MASK LEVEL`), which the README explicitly says are shared vocabulary — so
out of the documented Task-4 scope. **Action:** documented as A11 residual; no code
change (would diverge from the grader on in-vocab).

---

## Phase 2 — Assumption Elimination Audit

Source: `ASSUMPTIONS.md` (A1–A11) + assumptions found in audit. "Deterministic
route" = we now fail loudly / validate rather than assume silently.

| Assumption | Source | Verified? | Risk | Action / Deterministic route |
|---|---|---|---|---|
| A1/A2 eval CSV schema (cols, pipe-sep) | §5.1 | ❌ files undistributed | HIGH | ✅ `inference._read_eval` validates schema → **raises** on mismatch (no silent empty submission) |
| A3 FAMILY∈{MOSFET,IGBT,IC}; 4th→[UNK] | inferred | ✅ handled | LOW | ✅ tokenizer/RF fallback, no crash |
| A4 Task-2 = completion only | §5.3 | ✅ | LOW | ✅ `complete_sequence` returns post-cut only |
| A5 SCORE = P(valid) | §5.3 | ⚠️ plausible, unconfirmed | MED | ⬜ cannot resolve w/o grader; documented; value emitted |
| A6 filenames nextstep/completion/anomaly.csv | SUBMISSION | ✅ | LOW | ✅ |
| A7 comma CSV, UTF-8 BOM-tolerant | standard | ✅ | LOW | ✅ utf-8-sig readers |
| A8 Block-accuracy definition | NOT specified | ⚠️ our proxy | MED | ⬜ documented; internal metric only, not in submission |
| A10 engine ≡ reference (10 rules/windows) | §3 | ✅ **proven** | — | ✅ differential_fuzz 0/8500 |
| A11 4th family "mostly shares vocabulary"; vocab-routing | README L163 | ⚠️ single-source | MED | ✅ routing deterministic; ✅ OOD benchmark quantifies; residual documented |
| Window sizes 12/15/6 | ref code | ✅ copied verbatim | — | ✅ not invented |
| OOD keyword fallbacks (PROBE/PAD/BACKSIDE) | derived | ✅ whole-word+category-gated | LOW | ✅ FP regression fixed + verified |

**Objective met:** every removable undocumented assumption is either eliminated
or converted to a loud deterministic check. The two irreducible ones (A5 SCORE
polarity, A8 Block definition) depend on the organizers' un-distributed scorer and
are flagged, not silently baked in.

---

## Phase 3 — Deterministic Engine Audit

`physics/` (state_machine, process_knowledge, ontology, parameters, spec_strict).

- **Every rule explicit** — declarative KB (`WINDOWED_RULES`, `ORDERING_RULES`,
  `LITHO_RULE`) interpreted by one generic engine; each carries `physical_reason`. ✅
- **Every rejection explainable** — `explain.py` + `fix.analyze` emit rule id +
  step index + physical why + concrete fix. ✅
- **Every failure observable** — degradation paths now `_warn_once` to stderr
  (missing vocab / missing reference); no silent fallback. ✅
- **Traceable** — verdict carries source ("(deterministic rule check)" when the
  reference path is used). ✅
- **No hallucination** — pure rule logic + human-authored templates; no generative
  text anywhere in the scored path. ✅
- **No silent invalid pass** — `differential_fuzz` proves it never under-flags
  in-vocab; `spec_strict` adds advisory over-checks. ✅

**Residual:** the `physical_reason` strings occasionally state a *convention* as
universal physics (anneal-as-clean, test-before-*any*-passivation). They mirror
the grader faithfully but read as fab law. **Action (P2):** annotate these strings
as "grader-rule, not universal physics."

---

## Phase 4 — Heuristic Engine Audit

The "heuristic" layer = (a) the category-based OOD reasoning in the engine, and
(b) the neural model (proposer only).

- **Deterministic > heuristic (enforced):** `validate_sequence_combined` routes
  all-in-vocab sequences to the **exact reference** (deterministic wins); the
  category engine is used only where the reference is undefined (novel vocab). The
  neural model **never** decides validity — it only proposes steps that the
  deterministic refinery then vetoes (`refinery.constrained_decode` guarantees a
  valid completion; `predict_next_steps` re-ranks legal-first). ✅
- **Disagreement handling:** on in-vocab the two are proven identical (0/8500). Any
  future disagreement is a `differential_fuzz` failure → becomes a test. ✅
- **Anti-shortcut safeguards:** UNK-dropout (forces context use), pseudo-family
  training (novel vocab), category aux head (learn function not names), and the
  OOD benchmark (measures transfer, not memorization). ✅
- **Token leakage:** `predict_next_steps` guarantees real step names, never leaks
  `[UNK]`/specials; `validate_submission` checks for special-token leak. ✅

**Residual (⚠️):** routing is whole-sequence — one novel token sends the *entire*
sequence to the category engine. Mitigated (engine≡reference proven; OOD benchmark
≈100% at realistic novelty; one-novel-token 250/250). **Action (P2):** optional
per-step hybrid routing for mixed-vocab sequences.

---

## Phase 5 — Novel-Token Robustness (design + status)

Explicit policy (implemented):

| Situation | Behavior | Status |
|---|---|---|
| Single novel token in known family | classified by physical category; if inert (inspect/measure) → no verdict change; verified **250/250 valid stay valid, 250/250 invalid stay caught** | ✅ `ood_benchmark.py` |
| Multiple novel tokens / partially-known family | category engine handles all steps uniformly; `ood_benchmark` sweeps f=0.1→1.0 → **Acc 1.000 at f=0.30, 0.998 at f=1.00** | ✅ |
| Entirely novel family (shared structure, novel materials) | category reasoning; **Family-transfer Acc/F1 ≈ 1.000** | ✅ |
| Truly opaque novel verb (no category cue) | classified **UNKNOWN** → inert (never a trigger/enabler) → explicit non-fabrication | ⚠️ may miss/over-flag; honest |

**Explicit uncertainty states** exist at the classification layer (`UNKNOWN`).
**Gap (⬜, P1):** these are not yet surfaced as a first-class *verdict* state
(see Phase 8). A deposition with an UNKNOWN-classified enabler should ideally
return `INSUFFICIENT_INFORMATION`, not a confident pass/fail.

---

## Phase 6 — Task-4 / OOD Generalization Evaluation

`ood_benchmark.py` — concrete, **verifiable** (ground truth inherited from the
reference, non-circular), no LLM. Models the README's "mostly-shared-vocabulary"
4th family by renaming device materials (structure/verbs kept).

| Metric | Definition | Result |
|---|---|---|
| Family-Transfer Accuracy | correct valid/invalid on renamed family, f=0.30 | **1.000** (target ≥80% ✅) |
| Constraint-Detection Recall | invalid sequences caught | 0.997–1.000 |
| Constraint-Detection Precision | 1 − false-positive rate | 1.000 (0 FP) |
| Rule-Attribution Accuracy | correct rule named | 1.000 |
| Novel-Token Handling | one benign novel token | 250/250 (0 FP, 0 lost) |
| Failure Explainability | rule id + why + fix on every flag | ✅ qualitative |

**Pathways to 85/90/95 (already met for the documented scenario):** the realistic
Task-4 regime is at ~100%. The **frontier** is the *out-of-scope* fully-opaque-verb
case (no shared vocabulary), where UNKNOWN handling limits recall. **Plan (P2):**
extend the ontology's category keyword coverage from real fab-verb corpora (each
addition gated by differential_fuzz=0 to preserve in-vocab equivalence) — raises
the opaque-verb floor without compromising determinism.

---

## Phase 7 — Large-Scale Evaluation Framework

**Existing (✅):** `differential_fuzz` (8.5k mutations, all rules), `exhaustive_test`
(3003 provided + boundaries + repair), `ood_benchmark` (600 × 7 novelty levels),
`bad_data_generator` (every rule×strategy×family, verified-labeled).

**Plan (⬜, P1) — unified 10k–50k reproducible harness** (`mega_eval.py`):

| Bucket | Source | Generator |
|---|---|---|
| Positive | provided variants | sample N |
| Negative | bad_data_generator | all rule×strategy |
| Boundary | window-edge (k−1 vs k) | exhaustive_test boundary gen, scaled |
| Adversarial | multi-violation / rule-confusing | bad_data_generator tier-2 |
| OOD | ood_benchmark renamer at varied f | reuse |
| Corrupted | truncated / missing-field | robustness fuzz, scaled |
| Mutation | auto-mutate valid → invalid | differential_fuzz mutators |

Track precision/recall/F1, rule-level acc, family-level acc, failure-detection,
unknown-detection. **Reproducible** (seeds now stable after the hash-salting fix).
Justification: the spec (§8 "Known Pitfalls": *a benchmark that measures more than
memorization*) and Level-3 stretch demand scale + reproducibility.

---

## Phase 8 — Silent-Failure Elimination Audit

| Location | Before | Now | Status |
|---|---|---|---|
| `validate_sequence_combined` vocab/reference import | silent fallthrough | `_warn_once` loud | ✅ |
| `inference._read_eval` (eval schema) | crash mid-run / silent | **raises** on bad schema / 0 rows | ✅ |
| `random_forest.load` (pickle) | silent | fail-safe + security warning | ✅ |
| `transition_model.build` (corrupt cache) | crash | rebuild (logged) | ✅ |
| Task-3 SCORE | constant (AUC=1 by construction) | model-only signal reported separately, honestly | ✅ |

**Gap (⬜, P1) — explicit verdict-state enum.** Today the system returns
valid/invalid (+ advisory). Recommendation: a first-class result type
`{VALID, INVALID, UNKNOWN, UNSUPPORTED, INSUFFICIENT_INFORMATION, CONFLICTING_RULES}`
returned whenever confidence is low (e.g. a trigger/enabler classified UNKNOWN),
so low-confidence cases never silently pass or fail. Source justification: the
spec's emphasis on generalization honesty + this prompt's Phase 8/5 requirements.

---

## Phase 9 — Experimental Dashboard Features (🧪 never affect scoring)

Implemented in `advisory.py`:

- **Warning layer (deterministic, 🧪 informational):** `spec_strict` real-fab
  warnings — passes the grader but breaks documented grammar (no strip, no anneal,
  block order). Two tiers (`physical` / `convention`). Cannot change validity. ✅
- **LLM explanation layer (🧪 non-deterministic, off by default):** pluggable
  `llm(prompt)` backend; returns a free-text explain/repair **suggestion** wrapped
  in an `EXPERIMENTAL / NON-DETERMINISTIC / UNVERIFIED` banner, shown **alongside**
  (never instead of) the engine-verified repair. The LLM is never called unless a
  caller explicitly wires it. ✅

**Plan (⬜, P2):** wire a concrete backend for the live demo; optionally verify the
LLM's proposed repaired sequence through the engine and display "verified ✓/✗".

---

## Phase 10 — Risk Register & Prioritized Roadmap

### Risk register

| # | Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| R1 | Real eval schema ≠ assumed (A1/A2) | Med | High | loud schema validation; one-switch parser | ⚠️ mitigated, not eliminated |
| R2 | SCORE polarity inverted (A5) | Low | Med | documented; trivial to flip post-hoc | ⚠️ |
| R3 | 4th family uses novel litho/clean vocabulary (breaks A11) | Low (README says shared) | Med | OOD benchmark quantifies; category engine | ⚠️ residual |
| R4 | Opaque novel verbs → UNKNOWN miss/over-flag | Med | Med | UNKNOWN is explicit/inert; verdict-enum (P1) | ⚠️ |
| R5 | Model overfits / memorizes (shortcut) | Med | Low (model is proposer only) | UNK-dropout, pseudo-families, OOD bench | ✅ |
| R6 | LLM advisory mistaken for truth | Low | High | off by default, hard banner, never scored | ✅ |
| R7 | Block-accuracy proxy ≠ official (A8) | Med | Low | internal metric only | ⚠️ |
| R8 | Model undertrained (tiny/CPU) | High | Med (Task-1/2 ID/OOD) | scaling run on Leonardo (external) | ⬜ |

### Prioritized roadmap (each: justification · source · impact)

**P0 — none open.** All scored-path correctness items are ✅ (engine≡grader proven,
litho enforced everywhere, no silent failures, submission format validated).

**P1 (high value, deterministic, no training):**
1. **Explicit verdict-state enum** (Phase 8/5). *Source:* this prompt + spec
   generalization honesty. *Impact:* eliminates the last class of silent
   low-confidence success on OOD; turns UNKNOWN-enabler cases into
   `INSUFFICIENT_INFORMATION`. 
2. **`mega_eval.py` 10k–50k unified harness** (Phase 7). *Source:* §8 "more than
   memorization" + Level-3. *Impact:* reproducible, large-scale confidence numbers
   across all buckets; reviewer-facing evidence.

**P2 (robustness / polish):**
3. Per-step hybrid routing for mixed-vocab sequences (Phase 4 residual). *Impact:*
   keeps the proven reference path for the known portion of a partially-novel seq.
4. Ontology fab-verb coverage expansion, each gated by differential_fuzz=0
   (Phase 6 frontier). *Impact:* raises opaque-verb OOD floor without touching
   in-vocab equivalence.
5. Annotate `physical_reason` strings as grader-rule vs universal physics
   (Phase 3 residual). *Impact:* honesty for any reader using the KB as reference.
6. Wire + engine-verify a live LLM backend for the demo (Phase 9). *Impact:* demo
   stretch feature, still non-authoritative.

**External (not in this repo's control):**
7. Full-size training + scaling curve on Leonardo (R8); demo video + slides PDF
   (submission requirements); run against the organizers' real `eval_metrics.py`.

### Bottom line
The **scored path is correct, traceable, and grader-equivalent** (proven), with
**no silent failures** and **zero hallucination** in anything that affects a label.
The remaining work is (a) two deterministic robustness upgrades (verdict-enum,
mega-harness) that increase *confidence and honesty* without changing the proven
core, and (b) external runs (scale, real grader, demo media) outside the code's
control.
