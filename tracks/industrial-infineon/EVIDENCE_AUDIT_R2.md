# Evidence-First Re-Audit — Round 2 (post-fix)

Same method as `EVIDENCE_AUDIT.md`: evidence (command output / file:line) before
conclusions; adversarial intent. This round re-runs the exact attacks that
previously broke the system and records whether each is now closed. Commands were
run this session against the current tree (after the P0–P3 fix commits
`f0bd1c9`, `3f0cc25`, `b833575`).

---

## A. The two P0 correctness defects — re-attacked

### R1 — litho casing/whitespace divergence (was: scored path ≠ grader on lowercase)
Fix: `canonicalize_step` (upper + collapse whitespace) in `apply_step`
(`state_machine.py`) and at the top of `validate_sequence_combined`.
Re-attack (this session):
```
combined: ['RULE_LITHO_LEVEL_SKIP']   ref(canon): ['RULE_LITHO_LEVEL_SKIP']   -> AGREE
```
And `differential_fuzz.py` now includes `lowercase`/`titlecase`/`double_space`
mutations and feeds the reference canonicalised input (mirroring production):
```
differential_fuzz --n 8000 : RESULT PASS — engine binary-equivalent to the reference
```
Status: **CLOSED.** The earlier lowercase DISAGREE no longer reproduces; the
equivalence proof now exercises casing/whitespace noise.

### R2 — opaque novel token silently passed (was: no uncertainty state)
Fix: `validate_with_confidence` → `VALID / INVALID / INSUFFICIENT_INFORMATION`;
`detect_anomaly` emits `verdict` + `insufficient_information` + `unknown_tokens`
and caps score at 0.5. Re-attack:
```
opaque token, no violation -> INSUFFICIENT_INFORMATION   (unknowns=['ZEPHYR FORMATION'])
```
No-regression evidence: 0/200 known-vocab steps classify UNKNOWN; all 600 provided
valid sequences return verdict=VALID (no spurious uncertainty).
Status: **CLOSED** (the opaque case is now explicit, not a silent confident pass).
Honest limit: the engine still cannot *validate* an opaque token — by design it
returns INSUFFICIENT_INFORMATION rather than guessing.

---

## B. Full regression (this session, current tree)

| Gate | Command | Result |
|---|---|---|
| Rule engine vs reference | `exhaustive_test.py` | **42/42 PASS** |
| Engine ≡ reference + casing/ws | `differential_fuzz.py --n 8000` | **PASS** (0 disagreements) |
| OOD token transfer (f=0.30) | `ood_benchmark.py` | Acc/F1 **1.000** |
| OOD one-novel-token | `ood_benchmark.py` | valid 250/250 (0 FP) · invalid 250/250 (0 lost) |
| OOD **structure+token** transfer | `ood_benchmark.py` | Acc/F1 **1.000** (n=500, 0 FP/FN) |
| Malformed/novel input | `robustness_test.py` | **PASS**, incl. mixed known-violation+novel caught |
| spec_strict false alarms | 3000 valid sequences | **0** physical advisories |
| End-to-end submission | inference → validate → self_score | see §C |

## C. End-to-end (schema-validated readers, post-fix)
`make_sample_eval` → `inference` → `validate_submission` → `self_score` (this session):
```
Task 1/2/3 submissions: 36 rows each -> outputs_M1/submissions/*.csv
validate_submission: RESULT PASS — all three files conform to the spec
T1 next-step : Top-1=0.722 Top-3=1.000 Top-5=1.000 MRR=0.856 (n=36)
T2 completion: NormEditDist=0.221  ExactMatch=0.000 (n=36)
T3 anomaly   : Acc=1.000 Prec=1.000 Rec=1.000 F1=1.000 RuleAttr=18/18  TP/FP/TN/FN=18/0/18/0
```
Identical to the pre-fix run — the P0–P3 changes introduced no regression in the
scored pipeline.

---

## D. Rule coverage matrix — updated statuses

The casing fix promotes the litho rule's scope. "VERIFIED (in-vocab)" now means:
canonical **and** case/whitespace-variant in-vocab, per the augmented differential
fuzz. All 10 rules: **VERIFIED (in-vocab, incl. case/ws)** via differential_fuzz +
exhaustive. Still **UNVERIFIED** against the organizers' real `eval_metrics.py`
(files undistributed) and **PARTIAL** on opaque-novel-vocab (now surfaced as
INSUFFICIENT_INFORMATION rather than a silent miss).

---

## E. Risk register — rebuilt (post-fix, evidence-backed)

| # | Risk | Likelihood | Impact | Detectability | Status / evidence |
|---|---|---|---|---|---|
| R1 | casing/ws divergence from grader | — | — | — | **CLOSED** (§A; differential+case PASS) |
| R2 | opaque-token silent pass | — | — | — | **CLOSED→explicit** (§A; INSUFFICIENT_INFORMATION) |
| R3 | eval schema / SCORE polarity ≠ assumed | Med | High | Med | mitigated: loud schema raise + `SCORE_IS_P_VALID` one-switch; still **ASSUMPTION** until real file |
| R4 | OOD was token-only | — | — | — | **CLOSED**: structure+token benchmark added, Acc/F1 1.000 |
| R5 | silent degradations (pk:62, inf:75) | Low | Med | — | **CLOSED**: both now warn to stderr |
| R6 | category_eval not reproducible from clean checkout | High | Low | — | mitigated: actionable message + documented train step |
| R7 | LLM advisory mistaken for truth | Low | High | High | banner now EXPERIMENTAL/UNVERIFIED/ADVISORY ONLY |
| R8 | model undertrained (tiny/CPU) | High | Med | High | **OPEN — external** (Leonardo run) |
| R9 | grader's real labels unverified | Med | High | Low | **OPEN — external** (files undistributed) |

### P0 status
With R1 and R2 closed (evidence above), there are **no open in-repo P0 correctness
risks**. The remaining open items (R8 model scale, R9 real-grader verification)
are **external** — they cannot be closed from this repository's code; they need the
Leonardo training run and the organizers' eval files respectively.

---

## F. Final classification (post-fix)

### VERIFIED FACTS (evidence shown this session)
- Engine ≡ reference on in-vocab mutations **including case/whitespace** (differential_fuzz PASS, R1 attack no longer reproduces).
- Opaque novel tokens yield an explicit INSUFFICIENT_INFORMATION verdict, not a silent pass (reproduced); 0/200 known steps are UNKNOWN, so no in-vocab regression.
- exhaustive 42/42; robustness PASS incl. mixed known-violation+novel; spec_strict 0 physical false alarms on 3000 valid sequences.
- OOD: token transfer AND structure+token transfer both Acc/F1 1.000 against reference-relabeled ground truth; one-novel-token 250/250 both directions.
- Silent degradations warn loudly; LLM layer is off by default with the EXPERIMENTAL/UNVERIFIED/ADVISORY-ONLY banner.

### ASSUMPTIONS (documented, not proven from docs)
- Eval schema (A1/A2) and SCORE polarity (A5) — both now have a loud check / one-switch, but the real files are unseen.
- "4th family mostly shares vocabulary" (README, one sentence) — OOD benchmarks quantify the token+structure case; truly-opaque vocab yields INSUFFICIENT_INFORMATION.

### HYPOTHESES (plausible, untested)
- The real eval is uppercase/canonical — now moot for correctness (canonicalisation makes the engine casing-agnostic and grader-consistent).

### FUTURE WORK / external
- Run against the organizers' real `eval_metrics.py` (R9).
- Full-size training + scaling curve on Leonardo (R8).
- Demo video + slides PDF; fill LICENSE team name.

### Headline
The two P0 defects from the prior evidence audit (litho casing divergence, opaque-
token silent pass) are **closed with reproduced evidence**, with **no in-vocab
regression** and all gates green. The only open risks are **external** (real grader,
model scale), which this repository's code cannot resolve.
