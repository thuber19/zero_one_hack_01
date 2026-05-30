# Provenance & Sources

Every piece of knowledge in this project is traced to one of three tiers:

- **Tier 1 — Challenge-authoritative.** Taken verbatim from the repo. These are
  the ground truth for *this* hackathon; no external citation needed (and the
  organizers' scoring uses the same definitions).
- **Tier 2 — Established process physics.** The *reason* each rule exists is
  standard semiconductor-manufacturing science, citable to canonical textbooks
  and references below.
- **Tier 3 — Our additions.** Things we built on top (category ontology, keyword
  fallback, pseudo-families, repair). Clearly flagged as ours, with the physical
  principle each rests on.

---

## Tier 1 — Challenge-authoritative (from this repo)

| What | Exact source in repo |
|---|---|
| The 10 rules, trigger sets, window sizes (12/15/12/15/6) | `training_data/generation_rules.md` §3; checker `generate_sequences.py::validate_sequence` |
| Step vocabulary & 12 functional categories | `generation_rules.md` §1 |
| Process grammar / block order | `generation_rules.md` §2 |
| Variation axes | `generation_rules.md` §4 |
| Eval protocol, metrics, submission formats | `generation_rules.md` §5; `training_data/README.md` |
| Step physical **descriptions** | `*_Longdescr.csv` (absorbed by `physics/step_semantics.py`) |
| Realistic **fab parameters** | `*_longdescription_parameters.csv` (absorbed by `step_semantics.py`) |
| 3000 validated reference sequences + 3 canonical refs | `*_variants.csv`, `synthetic_*.csv` |

Our validator (`physics/state_machine.py`) is verified bit-for-bit against this
Tier-1 ground truth: **3003/3003** provided sequences pass, all 10 rules match,
windows flip at the exact documented step.

---

## Tier 2 — Why each rule is real physics (citable)

Each challenge rule encodes an established manufacturing constraint. The repo's
own parameter CSV independently corroborates many of these (e.g. it specifies
RCA SC-1 as `NH₄OH:H₂O₂:H₂O 1:1:5, 75 °C` — exactly the literature recipe).

| Rule | Physical principle | Source |
|---|---|---|
| `RULE_DEP_NO_CLEAN` | Wafers must be RCA/HF cleaned before oxidation/CVD/epi; contaminants left on the surface diffuse into and nucleate defects in the grown film. | Kern & Puotinen, *RCA Review* 31 (1970); [RCA clean (Wikipedia)](https://en.wikipedia.org/wiki/RCA_clean); Plummer/Deal/Griffin, *Silicon VLSI Technology* (ISBN 0-13-085037-3) |
| `RULE_ETCH_NO_MASK` / `RULE_METAL_ETCH_NO_LITHO` | Patterned etch removes material everywhere not protected by developed photoresist; the litho expose+develop creates that stencil. | Plummer/Deal/Griffin; Campbell, *Fabrication Engineering at the Micro and Nanoscale* |
| `RULE_LITHO_LEVEL_SKIP` | Each mask level registers to alignment marks/structures from the previous level; levels must be built in order. | Plummer/Deal/Griffin (lithography & overlay) |
| `RULE_IMPLANT_NO_MASK` | Oxide/photoresist masks block implant ions; doping reaches the substrate only through a deliberately opened window (selective doping). | [MicroChemicals, *Ion Implantation with Photoresist Masks*](https://www.microchemicals.com/dokumente/application_notes/ion_implantation_photoresist.pdf); [implant mask thickness](https://www.memsolver.com/Help/tochtml/impmask.htm); Sze & Ng, *Physics of Semiconductor Devices* |
| `RULE_CMP_NO_DEP` | CMP polishes deposited overburden down to a plane; with nothing deposited it grinds the underlying structure. | Wolf & Tauber, *Silicon Processing for the VLSI Era*; Quirk & Serda, *Semiconductor Manufacturing Technology* |
| `RULE_PAD_OPEN_BEFORE_DEP` | The pad etch opens bond pads *through* the passivation; the layer must exist and be cured (cross-linked) first. | Quirk & Serda; standard back-end-of-line practice |
| `RULE_TEST_BEFORE_PASSIVATION` | Wafer-sort probing stresses the die; passivation seals interconnects/active areas before probe contact. | May & Spanos, *Fundamentals of Semiconductor Manufacturing*; Quirk & Serda |
| `RULE_SHIP_BEFORE_TEST` | Wafer sort screens defective dice; shipping before sort releases untested product (a QC gate). | May & Spanos (yield/test); standard fab QC |
| `RULE_BACKSIDE_BEFORE_PASSIVATION` | Backside metallisation applies thermal/mechanical/chemical stress; the front must be sealed first. | Quirk & Serda; power-device back-end practice |

**Canonical textbooks (all real, widely adopted):**
- J. D. Plummer, M. Deal, P. Griffin, *Silicon VLSI Technology: Fundamentals,
  Practice and Modeling*, Prentice Hall (ISBN 0-13-085037-3) — MIT-adopted.
- S. Campbell, *Fabrication Engineering at the Micro and Nanoscale*, Oxford UP.
- S. M. Sze & K. K. Ng, *Physics of Semiconductor Devices*, Wiley.
- S. Wolf & R. Tauber, *Silicon Processing for the VLSI Era*, Lattice Press.
- G. May & C. Spanos, *Fundamentals of Semiconductor Manufacturing and Process
  Control*, Wiley.
- M. Quirk & J. Serda, *Semiconductor Manufacturing Technology*, Prentice Hall.

---

## Tier 3 — Our additions (clearly ours) and why they're sound

| What we built | Status | Physical justification |
|---|---|---|
| 14-category physical ontology (`physics/ontology.py`) | extends the repo's 12 categories | The repo already organises steps into functional categories (`generation_rules.md` §1); we add `ETCH_BLANKET` (spacer etch is non-patterned — the doc itself flags this exclusion) and `PATTERN_INSPECT`. |
| Keyword fallback for unknown steps | ours | Rests on the fact that process **function** is universal across thin-film technologies: a "GROW…"/"DEPOSIT…" step is a deposition whatever the material (Si, GaN, SiC). Same physics → same precondition. Grounded in the universality emphasised across all Tier-2 texts. |
| Pseudo-families (`pseudo_family.py`) | ours | Category-preserving renaming produces unseen vocabulary that is still physically valid (verified by the engine). Justified because the 10 rules are material-agnostic — they constrain *function order*, not specific chemistries. |
| Fix suggestions + repair (`fix.py`) | ours | Each repair inserts the exact missing enabler the rule names (a clean, a develop, a fill) or restores the documented block order — i.e. it makes the route satisfy the Tier-1 rule, verified by re-running the checker. |
| Verifier-as-reward (`reward.py`) | ours | Standard verifier-guided RL; the verifier is the Tier-1 rule set, so the reward is exactly the challenge's own correctness definition. |

**Honesty note.** Where the fab CSVs gave us specific parameters (temperatures,
doses, chemistries) we use them verbatim (Tier 1). Where we generalise to unseen
families we rely only on the *category-level* physics that every Tier-2 source
treats as universal — we do **not** invent new chemistries or numbers for
unknown families; we only assert that an unknown deposition still needs a clean,
an unknown etch still needs a mask, etc.

---

## How to verify any claim yourself
- Rule definitions: `python physics/process_knowledge.py --export knowledge/PROCESS_MODEL.md`
- Rules vs reference: `python exhaustive_test.py` (3003/3003 real sequences; 42/42 checks)
- Parameter provenance: `python physics/step_semantics.py` (prints the CSV-sourced
  descriptions/parameters per step)

Sources:
- [RCA clean — Wikipedia](https://en.wikipedia.org/wiki/RCA_clean)
- [MicroChemicals — Ion Implantation with Photoresist Masks](https://www.microchemicals.com/dokumente/application_notes/ion_implantation_photoresist.pdf)
- [Implant mask thickness — MEMSolver](https://www.memsolver.com/Help/tochtml/impmask.htm)
- [Silicon VLSI Technology (Plummer/Deal/Griffin)](https://www.amazon.com/Silicon-VLSI-Technology-Fundamentals-Practice/dp/0130850373)
