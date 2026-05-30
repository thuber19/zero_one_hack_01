# Process Knowledge Model

_Auto-generated from `physics/process_knowledge.py` — the single source of truth. Edit the data there, regenerate this._

## How the process works

1. **Incoming** — Receive the lot, identify it, inspect and measure the bare wafer.
2. **Pre-clean** — Wet/chemical cleans (RCA, HF dip) remove organics and native oxide — the wafer must be clean before anything is grown.
3. **Substrate prep** — Family-specific: epitaxy (MOSFET/IGBT) or backside grind/clean (IC) to set up the starting material.
4. **First oxidation** — Grow the first thermal oxide; it both protects the surface and provides a clean base for what follows.
5. **Litho–etch–implant cycles** — The heart of the process: for each device region, coat→align→expose→develop resist, etch the opening, strip, clean, implant dopants, anneal. Repeated per mask level in ascending order.
6. **Interlayer dielectric** — Deposit dielectric, densify, CMP flat — the insulating layer between devices and wiring.
7. **Vias** — Litho + etch contact holes through the dielectric, then fill with barrier/seed/metal and CMP.
8. **Metallisation** — Deposit metal, pattern it with full lithography, etch, strip, clean — the interconnect wiring.
9. **Passivation** — Deposit and cure the protective top layer, then open windows to the bond pads.
10. **Backside** — Thin the wafer and form the backside contact — only after the front is sealed.
11. **Test & ship** — Final clean and inspection, electrical + sort test, then release and ship — never before sort.

## Wafer state variables (what the process reasons about)

### `surface_cleanliness`
- **What:** Whether the top surface is chemically clean and well-defined (set by a clean/prep/anneal/CMP/thermal-oxidation step; decays as later steps add contamination).
- **Why it matters:** Thin-film growth nucleates on the existing surface. Contaminants get buried in the film as defects, so a clean surface must exist shortly before any deposition.

### `resist_pattern`
- **What:** The photoresist lifecycle: coated → exposed → developed (a physical mask with openings).
- **Why it matters:** Etchants and implants act everywhere they can reach. A developed resist pattern is the physical stencil that decides where material is removed or doped.

### `mask_level`
- **What:** The highest lithography mask level aligned so far.
- **Why it matters:** Each level's features register to structures built by the previous level. Levels must advance in order or the geometry mis-registers.

### `implant_window`
- **What:** Whether an opening exists (oxide etch or developed resist) through which ions can reach the substrate.
- **Why it matters:** Oxide and resist block implant ions. Doping only lands where a window was recently opened.

### `overburden`
- **What:** Whether excess material was recently deposited or filled above the target plane.
- **Why it matters:** CMP polishes material down to a plane. With nothing deposited above it, CMP grinds into the structure instead of levelling overburden.

### `passivation`
- **What:** The protective top layer's lifecycle: deposited → cured.
- **Why it matters:** The final passivation seals the device. Pad-window opening, electrical test, and backside work all require it to exist and be cured first.

### `sort_tested`
- **What:** Whether wafer sort test has screened the dice.
- **Why it matters:** Shipping before sort sends untested, possibly defective product to the customer.

## Operation (event) classes

Membership is hybrid: exact reference vocabulary for known steps, physical category for unknown 4th-family steps.

- **DEPOSITION** — Grows or deposits a new material layer.  
  _matched by:_ 19 known steps; OOD categories {DEPOSIT}
- **PATTERNED_ETCH** — Removes material through a resist pattern.  
  _matched by:_ 13 known steps; OOD categories {ETCH}
- **METAL_ETCH** — Patterns the metal interconnect (needs full lithography).  
  _matched by:_ 2 known steps
- **IMPLANT** — Drives dopant ions into the substrate.  
  _matched by:_ 9 known steps; OOD categories {IMPLANT}
- **CMP** — Chemical-mechanical planarisation.  
  _matched by:_ 4 known steps; OOD categories {CMP}
- **PAD_WINDOW_OPEN** — Opens a window to the bond pads through passivation.  
  _matched by:_ 4 known steps
- **ELECTRICAL_TEST** — Probe-based electrical characterisation.  
  _matched by:_ 6 known steps
- **SHIP** — Releases the lot to the customer.  
  _matched by:_ 1 known steps
- **BACKSIDE_METAL** — Deposits the backside metal contact.  
  _matched by:_ 1 known steps
- **CLEAN_SURFACE** — Leaves a deposition-ready surface.  
  _matched by:_ 32 known steps; OOD categories {ANNEAL, CLEAN, CMP, PREP}
- **DEVELOP** — Develops the resist, creating the physical mask pattern.  
  _matched by:_ 2 known steps; OOD flags develops_resist
- **EXPOSE** — UV-exposes the resist, writing the latent image.  
  _matched by:_ prefixes EXPOSE LITHO LEVEL; OOD flags exposes_resist
- **IMPLANT_OPENER** — Opens an implant window (oxide etch or develop).  
  _matched by:_ 4 known steps; OOD union of PATTERNED_ETCH, DEVELOP
- **DEPOSIT_OR_FILL** — Leaves overburden for CMP to planarise.  
  _matched by:_ 21 known steps; OOD categories {DEPOSIT, FILL}

## Rules (why each violation is impossible)

### Windowed rules — a trigger needs an enabler within N steps

#### RULE_DEP_NO_CLEAN — Deposition needs a clean surface
- **Trigger:** DEPOSITION
- **Requires:** CLEAN_SURFACE within 12
- **Plain:** A deposition has no cleaning step in the prior 12 steps.
- **Why:** Thin-film deposition nucleates on the existing surface; contamination from a prior etch or handling becomes buried defects. A clean, well-defined surface must exist shortly before any deposition. This is universal to all thin-film processes, which is why it transfers to unseen families.

#### RULE_METAL_ETCH_NO_LITHO — Metal etch needs full lithography
- **Trigger:** METAL_ETCH
- **Requires:** EXPOSE within 15, DEVELOP within 15
- **Plain:** A metal etch is missing its EXPOSE and/or DEVELOP within 15 steps.
- **Why:** Metal patterning needs an exact resist image: exposure writes the latent image, development turns it into a physical mask. Both must be present and recent, or the etch clears the whole metal layer.

#### RULE_ETCH_NO_MASK — Patterned etch needs a developed mask
- **Trigger:** PATTERNED_ETCH
- **Requires:** DEVELOP within 12
- **Plain:** A patterned etch has no DEVELOP in the prior 12 steps.
- **Why:** Etchants attack every exposed surface. Developed resist physically shields what must survive. Without it the etch removes material uniformly and no device geometry is defined.

#### RULE_IMPLANT_NO_MASK — Implant needs an open window
- **Trigger:** IMPLANT
- **Requires:** IMPLANT_OPENER within 15
- **Plain:** An implant has no mask opening (oxide etch or develop) within 15 steps.
- **Why:** Oxide and resist block implant ions; doping only reaches the substrate through a recently opened window. Without one, doping is misplaced or absent.

#### RULE_CMP_NO_DEP — CMP needs something to planarise
- **Trigger:** CMP
- **Requires:** DEPOSIT_OR_FILL within 6
- **Plain:** A CMP step has no deposition or fill in the prior 6 steps.
- **Why:** CMP polishes material down to a target plane. With no overburden it grinds into the underlying structure.

### Litho-level rule (numeric ordering)

#### RULE_LITHO_LEVEL_SKIP — Mask levels advance in order
- **Plain:** A mask level is skipped or decreases.
- **Why:** Each lithography level patterns features that register to structures built by the previous level. Skipping a level means those structures and alignment marks were never created; decreasing a level would overwrite completed structures.

### Ordering rules — a trigger needs a milestone first

#### RULE_PAD_OPEN_BEFORE_DEP — Pad window needs deposited+cured passivation
- **Trigger:** PAD_WINDOW_OPEN
- **Requires milestones:** passivation_deposited, passivation_cured
- **Plain:** A pad window is opened before passivation is deposited and cured.
- **Why:** The pad-window etch opens access to the bond pads through the passivation. The layer must exist and be cross-linked (cured) first, or the etch attacks the metal/dielectric beneath.

#### RULE_TEST_BEFORE_PASSIVATION — Electrical test needs cured passivation
- **Trigger:** ELECTRICAL_TEST
- **Requires milestones:** passivation_cured
- **Plain:** An electrical test runs before passivation is cured.
- **Why:** Probe needles contact the pads; without cured passivation sealing the interconnects and active areas, probing causes contamination, leakage and irreversible damage.

#### RULE_SHIP_BEFORE_TEST — Ship needs sort test
- **Trigger:** SHIP
- **Requires milestones:** sort_test_done
- **Plain:** The lot ships before wafer sort test.
- **Why:** Sort test screens defective dice. Shipping first sends untested, possibly defective product to the customer.

#### RULE_BACKSIDE_BEFORE_PASSIVATION — Backside metal needs cured passivation
- **Trigger:** BACKSIDE_METAL
- **Requires milestones:** passivation_cured
- **Plain:** Backside metal is deposited before passivation is cured.
- **Why:** Backside metallisation subjects the wafer to reactive sputtering, thermal stress and handling; without cured passivation the finished front-side devices delaminate, crack, or get contaminated.

## Milestones (one-way state flags)

- **passivation_deposited** — Protective passivation layer has been deposited.  (set by: DEPOSIT PASSIVATION, DEPOSIT PASSIVATION LAYER)
- **passivation_cured** — Passivation has been cured (cross-linked, mechanically stable).  (set by: CURE PASSIVATION)
- **sort_test_done** — Wafer sort test has screened the dice.  (set by: WAFER SORT TEST)
