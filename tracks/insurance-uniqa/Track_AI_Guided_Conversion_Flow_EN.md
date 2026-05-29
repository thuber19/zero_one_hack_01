# Track: Insurance AI
## Use Case Title: AI-Guided Conversion Flow — The Intelligent Conversion Coach

**Challenge Owner:** UNIQA
**Mentor(s):** TBD (Lumos + UNIQA domain expert)
**Difficulty:** Intermediate to Advanced
**Estimated Scope:** Yes, the case is realistic in 36h. The domain-specific advisory logic is provided by an existing chatbot, so the hackathon focuses on two clearly scoped building blocks: the intervention logic of the Conversion Coach and persona-based simulation on the cluster.

---

### 🎯 Scope Constraint — READ THIS FIRST

There is a hard scope boundary for this track. The Conversion Coach operates **only** on the path users can complete online by themselves:

| In Scope ✅ | Out of Scope ❌ |
|---|---|
| **Private-doctor tariffs** ("Bei Arztbesuchen" — Start & Optimal) | **Hospital tariffs** ("Im Krankenhaus" — hospital/Sonderklasse path) |
| **"Myself only"** — insurance for yourself only | **"Other persons"** — insurance for others (auto-routes to advisor) |
| **Online-purchasable tariffs** (Start & Optimal) | **Advisor-required tariffs** (Opt. Plus & Premium — routes to appointment booking) |
| **All information currently collected in the calculator must still be collected** — no steps may be removed | Advisor handoff is a valid exit route but **does not count as conversion** for this track |
| Users who can complete online → Coach helps them convert | Users who cannot complete online → route to advisor (no further coaching) |

**Conversion for this track = online purchase completion (Start or Optimal tariff).** Anything that routes to an advisor is outside the coaching scope — it's a clean handoff, not a conversion win.

This means: the hospital path (Step 5 add-ons), the "other persons" branch (Step 2), and the Opt. Plus/Premium tariffs are **explicitly excluded** from the Coach's active intervention scope. Users selecting these are routed to an advisor and exit the funnel. The Coach does not try to save them — it only helps users who *can* complete online actually complete.

All information currently asked in the calculator must still be collected — no steps may be removed from the in-scope path.

---

### 1. Problem Statement (3–5 sentences)

Customers arrive on the UNIQA website with concrete interest in a health insurance product, run through a 15-step online calculator, and drop off in massive numbers — out of 1,000 people who start, only around 56 complete an online purchase (5.6% conversion rate). The drop-off points are known (most notably the initial price display at 66% drop-off and the final price at 78%), but the *reasons* are not. Today the journey doesn't react to whether someone is hesitating, comparing, or just checking the price — everyone gets the same static funnel. This hackathon develops a **Conversion Coach** that detects uncertainty and abandonment intent in real time and intervenes appropriately, together with a **synthetic persona setup** that simulates different user intentions realistically and lets teams test interventions against each other.

### 2. Why It Matters (Business Context)

- Most customers who abandon were actually interested — they just needed the right support at the right moment. A funnel that adapts to the person (instead of the other way round) lifts conversion *and* improves CX at a touchpoint that today creates frustration.
- UNIQA benefits directly (higher online conversion, better lead quality for advisor handoffs), and the logic is portable to other insurers with complex online journeys.
- Connection to **European AI Sovereignty**: the focus is on independently developed persona simulations on European compute infrastructure, custom intervention logic, and a testable open system rather than an off-the-shelf black box. This is explicitly *not* an LLM-wrapper case — the intelligence sits in the detection/decision logic and in the persona setup.

### 3. Expected Outcome / Definition of Done

- **Minimum Viable Result:** A working Conversion Coach prototype (built on top of an existing insurance-domain chatbot) that detects behavioral signals and intervenes contextually, plus a runnable persona simulation setup with at least three personas and a documented before-vs-after comparison of conversion with and without the Coach.
- **Stretch Goals:** Generate persona variants (e.g., demographic sub-variants per segment), systematically optimize intervention timing, treat journey speed as an additional test dimension, surface new drop-off patterns not yet visible in UNIQA's current data.
- **Learning goals for participants:** Building rule-based or learned intervention logic, designing synthetic personas with realistic reaction patterns, large-scale journey simulation on cluster infrastructure, systematic A/B testing of logics, clean measurement of conversion effects.
- **Format:** Working prototype plus simulation report; optionally complemented by a live demo where a persona runs through the journey and the Coach intervenes at typical drop-off points.
- **Demonstrator Stretch Goal:** A side-by-side demo showing two identical persona runs — one with, one without the Coach — making it visible at which step the intervention made the difference.

### 4. System Specification

- **Architecture:** Two layers. A detection and decision layer (the Coach) sits on top of an existing insurance-domain chatbot. The chatbot answers domain questions; the Coach decides *when* and *how* to intervene.
- **Domain logic:** Assumed given — the chatbot can handle product questions, term explanations, and tariff comparisons. This capability is not built from scratch in the hackathon.
- **Conversion Coach (build focus):**
  - Detects behavioral signals such as inactivity, backwards navigation, repeated changes, unusually long dwell time, hover patterns on price elements
  - Triggers contextual interventions: simplified explanations, trust signals, alternative framings, market comparisons — **advisor handoff only as an exit for out-of-scope paths** (hospital, other persons, Opt. Plus/Premium), not as a coaching goal
  - Designed so different persona types with different intentions can be systematically run through and compared
  - **Scope boundary:** The Coach only coaches the private-doctor/"myself only"/online-purchasable path. Users on hospital, "other persons", or Opt. Plus/Premium paths are routed to an advisor — no coaching.
- **Persona setup:** Synthetic personas with different intentions (purchase, orientation, comparison, price check) and decision logics. Personas are not just test data — they are a central development instrument.
- **Training approaches:** Free choice. Natural candidates include rule-based intervention logic, classical ML classifiers for abandonment probability, LLM-based persona bots, RL for intervention timing, or hybrid approaches.
- **Constraints:** Reproducibility on cluster infrastructure, open and traceable logic instead of black-box recommendations. Explicitly *not* a pure LLM-wrapper case — the Coach logic must be a substantive technical component in its own right.

### 5. Task Structure (Levels)

- **Level 1:** Understand the existing journey (walk through the live calculator, read the journey documentation), turn the three provided personas into runnable persona bots, build a first Conversion Coach logic with clearly defined trigger rules and intervention types.
- **Level 2:** Test the logic in the persona setup, compare at least three intervention variants against each other, measure before/after conversion, formulate initial hypotheses about which intervention works for which persona and why.
- **Level 3 / Stretch:** Run large-scale simulations on the cluster (thousands of journeys, persona variants, timing permutations), uncover new drop-off patterns not visible in UNIQA's current data, formulate validated recommendations for moving the Coach into production.

### 6. Data & Resources

- **Personas:** Three detailed persona profiles (Judith / Franz / Peter) are provided as markdown briefings, complemented by a structured `personas.json` with quantitative values from UNIQA's segmentation research (demographics, insurance behavior, decision criteria, channel preference per CJ step, life-event triggers). Teams can use these profiles directly as system prompts for persona bots or extend them with their own variants. **Note on channel data:** `personas.json` provides `channel_preference_per_journey_step_pct_dominant_channel` — the most preferred channel per step with its percentage share (e.g., consultation: via advisor 90%). The full 3-channel breakdown (online / advisor / customer_service per step) is available only for Segment 1 in the original segmentation booklet and was not transcribed for this dataset. Teams needing secondary-channel distributions should estimate from the dominant value and segment behavioral patterns.
- **Segment distribution in the online funnel:** Estimated 50% Segment 2 (Online Affine), 30% Segment 1 (Rising Hybrids), 20% Segment 3 (Service Affine). Source: UNIQA internal estimate, not hard tracking.
- **Journey documentation:** A companion markdown document describes the 15-step UNIQA health insurance journey with its four visible phases (Inputs → Product → Recommendation → Closing), the known drop-off steps, and the UI elements. **Important:** Only the private-doctor/"myself only"/online-purchasable path is in coaching scope. The hospital path, the "other persons" branch, and the Opt. Plus/Premium tariffs automatically route to appointment booking and are **outside the active coaching scope**. They remain available in the calculator, but the Coach does not try to keep users on these paths — they are routed to an advisor.
- **Product data (health insurance):** Four tariffs (Start / Optimal / Opt. Plus / Premium) with estimated monthly premiums between €38.74 and €140.16, plus per-coverage-area reimbursement caps (medical services, medications, therapeutic treatments, aids, refractive eye surgery). **In scope for coaching are only Start (€38.74) and Optimal (€68.14)** — the two online-purchasable private-doctor tariffs. Opt. Plus and Premium are relevant for simulation as selectable options (users can click them → Coach routes to advisor), but they are not conversion targets. Data is publicly visible at [uniqa.at/rechner/krankenversicherung](https://www.uniqa.at/rechner/krankenversicherung/) and may be used for simulation and synthetically extended.
- **Drop-off data (real):** From UNIQA funnel analysis (Dec 10, 2025 – Feb 1, 2026):

  | Step | What happens | Drop-off |
  |---|---|---|
  | Tariff selection: initial price display | Initial Price | **66%** |
  | Add-on coverage selection | Additional Coverage | 24% |
  | Personal data entry: final price | Final Price | **78%** |

  This yields a **current online conversion baseline of ~5.6%** (out of 1,000 starters, ~56 complete). **Important:** All drop-off rates are conditional on reaching that step (i.e., relative to the cohort that arrived at that step, not to the original 1,000 starters). The survival calculation: 1,000 starters → 340 survive Step 4 (34% survive the 66% drop-off) → 258 survive Step 5 (76% survive the 24% drop-off) → ~57 survive Step 7 (22% survive the 78% drop-off) ≈ 56 completing = 5.6%.

- **Traffic sources:** ~80% of calculator traffic comes from paid and organic search, 70%+ between 9 a.m. and 8 p.m. Users mostly arrive with concrete search intent.
- **Persona bot "Team Tina":** Optional asset from a previous UNIQA hackathon — a Claude-based persona bot built on segmentation data. Access or data export has been requested from UNIQA; if not available, teams start with the provided persona briefings.
- **Compute:** Training and simulation on the Leonardo cluster are explicitly part of the case. GPU quota per team: TBD.
- **NDAs / data privacy:** All provided data is cleared for use. No personally identifiable real-customer data is used.

### 7. Evaluation & Benchmarking

- **Eval setup:** Each team measures the impact of the Conversion Coach against a clearly defined baseline (same persona set, same number of runs, journey without Coach). **Conversion for this track = online purchase completion (Start or Optimal).** An advisor handoff is a valid exit path for out-of-scope users (hospital, other persons, Opt. Plus/Premium), but it **does not** count as a conversion success for the coaching goal. The coaching goal is: help users who *can* complete online actually complete.
- **Three central evaluation dimensions:**

  | # | Dimension | What we measure | Metric(s) |
  |---|---|---|---|
  | 1 | **Conversion uplift** | Does the Coach work? | Conversion rate vs. baseline, drop-off reduction per critical step |
  | 2 | **Persona differentiation** | Does it work across all three personas? | Conversion per persona, performance drop between personas |
  | 3 | **Intervention quality** | When does it help, when does it annoy? | Trigger precision/recall, "annoyance rate" (unnecessary interventions) |

- **Comparability:** All teams work with the same three personas and the same documented journey, so results are directly comparable. Custom persona variants are welcome but must be clearly documented in the final report.
- **Visualization:** At minimum drop-off comparisons (with/without Coach per step), conversion tables per persona, and one qualitative before/after example of a complete persona journey.
- **Test frequency:** Iterative simulation runs during development make sense. Teams decide their own cadence.

**Demonstrator output examples:**

- **Without Coach (persona Franz, Segment 2 — Online Affine):** Sees the initial price display, compares 4 tariffs, clicks Premium once ("advisory required"), navigates back, closes the tab → abandonment.
- **With Coach (persona Franz, same situation):** Coach detects the backwards navigation from the Premium tariff, shows an explanation ("Premium requires a short advisory call; you can complete Optimal fully online at any time"), surfaces a market comparison for Optimal → Franz selects Optimal and converts **online** → ✅ conversion success.
- **Without Coach (persona Judith, Segment 1 — Rising Hybrid):** Sees the final price after the health questions, which is higher than the initially displayed price, abandons.
- **With Coach (same situation):** Coach detects long dwell time on the final price page plus a hover on "cancel," transparently explains why the price changed, supports Judith in completing the online purchase → ✅ conversion success.
- **Out-of-scope example (persona selects hospital or other persons):** Coach detects the path selection, cleanly routes to advisor → no coaching, no conversion success for this track, but correct exit.

### 8. Technical Notes

- **Suggested tech stack:** Python for simulation and eval, an LLM framework of your choice (OpenAI/Anthropic/local) for persona bots, optionally a lightweight frontend (Streamlit, Gradio) for the live demo. Persona bots don't require heavy architecture — good prompts beat complex setups.
- **Mandatory discovery (first 30 minutes):** Every team walks through the live calculator at [uniqa.at/rechner/krankenversicherung](https://www.uniqa.at/rechner/krankenversicherung/) — but **only the private-doctor/"myself" path**. The hospital path and the "other persons" branch are out of scope and do not need to be walked through in detail.
- **Known pitfalls:** Persona bots become too generic when prompts are too short (use the full briefings), intervention triggers fire too often and become annoying, the temptation to build too many things at once instead of validating one axis cleanly, fair baseline comparisons require identical persona seeds. **Scope pitfall:** Do not attempt to coach the hospital path or the "other persons" branch — these are out of scope.
- **Known baseline:** ~5.6% online conversion on the real journey; 66% drop-off at initial price and 78% at final price are the two targets. A Coach logic that visibly reduces these drop-offs (even just for one persona group) is already a strong result.

### 9. Evaluation Criteria (track-specific)

- Technical depth of the Conversion Coach logic and traceable decision rules
- Quality and realism of the persona setup (custom personas or variants are welcome)
- Strength of the baseline vs. Coach comparison in simulation
- Reproducibility and clarity of evaluation (persona seeds, logging, metric definitions)
- Robustness of conclusions — which hypotheses are recommended for production and how are they validated?
- Quality of demo, visualization, and presentation of results

### 10. Contact & Support During the Event

- **Challenge Owner:** Catarina, UNIQA (Slack channel `#help-insurance`)
- **On-site mentor:** TBD
- **Domain expert (Slack/phone):** TBD on UNIQA side
- **Emergency contact:** Lumos desk in the lobby