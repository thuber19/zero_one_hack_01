# 📁 Hackathon Case Materials — AI-Guided Conversion Flow

> UNIQA Health Insurance — Intelligent Conversion Coach  
> Read this first. Estimated read: 3 minutes.

---

## The Challenge in One Sentence

Build a **Conversion Coach** that detects when users are about to abandon UNIQA's health insurance calculator and intervenes in real time — then prove it works using synthetic persona simulations.

---

## 🎯 Scope Constraint — READ THIS

There is a hard scope boundary for this track. The Conversion Coach operates **only** on the path users can complete online themselves:

| In Scope ✅ | Out of Scope ❌ |
|---|---|
| **Privatarzt tariffs** ("Bei Arztbesuchen" — Start & Optimal) | **Krankenhaus tariffs** ("Im Krankenhaus" — hospital/Sonderklasse path) |
| **"Ich selbst"** — insurance for yourself only | **"Andere Personen"** — insurance for others (routes to advisor) |
| **Online-purchasable tariffs** (Start & Optimal) | **Advisor-required tariffs** (Opt. Plus & Premium — routes to appointment booking) |
| **All information currently collected in the calculator** must still be collected | Advisor handoff is a valid exit route but **not counted as conversion** for this track |
| Users who can complete online → help them convert | Users who can't → route them to advisor (no further coaching) |

**Conversion for this track = online purchase completion (Start or Optimal tariff).** Anything that routes to an advisor is outside the coaching scope — it's a clean handoff, not a conversion win.

This means: the Krankenhaus path (Step 5 add-ons), the "andere Personen" branch (Step 2), and Opt. Plus/Premium tariffs are **explicitly excluded** from the Coach's active intervention scope. Users selecting these are routed to an advisor and exit the funnel. The Coach does not try to save them — it only helps users who *can* complete online actually complete.

All information currently asked in the calculator must still be collected — no steps may be removed from the in-scope path.

---

## Reading Order

| Order | File | What you get | Time |
|---|---|---|---|
| **1** | This README | Folder structure, what matters, what doesn't | 3 min |
| **2** | `Track_AI_Guided_Conversion_Flow_EN.md` | Full case spec — problem, architecture, deliverables, eval criteria | 10 min |
| **3** | `uniqa-funnel-doc_german.md` | The 15-step calculator journey — every screen, every drop-off point | 7 min |
| **4** | `personas_comparison_matrix.md` | Cheat sheet — all three personas side-by-side | 5 min |
| **5** | `persona_judith_segment_1.md` / `_franz_` / `_peter_` | Full persona briefings (use as system prompts for persona bots) | 5 min each |
| **6** | `personas.json` | All quantitative data — demographics, channel prefs, purchase drivers, life events | reference |

**During the event**, also walk through the live calculator: [uniqa.at/rechner/krankenversicherung](https://www.uniqa.at/rechner/krankenversicherung/) — mandatory in the first 30 minutes.

---

## File Map

### Core case documents (you must read these)

| File | Role | Language |
|---|---|---|
| `Track_AI_Guided_Conversion_Flow_EN.md` | **Case specification** — problem statement, architecture, deliverables, evaluation criteria, technical notes. This is the contract. | English |
| `Track_AI_Guided_Conversion_Flow.md` | Same case spec in German. | German |
| `uniqa-funnel-doc_german.md` | **Funnel documentation** — the 15-step journey, UI description, drop-off points, conversion-killer hypotheses. No English version exists — use translation if needed. | German |

### Persona files (you need all three)

| File | Role |
|---|---|
| `persona_judith_segment_1.md` | **Judith Berger**, Segment 1 — Rising Hybrids (30% of funnel traffic). Researches online, commits in person. Drops off at initial price display. |
| `persona_franz_segment_2.md` | **Franz Huber**, Segment 2 — Online Affine (50% of funnel traffic). Digital-first, hates friction. Drops off at final price. |
| `persona_peter_segment_3.md` | **Peter Wagner**, Segment 3 — Service Affine (20% of funnel traffic). Overwhelmed by complexity, wants someone to just tell him what to do. Drops off early. |
| `personas_comparison_matrix.md` | **Cheat sheet** — all three personas compared side-by-side. Start here before reading the full briefings. |
| `personas.json` | **Structured data** — every quantitative value from the UNIQA segmentation booklet (n=4,004). Demographics, insurance behavior, decision drivers, channel preferences, life events, financial assets, UNIQA market share, and more. |

### Source presentations (reference only — data already extracted)

| File | What it is | Do you need it? |
|---|---|---|
| `SegmentierungNeu_SegmentBooklet_Final_März_2026.pptx` | Original UNIQA segmentation booklet (35 slides). Source of all quantitative data in `personas.json`. | **No** — all relevant data is in `personas.json`. Consult only if you need to verify a specific number or see charts/visuals. |
| `Analyse KV ambulant_AI Hackathon.pptx` | UNIQA funnel analysis (4 slides). Shows drop-off rates and traffic sources. | **No** — same data is in the track spec and funnel doc. |
| `Segment Prototypen bzw. Personas Mai 2026.pptx` | Segment persona summaries (3 slides). One slide per segment with short labels. | **No** — content is fully covered by the markdown personas. |

---

## Key Numbers to Know

These are the numbers you'll reference all weekend. Memorize them.

**The funnel:**
- **5.6%** online conversion rate (1,000 starters → ~56 completions)
- **66%** drop off at initial price display (Step 4 — conditional on reaching this step)
- **24%** drop off at add-on coverage selection (Step 5 — conditional)
- **78%** drop off at final price (Step 7 — conditional)
- Survival math: 1,000 → 340 → 258 → ~57 ≈ 56

**The personas:**

| | Judith (S1) | Franz (S2) | Peter (S3) |
|---|---|---|---|
| Funnel share | 30% | **50%** | 20% |
| Dominant channel at purchase | Advisor (78%) | **Online (89%)** | Customer service (59%) |
| NPS | +17 | +1 | **–6** |
| KV purchase intent 3y | 18% | 16% | 13% |
| Primary drop-off step | Initial price | **Final price** | Early — before price |
| Conversion = (in scope) | **Online purchase only** | **Online purchase only** | **Online purchase only** |
| Out-of-scope exit | Advisor handoff | Advisor handoff | Advisor handoff |
| Coach must NOT | Push online-only | **Push advisor** | Push self-service |

**The product (⚠️ only Start & Optimal are in scope for coaching):**
- 4 tariffs exist, but only **2 are online-purchasable** → Start (€38.74) / Optimal (€68.14)
- Opt. Plus (€96.66) + Premium (€140.16): advisory required → routes out of scope to appointment booking
- Krankenhaus (hospital) path: also routes to advisor, not online-purchasable
- "Andere Personen" (insurance for others): also routes to advisor, not online-purchasable
- ~80% of traffic arrives via paid/organic search
- **All calculator steps must still be collected — no simplification of the data-gathering flow**

---

## What "Conversion" Means (Important — Scoped)

For this track, **conversion = online purchase only.** Completing the calculator and signing up online for a Start or Optimal tariff.

Advisor handoffs (for Krankenhaus, andere Personen, Opt. Plus/Premium) are a **valid exit route** but **do not count as conversion for this track.** The Coach routes those users away cleanly — it does not coach them.

The 5.6% baseline already reflects online-only completions. The Coach's goal is to increase that number by helping users who *can* complete online actually complete, not by routing more people to advisors.

> **Note:** In the broader UNIQA context, advisor handoffs are a valid business outcome. For this hackathon track, we focus exclusively on the online-purchasable path to keep the problem scope tight.

---

## How the Files Connect

```
Track spec (EN/DE)
  ├── references ──→ personas.json (data source)
  ├── references ──→ persona_*.md (persona briefings)
  └── references ──→ uniqa-funnel-doc (journey description)

personas.json
  ├── quantifies ──→ persona_judith_segment_1.md
  ├── quantifies ──→ persona_franz_segment_2.md
  └── quantifies ──→ persona_peter_segment_3.md

personas_comparison_matrix.md
  └── summarizes ──→ personas.json (side-by-side)

uniqa-funnel-doc_german.md
  └── describes ──→ the actual calculator journey (screens, steps, drop-offs)

PPTX files (3×)
  └── source documents for the above — data already extracted
```

**Rule of thumb:** If a number appears in both `personas.json` and a markdown persona, the JSON is the authoritative source. If a narrative in a markdown persona conflicts with JSON data, the JSON is the population-level truth but the markdown captures the archetype — see each persona's source note for guidance.

---

## Architecture at a Glance

```
                    ┌──────────────────────┐
                    │   Insurance Chatbot  │  ← existing, handles domain questions
                    │   (product, terms,   │
                    │    tariff compare)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Conversion Coach   │  ← YOU BUILD THIS
                    │                      │
                    │  Detection layer:    │     Detects: dwell time, back-nav,
                    │  When to intervene?  │     repeated changes, hover patterns
                    │                      │
                    │  Decision layer:     │     Interventions: explanations,
                    │  How to intervene?   │     trust signals, handoffs, price
                    │                      │     reframes, simplification
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Persona Bots (×3)   │  ← YOU BUILD THESE
                    │  Judith / Franz /    │     Synthetic users with different
                    │  Peter               │     intentions and reaction patterns
                    └──────────────────────┘
```

---

## Quick-Start Checklist for First 30 Minutes

- [ ] Read this README
- [ ] Skim `personas_comparison_matrix.md` (5 min)
- [ ] Walk through the live calculator at [uniqa.at/rechner/krankenversicherung](https://www.uniqa.at/rechner/krankenversicherung/) — experience the drop-off points yourself
- [ ] Read the track spec (`Track_AI_Guided_Conversion_Flow_EN.md`)
- [ ] Read the funnel doc (`uniqa-funnel-doc_german.md`)
- [ ] Pick a persona to build first (start with Franz — he's 50% of funnel traffic and the simplest channel logic)

---

## Known Gaps (Things we don't have data for)

These are documented unknowns, not oversights. Teams should flag them in their reports:

| Gap | Impact | What to do |
|---|---|---|
| No drop-off data for advisor-booking sub-funnel (Steps 8–11) | **Out of scope** — advisor path is not coached, only routed | Not needed — not part of coaching scope |
| Step 6 (health questions) not fully documented | Medium — sits between two biggest drop-offs | Walk through live calculator to capture |
| Step 12+ (online closing) not fully documented | Low — affects final conversion math only | Walk through live calculator to capture |
| No distribution of price delta (initial vs. final price) | High — the gap drives the 78% drop-off | Use synthetic assumptions; document them |
| Channel data shows only dominant channel per step, not full 3-way split | Medium — limits multi-channel simulation | Estimate secondary channels from dominant value and segment profile |
| Krankenhaus (hospital) path after Step 5 | **Out of scope** — Krankenhaus path routes to advisor, Coach does not intervene here | Not simulated — routes to advisor exit |

---

## File Sizes & Formats

| File | Size | Format |
|---|---|---|
| `personas.json` | ~37 KB | JSON — directly loadable in Python/JS |
| `Track_AI_Guided_Conversion_Flow_EN.md` | ~14 KB | Markdown |
| `Track_AI_Guided_Conversion_Flow.md` | ~15 KB | Markdown (German) |
| `uniqa-funnel-doc_german.md` | ~10 KB | Markdown (German) |
| `persona_judith_segment_1.md` | ~7 KB | Markdown |
| `persona_franz_segment_2.md` | ~7 KB | Markdown |
| `persona_peter_segment_3.md` | ~11 KB | Markdown |
| `personas_comparison_matrix.md` | ~9 KB | Markdown |
| `SegmentierungNeu_...pptx` | ~12 MB | PowerPoint — source booklet (35 slides) |
| `Analyse KV_...pptx` | ~2.6 MB | PowerPoint — funnel analysis (4 slides) |
| `Segment Prototypen_...pptx` | ~2.2 MB | PowerPoint — persona summaries (3 slides) |