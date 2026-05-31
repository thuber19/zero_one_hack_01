# Zero One Hack_01

> ## 🏁 Team **TBD** — Industrial AI (Infineon) submission
> *Tobias Huber · Mina Mikail · Khaled El Yamany · Fathy Shalaby*
>
> **What:** **procseq** — two from-scratch neural models (Llama-style decoder for
> next-step/completion, DeBERTa-style encoder for anomaly) that *learn* semiconductor
> process logic, wrapped in a physics verification layer (*model proposes, physics disposes*).
>
> **Read first:** [`REPORT.md`](REPORT.md) (full write-up + results) ·
> submission code in [`tracks/industrial-infineon/solution/`](tracks/industrial-infineon/solution/) ·
> deliverable CSVs + scores in [`tracks/industrial-infineon/solution/artifacts/`](tracks/industrial-infineon/solution/artifacts/).
>
> **Run it:**
> ```bash
> pip install -r requirements.txt
> cd tracks/industrial-infineon/solution
> make smoke                                              # CPU ~30s sanity check
> python -m procseq.run_all --config configs/leonardo_decoder.yaml   # full pipeline (GPU)
> ```
> Headline results (held-out self-eval): next-step **Top-1 0.77 / Top-5 1.00**
> (category 0.96); completion **block-acc 0.92, 100% rule-valid**; anomaly via
> physics hybrid. See `REPORT.md` for the honest breakdown.

---

**36 hours. Real infrastructure. European AI sovereignty.**

Welcome to the central repository for Zero One Hack_01, hosted by [Lumos Consulting](https://lumos-consulting.at) at [AI Factory Austria](https://aifactory.at) in Vienna, with compute provided by CINECA on the Leonardo GPU Cluster.

---

## Quick links

- 🌐 **Docs**: [docs.zero-one.lumos-consulting.at](https://docs.zero-one.lumos-consulting.at/)
- 💬 **Discord**: https://discord.gg/e6rrVbcD5
- 📍 **Venue**: AI Factory Austria (AI:AT), Vienna

---

## The three tracks

| Track                | Partner  | What you'll build                                                                                                               |
| -------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 🧾 **Insurance AI**   | UNIQA    | An AI-guided conversion flow that replaces a static form-based insurance calculator. Persona-based simulations on Leonardo.     |
| ⚙️ **Industrial AI**  | Infineon | Train and benchmark sequence models on semiconductor process flows. Does your model learn real process logic, or just memorize? |
| 📈 **Forecasting AI** | Sybilion | Build a decision agent on top of a probabilistic forecasting API. Live mid-run plot twist on Sunday.                            |

Each track's full briefing, data, and starter materials live in [`/tracks/`](./tracks/). 

---

## What's provided

- **Compute**: Leonardo GPU Cluster (A100s). 
- **Workspace**: Power, fast WiFi, monitors on request, breakout rooms for team calls.
- **Mentors**: Domain experts from each partner company, plus ML/infra mentors from Lumos and HPE.
- **API credits and tokens**: Track-specific, documented in each track's README.


---
## How submissions work

1. **Fill out the Tally submission form** by Sunday 10:00 — link will be shared in `#announcements`
2. The form takes four fields: team name, repository URL, slides (PDF), and demo video (file or link, max 2 minutes)
3. The Tally form timestamp is your official submission time
4. After 10:00 the form closes. No late submissions.

Full submission details, requirements, and the pre-submission checklist live in [`/submission/SUBMISSION.md`](./submission/SUBMISSION.md).

---

## Judging

Each track has its own rubric in [`/judging/rubrics.md`](./judging/rubrics.md). All tracks share these baseline expectations:

- **Working artifact** — not a slideware demo, something that actually runs
- **Reproducibility** — your repo should let someone else re-run your work
- **Honest evaluation** — show what worked, show what didn't, show what you measured
- **Visible reasoning** — explain *why* you made the technical choices you did

---

## Code of conduct & house rules

- Be kind. Be useful. Be honest about your work.
- AI Factory Austria is a working facility — respect equipment, doors, quiet hours.
- Mentors are here to unblock you, not to write your code. Use them well.
- The Leonardo cluster is shared infrastructure. No cryptomining, no training on copyrighted data, no abuse of compute. Violations = disqualification.
- See [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) for the full version.

---

## Get help

| Channel                                    | Use for                             |
| ------------------------------------------ | ----------------------------------- |
| `#announcements`                           | Schedule changes, important updates |
| `#industrial`,`#insurance`, `#forecasting` | Track-specific questions            |
| `#infra`                                   | Leonardo, GPU quota, WiFi, hardware |
| `#general`                                 | Everything else                     |
| In-person Lumos desk (lobby)               | Anything urgent                     |

---

*Looking forward to seeing what you build.* 🚀