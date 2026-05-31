# Industrial AI (Infineon) — Team TBD submission

> **Can a model learn the hidden *logic* of semiconductor manufacturing — the order
> constraints that make a process route valid — instead of just memorizing sequences?**

Our answer is **procseq**: two from-scratch neural models that learn the grammar of
fab routes, wrapped in a deterministic **physics verification layer**
(*"the model proposes, physics disposes"*). A Llama-style **decoder** does next-step
prediction & completion; a DeBERTa-style **encoder** does anomaly detection; a rule
engine guarantees every emitted route is physically valid.

📄 **The full write-up + results is in the root [`REPORT.md`](../../REPORT.md).**

---

## Where everything is

| Path | What |
|---|---|
| **[`solution/`](solution/)** | **The submission** — procseq (models, training, inference, hybrid, dashboard, tests) |
| `solution/README.md` | Setup + how to run / reproduce |
| `solution/artifacts/` | Deliverables: `nextstep.csv`, `completion.csv`, `anomaly.csv`, `metrics.json`, training logs |
| `solution/pitch/ProcSeq_Pitch.pdf` | Pitch slides |
| `physics/`, `refinery.py`, `fix.py`, `pseudo_family.py` | The **verification companion** procseq imports (rule engine + physics-vetoed decode) |
| `data/` | Organizer data: family variants, eval inputs, `generate_sequences.py`, `generation_rules.md` (the 10 rules) |

## Run it

```bash
pip install -r ../../requirements.txt
cd solution
make smoke                                              # CPU, ~30s sanity check → "SMOKE OK"
python -m procseq.run_all --config configs/leonardo_decoder.yaml   # full pipeline (GPU)
python -m procseq.run_all --config configs/leonardo_decoder.yaml --skip-train  # inference-only
```

## Results (held-out self-eval — see `REPORT.md` for the honest breakdown)

| Task | Headline |
|---|---|
| 1 · Next-step | **Top-1 0.937 · Top-5 1.000 · MRR 0.97**, next-operation accuracy **0.998** |
| 2 · Completion | **Block-level 0.94**, completions **100% rule-valid** |
| 3 · Anomaly | physics hybrid: **F1 1.0, rule-attribution 0.97**; learned encoder alone ≈ chance (AUC 0.49, honest) |

## The three tasks (+ hidden 4th family)

1. **Next-step prediction** — predict the next step of a partial route (Top-5).
2. **Sequence completion** — finish a partial route (must stay rule-valid).
3. **Anomaly detection** — flag routes that break the 10 process-logic rules, attribute the rule.
4. **OOD** *(post-submission)* — generalization to an unseen 4th product family.

The 10 forbidden patterns and the process grammar are documented in
[`data/generation_rules.md`](data/generation_rules.md).
