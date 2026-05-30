# Benchmark Results (measured)

All numbers below are **real, measured on this machine** (tiny models, CPU, few
epochs). They **rank METHODS, not final scores** — full-size convergence happens
on Leonardo. Raw artifacts are committed alongside this file:
`outputs*/training_history.json`, `outputs_M3/category_eval.json`. Reproduce the
model-vs-model ranking with `python benchmark_models.py`.

> Honest caveats up front: (1) M0/M1 use the 205-token vocab; M2/M3 use the
> 673-token integrated vocab (real + pseudo-family) — so raw val-accuracy is NOT
> directly comparable across those pairs. (2) No run against the organizers' real
> grader has happened yet. (3) Task-3 F1/AUC of 1.000 reflect the **rule engine**
> (which on in-vocab is the reference checker), not neural skill — the model's
> *unaided* anomaly signal is reported separately below.

## 1. Training runs (raw `training_history.json`)
| Model | What | size | params | vocab | epochs | seqs | best val-acc | best val-loss | RF top-15 | aux head |
|---|---|---|---|---|---|---|---|---|---|---|
| **M0** | baseline (colleague's, no integration) | tiny | 581,248 | 205 | 10 | 6000 | **0.8095** | 0.3308 | **1.0000** | no |
| **M1** | continue + integration (UNK-dropout) | tiny | 581,248 | 205 | 5 | 3000 | 0.8037 | 0.3610 | 0.9999 | no |
| **M2** | scratch + integration (real + pseudo) | tiny | 641,152 | 673 | 6 | 2300 | 0.5281 | 2.398 | 0.9951 | no |
| **M3** | scratch + integration + **category aux head** | tiny | 643,474 | 673 | 12 | 3600 | 0.7595 | 0.6806 | — | **yes** |

## 2. Next-step ranking — `benchmark_models.py` (raw model, physics OFF, same seed)
| Model | ID Top-1 | ID Top-5 | OOD Top-1 | OOD Top-5 |
|---|---|---|---|---|
| M0 baseline       | 0.786 | 1.000 | 0.271 | 0.364 |
| M1 continue+integ | **0.808** | 1.000 | 0.224 | 0.393 |
| M2 scratch+integ  | 0.642 | 0.974 | 0.280 | **0.477** |

- Integration **helps OOD** (both beat baseline on OOD Top-5).
- **M1 wins in-distribution**; **M2 wins OOD** (despite being undertrained).
- For reference, the n-gram model's OOD next-step Top-1 ≈ 0.25 (`real_family_benchmark.py`) — low is expected; the **physics layer** is what generalises OOD, not the raw LM.

## 3. M3 model "understanding" — `category_eval.py` (`outputs_M3/category_eval.json`)
Does the model learn the *function* of a step (not just its name), and does it transfer?
| Metric | Value | Note |
|---|---|---|
| random-guess baseline | 0.056 | 18 categories |
| **ID next-CATEGORY acc** | **0.945** | strong in-distribution functional learning |
| **OOD next-CATEGORY acc** | **0.411** | unseen families — **7.3× random** (real transfer) |
| OOD next-NAME (lexical) acc | 0.152 | category ≫ name ⇒ it learned function, not memorised names |

## 4. Integrated self-eval (physics ON) — `src/evaluate.py --self-eval`
- **Task 1**: Top-1 0.657 · Top-5 **0.997** · MRR 0.823
- **Task 2**: Normalized Edit Distance 0.243 (lower=better); completions guaranteed physically valid
- **Task 3** (rule engine): Acc/Prec/Rec/**F1 1.000** · Rule-Attribution 1.000 · ROC-AUC 1.000
  - **Model-ONLY signal (honest, no physics)**: transformer per-token loss separates classes (valid mean ≈ 0.33 vs invalid ≈ 0.53); RF-violation ROC-AUC ≈ 0.875. (Well below the rule-engine's 1.000 — the engine is what makes detection exact.)

## 5. End-to-end sample-eval dry-run self-score (M1) — `make_sample_eval.py` → `self_score.py`
Held-out spec-format set (36 T1/T2 rows, 36 anomaly rows), submissions format-validated:
- **Task 1**: Top-1 0.722 · Top-3 1.000 · Top-5 1.000 · MRR 0.856
- **Task 2**: Normalized Edit Distance 0.221 · Exact-Match 0.000
- **Task 3**: Acc/Prec/Rec/F1 1.000 · Rule-Attribution 18/18 · confusion TP18 FP0 TN18 FN0

## 6. Correctness/robustness gates (not model scores, but part of the harness)
- `exhaustive_test.py`: **42/42**
- `differential_fuzz.py`: engine ≡ reference on **8,500+ in-vocab mutations, all 10 rules, 0 disagreements**
- `fix.repair`: **0 unrepaired / 9,000** fuzzed invalid routes
- `constrained_decode`: provably valid completion even under an adversarial scorer
- `robustness_test.py`: no crash on empty / 320-step / unknown-family / novel-vocab / blank input

## Recommended pick
- **In-distribution model**: M1 (continue + integration) — best ID Top-1.
- **OOD / Task-4 model**: M2/M3 method (scratch + integration [+ aux head]) — best OOD Top-5; train to convergence at full size on Leonardo (`jobs/train_integrated.sh`, sizes tiny/small/medium for the scaling curve).
