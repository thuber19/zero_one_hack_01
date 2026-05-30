# Ready-to-Train Guide (integrated system)

Everything needed to train the model and save results, with the integration
fully wired and verified. The physics harness is stdlib (runs anywhere); the
neural pipeline needs `requirements.txt` (torch/sklearn) and ideally a GPU
(Leonardo) for full-size runs.

## What's verified (run locally, real numbers)
- Harness correctness: `exhaustive_test.py` = **42/42** (3000 provided sequences
  validate; every window boundary exact; all 10 rules detect/explain/repair;
  0 false positives; OOD on 5 real published families = 0 FP).
- Real trained pipeline (tiny, CPU): RF top-15 **0.9999**; transformer val-acc
  **~0.80**; integrated self-eval Task 1 Top-5 **0.997**, Task 3 F1/RuleAttr
  **1.000**, with **per-family breakdown + ROC-AUC + Block-level accuracy**.
- Security/sanitization: no eval/exec/secrets; pickle cache fail-safe.

## The three training methods (compared)
| Method | How | Command |
|---|---|---|
| **M0 baseline** | colleagues' original (no integration) | `OUTPUT_DIR=outputs python src/train.py --model-size small --epochs 50` |
| **M1 continue + integration** | fine-tune M0 checkpoint + UNK-dropout (OOD lever, same vocab) | `OUTPUT_DIR=outputs_M1 python src/train.py --model-size small --epochs 30 --init-from outputs/best_transformer.pt --unk-dropout 0.15` |
| **M2 scratch + integration** | integrated data (real + pseudo-family) + UNK-dropout | `python src/generate_integrated_data.py --extra-data 5000 --ood 1500 --output-dir outputs_M2 && OUTPUT_DIR=outputs_M2 python src/train.py --model-size small --epochs 50 --unk-dropout 0.15` |

**Controlled tiny-CPU comparison (ranks methods, not final scores):**

| Model | ID Top-1 | ID Top-5 | OOD Top-1 | OOD Top-5 |
|---|---|---|---|---|
| M0 baseline        | 0.786 | 1.000 | 0.271 | 0.364 |
| M1 continue+integ  | **0.808** | 1.000 | 0.224 | 0.393 |
| M2 scratch+integ   | 0.642 | 0.974 | 0.280 | **0.477** |

- Integration **helps OOD** (both beat baseline on OOD Top-5).
- **M1 (continue+integration) wins in-distribution** (0.808 Top-1).
- **M2 (scratch+integration) wins OOD** (0.477 Top-5) despite being undertrained.
- Task 3 identical (physics) for all → F1 1.000.

**Recommended Leonardo run:** scratch+integration (M2 method) trained to
convergence at full size for best Task-4 OOD; keep continue+integration (M1) as
the strong in-distribution model. Use `jobs/train_integrated.sh` (sizes
tiny/small/medium for the scaling curve).

## End-to-end (one path)
```bash
cd tracks/industrial-infineon
# data
python src/generate_integrated_data.py --extra-data 5000 --ood 1500 --output-dir outputs_run
# train (writes model_config.json so inference loads the right size)
OUTPUT_DIR=outputs_run python src/train.py --model-size small --epochs 50 --unk-dropout 0.15
# evaluate (per-family + AUC + block-level, physics-integrated)
OUTPUT_DIR=outputs_run python src/evaluate.py --self-eval --output-dir outputs_run --model-size small
# plots
python make_plots.py
# official eval files when distributed -> submission CSVs
python src/inference.py --eval-dir <eval_files> --output-dir outputs_run
#   -> outputs_run/submissions/{nextstep,completion,anomaly}.csv
# demo
python demo.py --output-dir outputs_run
```

## Integration points (model proposes, physics disposes)
- Task 1: `refinery.rerank` over the model's top-15 (physics-legal first).
- Task 2: `refinery.constrained_decode` (vetoes invalid steps, guarantees
  termination — completions are ALWAYS physically valid).
- Task 3: `validate_sequence_combined` (exact for known families, category-based
  for OOD incl. the now-OOD-robust ordering rules) + `fix.analyze` for
  explanation + repair.
Toggle with `use_physics=False` for ablation.

## Enrichments (this round)
- **Next-category auxiliary head — DONE** (optional, backward-compatible).
  Train with: `python src/train.py --model-size small --epochs 50 --aux-category`.
  Verified: joint loss back-props, and existing checkpoints still load (n_cat=0).
- **Parameter plausibility** (`physics/parameters.py`) — additive sanity check on
  fab numbers (temp/dose/energy/thickness); 0 false positives on 304 real
  parameters; flags absurdities. NOT one of the 10 rules; never affects scoring.
- **Causal graph + judge artifact** — `knowledge/PROCESS_LOGIC.md` (Mermaid +
  the physics "why" per rule) and `knowledge/logic_graph.png`.
- **Structural diversity** (second-metal-layer axis), **window-edge contrastive
  pairs**, **curated synonyms** (coverage 159/180), **+3 hypothesis OOD families**.

## Deliberately NOT done (anti-overengineering calls)
- **RoPE**: sequences are ≤200 steps where learned positional embeddings are
  sufficient; RoPE needs a custom-attention rewrite (risk) for marginal gain.
- **Factorized INPUT embedding**: scoped-next — it requires the inference path to
  also supply per-token categories; the aux *output* head already delivers the
  category signal with zero inference change.

## Remaining (next steps)
- Train **M3 = scratch + integration + `--aux-category`** at full size; 4-way benchmark.
- Full Leonardo scaling sweep + GRPO (`reward.py`) fine-tune.
- Fill `LICENSE` with the real team name; record demo using `demo.py` +
  `knowledge/logic_graph.png`.
```
