#!/usr/bin/env python3
"""
benchmark_models.py — objective, apples-to-apples comparison of the 3 training
methods. Task 3 is identical (physics) across all, so the differentiator is the
MODEL's next-step prediction, measured both in-distribution and in an OOD context
(real-family flows). Reported with no skew: OOD accuracy is computed only over
positions whose true next step IS in the model's vocabulary (otherwise it is
unpredictable by construction), with n shown.
"""

from __future__ import annotations
import random, sys
from pathlib import Path

_SUB = Path(__file__).resolve().parent
for _p in (str(_SUB), str(_SUB / "src"), str(_SUB / "training_data")):
    sys.path.insert(0, _p)
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

from generate_sequences import generate_sequence, validate_sequence
from inference import ProcessPredictor
from real_family_benchmark import FAMILIES   # the 5 real OOD flows

MODELS = {"M0_baseline": "outputs_test", "M1_continue+integ": "outputs_M1",
          "M2_scratch+integ": "outputs_M2"}


def id_eval(pred, n_per=8, seed=999):
    # use_physics=False: compare the RAW model ranking. The physics rerank is
    # identical across all models, so including it would only mask model
    # differences — for a fair model-vs-model comparison we isolate the model.
    rng = random.Random(seed)
    rows = []
    for fam in ("mosfet", "igbt", "ic"):
        for _ in range(n_per):
            s = generate_sequence(fam, rng)
            if validate_sequence(s):
                continue
            rows.append((fam, s))
    n = t1 = t5 = 0
    for fam, s in rows:
        for i in range(4, len(s) - 1, 2):   # every 2nd position (speed)
            ranks = [r[0] for r in pred.predict_next_steps(s[:i], fam, top_k=5, use_physics=False)]
            n += 1; t1 += s[i] == ranks[0] if ranks else 0; t5 += s[i] in ranks
    return t1 / max(n, 1), t5 / max(n, 1), n


def ood_eval(pred):
    # next-step on real-family flows; only count positions whose target is in vocab
    vocab = set(pred.tokenizer.token2id)
    n = t1 = t5 = skipped = 0
    for fam, seq in FAMILIES.items():
        f = "mosfet"  # arbitrary family token; these are OOD anyway
        for i in range(4, len(seq) - 1):
            tgt = seq[i]
            if tgt not in vocab:        # unpredictable by construction — exclude, count it
                skipped += 1
                continue
            ranks = [r[0] for r in pred.predict_next_steps(seq[:i], f, top_k=5, use_physics=False)]
            n += 1; t1 += tgt == ranks[0] if ranks else 0; t5 += tgt in ranks
    return t1 / max(n, 1), t5 / max(n, 1), n, skipped


def main():
    print(f"{'MODEL':<20}{'ID Top-1':>10}{'ID Top-5':>10}{'OOD Top-1':>11}{'OOD Top-5':>11}{'OODn':>7}")
    print("=" * 72)
    results = {}
    for name, d in MODELS.items():
        path = _SUB / d
        if not (path / "best_transformer.pt").exists():
            print(f"{name:<20}  (not trained yet — skipped)")
            continue
        try:
            pred = ProcessPredictor.load(path, device="cpu")
        except Exception as e:
            print(f"{name:<20}  load error: {type(e).__name__}: {e}")
            continue
        idt1, idt5, idn = id_eval(pred)
        ot1, ot5, on, sk = ood_eval(pred)
        results[name] = dict(id1=idt1, id5=idt5, o1=ot1, o5=ot5, on=on, sk=sk)
        print(f"{name:<20}{idt1:>10.3f}{idt5:>10.3f}{ot1:>11.3f}{ot5:>11.3f}{on:>7}")
    print("=" * 72)
    print("Task 3 (anomaly) decision is the rule engine, which is model-INDEPENDENT,")
    print("so it is identical across these models (this script only ranks the models'")
    print("next-step skill). For the model's UNAIDED anomaly signal and the rule-engine")
    print("numbers, see `python src/evaluate.py --self-eval` (model-only ROC-AUC).")
    if results:
        best = max(results, key=lambda k: (results[k]["o5"], results[k]["o1"], results[k]["id5"]))
        print(f"\nBest OOD generalisation (model next-step): {best}")
    print("NB: tiny CPU models, few epochs — this ranks METHODS, not final scores.")


if __name__ == "__main__":
    main()
