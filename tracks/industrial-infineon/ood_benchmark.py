#!/usr/bin/env python3
"""
ood_benchmark.py — a CONCRETE, VERIFIABLE Task-4 (OOD) benchmark. No LLM.

Idea (and why the number means something):
  The README says the hidden 4th family "mostly shares vocabulary; differences
  are which optional blocks appear and cycle counts." So we model a 4th family by
  taking REFERENCE-LABELLED sequences (valid ones from the provided variants;
  invalid ones from bad_data_generator, each with its known violated rule) and
  RENAMING a controllable fraction `f` of the distinct step types to NOVEL but
  category-preserving names (keep the functional verb, swap the material — exactly
  how a real new device family is named). The TRUE label is INHERITED from the
  reference verdict on the ORIGINAL sequence (a structural violation survives a
  rename; a valid flow stays valid IF the engine still recognises the renamed
  steps' physical roles).

Why this isn't circular: the label comes from the reference checker on the
ORIGINAL (known) names, while the engine must RE-DERIVE each renamed step's
category from its NOVEL name. If classification fails (e.g. a renamed clean is no
longer seen as a clean), the engine gives the wrong verdict and the benchmark
counts it — which is precisely the OOD robustness we want to measure.

We sweep f = 0.10 → 1.00 and report binary valid/invalid Accuracy/Precision/
Recall/F1 + rule-attribution at each, with the full confusion matrix and every
miss. f≈0.2-0.4 is the realistic "mostly shared vocabulary" regime.

Usage:  python ood_benchmark.py --n-per-class 300 --seed 11
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "training_data"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from generate_sequences import read_csv_sequences
from physics.state_machine import validate_by_state_machine as ENGINE
from physics.known_vocab import KNOWN_VOCAB
import bad_data_generator as BDG

# Material / device-specific words we swap for novel ones. We deliberately KEEP
# the functional verbs (DEPOSIT/ETCH/IMPLANT/CLEAN/ANNEAL/CMP/EXPOSE/DEVELOP/
# STRIP/CURE/...) because a real 4th family keeps those too — it changes the
# materials and device regions. This is the realistic OOD model.
# Device-MATERIAL / region words a new family legitimately changes. We do NOT
# include shared functional/structural vocabulary (PASSIVATION, the litho keywords
# ALIGN/MASK/LEVEL/EXPOSE/LITHO/DEVELOP, CLEAN, TEST, SORT, ...) — the README says
# those are "mostly shared", and renaming them inconsistently would be an artifact,
# not a real OOD family. Swapping these material words keeps the functional verb
# intact (so the physical CATEGORY is still inferable) while making the token novel.
_MATERIALS = {
    "POLYSILICON", "OXIDE", "NITRIDE", "TUNGSTEN", "ALUMINUM", "COPPER",
    "SILICON", "EPITAXIAL", "SPACER", "FIELD", "GATE", "WELL", "BODY",
    "BUFFER", "SOURCE", "DRAIN", "LDD", "CHANNEL", "DIELECTRIC", "BARRIER",
    "INTERLAYER", "INTERLEVEL", "TITANIUM", "JUNCTION", "EMITTER",
}
_NOVEL_MATS = ["XENULON", "ZORBITE", "KORRIUM", "VANTREX", "QUBALT",
               "NYRELIUM", "THARNIUM", "OVELLIUM", "DRACALITE", "FENNIUM"]


def _novelize(step: str, matmap: dict, rng: random.Random):
    """If `step` has a device-material word, return a category-preserving NOVEL
    variant (verb kept, material swapped consistently) and True. Otherwise return
    (step, False) — shared functional/structural steps are NOT renamed, modelling
    the README's 'step names are mostly shared'."""
    words = step.split()
    if not any(wd in _MATERIALS for wd in words):
        return step, False
    out = [matmap.setdefault(wd, rng.choice(_NOVEL_MATS)) if wd in _MATERIALS else wd
           for wd in words]
    novel = " ".join(out)
    if novel in KNOWN_VOCAB:               # collided with a real step -> force novelty
        novel = novel + " " + matmap["__tag__"]
    return novel, True


def make_ood_family(steps: list[str], f: float, rng: random.Random) -> list[str]:
    """Rename a fraction `f` of the distinct MATERIAL-BEARING step types to novel,
    category-preserving names (consistent within the sequence). Shared structural
    steps (litho/clean/test/passivation) are left intact — the realistic Task-4
    'mostly shared vocabulary' model."""
    matmap = {"__tag__": rng.choice(["ZX", "Qn", "Vx", "Kp", "Ho"])}
    # distinct step types that actually CAN be novelised (have a material word)
    renamable = sorted({s for s in set(steps)
                        if any(w in _MATERIALS for w in s.split())})
    n_rename = int(round(f * len(renamable)))
    chosen = set(rng.sample(renamable, min(n_rename, len(renamable))))
    rename = {}
    for s in chosen:
        nv, ok = _novelize(s, matmap, rng)
        if ok:
            rename[s] = nv
    return [rename.get(s, s) for s in steps]


def evaluate_at_f(valid_seqs, bad_recs, f, rng):
    """Return metrics dict for one fraction f."""
    tp = fp = tn = fn = 0
    rule_ok = rule_tot = 0
    misses = []
    # valid (label = valid)
    for s in valid_seqs:
        ood = make_ood_family(s, f, rng)
        v = ENGINE(ood)
        if v:  # engine flagged a valid flow -> false positive
            fp += 1
            misses.append(("FP/valid-flagged", f, [x.rule for x in v][:2]))
        else:
            tn += 1
    # invalid (label = invalid, known first_rule)
    for rec in bad_recs:
        ood = make_ood_family(rec["steps"], f, rng)
        v = ENGINE(ood)
        if v:
            tp += 1
            rule_tot += 1
            if any(x.rule == rec["first_rule"] for x in v):
                rule_ok += 1
        else:  # missed a real violation
            fn += 1
            misses.append(("FN/violation-missed", f, rec["first_rule"]))
    n = tp + fp + tn + fn
    prec = tp / max(tp + fp, 1)
    rec_ = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec_ / max(prec + rec_, 1e-9)
    return dict(f=f, acc=(tp + tn) / max(n, 1), prec=prec, rec=rec_, f1=f1,
                tp=tp, fp=fp, tn=tn, fn=fn,
                rule_attr=(rule_ok / rule_tot if rule_tot else 0.0), misses=misses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--show-misses", type=int, default=6)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # reference-labelled corpus
    valid_all = []
    for p in sorted((_ROOT / "training_data").glob("*_variants.csv")):
        valid_all += list(read_csv_sequences(p).values())
    rng.shuffle(valid_all)
    valid_seqs = valid_all[: args.n_per_class]
    bad_recs, _ = BDG.build(per_combo=max(2, args.n_per_class // 30), seed=args.seed)
    rng.shuffle(bad_recs)
    bad_recs = bad_recs[: args.n_per_class]

    print(f"OOD (Task-4) benchmark — ground truth INHERITED from the reference checker")
    print(f"  {len(valid_seqs)} valid + {len(bad_recs)} invalid reference-labelled sequences")
    print(f"  modelling a 4th family by renaming fraction f of step types to novel,")
    print(f"  category-preserving names (verbs kept, materials swapped).\n")
    print(f"  {'f':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'RuleAttr':>9}   confusion(TP/FP/TN/FN)")
    realistic = None
    for f in (0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
        m = evaluate_at_f(valid_seqs, bad_recs, f, random.Random(args.seed + int(f * 100)))
        print(f"  {f:>5.2f} {m['acc']:>7.3f} {m['prec']:>7.3f} {m['rec']:>7.3f} {m['f1']:>7.3f} "
              f"{m['rule_attr']:>9.3f}   {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")
        if abs(f - 0.30) < 1e-9:
            realistic = m
    print()
    if realistic:
        verdict = "PASS (>=80%)" if realistic["acc"] >= 0.80 else "BELOW 80% — needs classifier work"
        print(f"Realistic 'mostly-shared-vocabulary' regime (f=0.30): "
              f"Acc={realistic['acc']:.3f}  F1={realistic['f1']:.3f}  -> {verdict}")
        ex = realistic["misses"][: args.show_misses]
        if ex:
            print("  example misses at f=0.30:")
            for kind, ff, info in ex:
                print(f"    {kind}: {info}")

    # ── ONE-NOVEL-TOKEN robustness: a single unknown step inserted into an
    #    otherwise in-vocab sequence must NOT flip the verdict (no weak logic). ──
    print("\nOne-novel-token robustness (single unknown step inserted):")
    rng2 = random.Random(args.seed + 7)
    # genuinely INERT novel steps (inspection/metrology) — these are neither a
    # rule trigger nor an enabler, so inserting one must never change the verdict.
    # (A novel DEPOSITION legitimately WOULD need a clean, so it is not "benign".)
    novel_benign = ["INSPECT NOVEL FEATURE", "MEASURE XENULON THICKNESS",
                    "INSPECT XENULON SURFACE", "MEASURE NOVEL OVERLAY"]
    vfp = 0
    for s in valid_seqs:
        s2 = list(s); s2.insert(rng2.randrange(1, len(s2)), rng2.choice(novel_benign))
        if ENGINE(s2):
            vfp += 1
    bmiss = 0
    for rec in bad_recs:
        s2 = list(rec["steps"]); s2.insert(rng2.randrange(1, len(s2)), rng2.choice(novel_benign))
        if not ENGINE(s2):
            bmiss += 1
    print(f"  valid+1 novel token  -> still valid: {len(valid_seqs)-vfp}/{len(valid_seqs)} "
          f"(false positives: {vfp})")
    print(f"  invalid+1 novel token -> still caught: {len(bad_recs)-bmiss}/{len(bad_recs)} "
          f"(violations lost: {bmiss})")
    print("  (a single novel token must not create or hide a violation; "
          "the routing/category engine handles known + novel steps together.)")


if __name__ == "__main__":
    main()
