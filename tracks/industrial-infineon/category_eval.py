#!/usr/bin/env python3
"""
category_eval.py — Does the MODEL itself learn the FUNCTION of a step, and does
that understanding TRANSFER to unseen families?

This is the honest test behind the claim "we taught the model to understand the
process, not just memorise step names." The model trained with --aux-category
has a head that predicts the *physical category* of the next step
(DEPOSITION / ETCH / LITHOGRAPHY / IMPLANT / ...). We measure, with NO physics
post-processing:

  * ID  : next-CATEGORY top-1 accuracy on held-out MOSFET / IGBT / IC.
  * OOD : the SAME metric on 5 real published families the model has NEVER seen
          (GaN HEMT, solar cell, BJT, SiC MOSFET, Schottky). The step *names*
          are mostly novel tokens -> the model cannot memorise them; predicting
          the right category here means it learned WHAT a step does from context.

We also report next-STEP top-1 (the ordinary LM metric) for reference, so the
gap between "knows the exact name" (low OOD) and "knows the function" (the
question) is visible and honest.

Usage:  python category_eval.py --model-dir outputs_M3 --model-size tiny
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    sys.path.insert(0, _p)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tokenizer import StepTokenizer, BOS_ID, FAMILY_TOKENS
from transformer_model import create_model
from train import build_id2cat
from physics.ontology import classify_step

# Real OOD families (faithful published flows) — reused from real_family_benchmark
import real_family_benchmark as rfb


def _cat_index_map(tokenizer):
    """Reconstruct the EXACT category ordering build_id2cat uses, so we can map a
    step NAME (incl. novel OOD names) -> the same category index the model was
    trained against."""
    reals = [t for t in tokenizer.id2token.values()
             if not (t.startswith("[") and t.endswith("]"))]
    cats = sorted({classify_step(t) for t in reals} | {"UNKNOWN"})
    return {c: i for i, c in enumerate(cats)}


@torch.no_grad()
def eval_family(model, tokenizer, cat2idx, family, sequences, min_ctx=4):
    """Return (cat_correct, cat_total, step_correct, step_total) over all
    teacher-forced next-step positions in the given sequences."""
    cat_c = cat_t = step_c = step_t = 0
    fam_tok = FAMILY_TOKENS.get(family.lower(), "[UNK]")
    fam_id = tokenizer.encode_step(fam_tok)
    for steps in sequences:
        step_ids = [tokenizer.encode_step(s) for s in steps]
        for t in range(min_ctx, len(steps) - 1):
            # prefix = [BOS, family, steps[0..t]] -> predict steps[t+1]
            prefix = [BOS_ID, fam_id] + step_ids[: t + 1]
            inp = torch.tensor([prefix], dtype=torch.long)
            attn = torch.ones_like(inp)
            logits, cat_logits = model(inp, attn, return_cat=True)
            # next-step (name) prediction
            pred_id = int(logits[0, -1].argmax())
            step_t += 1
            if pred_id == step_ids[t + 1]:
                step_c += 1
            # next-category (function) prediction
            true_cat = cat2idx.get(classify_step(steps[t + 1]))
            if true_cat is None:
                continue
            pred_cat = int(cat_logits[0, -1].argmax())
            cat_t += 1
            if pred_cat == true_cat:
                cat_c += 1
    return cat_c, cat_t, step_c, step_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="outputs_M3")
    ap.add_argument("--model-size", default="tiny")
    ap.add_argument("--id-per-family", type=int, default=40)
    args = ap.parse_args()

    mdir = Path(args.model_dir)
    tokenizer = StepTokenizer.load(mdir / "tokenizer.txt")
    id2cat, ncat = build_id2cat(tokenizer)
    cat2idx = _cat_index_map(tokenizer)
    print(f"Loaded tokenizer: vocab={tokenizer.vocab_size}, categories={ncat}")

    model = create_model(tokenizer.vocab_size, size=args.model_size, n_categories=ncat)
    sd = torch.load(mdir / "best_transformer.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if model.cat_head is None:
        print("ERROR: model has no category head — was it trained with --aux-category?")
        sys.exit(1)
    if any("cat_head" in k for k in missing):
        print("ERROR: checkpoint has no cat_head weights — not an aux-category model.")
        sys.exit(1)
    model.eval()
    print(f"Loaded model (missing={len(missing)}, unexpected={len(unexpected)})")

    # ── ID: held-out MOSFET / IGBT / IC ──
    id_seqs = json.load(open(mdir / "sequences.json"))
    print("\n=== IN-DISTRIBUTION (MOSFET / IGBT / IC) ===")
    id_cat_c = id_cat_t = id_step_c = id_step_t = 0
    for fam, seqs in id_seqs.items():
        sample = seqs[: args.id_per_family]
        cc, ct, sc, st = eval_family(model, tokenizer, cat2idx, fam, sample)
        id_cat_c += cc; id_cat_t += ct; id_step_c += sc; id_step_t += st
        print(f"  {fam.upper():7s}  category-acc {cc/ct:.3f}   step-acc {sc/st:.3f}  (n={ct})")
    print(f"  {'OVERALL':7s}  category-acc {id_cat_c/id_cat_t:.3f}   "
          f"step-acc {id_step_c/id_step_t:.3f}")

    # ── OOD: 5 real unseen families ──
    ood = {
        "GaN HEMT": rfb.GAN_HEMT, "Solar cell": rfb.SOLAR_CELL, "BJT": rfb.BJT,
        "SiC MOSFET": rfb.SIC_MOSFET, "Schottky": rfb.SCHOTTKY,
    }
    print("\n=== OUT-OF-DISTRIBUTION (5 real unseen families) ===")
    print("  (step names are mostly NOVEL tokens; only CATEGORY can be inferred)")
    od_cat_c = od_cat_t = od_step_c = od_step_t = 0
    for name, seq in ood.items():
        # family unknown -> encoded as [UNK]; that is the Task-4 condition
        cc, ct, sc, st = eval_family(model, tokenizer, cat2idx, "__ood__", [seq])
        od_cat_c += cc; od_cat_t += ct; od_step_c += sc; od_step_t += st
        print(f"  {name:11s}  category-acc {cc/ct:.3f}   step-acc {sc/st:.3f}  (n={ct})")
    print(f"  {'OVERALL':11s}  category-acc {od_cat_c/od_cat_t:.3f}   "
          f"step-acc {od_step_c/od_step_t:.3f}")

    # ── honest summary ──
    base = 1.0 / ncat
    print("\n=== HONEST READOUT ===")
    print(f"  Random-guess category baseline: {base:.3f} ({ncat} categories)")
    print(f"  ID  functional (category) acc : {id_cat_c/id_cat_t:.3f}")
    print(f"  OOD functional (category) acc : {od_cat_c/od_cat_t:.3f}")
    print(f"  OOD lexical (exact-name)  acc : {od_step_c/od_step_t:.3f}")
    print("  Interpretation: category >> name on OOD ==> the model learned the")
    print("  FUNCTION of steps (transferable), not just memorised names.")

    out = {
        "n_categories": ncat, "random_baseline": base,
        "id_category_acc": id_cat_c / id_cat_t, "id_step_acc": id_step_c / id_step_t,
        "ood_category_acc": od_cat_c / od_cat_t, "ood_step_acc": od_step_c / od_step_t,
    }
    (mdir / "category_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {mdir / 'category_eval.json'}")


if __name__ == "__main__":
    main()
