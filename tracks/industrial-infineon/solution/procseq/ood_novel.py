"""True novel-4th-family OOD stress for procseq's models — reuses the team's
`pseudo_family` generator (Mina's) instead of inventing our own.

This directly operationalizes the audit's keyword-free attack (the "OP_5" rename):
`pseudo_family.pseudo_sequence(..., rename_fraction)` renames device steps to NOVEL
vocabulary; at rename_fraction=1.0 the names carry no recognizable keyword. We then
measure how procseq's decoder (next-step) and encoder (anomaly) degrade vs.
in-distribution, across the novelty spectrum (0.0 -> 0.5 -> 1.0).

Honest by construction: renamed steps are `[UNK]` to our tokenizer and `UNKNOWN`
to `classify_step`, so BOTH lexical and category metrics fall — we *report* the
cliff, we don't hide it. The `n_invalid_constructible` field also exposes the
auditor's point: when names are keyword-free, the physics rules can't even build
a labelled violation.

Run (after a Leonardo training run):
    python -m procseq.ood_novel --decoder-ckpt <dir>/decoder --encoder-ckpt <dir>/encoder
"""
import argparse
import json
import random
from pathlib import Path

from procseq import grammar          # noqa: F401  (sets up the data/ import path)
from procseq import external         # noqa: F401  (adds the track root for physics/pseudo_family)
from procseq.tokenizer import load_tokenizer
from procseq import infer, eval_metrics as em
from procseq.grammar import RULE_IDS

try:
    import pseudo_family as PF
    _PF_OK = True
    _PF_ERR = ""
except Exception as e:  # pragma: no cover
    _PF_OK = False
    _PF_ERR = repr(e)

_FAMS = ["mosfet", "igbt", "ic"]


def _novel_valid(n, rename_fraction, seed):
    """n novel-vocabulary valid sequences at a given rename_fraction.
    Returns (base_family_token, steps) — we condition on the base family the
    structure came from, since procseq has no token for a brand-new family."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        fam = rng.choice(_FAMS)
        out.append((fam.upper(), PF.pseudo_sequence(fam, "OOD", rng,
                                                     rename_fraction=rename_fraction)))
    return out


def _eval_decoder(dec, tok, novel):
    preds, gold = {}, {}
    for i, (fam, s) in enumerate(novel):
        cut = max(1, int(len(s) * 0.8))
        if cut >= len(s):
            continue
        preds[f"x{i}"] = infer.predict_next_step(dec, tok, s[:cut], fam, k=5)
        gold[f"x{i}"] = s[cut]
    return em.score_nextstep(preds, gold)


def _eval_encoder(enc, tok, novel, seed):
    from procseq.infer_anomaly import classify_sequence
    rng = random.Random(seed + 7)
    pred, gold = {}, {}
    n_inv = 0
    for i, (fam, s) in enumerate(novel):
        iv, sc, rule = classify_sequence(enc, tok, s, fam, RULE_IDS)
        pred[f"v{i}"] = (iv, sc, rule); gold[f"v{i}"] = (1, "")
        res = PF.inject_violation(s, rng)        # category rules; may fail if keyword-free
        if res:
            broken, gr = res
            iv2, sc2, rule2 = classify_sequence(enc, tok, broken, fam, RULE_IDS)
            pred[f"b{i}"] = (iv2, sc2, rule2); gold[f"b{i}"] = (0, gr); n_inv += 1
    r = em.score_anomaly(pred, gold)
    r["n_invalid_constructible"] = n_inv
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder-ckpt", default=None)
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--fractions", default="0.0,0.5,1.0",
                    help="novelty spectrum: 0.0=in-dist vocab .. 1.0=keyword-free")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/ood_novel.json")
    a = ap.parse_args(argv)
    if not _PF_OK:
        raise SystemExit(f"pseudo_family unavailable ({_PF_ERR}); need the team's "
                         f"physics/ + pseudo_family.py on the track root.")
    if not (a.decoder_ckpt or a.encoder_ckpt):
        raise SystemExit("pass --decoder-ckpt and/or --encoder-ckpt")

    dec = dec_tok = enc = enc_tok = None
    if a.decoder_ckpt:
        from transformers import LlamaForCausalLM
        dec_tok = load_tokenizer(Path(a.decoder_ckpt))
        dec = LlamaForCausalLM.from_pretrained(a.decoder_ckpt)
    if a.encoder_ckpt:
        from procseq.infer_anomaly import _load_encoder
        enc_tok = load_tokenizer(Path(a.encoder_ckpt))
        enc = _load_encoder(a.encoder_ckpt, enc_tok)

    results = {}
    for frac in [float(x) for x in a.fractions.split(",")]:
        novel = _novel_valid(a.n, frac, a.seed)
        rec = {"rename_fraction": frac, "n": len(novel)}
        if dec is not None:
            d = _eval_decoder(dec, dec_tok, novel)
            rec["nextstep_top1"] = d["top1"]
            rec["nextstep_top1_category"] = d["top1_category"]
        if enc is not None:
            e = _eval_encoder(enc, enc_tok, novel, a.seed)
            rec["anomaly_binary_acc"] = e["binary_accuracy"]
            rec["anomaly_f1"] = e["f1"]
            rec["n_invalid_constructible"] = e["n_invalid_constructible"]
        results[f"rename_{frac}"] = rec
        print(json.dumps(rec), flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
