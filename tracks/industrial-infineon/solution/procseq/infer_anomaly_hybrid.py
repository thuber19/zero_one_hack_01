"""Hybrid Task-3 anomaly: the official rule engine decides VALID/INVALID + which
rule (deterministic, ~perfect in-distribution), and the learned encoder supplies a
CONTINUOUS score so ROC-AUC isn't degenerate. "Model gives a confidence, physics
gives the verdict."

This is the high-accuracy Task-3 submission. procseq's pure-encoder
`infer_anomaly.py` stays as the honest "what the model learned alone" ablation.
The from-scratch DeBERTa is hard to train (it sits near 0.5 binary acc until it
converges); the rule engine is the right tool for the *scored* verdict.

    python -m procseq.infer_anomaly_hybrid --config <cfg>           # our mirror
    python -m procseq.infer_anomaly_hybrid --config <cfg> --real    # data/eval_input_anomaly.csv
"""
import argparse
import csv
from pathlib import Path

from procseq import grammar          # noqa: F401  data/ path
from procseq import external         # noqa: F401  track-root path
from procseq.config import load_config
from procseq.grammar import validate_sequence


def _maybe_encoder(cfg):
    """Load the trained encoder for a continuous SCORE, if the checkpoint exists."""
    try:
        from procseq.tokenizer import load_tokenizer
        from procseq.infer_anomaly import _load_encoder, classify_sequence
        from procseq.grammar import RULE_IDS
        ck = Path(cfg["encoder_ckpt"])
        if not (ck / "pytorch_model.bin").exists():
            return None
        tok = load_tokenizer(ck)
        model = _load_encoder(str(ck), tok)
        return (model, tok, RULE_IDS, classify_sequence)
    except Exception as e:  # pragma: no cover
        print(f"  [task3-hybrid] encoder unavailable for SCORE ({e!r}); using constant score", flush=True)
        return None


def run(cfg, real=False):
    art = Path(cfg["artifacts"])
    if real:
        src = Path(grammar.TRAINING_DATA_DIR) / "eval_input_anomaly.csv"
        out = art / "submission_task3_hybrid_real.csv"
    else:
        src = art / "eval_input_anomaly.csv"
        out = art / "submission_task3_hybrid.csv"
    enc = _maybe_encoder(cfg)
    with src.open() as f:
        in_rows = list(csv.DictReader(f))
    print(f"  [task3-hybrid{'/real' if real else ''}] {len(in_rows)} sequences "
          f"(rule-engine verdict + {'model' if enc else 'constant'} score)...", flush=True)
    rows = []
    for i, r in enumerate(in_rows, 1):
        steps = r["SEQUENCE"].split("|")
        v = validate_sequence(steps)
        is_valid = 1 if not v else 0
        rule = v[0].rule if v else ""
        if enc is not None:
            model, tok, RULE_IDS, classify = enc
            _iv, score, _r = classify(model, tok, steps, r["FAMILY"], RULE_IDS)  # P(valid)
        else:
            score = 0.95 if is_valid else 0.05
        rows.append([r["EXAMPLE_ID"], is_valid, f"{score:.4f}", rule])
        if i % 200 == 0 or i == len(in_rows):
            print(f"    task3 {i}/{len(in_rows)}", flush=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        w.writerows(rows)
    print(f"-> {out} ({len(rows)} rows)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args(argv)
    run(load_config(a.config), real=a.real)


if __name__ == "__main__":
    main()
