"""End-to-end data prep: scale, split, build tokenizer + eval mirrors."""
import argparse
import random
from pathlib import Path
from procseq import data, make_eval
from procseq.tokenizer import build_tokenizer, DEFAULT_DIR
from procseq.grammar import FAMILIES, write_csv

ART = Path(__file__).resolve().parents[1] / "artifacts"

def run(n_per_family: int, seed: int, smoke: bool):
    ART.mkdir(exist_ok=True)
    rng = random.Random(seed)
    build_tokenizer(DEFAULT_DIR)
    eval_valid_rows, anomaly_rows = [], []
    for fam in FAMILIES:
        seqs = {f"{fam}_{i:05d}": s for i, s in
                enumerate(data.scale_family(fam, n_per_family, seed))}
        tr, va, te = data.split_ids(list(seqs), 0.1, 0.1, seed)
        (ART / "splits").mkdir(exist_ok=True)
        write_csv(ART / "splits" / f"{fam}_train.csv", [seqs[i] for i in tr])
        write_csv(ART / "splits" / f"{fam}_val.csv", [seqs[i] for i in va])
        write_csv(ART / "splits" / f"{fam}_test.csv", [seqs[i] for i in te])
        held = {i: seqs[i] for i in te}
        k = 5 if smoke else 100
        sample_ids = list(held)[:k]
        eval_valid_rows += make_eval.build_valid_rows(
            {i: held[i] for i in sample_ids}, fam, (0.6, 0.8), rng)
        nv, ni = (5, 3) if smoke else (200, 129)
        anomaly_rows += make_eval.build_anomaly_rows(held, fam, nv, ni, rng)
    make_eval.write_valid_files(eval_valid_rows,
        ART / "eval_input_valid.csv", ART / "eval_valid_groundtruth.csv")
    make_eval.write_anomaly_files(anomaly_rows,
        ART / "eval_input_anomaly.csv", ART / "eval_anomaly_labels.csv")
    print(f"Built data: {len(eval_valid_rows)} valid-eval rows, "
          f"{len(anomaly_rows)} anomaly rows -> {ART}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.n_per_family if not a.smoke else 20, a.seed, a.smoke)
