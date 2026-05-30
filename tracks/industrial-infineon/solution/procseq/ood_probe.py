"""Leave-one-family-out: train on 2 families, score Task1 on the held-out 3rd.
Reports ID vs OOD Top-1 and the drop Delta, self-simulating Task 4."""
import argparse, json
from pathlib import Path
import torch
from transformers import LlamaForCausalLM
from procseq.config import Config
from procseq.tokenizer import build_tokenizer, DEFAULT_DIR
from procseq.models.decoder import build_decoder
from procseq.datasets import ClmDataset, clm_collate
from procseq.data import scale_family
from procseq import train_decoder, infer, eval_metrics as em
from procseq.grammar import FAMILIES

def _score_family_top1(model, tok, family, n, seed):
    seqs = scale_family(family, n, seed)
    preds, gold = {}, {}
    for i, s in enumerate(seqs[:50]):
        cut = max(1, int(len(s) * 0.8))
        if cut >= len(s):
            continue
        eid = f"{family}_{i}"
        preds[eid] = infer.predict_next_step(model, tok, s[:cut], family, k=5)
        gold[eid] = s[cut]
    return em.score_nextstep(preds, gold)["top1"]

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True, choices=FAMILIES)
    ap.add_argument("--data-per-family", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/ood")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"decoder_no_{a.holdout}"
    cfg = Config({"run_name": f"ood_no_{a.holdout}", "seed": a.seed,
                  "precision": "bf16", "artifacts": str(out),
                  "decoder_ckpt": str(ckpt),
                  "decoder": {"size": "base", "max_len": 256,
                              "data_per_family": a.data_per_family, "batch_size": 64,
                              "lr": 6e-4, "max_steps": a.max_steps, "eval_every": 500},
                  "_ood_holdout": a.holdout})
    import yaml; cfgp = out / f"cfg_no_{a.holdout}.yaml"; cfgp.write_text(yaml.safe_dump(dict(cfg)))
    # train_decoder uses all families by default; monkeypatch loader to exclude holdout:
    orig = train_decoder._load_training_pairs
    def filtered(n, seed):
        return [(s, f) for (s, f) in orig(n, seed) if f != a.holdout]
    train_decoder._load_training_pairs = filtered
    try:
        train_decoder.main(["--config", str(cfgp)])
    finally:
        train_decoder._load_training_pairs = orig
    tok = build_tokenizer(DEFAULT_DIR)
    model = LlamaForCausalLM.from_pretrained(str(ckpt))
    id_fams = [f for f in FAMILIES if f != a.holdout]
    id_top1 = sum(_score_family_top1(model, tok, f, a.data_per_family, a.seed)
                  for f in id_fams) / len(id_fams)
    ood_top1 = _score_family_top1(model, tok, a.holdout, a.data_per_family, a.seed)
    rec = {"holdout": a.holdout, "id_top1": id_top1, "ood_top1": ood_top1,
           "delta": id_top1 - ood_top1}
    (out / f"ood_{a.holdout}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))

if __name__ == "__main__":
    main()
