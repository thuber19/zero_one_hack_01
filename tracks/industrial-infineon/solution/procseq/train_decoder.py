"""Train the decoder with Accelerate (DeepSpeed via accelerate config on cluster)."""
import argparse
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from accelerate import Accelerator
from accelerate.utils import set_seed
from procseq.config import load_config
from procseq.tokenizer import build_tokenizer, DEFAULT_DIR
from procseq.models.decoder import build_decoder
from procseq.datasets import ClmDataset, clm_collate
from procseq.data import scale_family, ucbs_weights
from procseq.grammar import FAMILIES

def _load_training_pairs(n_per_family, seed):
    pairs = []
    for fam in FAMILIES:
        for s in scale_family(fam, n_per_family, seed):
            pairs.append((s, fam))
    return pairs

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config); dc = cfg["decoder"]
    set_seed(cfg.get("seed", 42))
    acc = Accelerator(mixed_precision=("bf16" if cfg.get("precision")=="bf16" else "no"),
                      log_with="tensorboard", project_dir=cfg.get("artifacts", "artifacts"))
    acc.init_trackers(cfg.get("run_name", "decoder"))
    tok = build_tokenizer(DEFAULT_DIR)
    model = build_decoder(dc["size"], tok, dc.get("max_len", 256))
    pairs = _load_training_pairs(dc.get("data_per_family", 20), cfg.get("seed", 42))
    ds = ClmDataset(pairs, tok, dc.get("max_len", 256))
    weights = ucbs_weights([p[0] for p in pairs])
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    dl = DataLoader(ds, batch_size=dc.get("batch_size", 4), sampler=sampler,
                    collate_fn=partial(clm_collate, pad_id=tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=dc.get("lr", 3e-3))
    model, opt, dl = acc.prepare(model, opt, dl)
    model.train(); step = 0; max_steps = dc.get("max_steps", 5)
    while step < max_steps:
        for batch in dl:
            out = model(**batch); loss = out.loss
            acc.backward(loss); opt.step(); opt.zero_grad()
            acc.log({"train/loss": loss.item()}, step=step)
            step += 1
            if step >= max_steps: break
    out_dir = Path(cfg.get("decoder_ckpt", "artifacts/decoder"))
    acc.wait_for_everyone()
    unwrapped = acc.unwrap_model(model)
    unwrapped.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    acc.end_training()
    if acc.is_main_process:
        print(f"Saved decoder -> {out_dir}")

if __name__ == "__main__":
    main()
