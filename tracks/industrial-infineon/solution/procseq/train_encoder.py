"""Train the anomaly encoder (binary BCE + multi-label rule BCE)."""
import argparse
from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from procseq.config import load_config
from procseq.tokenizer import build_tokenizer, DEFAULT_DIR
from procseq.models.encoder import build_encoder
from procseq.datasets import ClsDataset, cls_collate
from procseq.anomaly_data import build_anomaly_training
from procseq.grammar import RULE_IDS
from procseq.contrastive import supcon_loss

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(argv); cfg = load_config(a.config); ec = cfg["encoder"]
    set_seed(cfg.get("seed", 42))
    acc = Accelerator(mixed_precision=("bf16" if cfg.get("precision")=="bf16" else "no"),
                      log_with="tensorboard", project_dir=cfg.get("artifacts","artifacts"))
    acc.init_trackers(cfg.get("run_name","encoder") + "_enc")
    tok = build_tokenizer(DEFAULT_DIR)
    model = build_encoder(ec["size"], tok, n_rules=len(RULE_IDS), max_position_embeddings=ec.get("max_len",256))
    items = build_anomaly_training(ec.get("data_per_family", 20), cfg.get("seed", 42))
    ds = ClsDataset(items, tok, RULE_IDS, ec.get("max_len", 256))
    dl = DataLoader(ds, batch_size=ec.get("batch_size",4), shuffle=True,
                    collate_fn=partial(cls_collate, pad_id=tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=ec.get("lr", 3e-3))
    bce = nn.BCEWithLogitsLoss()
    # Supervised-contrastive term (optional): pulls valid embeddings together and
    # pushes rule-violating twins away. Hard negatives come for free from anomaly_inject.
    con = ec.get("contrastive", {}) or {}
    con_on = bool(con.get("enabled", False))
    con_w = float(con.get("weight", 0.5))
    con_temp = float(con.get("temperature", 0.1))
    model, opt, dl = acc.prepare(model, opt, dl)
    model.train(); step = 0; max_steps = ec.get("max_steps", 5)
    while step < max_steps:
        for b in dl:
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            loss = bce(out["invalid_logit"], b["invalid"]) + \
                   bce(out["rule_logits"], b["rules"])
            logs = {"train/enc_loss": loss.item()}
            if con_on:
                closs = supcon_loss(out["embed"], b["invalid"], temperature=con_temp)
                loss = loss + con_w * closs
                logs["train/contrastive"] = float(closs.detach())
            acc.backward(loss); opt.step(); opt.zero_grad()
            acc.log(logs, step=step); step += 1
            if step >= max_steps: break
    out_dir = Path(cfg.get("encoder_ckpt","artifacts/encoder")); out_dir.mkdir(parents=True, exist_ok=True)
    acc.wait_for_everyone()
    acc.save(acc.unwrap_model(model).state_dict(), out_dir / "pytorch_model.bin")
    tok.save_pretrained(out_dir)
    acc.end_training()
    if acc.is_main_process: print(f"Saved encoder -> {out_dir}")

if __name__ == "__main__":
    main()
