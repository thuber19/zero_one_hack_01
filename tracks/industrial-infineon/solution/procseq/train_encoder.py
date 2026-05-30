"""Train the anomaly encoder (binary BCE + rule BCE + optional contrastive),
with live stdout progress logging."""
import argparse
import time
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
from procseq.vocab import FAMILY_TOKEN, step_to_token
from procseq.grammar import RULE_IDS
from procseq.contrastive import supcon_loss


@torch.no_grad()
def _val_anomaly_acc(model, tok, val_items, max_len, n=128):
    """Quick binary anomaly accuracy on a held-out sample (progress signal)."""
    model.eval()
    dev = next(model.parameters()).device
    correct = total = 0
    for steps, fam, is_valid, rule in val_items[:n]:
        text = " ".join([tok.cls_token, FAMILY_TOKEN[fam]]
                        + [step_to_token(s) for s in steps] + [tok.sep_token])
        ids = torch.tensor([tok.encode(text)[:max_len]], device=dev)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
        pred_invalid = 1 if torch.sigmoid(out["invalid_logit"])[0].item() >= 0.5 else 0
        gold_invalid = 0 if is_valid else 1
        correct += int(pred_invalid == gold_invalid)
        total += 1
    model.train()
    return correct / max(1, total)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(argv); cfg = load_config(a.config); ec = cfg["encoder"]
    set_seed(cfg.get("seed", 42))
    acc = Accelerator(mixed_precision=("bf16" if cfg.get("precision") == "bf16" else "no"),
                      log_with="tensorboard", project_dir=cfg.get("artifacts", "artifacts"))
    acc.init_trackers(cfg.get("run_name", "encoder") + "_enc")
    max_len = ec.get("max_len", 256)
    tok = build_tokenizer(DEFAULT_DIR)
    model = build_encoder(ec["size"], tok, n_rules=len(RULE_IDS), max_position_embeddings=max_len)
    items = build_anomaly_training(ec.get("data_per_family", 20), cfg.get("seed", 42))
    n_val = min(128, len(items) // 5) if len(items) > 16 else 0
    val_items = items[-n_val:] if n_val else items[:8]
    train_items = items[:-n_val] if n_val else items
    ds = ClsDataset(train_items, tok, RULE_IDS, max_len)
    dl = DataLoader(ds, batch_size=ec.get("batch_size", 4), shuffle=True,
                    collate_fn=partial(cls_collate, pad_id=tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=ec.get("lr", 3e-3))
    bce = nn.BCEWithLogitsLoss()
    con = ec.get("contrastive", {}) or {}
    con_on = bool(con.get("enabled", False))
    con_w = float(con.get("weight", 0.5))
    con_temp = float(con.get("temperature", 0.1))

    max_steps = ec.get("max_steps", 5)
    log_every = ec.get("log_every", 25)
    eval_every = ec.get("eval_every", 250)
    n_params = sum(p.numel() for p in model.parameters())
    if acc.is_main_process:
        print("=" * 64, flush=True)
        print(f"  ENCODER '{ec['size']}' | params={n_params/1e6:.2f}M | vocab={len(tok)}", flush=True)
        print(f"  device={acc.device} | precision={cfg.get('precision','no')} | "
              f"max_steps={max_steps} | bs={ec.get('batch_size',4)} | lr={ec.get('lr',3e-3)} | "
              f"contrastive={'ON' if con_on else 'OFF'}", flush=True)
        print(f"  train items={len(train_items)} | val sample={min(n_val or 8, 128)}", flush=True)
        print("=" * 64, flush=True)

    model, opt, dl = acc.prepare(model, opt, dl)
    model.train()
    step = 0
    rl_bin = rl_rule = rl_con = 0.0
    t0 = time.time(); t_log = t0
    while step < max_steps:
        for b in dl:
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            l_bin = bce(out["invalid_logit"], b["invalid"])
            l_rule = bce(out["rule_logits"], b["rules"])
            loss = l_bin + l_rule
            logs = {"train/binary": l_bin.item(), "train/rule": l_rule.item()}
            rl_bin += l_bin.item(); rl_rule += l_rule.item()
            if con_on:
                closs = supcon_loss(out["embed"], b["invalid"], temperature=con_temp)
                loss = loss + con_w * closs
                logs["train/contrastive"] = float(closs.detach())
                rl_con += float(closs.detach())
            acc.backward(loss); opt.step(); opt.zero_grad()
            acc.log(logs, step=step); step += 1

            if acc.is_main_process and step % log_every == 0:
                now = time.time(); sps = log_every / max(1e-6, now - t_log); t_log = now
                eta = (max_steps - step) / max(1e-6, sps)
                msg = (f"  step {step:5d}/{max_steps} | bin={rl_bin/log_every:.4f} | "
                       f"rule={rl_rule/log_every:.4f}")
                if con_on:
                    msg += f" | con={rl_con/log_every:.4f}"
                msg += f" | {sps:.1f} it/s | elapsed={now-t0:.0f}s | eta={eta:.0f}s"
                print(msg, flush=True)
                rl_bin = rl_rule = rl_con = 0.0

            if step % eval_every == 0 or step >= max_steps:
                va = _val_anomaly_acc(acc.unwrap_model(model), tok, val_items, max_len)
                acc.log({"val/anomaly_acc": va}, step=step)
                if acc.is_main_process:
                    print(f"  --- val anomaly binary-acc = {va:.4f}  (step {step}) ---", flush=True)
            if step >= max_steps:
                break

    out_dir = Path(cfg.get("encoder_ckpt", "artifacts/encoder")); out_dir.mkdir(parents=True, exist_ok=True)
    acc.wait_for_everyone()
    acc.save(acc.unwrap_model(model).state_dict(), out_dir / "pytorch_model.bin")
    tok.save_pretrained(out_dir)
    acc.end_training()
    if acc.is_main_process:
        print(f"Saved encoder -> {out_dir}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
