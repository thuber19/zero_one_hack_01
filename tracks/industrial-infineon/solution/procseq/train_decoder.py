"""Train the decoder with Accelerate, with live stdout progress logging."""
import argparse
import time
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from accelerate import Accelerator
from accelerate.utils import set_seed
from procseq.config import load_config
from procseq.tokenizer import build_tokenizer, load_tokenizer, DEFAULT_DIR, encode_sequence
from procseq.models.decoder import build_decoder
from procseq.datasets import ClmDataset, clm_collate
from procseq.data import scale_family, ucbs_weights
from procseq.vocab import FAMILY_TOKEN, SPECIAL_TOKENS, token_to_step
from procseq.grammar import FAMILIES


def _load_training_pairs(n_per_family, seed):
    pairs = []
    for fam in FAMILIES:
        for s in scale_family(fam, n_per_family, seed):
            pairs.append((s, fam))
    return pairs


@torch.no_grad()
def _val_next_step_acc(model, tok, val_pairs, max_len, n=64):
    """Quick next-step top-1 accuracy on a small val sample (progress signal)."""
    model.eval()
    dev = next(model.parameters()).device
    special_ids = {tok.convert_tokens_to_ids(s) for s in SPECIAL_TOKENS}
    correct = total = 0
    for steps, fam in val_pairs[:n]:
        cut = max(1, int(len(steps) * 0.8))
        if cut >= len(steps):
            continue
        gold = steps[cut]
        ids = tok.encode(" ".join([tok.bos_token, FAMILY_TOKEN[fam]])) + \
            encode_sequence(tok, steps[:cut], family=None, add_bos_eos=False)
        x = torch.tensor([ids[:max_len]], device=dev)
        logits = model(input_ids=x).logits[0, -1]
        pred = None
        for tid in torch.argsort(logits, descending=True).tolist():
            if tid in special_ids:
                continue
            pred = token_to_step(tok.convert_ids_to_tokens(tid))
            break
        correct += int(pred == gold)
        total += 1
    model.train()
    return correct / max(1, total)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config); dc = cfg["decoder"]
    set_seed(cfg.get("seed", 42))
    acc = Accelerator(mixed_precision=("bf16" if cfg.get("precision") == "bf16" else "no"),
                      log_with="tensorboard", project_dir=cfg.get("artifacts", "artifacts"))
    acc.init_trackers(cfg.get("run_name", "decoder"))
    max_len = dc.get("max_len", 256)
    tok = load_tokenizer(DEFAULT_DIR)   # load pre-built tokenizer (no write race under DDP/parallel)
    model = build_decoder(dc["size"], tok, max_len)
    pairs = _load_training_pairs(dc.get("data_per_family", 20), cfg.get("seed", 42))
    val_pairs = pairs[::max(1, len(pairs) // 64)][:64]   # ~64, spread across families
    ds = ClmDataset(pairs, tok, max_len)
    weights = ucbs_weights([p[0] for p in pairs])
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    dl = DataLoader(ds, batch_size=dc.get("batch_size", 4), sampler=sampler,
                    collate_fn=partial(clm_collate, pad_id=tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=dc.get("lr", 3e-3))
    max_steps = dc.get("max_steps", 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-6)

    log_every = dc.get("log_every", 25)
    eval_every = dc.get("eval_every", 250)
    n_params = sum(p.numel() for p in model.parameters())
    if acc.is_main_process:
        print("=" * 64, flush=True)
        print(f"  DECODER '{dc['size']}' | params={n_params/1e6:.2f}M | vocab={len(tok)}", flush=True)
        print(f"  device={acc.device} | precision={cfg.get('precision','no')} | "
              f"max_steps={max_steps} | bs={dc.get('batch_size',4)} | lr={dc.get('lr',3e-3)}", flush=True)
        print(f"  torch={torch.__version__} | cuda_available={torch.cuda.is_available()}"
              + ("  *** WARNING: training on CPU — GPU not visible to torch! ***"
                 if str(acc.device) == "cpu" else ""), flush=True)
        print(f"  train pairs={len(pairs)} (~{len(pairs)//len(FAMILIES)}/family) | "
              f"val sample={len(val_pairs)}", flush=True)
        print("=" * 64, flush=True)

    model, opt, dl = acc.prepare(model, opt, dl)
    model.train()
    step = 0
    run_loss = 0.0
    t0 = time.time(); t_log = t0
    while step < max_steps:
        for batch in dl:
            out = model(**batch); loss = out.loss
            acc.backward(loss); opt.step(); opt.zero_grad(); scheduler.step()
            run_loss += loss.item()
            acc.log({"train/loss": loss.item()}, step=step)
            step += 1

            if acc.is_main_process and step % log_every == 0:
                avg = run_loss / log_every; run_loss = 0.0
                now = time.time()
                sps = log_every / max(1e-6, now - t_log); t_log = now
                eta = (max_steps - step) / max(1e-6, sps)
                lr = opt.param_groups[0]["lr"]
                print(f"  step {step:5d}/{max_steps} | loss={avg:.4f} | lr={lr:.2e} | "
                      f"{sps:.1f} it/s | elapsed={now-t0:.0f}s | eta={eta:.0f}s", flush=True)

            if step % eval_every == 0 or step >= max_steps:
                acc_top1 = _val_next_step_acc(acc.unwrap_model(model), tok, val_pairs, max_len)
                acc.log({"val/next_step_top1": acc_top1}, step=step)
                if acc.is_main_process:
                    print(f"  --- val next-step top-1 = {acc_top1:.4f}  (step {step}) ---", flush=True)
            if step >= max_steps:
                break

    out_dir = Path(cfg.get("decoder_ckpt", "artifacts/decoder"))
    acc.wait_for_everyone()
    if acc.is_main_process:   # only rank 0 writes the checkpoint (DDP-safe)
        acc.unwrap_model(model).save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
    acc.end_training()
    if acc.is_main_process:
        print(f"Saved decoder -> {out_dir}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
