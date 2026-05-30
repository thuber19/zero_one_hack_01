#!/usr/bin/env python3
"""
train_grpo.py — make the MODEL internalize the process constraints (not just the
external rule engine) via GRPO (group-relative policy optimization) with the
physics verifier as the reward.

Why this exists: SFT (next-token + aux-category) teaches the model the *patterns*;
GRPO teaches it to *prefer legal continuations* by rewarding sequences the
deterministic verifier (reward.py) judges physically valid. No human labels, no
value network — the reward is the physics.

Loop per step:
  1. take a batch of valid prefixes (cut from generated sequences),
  2. sample K completions per prefix from the current policy (temperature),
  3. score each with the verifier reward (fraction of legal completion steps +
     a terminal bonus for reaching SHIP LOT),
  4. group-relative advantage A_i = r_i - mean(r over the K samples of that prefix),
  5. policy-gradient update: maximize sum_i A_i * logprob(sample_i).

Recommended use (recipe): SFT first (train.py --aux-category --unk-dropout), THEN
  OUTPUT_DIR=outputs_run python src/train_grpo.py --init-from outputs_run/best_transformer.pt \
      --data-dir outputs_run --steps 2000 --group-size 8 --device cuda

Smoke (CPU, tiny, proves the loop + reward flow):
  OUTPUT_DIR=outputs_M3 python src/train_grpo.py --init-from outputs_M3/best_transformer.pt \
      --data-dir outputs_M3 --steps 3 --group-size 4 --prefixes-per-step 4 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))            # repo root for reward.py
from tokenizer import StepTokenizer, FAMILY_TOKENS, BOS_ID, EOS_ID
from transformer_model import create_model
import reward as R
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

_TERMINAL = "SHIP LOT"


def _encode_prefix(tok: StepTokenizer, family: str, steps: list[str]) -> list[int]:
    fam = FAMILY_TOKENS.get(family.lower(), "[UNK]")
    return [BOS_ID, tok.encode_step(fam)] + [tok.encode_step(s) for s in steps]


@torch.no_grad()
def sample_completion(model, tok, prefix_ids, max_new, temp, device, real_ids):
    """Sample one completion (token ids) from the policy. real_ids restricts
    sampling to real step tokens + EOS (never PAD/UNK/family specials)."""
    ids = list(prefix_ids)
    gen = []
    for _ in range(max_new):
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(inp, torch.ones_like(inp))[0, -1] / max(temp, 1e-6)
        mask = torch.full_like(logits, float("-inf"))
        mask[real_ids] = 0.0
        probs = F.softmax(logits + mask, dim=-1)
        nxt = int(torch.multinomial(probs, 1))
        gen.append(nxt)
        ids.append(nxt)
        if nxt == EOS_ID:
            break
        if tok.id2token.get(nxt, "") == _TERMINAL:
            break
    return gen


def logprob_sum(model, prefix_ids, gen_ids, device):
    """Sum log p(token) over the generated tokens, WITH grad (policy gradient)."""
    full = list(prefix_ids) + list(gen_ids)
    inp = torch.tensor([full], dtype=torch.long, device=device)
    logits = model(inp, torch.ones_like(inp))[0]            # (T, V)
    lp = F.log_softmax(logits, dim=-1)
    start = len(prefix_ids) - 1                             # token i predicted by logits[i-1]
    total = 0.0
    for k, tid in enumerate(gen_ids):
        total = total + lp[start + k, tid]
    return total


def completion_reward(prefix_steps, gen_steps):
    """Verifier reward: fraction of the COMPLETION's steps that are legal given the
    prefix, plus a terminal bonus for legally reaching SHIP LOT. Pure physics."""
    full = list(prefix_steps) + list(gen_steps)
    flags = R.per_step_legality(full)
    comp_flags = flags[len(prefix_steps):] or [0]
    r = sum(comp_flags) / len(comp_flags)
    if gen_steps and gen_steps[-1].upper() == _TERMINAL and comp_flags[-1] == 1:
        r = min(1.0, r + 0.2)
    return r


def _load_pairs(data_dir: Path):
    """(family, steps) pairs from EITHER convention: train_sequences.csv (unified
    main pipeline) or sequences.json (our integrated generator)."""
    csv_path = data_dir / "train_sequences.csv"
    if csv_path.exists():
        from data_pipeline import load_train_csv
        return load_train_csv(csv_path)
    sj = data_dir / "sequences.json"
    fam_seqs = json.loads(sj.read_text())
    return [(fam, s) for fam, lst in fam_seqs.items() for s in lst]


def load_prefixes(data_dir: Path, n: int, rng):
    seqs = list(_load_pairs(data_dir))
    rng.shuffle(seqs)
    out = []
    for fam, s in seqs[: n * 3]:
        if len(s) < 8:
            continue
        cut = rng.randint(4, max(5, int(len(s) * 0.7)))
        out.append((fam, s[:cut]))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", type=str, default=None, help="SFT checkpoint to continue (recommended)")
    ap.add_argument("--data-dir", type=str, required=True, help="dir with sequences.json + tokenizer.txt")
    ap.add_argument("--model-size", default=None, help="overrides model_config.json")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--prefixes-per-step", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8, help="K completions per prefix")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(os.environ.get("OUTPUT_DIR", str(data_dir)))
    device = args.device

    tok = StepTokenizer.load(data_dir / "tokenizer.txt")
    size = args.model_size
    cfg = data_dir / "model_config.json"
    if size is None and cfg.exists():
        size = json.loads(cfg.read_text()).get("model_size", "small")
    size = size or "small"
    model = create_model(tok.vocab_size, size=size)
    if args.init_from and Path(args.init_from).exists():
        sd = torch.load(args.init_from, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"continued from {args.init_from} (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    else:
        print("WARNING: no --init-from; GRPO from random init is far weaker than "
              "continuing from an SFT checkpoint.", flush=True)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # restrict sampling to REAL step tokens (id>=7) + EOS — never PAD/BOS/UNK/family
    real_ids = [i for i in range(tok.vocab_size)
                if not (tok.id2token.get(i, "").startswith("[") and tok.id2token.get(i, "").endswith("]"))]
    real_ids.append(EOS_ID)

    print(f"GRPO: size={size} vocab={tok.vocab_size} steps={args.steps} "
          f"K={args.group_size} prefixes/step={args.prefixes_per_step} device={device}", flush=True)

    for step in range(1, args.steps + 1):
        prefixes = load_prefixes(data_dir, args.prefixes_per_step, rng)
        opt.zero_grad()
        batch_loss = 0.0
        batch_reward = 0.0
        n = 0
        for family, pre_steps in prefixes:
            pre_ids = _encode_prefix(tok, family, pre_steps)
            samples = []   # (gen_ids, gen_steps, reward)
            for _ in range(args.group_size):
                gen_ids = sample_completion(model, tok, pre_ids, args.max_new, args.temp, device, real_ids)
                gen_steps = [tok.id2token[g] for g in gen_ids
                             if not (tok.id2token.get(g, "[").startswith("[") )]
                r = completion_reward(pre_steps, gen_steps)
                samples.append((gen_ids, gen_steps, r))
            rewards = [s[2] for s in samples]
            base = sum(rewards) / len(rewards)
            batch_reward += base
            n += 1
            for gen_ids, _gs, r in samples:
                adv = r - base
                if adv == 0 or not gen_ids:
                    continue
                lp = logprob_sum(model, pre_ids, gen_ids, device)
                batch_loss = batch_loss + (-adv * lp)
        if isinstance(batch_loss, torch.Tensor) and batch_loss.requires_grad:
            (batch_loss / max(n, 1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        meanr = batch_reward / max(n, 1)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"  step {step:4d}/{args.steps}  mean validity-reward={meanr:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "grpo_transformer.pt")
    print(f"GRPO DONE -> {out_dir / 'grpo_transformer.pt'}", flush=True)


if __name__ == "__main__":
    main()
