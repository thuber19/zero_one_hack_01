"""Hybrid inference: procseq's LEARNED decoder proposes, the team's PhysicsRefinery
disposes. Best of both worlds — learned generation + a physically-guaranteed-valid
output.

- Task 1 (next-step): decoder top-k -> refinery.rerank (physically-legal first).
- Task 2 (completion): refinery.beam_decode driven by the decoder's softmax ->
  a completion guaranteed rule-valid (beam search + legality + clean termination),
  which lowers edit distance vs. plain greedy.

This raises the *scored* metrics while procseq's pure-neural `infer.py` stays as the
honest "what the model learned alone" ablation. Run after a decoder training run:
    python -m procseq.infer_hybrid --config <cfg>           # on our mirrors
    python -m procseq.infer_hybrid --config <cfg> --real    # on data/eval_input_valid.csv
"""
import argparse
import csv
from pathlib import Path
import torch

from procseq import grammar      # noqa: F401  (sets up the data/ import path)
from procseq import external     # noqa: F401  (adds the track root for refinery/physics)
from procseq.config import load_config
from procseq.tokenizer import load_tokenizer, encode_sequence
from procseq.vocab import FAMILY_TOKEN, SPECIAL_TOKENS, token_to_step

import refinery as _ref          # the team's PhysicsRefinery (model proposes, physics disposes)


def _load_decoder(ckpt):
    from transformers import LlamaForCausalLM
    return LlamaForCausalLM.from_pretrained(ckpt)


@torch.no_grad()
def _logits_last(model, tok, steps, family):
    base = tok.encode(" ".join([tok.bos_token, FAMILY_TOKEN[family]]))
    ids = base + encode_sequence(tok, steps, family=None, add_bos_eos=False)
    x = torch.tensor([ids], device=next(model.parameters()).device)
    return model(input_ids=x).logits[0, -1]


def _ranked_names(model, tok, steps, family, k=15):
    logits = _logits_last(model, tok, steps, family)
    special = {tok.convert_tokens_to_ids(s) for s in SPECIAL_TOKENS}
    out = []
    for tid in torch.argsort(logits, descending=True).tolist():
        if tid in special:
            continue
        out.append(token_to_step(tok.convert_ids_to_tokens(tid)))
        if len(out) >= k:
            break
    return out


def _probs_pairs(model, tok, steps, family, k=20):
    """Ranked (step_name, prob) pairs — the score_fn the refinery's beam wants."""
    probs = torch.softmax(_logits_last(model, tok, steps, family), dim=-1)
    special = {tok.convert_tokens_to_ids(s) for s in SPECIAL_TOKENS}
    out = []
    for tid in torch.argsort(probs, descending=True).tolist():
        if tid in special:
            continue
        out.append((token_to_step(tok.convert_ids_to_tokens(tid)), float(probs[tid])))
        if len(out) >= k:
            break
    return out


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"-> {path} ({len(rows)} rows)", flush=True)


def run_hybrid(cfg, real=False, limit=None):
    refinery = _ref.PhysicsRefinery(category_mode="off")   # pure rule-legality (safe)
    tok = load_tokenizer(Path(cfg["decoder_ckpt"]))
    model = _load_decoder(cfg["decoder_ckpt"]); model.eval()
    art = Path(cfg["artifacts"])
    if real:
        src = Path(grammar.TRAINING_DATA_DIR) / "eval_input_valid.csv"; suf = "_real"
    else:
        src = art / "eval_input_valid.csv"; suf = ""
    with src.open() as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    max_new = cfg["decoder"].get("max_len", 256)

    # ── Task 1: decoder candidates, physics-legal first ──
    t1 = []
    for r in rows:
        steps = r["PARTIAL_SEQUENCE"].split("|")
        ranked = _ranked_names(model, tok, steps, r["FAMILY"], k=15)
        refined = refinery.rerank(steps, ranked, k=5)
        refined += [""] * (5 - len(refined))
        t1.append([r["EXAMPLE_ID"], *refined[:5]])
    _write(art / f"submission_task1_hybrid{suf}.csv",
           ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"], t1)

    # ── Task 2: physics beam search driven by the decoder ──
    print(f"  [hybrid task2{suf}] {len(rows)} sequences via physics beam-decode...", flush=True)
    t2 = []
    for i, r in enumerate(rows, 1):
        steps = r["PARTIAL_SEQUENCE"].split("|")
        fam = r["FAMILY"]
        def score_fn(pfx, _fam=fam):
            return _probs_pairs(model, tok, pfx, _fam, k=20)
        comp = refinery.beam_decode(steps, score_fn, beam=5, max_steps=max_new)
        t2.append([r["EXAMPLE_ID"], "|".join(comp)])
        if i % 100 == 0 or i == len(rows):
            print(f"    task2 {i}/{len(rows)}", flush=True)
    _write(art / f"submission_task2_hybrid{suf}.csv",
           ["EXAMPLE_ID", "PREDICTED_SEQUENCE"], t2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap #examples (debug)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    run_hybrid(cfg, real=a.real, limit=a.limit)


if __name__ == "__main__":
    main()
