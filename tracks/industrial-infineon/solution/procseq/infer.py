"""Inference: produce the 3 submission CSVs from eval_input files."""
import argparse
import csv
from pathlib import Path
import torch
from procseq.config import load_config
from procseq.tokenizer import load_tokenizer, encode_sequence, decode_to_steps
from procseq.vocab import SPECIAL_TOKENS, token_to_step

@torch.no_grad()
def _next_logits(model, tok, steps, family):
    ids = encode_sequence(tok, steps, family=family, add_bos_eos=False)
    # prepend BOS + family handled inside encode when add_bos_eos True; here we add BOS+fam manually
    from procseq.vocab import FAMILY_TOKEN
    prefix = tok.encode(" ".join([tok.bos_token, FAMILY_TOKEN[family]]))
    ids = prefix + ids
    x = torch.tensor([ids], device=next(model.parameters()).device)
    out = model(input_ids=x)
    return out.logits[0, -1]

def predict_next_step(model, tok, partial_steps, family, k=5):
    model.eval()
    logits = _next_logits(model, tok, partial_steps, family)
    special_ids = {tok.convert_tokens_to_ids(s) for s in SPECIAL_TOKENS}
    order = torch.argsort(logits, descending=True).tolist()
    out = []
    for tid in order:
        if tid in special_ids:
            continue
        tokn = tok.convert_ids_to_tokens(tid)
        out.append(token_to_step(tokn))
        if len(out) == k:
            break
    return out

def complete_sequence(model, tok, partial_steps, family, max_new=200):
    model.eval()
    steps = list(partial_steps)
    eos = tok.eos_token_id
    from procseq.vocab import FAMILY_TOKEN
    base = tok.encode(" ".join([tok.bos_token, FAMILY_TOKEN[family]]))
    ids = base + encode_sequence(tok, partial_steps, family=None, add_bos_eos=False)
    dev = next(model.parameters()).device
    produced = []
    for _ in range(max_new):
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            nxt = int(model(input_ids=x).logits[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        tokn = tok.convert_ids_to_tokens(nxt)
        if tokn in SPECIAL_TOKENS:
            continue
        produced.append(token_to_step(tokn))
    return produced

def _load_decoder(ckpt):
    from transformers import LlamaForCausalLM
    return LlamaForCausalLM.from_pretrained(ckpt)

def run_task1(cfg):
    tok = load_tokenizer(Path(cfg["decoder_ckpt"]))
    model = _load_decoder(cfg["decoder_ckpt"])
    art = Path(cfg["artifacts"])
    rows_out = []
    with (art / "eval_input_valid.csv").open() as f:
        for r in csv.DictReader(f):
            steps = r["PARTIAL_SEQUENCE"].split("|")
            ranked = predict_next_step(model, tok, steps, r["FAMILY"], k=5)
            ranked += [""] * (5 - len(ranked))
            rows_out.append([r["EXAMPLE_ID"], *ranked[:5]])
    out = art / "submission_task1.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["EXAMPLE_ID","RANK_1","RANK_2","RANK_3","RANK_4","RANK_5"])
        w.writerows(rows_out)
    print(f"Task1 -> {out} ({len(rows_out)} rows)")

def run_task2(cfg):
    tok = load_tokenizer(Path(cfg["decoder_ckpt"]))
    model = _load_decoder(cfg["decoder_ckpt"])
    art = Path(cfg["artifacts"])
    rows_out = []
    with (art / "eval_input_valid.csv").open() as f:
        for r in csv.DictReader(f):
            steps = r["PARTIAL_SEQUENCE"].split("|")
            suffix = complete_sequence(model, tok, steps, r["FAMILY"],
                                       max_new=cfg["decoder"].get("max_len", 256))
            rows_out.append([r["EXAMPLE_ID"], "|".join(suffix)])
    out = art / "submission_task2.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["EXAMPLE_ID","PREDICTED_SEQUENCE"])
        w.writerows(rows_out)
    print(f"Task2 -> {out} ({len(rows_out)} rows)")

def run_task3(cfg):
    from procseq.infer_anomaly import run_anomaly
    run_anomaly(cfg)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--task", choices=["1","2","3"])
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.all or a.task == "1": run_task1(cfg)
    if a.all or a.task == "2": run_task2(cfg)
    if a.all or a.task == "3": run_task3(cfg)

if __name__ == "__main__":
    main()
