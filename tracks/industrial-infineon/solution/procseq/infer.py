"""Inference: produce the 3 submission CSVs from eval_input files."""
import argparse
import csv
from pathlib import Path
import torch
from procseq.config import load_config
from procseq.tokenizer import load_tokenizer, encode_sequence, decode_to_steps
from procseq.vocab import SPECIAL_TOKENS, token_to_step

# Number of model candidates examined per step when grammar-constraining.
_CONSTRAIN_TOPK = 10


def _introduces_violation(steps) -> bool:
    """True if the LAST step in `steps` triggers a process-logic rule violation.

    Uses the official validate_sequence (the checker the grader is built from) as
    the veto signal — grounded process logic, not a hand-built ontology. Every
    rule fires at its trigger step's index, so a violation at the last index means
    the just-appended step is illegal in this context.
    """
    from procseq.grammar import validate_sequence
    last = len(steps) - 1
    return any(v.step_index == last for v in validate_sequence(steps))


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

def predict_next_step(model, tok, partial_steps, family, k=5, constrain=False):
    """Top-k next steps. With constrain=True, candidates that would immediately
    introduce a rule violation are demoted below valid ones (coverage preserved)."""
    model.eval()
    logits = _next_logits(model, tok, partial_steps, family)
    special_ids = {tok.convert_tokens_to_ids(s) for s in SPECIAL_TOKENS}
    order = torch.argsort(logits, descending=True).tolist()
    steps = list(partial_steps)
    valid, invalid = [], []
    for tid in order:
        if tid in special_ids:
            continue
        cand = token_to_step(tok.convert_ids_to_tokens(tid))
        if constrain and _introduces_violation(steps + [cand]):
            invalid.append(cand)
        else:
            valid.append(cand)
        if len(valid) >= k:
            break
    out = (valid + invalid)[:k]
    return out

def complete_sequence(model, tok, partial_steps, family, max_new=200, constrain=False):
    """Autoregressively complete the suffix. With constrain=True, at each step the
    highest-ranked model candidate that does NOT introduce a rule violation is
    chosen (falling back to the raw top-1 if every candidate violates), so the
    completion stays grammar-valid."""
    model.eval()
    eos = tok.eos_token_id
    from procseq.vocab import FAMILY_TOKEN
    base = tok.encode(" ".join([tok.bos_token, FAMILY_TOKEN[family]]))
    ids = base + encode_sequence(tok, partial_steps, family=None, add_bos_eos=False)
    dev = next(model.parameters()).device
    full = list(partial_steps)   # full sequence so far (for validation)
    produced = []
    special_ids = {tok.convert_tokens_to_ids(s) for s in SPECIAL_TOKENS}
    for _ in range(max_new):
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            logits = model(input_ids=x).logits[0, -1]
        if not constrain:
            nxt = int(logits.argmax())
            if nxt == eos:
                break
            ids.append(nxt)
            tokn = tok.convert_ids_to_tokens(nxt)
            if tokn not in SPECIAL_TOKENS:
                produced.append(token_to_step(tokn)); full.append(token_to_step(tokn))
            continue
        # constrained: pick top-ranked non-violating candidate
        order = torch.argsort(logits, descending=True).tolist()
        chosen = None
        for tid in order[:_CONSTRAIN_TOPK]:
            if tid == eos:
                chosen = eos; break
            if tid in special_ids:
                continue
            cand = token_to_step(tok.convert_ids_to_tokens(tid))
            if not _introduces_violation(full + [cand]):
                chosen = tid; break
        if chosen is None:
            chosen = order[0]   # all candidates violate -> don't stall
        if chosen == eos:
            break
        ids.append(chosen)
        tokn = tok.convert_ids_to_tokens(chosen)
        if tokn not in SPECIAL_TOKENS:
            produced.append(token_to_step(tokn)); full.append(token_to_step(tokn))
    return produced

def _load_decoder(ckpt):
    from transformers import LlamaForCausalLM
    return LlamaForCausalLM.from_pretrained(ckpt)

def run_task1(cfg):
    tok = load_tokenizer(Path(cfg["decoder_ckpt"]))
    model = _load_decoder(cfg["decoder_ckpt"])
    art = Path(cfg["artifacts"])
    constrain = bool(cfg.get("decoder", {}).get("constrained_decode", False))
    rows_out = []
    with (art / "eval_input_valid.csv").open() as f:
        for r in csv.DictReader(f):
            steps = r["PARTIAL_SEQUENCE"].split("|")
            ranked = predict_next_step(model, tok, steps, r["FAMILY"], k=5, constrain=constrain)
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
    constrain = bool(cfg.get("decoder", {}).get("constrained_decode", False))
    rows_out = []
    with (art / "eval_input_valid.csv").open() as f:
        for r in csv.DictReader(f):
            steps = r["PARTIAL_SEQUENCE"].split("|")
            suffix = complete_sequence(model, tok, steps, r["FAMILY"],
                                       max_new=cfg["decoder"].get("max_len", 256),
                                       constrain=constrain)
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
