"""Task 3 inference: classify each anomaly-eval sequence -> submission CSV."""
import csv
from pathlib import Path
import torch
from procseq.tokenizer import load_tokenizer
from procseq.models.encoder import build_encoder
from procseq.vocab import FAMILY_TOKEN, step_to_token
from procseq.grammar import RULE_IDS

@torch.no_grad()
def classify_sequence(model, tok, steps, family, rule_ids):
    model.eval()
    text = " ".join([tok.cls_token, FAMILY_TOKEN[family]] +
                    [step_to_token(s) for s in steps] + [tok.sep_token])
    ids = torch.tensor([tok.encode(text)], device=next(model.parameters()).device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    p_invalid = torch.sigmoid(out["invalid_logit"])[0].item()
    is_valid = 0 if p_invalid >= 0.5 else 1
    score = 1.0 - p_invalid  # P(valid) for AUC
    rule = ""
    if is_valid == 0:
        rule = rule_ids[int(out["rule_logits"][0].argmax())]
    return is_valid, score, rule

def _load_encoder(ckpt, tok):
    model = build_encoder("base", tok, n_rules=len(RULE_IDS))  # size from state shape
    sd = torch.load(Path(ckpt) / "pytorch_model.bin", map_location="cpu")
    # infer size by trying presets until shapes match
    for size in ("tiny", "small", "base"):
        m = build_encoder(size, tok, n_rules=len(RULE_IDS))
        try:
            m.load_state_dict(sd); return m
        except Exception:
            continue
    raise RuntimeError("encoder size mismatch")

def run_anomaly(cfg):
    art = Path(cfg["artifacts"])
    tok = load_tokenizer(Path(cfg["encoder_ckpt"]))
    model = _load_encoder(cfg["encoder_ckpt"], tok)
    rows = []
    with (art / "eval_input_anomaly.csv").open() as f:
        for r in csv.DictReader(f):
            steps = r["SEQUENCE"].split("|")
            iv, sc, rule = classify_sequence(model, tok, steps, r["FAMILY"], RULE_IDS)
            rows.append([r["EXAMPLE_ID"], iv, f"{sc:.4f}", rule])
    out = art / "submission_task3.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["EXAMPLE_ID","IS_VALID","SCORE","PREDICTED_RULE"])
        w.writerows(rows)
    print(f"Task3 -> {out} ({len(rows)} rows)")
