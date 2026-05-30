"""Torch datasets/collators for decoder (CLM) and encoder (classification)."""
import torch
from torch.utils.data import Dataset
from procseq.tokenizer import encode_sequence

class ClmDataset(Dataset):
    def __init__(self, seqs_with_family, tokenizer, max_len=256):
        self.items = seqs_with_family
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        steps, fam = self.items[i]
        ids = encode_sequence(self.tok, steps, family=fam, add_bos_eos=True)[:self.max_len]
        return {"input_ids": ids}

def clm_collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, labels, attn = [], [], []
    for b in batch:
        x = b["input_ids"]
        pad = [pad_id] * (maxlen - len(x))
        ids.append(x + pad)
        labels.append(x + [pad_id] * (maxlen - len(x)))
        attn.append([1] * len(x) + [0] * (maxlen - len(x)))
    ids = torch.tensor(ids); labels = torch.tensor(labels)
    labels[ids == pad_id] = -100
    return {"input_ids": ids, "attention_mask": torch.tensor(attn), "labels": labels}

class ClsDataset(Dataset):
    def __init__(self, items, tokenizer, rule_ids, max_len=256):
        # items: list of (steps, family, is_valid, rule_str)
        self.items = items; self.tok = tokenizer
        self.rule_index = {r: k for k, r in enumerate(rule_ids)}
        self.max_len = max_len

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        steps, fam, is_valid, rule = self.items[i]
        from procseq.vocab import FAMILY_TOKEN, step_to_token
        text = " ".join([self.tok.cls_token, FAMILY_TOKEN[fam]] +
                        [step_to_token(s) for s in steps] + [self.tok.sep_token])
        ids = self.tok.encode(text)[:self.max_len]
        rule_vec = [0.0] * len(self.rule_index)
        if not is_valid and rule in self.rule_index:
            rule_vec[self.rule_index[rule]] = 1.0
        return {"input_ids": ids, "invalid": float(0 if is_valid else 1), "rules": rule_vec}

def cls_collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, attn = [], []
    for b in batch:
        x = b["input_ids"]; pad = [pad_id] * (maxlen - len(x))
        ids.append(x + pad); attn.append([1]*len(x) + [0]*(maxlen-len(x)))
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn),
            "invalid": torch.tensor([b["invalid"] for b in batch]),
            "rules": torch.tensor([b["rules"] for b in batch])}
