"""MLMTokenizer — wraps FabTokenizer, adding [CLS], [SEP], [MASK] for BERT-style MLM.

Load order:
  1. $WORK/artifacts/001/vocab.json (Spec 001 FabTokenizer format)
  2. If [CLS]/[SEP]/[MASK] missing → append new IDs (existing IDs never renumbered)
  3. Fallback: build vocab from CSVs, save to $WORK/artifacts/002/vocab.json

Encoding format: [CLS] [VARIANT] step_1 ... step_N [SEP] [PAD] ... [PAD]  (padded to max_len)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.data.tokenizer import FabTokenizer, normalize_step, SPECIALS as FAB_SPECIALS

BERT_SPECIALS = ["[MASK]", "[CLS]", "[SEP]"]
_VARIANT_TOKEN_STRINGS = {"IC": "[IC]", "IGBT": "[IGBT]", "MOSFET": "[MOSFET]"}


@dataclass
class MLMTokenizer:
    id_to_step: list[str] = field(default_factory=list)
    step_to_id: dict[str, int] = field(default_factory=dict)

    # ---- properties --------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_step)

    @property
    def pad_id(self) -> int:
        return self.step_to_id["[PAD]"]

    @property
    def cls_id(self) -> int:
        return self.step_to_id["[CLS]"]

    @property
    def sep_id(self) -> int:
        return self.step_to_id["[SEP]"]

    @property
    def mask_id(self) -> int:
        return self.step_to_id["[MASK]"]

    @property
    def unk_id(self) -> int:
        return self.step_to_id.get("[UNK]", 3)

    def variant_id(self, variant: str) -> int:
        token = _VARIANT_TOKEN_STRINGS[variant]
        return self.step_to_id[token]

    # ---- construction ------------------------------------------------------

    @classmethod
    def _from_id_to_step(cls, id_to_step: list[str]) -> "MLMTokenizer":
        extended = list(id_to_step)
        for tok in BERT_SPECIALS:
            if tok not in extended:
                extended.append(tok)
        step_to_id = {s: i for i, s in enumerate(extended)}
        return cls(id_to_step=extended, step_to_id=step_to_id)

    @classmethod
    def load(cls, path: str | Path) -> "MLMTokenizer":
        with open(path) as f:
            d = json.load(f)
        id_to_step = d.get("id_to_step", d.get("id_to_token"))
        if id_to_step is None:
            raise ValueError(f"Unrecognised vocab format in {path}: missing 'id_to_step' key")
        return cls._from_id_to_step(id_to_step)

    @classmethod
    def build(cls, csv_paths: dict[str, str | Path]) -> "MLMTokenizer":
        fab = FabTokenizer.build(csv_paths)
        return cls._from_id_to_step(fab.id_to_step)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"id_to_step": self.id_to_step}, f, indent=2)

    # ---- encoding ----------------------------------------------------------

    def encode_step(self, s: str) -> int:
        return self.step_to_id.get(normalize_step(s), self.unk_id)

    def encode_mlm(self, variant: str, steps: list[str], max_len: int = 100) -> list[int]:
        """Encode as [CLS] [VARIANT] step... [SEP] [PAD]... padded to max_len."""
        ids = [self.cls_id, self.variant_id(variant)]
        ids.extend(self.encode_step(s) for s in steps)
        ids.append(self.sep_id)
        if len(ids) > max_len:
            # truncate steps, keep [CLS], [VARIANT], ..., [SEP]
            ids = ids[:max_len - 1] + [self.sep_id]
        ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_step[i] if 0 <= i < self.vocab_size else "[OOR]" for i in ids]

    # ---- compatibility check -----------------------------------------------

    def verify_compat(self, spec001_path: str | Path) -> None:
        """Assert all tokens in Spec 001 vocab appear at same ID in this vocab. Raises on mismatch."""
        with open(spec001_path) as f:
            d001 = json.load(f)
        id_to_step_001 = d001.get("id_to_step", d001.get("id_to_token", []))
        mismatches = []
        for i, tok in enumerate(id_to_step_001):
            if self.step_to_id.get(tok) != i:
                mismatches.append((i, tok, self.step_to_id.get(tok)))
        if mismatches:
            lines = "\n".join(f"  id={i} token={tok!r} → got id={got}" for i, tok, got in mismatches[:20])
            raise ValueError(f"Tokenizer compat check FAILED ({len(mismatches)} mismatches):\n{lines}")
