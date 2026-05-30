"""FabTokenizer — flat per-step vocabulary.

Specials reserve IDs 0..6:
  0 [PAD]  1 [BOS]  2 [EOS]  3 [UNK]  4 [IC]  5 [IGBT]  6 [MOSFET]

Step strings are normalized (strip + collapse whitespace + upper) before lookup.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
VARIANT_TOKENS = {"IC": 4, "IGBT": 5, "MOSFET": 6}

_WS_RE = re.compile(r"\s+")


def normalize_step(s: str) -> str:
    s = s.replace("﻿", "").strip().strip('"').strip()
    s = _WS_RE.sub(" ", s).upper()
    return s


@dataclass
class FabTokenizer:
    step_to_id: dict[str, int] = field(default_factory=dict)
    id_to_step: list[str] = field(default_factory=list)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_step)

    @classmethod
    def build(cls, csv_paths: dict[str, str | Path]) -> "FabTokenizer":
        steps: set[str] = set()
        for path in csv_paths.values():
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = [n.strip().upper() for n in reader.fieldnames or []]
                step_col = "STEP" if "STEP" in fieldnames else (reader.fieldnames or [""])[-1]
                for row in reader:
                    raw = row.get(step_col) or row.get("STEP") or ""
                    n = normalize_step(raw)
                    if n:
                        steps.add(n)
        ordered = SPECIALS + sorted(steps)
        return cls(
            step_to_id={s: i for i, s in enumerate(ordered)},
            id_to_step=ordered,
        )

    def encode_step(self, s: str) -> int:
        return self.step_to_id.get(normalize_step(s), UNK_ID)

    def encode_sequence(self, variant: str, steps: list[str]) -> list[int]:
        ids = [VARIANT_TOKENS[variant], BOS_ID]
        ids.extend(self.encode_step(s) for s in steps)
        ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_step[i] if 0 <= i < self.vocab_size else "[OOR]" for i in ids]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"id_to_step": self.id_to_step}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FabTokenizer":
        with open(path) as f:
            d = json.load(f)
        id_to_step = d["id_to_step"]
        return cls(
            step_to_id={s: i for i, s in enumerate(id_to_step)},
            id_to_step=id_to_step,
        )
