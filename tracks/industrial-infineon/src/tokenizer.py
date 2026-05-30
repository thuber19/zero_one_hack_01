"""
Custom tokenizer for semiconductor process sequences.

Each unique process step is a single token. Special tokens:
  [PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3, [MOSFET]=4, [IGBT]=5, [IC]=6
"""

import csv
from pathlib import Path


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[MOSFET]", "[IGBT]", "[IC]"]
FAMILY_TOKENS = {"mosfet": "[MOSFET]", "igbt": "[IGBT]", "ic": "[IC]"}

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class StepTokenizer:
    """Maps process step strings <-> integer token IDs."""

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        # Reserve special tokens
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token2id[tok] = i
            self.id2token[i] = tok

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    def encode_step(self, step: str) -> int:
        return self.token2id.get(step, UNK_ID)

    def encode_sequence(self, steps: list[str], family: str) -> list[int]:
        """Encode a sequence with [BOS] + family_token + steps + [EOS].

        ASSUMPTION: family is one of {mosfet, igbt, ic}. The hidden 4th family
        (Task 4) will NOT be in FAMILY_TOKENS, so we fall back to [UNK] instead
        of crashing with a KeyError. The physics layer is family-agnostic, so
        this is safe; see ASSUMPTIONS.md (A3)."""
        family_tok = FAMILY_TOKENS.get(family.lower(), "[UNK]")
        ids = [BOS_ID, self.encode_step(family_tok)]
        for s in steps:
            ids.append(self.encode_step(s))
        ids.append(EOS_ID)
        return ids

    def decode_ids(self, ids: list[int]) -> list[str]:
        return [self.id2token.get(i, "[UNK]") for i in ids]

    def decode_steps(self, ids: list[int]) -> list[str]:
        """Decode token IDs, filtering out special tokens."""
        special = set(range(len(SPECIAL_TOKENS)))
        return [self.id2token.get(i, "[UNK]") for i in ids if i not in special]

    def save(self, path: Path):
        with open(path, "w") as f:
            for token, idx in sorted(self.token2id.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{token}\n")

    @classmethod
    def load(cls, path: Path) -> "StepTokenizer":
        tok = cls()
        tok.token2id.clear()
        tok.id2token.clear()
        with open(path) as f:
            for line in f:
                idx_s, token = line.strip().split("\t", 1)
                idx = int(idx_s)
                tok.token2id[token] = idx
                tok.id2token[idx] = token
        return tok

    @classmethod
    def from_sequences(cls, sequences: list[list[str]]) -> "StepTokenizer":
        """Build tokenizer from a list of step sequences."""
        tok = cls()
        for seq in sequences:
            for step in seq:
                tok.add_token(step)
        return tok
