"""Canonical step vocabulary and atomic step<->token mapping."""
from pathlib import Path
from procseq.grammar import TRAINING_DATA_DIR, read_csv_sequences, FAMILIES

SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[CLS]", "[SEP]", "[MASK]",
    "[FAM_MOSFET]", "[FAM_IGBT]", "[FAM_IC]",
]
FAMILY_TOKEN = {"MOSFET": "[FAM_MOSFET]", "IGBT": "[FAM_IGBT]", "IC": "[FAM_IC]"}

def step_to_token(step: str) -> str:
    return step.strip().replace(" ", "_")

def token_to_step(token: str) -> str:
    return token.replace("_", " ")

_SOURCE_FILES = [
    "MOSFET_variants.csv", "IGBT_variants.csv", "IC_variants.csv",
    "synthetic_mosfet.csv", "syntheticIGBT.csv", "syntheticIC.csv",
]

def all_steps() -> list[str]:
    """Sorted union of every distinct STEP string in the provided CSVs."""
    steps: set[str] = set()
    for fname in _SOURCE_FILES:
        p = Path(TRAINING_DATA_DIR) / fname
        if not p.exists():
            continue
        for seq in read_csv_sequences(p).values():
            steps.update(seq)
    return sorted(steps)

def build_vocab() -> dict[str, int]:
    """token -> id. Specials first (stable ids), then step tokens sorted."""
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    step_tokens = sorted({step_to_token(s) for s in all_steps()})
    for tok in step_tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab
