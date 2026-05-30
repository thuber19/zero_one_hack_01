"""Dataset loading, scaling, splitting, and Uniform Case-Based Sampling."""
import hashlib
import random
from pathlib import Path
from procseq.grammar import (TRAINING_DATA_DIR, read_csv_sequences,
                             generate_dataset, write_csv)

_VARIANT_FILE = {"MOSFET": "MOSFET_variants.csv",
                 "IGBT": "IGBT_variants.csv", "IC": "IC_variants.csv"}

def load_provided(family: str) -> dict[str, list[str]]:
    return read_csv_sequences(Path(TRAINING_DATA_DIR) / _VARIANT_FILE[family])

def _seq_key(steps: list[str]) -> str:
    return "|".join(steps)

def scale_family(family: str, n: int, seed: int,
                 include_provided: bool = True) -> list[list[str]]:
    """Return up to n+provided unique valid sequences for a family."""
    seen: set[str] = set()
    out: list[list[str]] = []
    if include_provided:
        for s in load_provided(family).values():
            k = _seq_key(s)
            if k not in seen:
                seen.add(k); out.append(s)
    if n > 0:
        for s in generate_dataset(family.lower(), n, seed=seed, validate=True):
            k = _seq_key(s)
            if k not in seen:
                seen.add(k); out.append(s)
    return out

def split_ids(ids, val_frac=0.1, test_frac=0.1, seed=13):
    ids = sorted(ids)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return sorted(train), sorted(val), sorted(test)

def ucbs_weights(sequences: list[list[str]], n_buckets: int = 8) -> list[float]:
    """Per-sequence sampling weight so each length bucket has equal total mass."""
    if not sequences:
        return []
    lengths = [len(s) for s in sequences]
    lo, hi = min(lengths), max(lengths)
    span = max(1, hi - lo)
    def bucket(L): return min(n_buckets - 1, (L - lo) * n_buckets // span)
    counts: dict[int, int] = {}
    bidx = [bucket(L) for L in lengths]
    for b in bidx:
        counts[b] = counts.get(b, 0) + 1
    n_active = len(counts)
    return [1.0 / (counts[b] * n_active) for b in bidx]

def write_long_csv(path: Path, sequences: list[list[str]]) -> None:
    write_csv(path, sequences)
