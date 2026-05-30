"""Load variant CSVs into per-SEQUENCE_ID step lists and build splits."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .tokenizer import FabTokenizer, VARIANT_TOKENS


def load_variant_csv(path: str | Path) -> dict[str, list[str]]:
    """Returns {sequence_id: [step1, step2, ...]} preserving CSV row order."""
    seqs: dict[str, list[str]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("SEQUENCE_ID", "").strip()
            step = row.get("STEP", "").strip()
            if sid and step:
                seqs[sid].append(step)
    return dict(seqs)


def load_all_variants(csv_paths: dict[str, str | Path]) -> list[tuple[str, str, list[str]]]:
    """Returns list of (variant, sequence_id, steps)."""
    out: list[tuple[str, str, list[str]]] = []
    for variant, path in csv_paths.items():
        if variant not in VARIANT_TOKENS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {list(VARIANT_TOKENS)}")
        seqs = load_variant_csv(path)
        for sid, steps in seqs.items():
            out.append((variant, sid, steps))
    return out


def build_splits(
    records: list[tuple[str, str, list[str]]],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    """Stratified per-variant split. Returns {split: [(variant, sid), ...]}."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    by_variant: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for variant, sid, _ in records:
        by_variant[variant].append((variant, sid))
    rng = random.Random(seed)
    splits: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    for variant, items in by_variant.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        splits["train"].extend(items[:n_tr])
        splits["val"].extend(items[n_tr : n_tr + n_va])
        splits["test"].extend(items[n_tr + n_va :])
    # leakage check
    seen: set[tuple[str, str]] = set()
    for items in splits.values():
        for it in items:
            if it in seen:
                raise RuntimeError(f"Leakage: {it} in multiple splits")
            seen.add(it)
    return splits


def encode_split(
    records: list[tuple[str, str, list[str]]],
    split_keys: list[tuple[str, str]],
    tokenizer: FabTokenizer,
) -> list[list[int]]:
    by_key = {(v, sid): (v, steps) for v, sid, steps in records}
    out: list[list[int]] = []
    for key in split_keys:
        v, steps = by_key[key]
        out.append(tokenizer.encode_sequence(v, steps))
    return out


def pack_sequences(token_lists: list[list[int]], max_len: int) -> list[list[int]]:
    """Greedy pack: keep each whole sequence intact, never cross max_len. Pad short ones."""
    from .tokenizer import PAD_ID

    packed: list[list[int]] = []
    cur: list[int] = []
    for seq in token_lists:
        if len(seq) > max_len:
            # truncate end (keeps variant + BOS + early steps, drops tail+EOS)
            seq = seq[: max_len - 1] + [seq[-1]]
        if len(cur) + len(seq) > max_len:
            cur += [PAD_ID] * (max_len - len(cur))
            packed.append(cur)
            cur = []
        cur.extend(seq)
    if cur:
        cur += [PAD_ID] * (max_len - len(cur))
        packed.append(cur)
    return packed


def vocab_hash(tokenizer: FabTokenizer) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(tokenizer.id_to_step).encode())
    return h.hexdigest()[:16]
