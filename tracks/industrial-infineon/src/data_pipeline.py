"""
Data pipeline: load existing CSVs, generate more data, build tokenizer,
create PyTorch datasets for transformer training and feature matrices
for random forest training.
"""

import sys
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import numpy as np

# Add training_data dir so we can import generate_sequences
TRAINING_DATA_DIR = Path(__file__).resolve().parent.parent / "training_data"
sys.path.insert(0, str(TRAINING_DATA_DIR))

from generate_sequences import (  # noqa: E402
    generate_dataset,
    read_csv_sequences,
)
from tokenizer import StepTokenizer, FAMILY_TOKENS, BOS_ID, EOS_ID, PAD_ID  # noqa: E402


# ── Loading & generation ─────────────────────────────────────────────────

def load_existing_sequences(data_dir: Path | None = None) -> dict[str, list[list[str]]]:
    """Load pre-generated variant CSVs. Returns {family: [[step, ...], ...]}."""
    if data_dir is None:
        data_dir = TRAINING_DATA_DIR
    family_files = {
        "mosfet": "MOSFET_variants.csv",
        "igbt": "IGBT_variants.csv",
        "ic": "IC_variants.csv",
    }
    result: dict[str, list[list[str]]] = {}
    for family, fname in family_files.items():
        p = data_dir / fname
        if p.exists():
            seqs = read_csv_sequences(p)
            result[family] = list(seqs.values())
            print(f"Loaded {len(result[family])} {family.upper()} sequences from {fname}")
        else:
            result[family] = []
            print(f"  {fname} not found, skipping")
    return result


def generate_additional(family: str, count: int, seed: int = 12345) -> list[list[str]]:
    """Generate additional sequences using the grammar."""
    print(f"Generating {count} additional {family.upper()} sequences (seed={seed})...")
    return generate_dataset(family, count, seed=seed, validate=True)


def prepare_all_data(
    extra_per_family: int = 5000,
    seed: int = 12345,
) -> tuple[dict[str, list[list[str]]], StepTokenizer]:
    """
    Load existing + generate extra sequences. Build tokenizer.
    Returns (family_sequences, tokenizer).
    """
    family_seqs = load_existing_sequences()

    # Generate extra data
    for family in ["mosfet", "igbt", "ic"]:
        existing = len(family_seqs[family])
        if existing < extra_per_family:
            extra = generate_additional(
                family, extra_per_family - existing, seed=seed
            )
            family_seqs[family].extend(extra)

    # Build tokenizer from all sequences
    all_seqs = []
    for seqs in family_seqs.values():
        all_seqs.extend(seqs)
    tokenizer = StepTokenizer.from_sequences(all_seqs)
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Total sequences: {sum(len(s) for s in family_seqs.values())}")

    return family_seqs, tokenizer


# ── PyTorch dataset for transformer ──────────────────────────────────────

class ProcessSequenceDataset(Dataset):
    """
    Dataset for next-step prediction. Each sample is a tokenized sequence.
    Input: tokens[:-1], Target: tokens[1:]  (shifted by 1 for causal LM).
    """

    def __init__(
        self,
        sequences: list[tuple[str, list[str]]],  # [(family, [step, ...]), ...]
        tokenizer: StepTokenizer,
        max_len: int = 200,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encoded: list[list[int]] = []

        for family, steps in sequences:
            ids = tokenizer.encode_sequence(steps, family)
            if len(ids) > max_len:
                ids = ids[:max_len]
            self.encoded.append(ids)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.encoded[idx]
        # Pad to max_len
        padded = ids + [PAD_ID] * (self.max_len - len(ids))
        input_ids = torch.tensor(padded[:-1], dtype=torch.long)
        target_ids = torch.tensor(padded[1:], dtype=torch.long)
        # Attention mask: 1 for real tokens, 0 for padding
        attn_mask = torch.tensor(
            [1] * (len(ids) - 1) + [0] * (self.max_len - len(ids)),
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attn_mask,
        }


# ── Feature extraction for random forest ─────────────────────────────────

def extract_rf_features(
    sequences: list[tuple[str, list[str]]],
    tokenizer: StepTokenizer,
    context_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (features, labels) for random forest training.

    For each transition (step_t -> step_{t+1}) in each sequence, create:
    - Features: [family_id, current_step_id, step_{t-1}_id, step_{t-2}_id, litho_level, position_frac]
    - Label: next_step_id

    Returns (X, y) numpy arrays.
    """
    family_id_map = {"mosfet": 0, "igbt": 1, "ic": 2}
    X_rows = []
    y_rows = []

    for family, steps in sequences:
        fam_id = family_id_map.get(family.lower(), -1)   # 4th family -> -1 (no crash)
        ids = [tokenizer.encode_step(s) for s in steps]
        n = len(ids)
        litho_level = 0

        for t in range(n - 1):
            # Track litho level
            if steps[t].startswith("ALIGN MASK LEVEL "):
                parts = steps[t].split("ALIGN MASK LEVEL ")
                if len(parts) > 1 and parts[1].isdigit():
                    litho_level = int(parts[1])

            # Context: previous steps (padded with 0 if at start)
            prev1 = ids[t - 1] if t >= 1 else PAD_ID
            prev2 = ids[t - 2] if t >= 2 else PAD_ID

            position_frac = t / n  # how far into the sequence

            features = [fam_id, ids[t], prev1, prev2, litho_level, position_frac]
            X_rows.append(features)
            y_rows.append(ids[t + 1])

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64)


def build_transition_map(
    sequences: list[tuple[str, list[str]]],
) -> dict[str, set[str]]:
    """
    Build a mapping: (family, current_step) -> set of observed next steps.
    Used as a fallback/lookup for candidate generation.
    """
    transitions: dict[str, set[str]] = defaultdict(set)
    for family, steps in sequences:
        for i in range(len(steps) - 1):
            key = f"{family.lower()}|{steps[i]}"
            transitions[key].add(steps[i + 1])
    return dict(transitions)
