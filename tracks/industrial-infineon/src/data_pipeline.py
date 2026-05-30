"""
Data pipeline: load CSVs, create PyTorch datasets for training,
extract features for random forest.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import numpy as np

# Add src/ and data/ to path
_SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = _SRC_DIR.parent / "data"
for _p in (str(_SRC_DIR), str(DATA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_sequences import generate_dataset, read_csv_sequences  # noqa: E402
from tokenizer import StepTokenizer, FAMILY_TOKENS, BOS_ID, EOS_ID, PAD_ID  # noqa: E402
from block_classifier import classify_step_block_id  # noqa: E402


# ── Loading ──────────────────────────────────────────────────────────────

def load_existing_sequences(data_dir: Path | None = None) -> dict[str, list[list[str]]]:
    """Load pre-generated variant CSVs. Returns {family: [[step, ...], ...]}."""
    if data_dir is None:
        data_dir = DATA_DIR
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


def load_train_csv(path: Path) -> list[tuple[str, list[str]]]:
    """
    Load train_sequences.csv (SEQUENCE_ID, FAMILY, STEP long format).
    Returns list of (family, [step, ...]) pairs.
    """
    sequences: dict[str, tuple[str, list[str]]] = {}  # seq_id -> (family, steps)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["SEQUENCE_ID"].strip()
            family = row["FAMILY"].strip().lower()
            step = row["STEP"].strip()
            if sid not in sequences:
                sequences[sid] = (family, [])
            sequences[sid][1].append(step)

    result = list(sequences.values())
    families = defaultdict(int)
    for fam, _ in result:
        families[fam] += 1
    for fam, count in sorted(families.items()):
        print(f"  {fam.upper()}: {count} sequences")
    return result


# ── PyTorch dataset ──────────────────────────────────────────────────────

class ProcessSequenceDataset(Dataset):
    """
    Dataset for next-step prediction. Each sample is a tokenized sequence.
    Input: tokens[:-1], Target: tokens[1:]  (shifted by 1 for causal LM).
    """

    def __init__(
        self,
        sequences: list[tuple[str, list[str]]],
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
        padded = ids + [PAD_ID] * (self.max_len - len(ids))
        input_ids = torch.tensor(padded[:-1], dtype=torch.long)
        target_ids = torch.tensor(padded[1:], dtype=torch.long)
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
    max_samples: int = 5_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (features, labels) for random forest. Caps at max_samples."""
    family_id_map = {"mosfet": 0, "igbt": 1, "ic": 2}
    X_rows = []
    y_rows = []

    for family, steps in sequences:
        fam_id = family_id_map.get(family.lower(), -1)   # 4th family -> -1 (no crash)
        ids = [tokenizer.encode_step(s) for s in steps]
        n = len(ids)
        litho_level = 0

        for t in range(n - 1):
            if steps[t].startswith("ALIGN MASK LEVEL "):
                parts = steps[t].split("ALIGN MASK LEVEL ")
                if len(parts) > 1 and parts[1].isdigit():
                    litho_level = int(parts[1])
            prev1 = ids[t - 1] if t >= 1 else PAD_ID
            prev2 = ids[t - 2] if t >= 2 else PAD_ID
            prev3 = ids[t - 3] if t >= 3 else PAD_ID
            position_frac = t / n
            block_id = classify_step_block_id(steps[t])
            X_rows.append([fam_id, ids[t], prev1, prev2, prev3, litho_level, position_frac, block_id])
            y_rows.append(ids[t + 1])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)

    if len(X) > max_samples:
        print(f"  Subsampling RF data: {len(X)} -> {max_samples} transitions")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    return X, y


def build_transition_map(
    sequences: list[tuple[str, list[str]]],
) -> dict[str, set[str]]:
    """Build (family, current_step) -> set of observed next steps."""
    transitions: dict[str, set[str]] = defaultdict(set)
    for family, steps in sequences:
        for i in range(len(steps) - 1):
            key = f"{family.lower()}|{steps[i]}"
            transitions[key].add(steps[i + 1])
    return dict(transitions)
