"""Robust re-export of the track's generate_sequences module.

generate_sequences.py lives in ../../training_data which is not a package,
so we add it to sys.path once, here, and re-export the symbols we use.
"""
import sys
from pathlib import Path

_TRAINING_DATA = (Path(__file__).resolve().parents[2] / "training_data")
if str(_TRAINING_DATA) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DATA))

import generate_sequences as _gs  # noqa: E402

validate_sequence = _gs.validate_sequence
read_csv_sequences = _gs.read_csv_sequences
write_csv = _gs.write_csv
generate_sequence = _gs.generate_sequence
generate_dataset = _gs.generate_dataset
estimate_combinatorics = _gs.estimate_combinatorics
Violation = _gs.Violation

TRAINING_DATA_DIR = _TRAINING_DATA
RULE_IDS = [
    "RULE_DEP_NO_CLEAN", "RULE_METAL_ETCH_NO_LITHO", "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP", "RULE_IMPLANT_NO_MASK", "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP", "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST", "RULE_BACKSIDE_BEFORE_PASSIVATION",
]
FAMILIES = ["MOSFET", "IGBT", "IC"]
