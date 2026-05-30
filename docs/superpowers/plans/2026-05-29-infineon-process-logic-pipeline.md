# Infineon Process-Logic Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible end-to-end pipeline that learns semiconductor process-flow logic and is benchmarked on the 3 Infineon submission tasks (next-step, completion, anomaly), runnable locally as a smoke test and at scale on Leonardo.

**Architecture:** Two from-scratch HuggingFace models on a custom atomic-step WordLevel tokenizer — a Llama-style decoder (Tasks 1+2) and a DeBERTa-v2-style encoder with binary + 10-way rule heads (Task 3). Training via Accelerate + DeepSpeed ZeRO-2 + SLURM. A dependency-light eval harness self-scores all tasks plus an OOD probe and a "logic probe" (generated completions checked against the real `validate_sequence` rules). n-gram / perplexity / rule-oracle baselines bracket the results.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace `transformers` + `tokenizers` + `accelerate`, DeepSpeed, numpy, matplotlib, (optional) streamlit. Reuses the track's `generate_sequences.py` (grammar, `validate_sequence`, `read_csv_sequences`, `write_csv`).

**Key facts locked from exploration:**
- Working dir for all paths below: repo root `/Users/fathyshalaby/zero_one_hack_01`.
- Track data dir: `tracks/industrial-infineon/training_data/`.
- ~198 distinct STEP strings across families (litho levels already enumerated as literal strings, e.g. `ALIGN MASK LEVEL 3`), so vocab is built directly from the data union — no level synthesis needed.
- Families: `MOSFET`, `IGBT`, `IC`. Variant files: `MOSFET_variants.csv`, `IGBT_variants.csv`, `IC_variants.csv` (long format `SEQUENCE_ID,STEP`, 1000 seqs each).
- Sequences start `RECEIVE WAFER LOT`, end `SHIP LOT`. No commas inside step names.
- `validate_sequence(steps) -> list[Violation]`; `Violation` has `.rule, .description, .step_index, .step_name`. Rule IDs: `RULE_DEP_NO_CLEAN, RULE_METAL_ETCH_NO_LITHO, RULE_ETCH_NO_MASK, RULE_LITHO_LEVEL_SKIP, RULE_IMPLANT_NO_MASK, RULE_CMP_NO_DEP, RULE_PAD_OPEN_BEFORE_DEP, RULE_TEST_BEFORE_PASSIVATION, RULE_SHIP_BEFORE_TEST, RULE_BACKSIDE_BEFORE_PASSIVATION`.

**Conventions for every task:** run `pytest` from `tracks/industrial-infineon/solution/`. Commit after each task with the message shown. All randomness seeded. Artifacts go under `solution/artifacts/` (gitignored).

---

## Phase 0 — Scaffold

### Task 0.1: Create the solution package skeleton

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/__init__.py`
- Create: `tracks/industrial-infineon/solution/procseq/models/__init__.py`
- Create: `tracks/industrial-infineon/solution/tests/__init__.py`
- Create: `tracks/industrial-infineon/solution/requirements.txt`
- Create: `tracks/industrial-infineon/solution/.gitignore`
- Create: `tracks/industrial-infineon/solution/pytest.ini`

- [ ] **Step 1: Create directories and empty package files**

```bash
cd tracks/industrial-infineon/solution
mkdir -p procseq/models tests configs slurm dashboard artifacts
touch procseq/__init__.py procseq/models/__init__.py tests/__init__.py
echo "__version__ = '0.1.0'" > procseq/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
torch>=2.2
transformers>=4.44
tokenizers>=0.19
accelerate>=0.33
deepspeed>=0.14
numpy>=1.26
matplotlib>=3.8
pyyaml>=6.0
pytest>=8.0
streamlit>=1.36
```

- [ ] **Step 3: Write `.gitignore`**

```
artifacts/
__pycache__/
*.pyc
.venv/
wandb/
runs/
*.egg-info/
```

- [ ] **Step 4: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
```

- [ ] **Step 5: Commit**

```bash
git add tracks/industrial-infineon/solution
git commit -m "chore: scaffold procseq solution package"
```

### Task 0.2: Re-export shim for the track's generator

The generator lives in a non-package directory. Provide one robust import point so the rest of the package never manipulates `sys.path` ad-hoc.

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/grammar.py`
- Test: `tracks/industrial-infineon/solution/tests/test_grammar_import.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grammar_import.py
from procseq import grammar

def test_reexports_core_symbols():
    assert callable(grammar.validate_sequence)
    assert callable(grammar.read_csv_sequences)
    assert callable(grammar.generate_dataset)
    # Known-good MOSFET reference-ish sequence validates clean-ish:
    v = grammar.validate_sequence(["RECEIVE WAFER LOT", "SHIP LOT"])
    # SHIP LOT before WAFER SORT TEST -> at least the ship rule fires
    assert any(x.rule == "RULE_SHIP_BEFORE_TEST" for x in v)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tracks/industrial-infineon/solution && python -m pytest tests/test_grammar_import.py -v`
Expected: FAIL (`ModuleNotFoundError: procseq.grammar`).

- [ ] **Step 3: Write `procseq/grammar.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_grammar_import.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tracks/industrial-infineon/solution/procseq/grammar.py tracks/industrial-infineon/solution/tests/test_grammar_import.py
git commit -m "feat: grammar re-export shim for validate_sequence/generators"
```

---

## Phase 1 — Foundation: vocab, tokenizer, data, anomaly injection, metrics, baselines

### Task 1.1: Vocabulary

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/vocab.py`
- Test: `tracks/industrial-infineon/solution/tests/test_vocab.py`

Design: each STEP string becomes one token by replacing spaces with `_`. Special tokens reserved at fixed ids. Vocab built from the union of all distinct STEP strings in the provided variant + reference + longdescr CSVs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vocab.py
from procseq import vocab

def test_step_token_roundtrip():
    assert vocab.step_to_token("RECEIVE WAFER LOT") == "RECEIVE_WAFER_LOT"
    assert vocab.token_to_step("RECEIVE_WAFER_LOT") == "RECEIVE WAFER LOT"

def test_build_vocab_contains_core_steps_and_specials():
    v = vocab.build_vocab()
    assert "RECEIVE_WAFER_LOT" in v
    assert "SHIP_LOT" in v
    assert "ALIGN_MASK_LEVEL_3" in v
    for s in vocab.SPECIAL_TOKENS:
        assert s in v
    # specials occupy the first ids
    assert v["[PAD]"] == 0
    assert len(v) > 150
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vocab.py -v`
Expected: FAIL (`ModuleNotFoundError: procseq.vocab`).

- [ ] **Step 3: Write `procseq/vocab.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vocab.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add procseq/vocab.py tests/test_vocab.py
git commit -m "feat: atomic-step vocabulary builder"
```

### Task 1.2: Tokenizer

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/tokenizer.py`
- Test: `tracks/industrial-infineon/solution/tests/test_tokenizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokenizer.py
from procseq.tokenizer import build_tokenizer, encode_sequence, decode_to_steps

def test_each_step_is_one_token():
    tok = build_tokenizer()
    steps = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "SHIP LOT"]
    ids = encode_sequence(tok, steps, family="MOSFET", add_bos_eos=True)
    # [BOS][FAM_MOSFET] + 3 steps + [EOS] = 6 tokens
    assert len(ids) == 6

def test_roundtrip_steps():
    tok = build_tokenizer()
    steps = ["RECEIVE WAFER LOT", "THERMAL OXIDATION", "SHIP LOT"]
    ids = encode_sequence(tok, steps, family="IGBT", add_bos_eos=False)
    assert decode_to_steps(tok, ids) == steps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tokenizer.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `procseq/tokenizer.py`**

```python
"""Custom atomic-step WordLevel tokenizer (one token per process step)."""
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from procseq import vocab as _vocab

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer"

def build_tokenizer(save_dir: Path | None = None) -> PreTrainedTokenizerFast:
    v = _vocab.build_vocab()
    raw = Tokenizer(WordLevel(vocab=v, unk_token="[UNK]"))
    raw.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )
    tok.add_special_tokens({"additional_special_tokens":
        ["[FAM_MOSFET]", "[FAM_IGBT]", "[FAM_IC]"]})
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(save_dir))
    return tok

def load_tokenizer(save_dir: Path = DEFAULT_DIR) -> PreTrainedTokenizerFast:
    if (save_dir / "tokenizer.json").exists():
        return PreTrainedTokenizerFast.from_pretrained(str(save_dir))
    return build_tokenizer(save_dir)

def _steps_to_text(steps: list[str]) -> str:
    return " ".join(_vocab.step_to_token(s) for s in steps)

def encode_sequence(tok, steps, family=None, add_bos_eos=True) -> list[int]:
    pieces = []
    if add_bos_eos:
        pieces.append(tok.bos_token)
    if family:
        pieces.append(_vocab.FAMILY_TOKEN[family])
    pieces.append(_steps_to_text(steps))
    if add_bos_eos:
        pieces.append(tok.eos_token)
    text = " ".join(p for p in pieces if p)
    return tok.encode(text)

def decode_to_steps(tok, ids) -> list[str]:
    """Inverse of encode_sequence body: drop specials, map tokens->steps."""
    out = []
    specials = set(_vocab.SPECIAL_TOKENS)
    for t in tok.convert_ids_to_tokens(ids):
        if t in specials or t is None:
            continue
        out.append(_vocab.token_to_step(t))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tokenizer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add procseq/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: atomic-step WordLevel tokenizer with family conditioning"
```

### Task 1.3: Data layer — load, scale, dedup, split, UCBS

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/data.py`
- Test: `tracks/industrial-infineon/solution/tests/test_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
from procseq import data

def test_load_provided_returns_1000_each():
    d = data.load_provided("MOSFET")
    assert len(d) == 1000
    assert all(s[0] == "RECEIVE WAFER LOT" and s[-1] == "SHIP LOT"
               for s in list(d.values())[:5])

def test_split_is_deterministic_and_disjoint():
    seqs = {f"seq_{i:04d}": ["RECEIVE WAFER LOT", "SHIP LOT"] for i in range(100)}
    tr, va, te = data.split_ids(list(seqs), val_frac=0.1, test_frac=0.1, seed=7)
    tr2, va2, te2 = data.split_ids(list(seqs), val_frac=0.1, test_frac=0.1, seed=7)
    assert (tr, va, te) == (tr2, va2, te2)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert len(tr) + len(va) + len(te) == 100

def test_ucbs_buckets_balance_lengths():
    seqs = [["a"] * 10] * 5 + [["a"] * 100] * 5
    weights = data.ucbs_weights(seqs, n_buckets=2)
    assert len(weights) == 10
    # short and long get equal total weight
    assert abs(sum(weights[:5]) - sum(weights[5:])) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `procseq/data.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS (note: `test_load_provided` needs the real data files; it reads them directly).

- [ ] **Step 5: Commit**

```bash
git add procseq/data.py tests/test_data.py
git commit -m "feat: data loading, scaling, deterministic splits, UCBS weights"
```

### Task 1.4: Anomaly injection (10 rule violators)

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/anomaly_inject.py`
- Test: `tracks/industrial-infineon/solution/tests/test_anomaly_inject.py`

Each injector takes a valid sequence + RNG, returns a perturbed sequence that triggers a specific rule (verified with `validate_sequence`). `inject_random` picks an applicable injector.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_anomaly_inject.py
import random
from procseq import anomaly_inject as ai
from procseq.data import load_provided
from procseq.grammar import validate_sequence, RULE_IDS

def _a_valid_mosfet():
    for s in load_provided("MOSFET").values():
        if not validate_sequence(s):
            return s
    raise AssertionError("no clean MOSFET sequence found")

def test_each_injector_triggers_its_rule():
    base = _a_valid_mosfet()
    rng = random.Random(0)
    fired_any = False
    for rule, fn in ai.INJECTORS.items():
        res = fn(base, rng)
        if res is None:
            continue  # not applicable to this base sequence
        fired_any = True
        rules = {v.rule for v in validate_sequence(res)}
        assert rule in rules, f"{rule} expected, got {rules}"
    assert fired_any

def test_inject_random_returns_labeled_invalid():
    base = _a_valid_mosfet()
    rng = random.Random(1)
    seq, rule = ai.inject_random(base, rng)
    assert rule in RULE_IDS
    assert rule in {v.rule for v in validate_sequence(seq)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_anomaly_inject.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `procseq/anomaly_inject.py`**

```python
"""Inject each of the 10 process-logic violations into valid sequences.

Each injector returns a NEW perturbed list, or None if not applicable to the
given base sequence. All injectors are verified by validate_sequence in tests.
"""
import random
from procseq.grammar import validate_sequence

# Steps used as anchors (mirrors generate_sequences.py vocab sets).
_DEPOSITIONS = {"THERMAL OXIDATION", "DEPOSIT POLYSILICON", "DEPOSIT BARRIER METAL",
                "DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC",
                "DEPOSIT METAL 1", "DEPOSIT TOP METAL", "DEPOSIT PASSIVATION",
                "DEPOSIT PASSIVATION LAYER", "EPITAXIAL DEPOSITION"}
_CLEANS = {"PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS", "WAFER SURFACE CLEAN",
           "RCA CLEAN 1", "RCA CLEAN 2", "WET CLEAN RCA1", "WET CLEAN RCA2",
           "HF DIP", "OXIDE STRIP", "FRONTSIDE CLEAN", "BACKSIDE CLEAN",
           "CLEAN AFTER ETCH", "CLEAN AFTER OXIDE ETCH", "CLEAN AFTER POLY ETCH",
           "CLEAN AFTER VIA ETCH", "CLEAN AFTER METAL ETCH", "DRY WAFER"}
_ETCHES = {"OXIDE ETCH", "OXIDE ETCH DRY", "POLYSILICON ETCH", "POLYSILICON ETCH DRY",
           "VIA ETCH", "METAL ETCH", "METAL ETCH DRY", "FIELD OXIDE ETCH"}
_IMPLANTS = {"IMPLANT WELL", "IMPLANT SOURCE DRAIN", "IMPLANT LDD", "IMPLANT P BODY",
             "IMPLANT N BUFFER", "IMPLANT N-TYPE", "IMPLANT CHANNEL STOP"}
_CMP = {"CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC", "CMP METAL", "CMP VIA FILL"}
_METAL_ETCH = {"METAL ETCH", "METAL ETCH DRY"}

def _first_index(steps, predicate):
    for i, s in enumerate(steps):
        if predicate(s):
            return i
    return None

def _remove_preceding(steps, trigger_idx, targets, window):
    """Return a copy with all `targets` removed from the window before trigger."""
    lo = max(0, trigger_idx - window)
    return [s for i, s in enumerate(steps)
            if not (lo <= i < trigger_idx and s in targets)]

def inj_dep_no_clean(steps, rng):
    idx = _first_index(steps, lambda s: s in _DEPOSITIONS)
    if idx is None:
        return None
    return _remove_preceding(steps, idx, _CLEANS | {"RAPID THERMAL ANNEAL",
            "THERMAL OXIDATION", "GATE OXIDE PREP", "ANNEAL OXIDE", "EPITAXY ANNEAL"}, 12)

def inj_etch_no_mask(steps, rng):
    idx = _first_index(steps, lambda s: s in _ETCHES)
    if idx is None:
        return None
    return _remove_preceding(steps, idx, {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}, 12)

def inj_metal_etch_no_litho(steps, rng):
    idx = _first_index(steps, lambda s: s in _METAL_ETCH)
    if idx is None:
        return None
    out = _remove_preceding(steps, idx,
        {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}, 15)
    out = [s for i, s in enumerate(out)
           if not (s.startswith("EXPOSE LITHO LEVEL") and
                   out.index(s) < idx and abs(out.index(s) - idx) <= 15)]
    return out

def inj_litho_level_skip(steps, rng):
    # swap the order of two consecutive ALIGN MASK levels
    aligns = [i for i, s in enumerate(steps) if s.startswith("ALIGN MASK LEVEL")]
    if len(aligns) < 2:
        return None
    i, j = aligns[0], aligns[1]
    out = steps[:]
    out[i], out[j] = out[j], out[i]
    return out

def inj_implant_no_mask(steps, rng):
    idx = _first_index(steps, lambda s: s in _IMPLANTS)
    if idx is None:
        return None
    return _remove_preceding(steps, idx,
        {"OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW",
         "DEVELOP PHOTORESIST"}, 15)

def inj_cmp_no_dep(steps, rng):
    idx = _first_index(steps, lambda s: s in _CMP)
    if idx is None:
        return None
    return _remove_preceding(steps, idx,
        _DEPOSITIONS | {"FILL VIA METAL", "FILL VIA TUNGSTEN", "DEPOSIT METAL SEED",
                        "DEPOSIT TUNGSTEN SEED", "DEPOSIT BARRIER METAL"}, 6)

def _move_before(steps, src_pred, dst_pred):
    si = _first_index(steps, src_pred)
    di = _first_index(steps, dst_pred)
    if si is None or di is None or si <= di:
        return None
    out = steps[:]
    item = out.pop(si)
    out.insert(di, item)  # now item appears before dst
    return out

def inj_pad_open_before_dep(steps, rng):
    pad = lambda s: s in {"OPEN PAD WINDOW", "OPEN BOND PAD WINDOW", "PAD WINDOW LITHO"}
    dep = lambda s: s in {"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"}
    return _move_before(steps, pad, dep)

def inj_test_before_passivation(steps, rng):
    test = lambda s: s in {"PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST",
                           "LEAKAGE TEST", "SWITCHING TEST"}
    cure = lambda s: s == "CURE PASSIVATION"
    return _move_before(steps, test, cure)

def inj_ship_before_test(steps, rng):
    ship = lambda s: s == "SHIP LOT"
    sort = lambda s: s == "WAFER SORT TEST"
    return _move_before(steps, ship, sort)

def inj_backside_before_passivation(steps, rng):
    bsm = lambda s: s == "DEPOSIT BACKSIDE METAL"
    cure = lambda s: s == "CURE PASSIVATION"
    return _move_before(steps, bsm, cure)

INJECTORS = {
    "RULE_DEP_NO_CLEAN": inj_dep_no_clean,
    "RULE_ETCH_NO_MASK": inj_etch_no_mask,
    "RULE_METAL_ETCH_NO_LITHO": inj_metal_etch_no_litho,
    "RULE_LITHO_LEVEL_SKIP": inj_litho_level_skip,
    "RULE_IMPLANT_NO_MASK": inj_implant_no_mask,
    "RULE_CMP_NO_DEP": inj_cmp_no_dep,
    "RULE_PAD_OPEN_BEFORE_DEP": inj_pad_open_before_dep,
    "RULE_TEST_BEFORE_PASSIVATION": inj_test_before_passivation,
    "RULE_SHIP_BEFORE_TEST": inj_ship_before_test,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": inj_backside_before_passivation,
}

def inject_random(steps, rng):
    """Try injectors in random order; return (seq, rule) for the first that fires
    its intended rule. Falls back to ship-before-test (always applicable)."""
    rules = list(INJECTORS)
    rng.shuffle(rules)
    for rule in rules:
        res = INJECTORS[rule](steps, rng)
        if res is None:
            continue
        fired = {v.rule for v in validate_sequence(res)}
        if rule in fired:
            return res, rule
    # last resort
    res = inj_ship_before_test(steps, rng)
    return res, "RULE_SHIP_BEFORE_TEST"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_anomaly_inject.py -v`
Expected: PASS. If a specific injector's `assert rule in rules` fails, adjust that injector's window/targets to match the corresponding branch in `generate_sequences.py::validate_sequence` (the test pins behavior to the real checker).

- [ ] **Step 5: Commit**

```bash
git add procseq/anomaly_inject.py tests/test_anomaly_inject.py
git commit -m "feat: 10 rule-violation injectors verified against validate_sequence"
```

### Task 1.5: Eval-mirror builders

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/make_eval.py`
- Test: `tracks/industrial-infineon/solution/tests/test_make_eval.py`

Produces organizer-format files from a held-out split: `eval_input_valid.csv` + `eval_valid_groundtruth.csv`; `eval_input_anomaly.csv` + `eval_anomaly_labels.csv`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_make_eval.py
import csv, random
from pathlib import Path
from procseq import make_eval

def test_valid_mirror_shapes(tmp_path):
    seqs = {f"seq_{i:04d}": ["RECEIVE WAFER LOT"] + ["A"] * 8 + ["SHIP LOT"]
            for i in range(10)}
    rows = make_eval.build_valid_rows(seqs, family="MOSFET",
                                      fractions=(0.6, 0.8), rng=random.Random(0))
    assert len(rows) == 20  # 10 seqs x 2 fractions
    r = rows[0]
    assert set(r) >= {"EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION",
                      "PARTIAL_SEQUENCE", "FULL_SEQUENCE"}
    assert "|" in r["PARTIAL_SEQUENCE"]

def test_anomaly_mirror_balance(tmp_path):
    valid = {f"v_{i}": ["RECEIVE WAFER LOT", "PRE CLEAN WAFER", "THERMAL OXIDATION",
                        "CURE PASSIVATION", "WAFER SORT TEST", "SHIP LOT"]
             for i in range(20)}
    rows = make_eval.build_anomaly_rows(valid, family="MOSFET",
                                        n_valid=10, n_invalid=8, rng=random.Random(0))
    assert len(rows) == 18
    n_inv = sum(1 for r in rows if r["IS_VALID"] == 0)
    assert n_inv == 8
    assert all(r["PREDICTED_RULE"] for r in rows if r["IS_VALID"] == 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_make_eval.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `procseq/make_eval.py`**

```python
"""Build organizer-format internal eval mirrors from held-out sequences."""
import csv
import math
import random
from pathlib import Path
from procseq.anomaly_inject import inject_random

def _cut(steps: list[str], frac: float) -> int:
    return max(1, int(math.floor(len(steps) * frac)))

def build_valid_rows(seqs, family, fractions, rng):
    rows = []
    for sid, steps in seqs.items():
        for frac in fractions:
            k = _cut(steps, frac)
            rows.append({
                "EXAMPLE_ID": f"{family}_{sid}_{int(frac*100)}",
                "FAMILY": family,
                "COMPLETION_FRACTION": frac,
                "PARTIAL_SEQUENCE": "|".join(steps[:k]),
                "FULL_SEQUENCE": "|".join(steps),
            })
    return rows

def build_anomaly_rows(seqs, family, n_valid, n_invalid, rng):
    items = list(seqs.values())
    rng.shuffle(items)
    rows = []
    for i in range(min(n_valid, len(items))):
        rows.append({"EXAMPLE_ID": f"{family}_valid_{i:04d}", "FAMILY": family,
                     "SEQUENCE": "|".join(items[i]), "IS_VALID": 1, "PREDICTED_RULE": ""})
    pool = items[n_valid:] or items
    for i in range(n_invalid):
        base = pool[i % len(pool)]
        seq, rule = inject_random(base, rng)
        rows.append({"EXAMPLE_ID": f"{family}_inval_{i:04d}", "FAMILY": family,
                     "SEQUENCE": "|".join(seq), "IS_VALID": 0, "PREDICTED_RULE": rule})
    rng.shuffle(rows)
    return rows

def write_valid_files(rows, input_path: Path, gt_path: Path):
    with input_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"])
        for r in rows:
            w.writerow([r["EXAMPLE_ID"], r["FAMILY"], r["COMPLETION_FRACTION"],
                        r["PARTIAL_SEQUENCE"]])
    with gt_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "FULL_SEQUENCE"])
        for r in rows:
            w.writerow([r["EXAMPLE_ID"], r["FULL_SEQUENCE"]])

def write_anomaly_files(rows, input_path: Path, labels_path: Path):
    with input_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "FAMILY", "SEQUENCE"])
        for r in rows:
            w.writerow([r["EXAMPLE_ID"], r["FAMILY"], r["SEQUENCE"]])
    with labels_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "IS_VALID", "PREDICTED_RULE"])
        for r in rows:
            w.writerow([r["EXAMPLE_ID"], r["IS_VALID"], r["PREDICTED_RULE"]])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_make_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add procseq/make_eval.py tests/test_make_eval.py
git commit -m "feat: organizer-format eval-mirror builders (valid + anomaly)"
```

### Task 1.6: Eval metrics harness

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/eval_metrics.py`
- Test: `tracks/industrial-infineon/solution/tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_metrics.py
from procseq import eval_metrics as em

def test_mrr_and_topk():
    preds = {"a": ["X", "Y", "Z", "P", "Q"], "b": ["Y", "X", "Z", "P", "Q"]}
    gold = {"a": "X", "b": "X"}
    r = em.score_nextstep(preds, gold)
    assert r["top1"] == 0.5
    assert r["top3"] == 1.0
    assert abs(r["mrr"] - ((1.0 + 0.5) / 2)) < 1e-9

def test_normalized_edit_distance_identity():
    assert em.normalized_edit_distance(["a", "b"], ["a", "b"]) == 1.0
    assert em.normalized_edit_distance(["a", "b"], ["a", "c"]) == 0.5

def test_completion_exact_match_and_token_acc():
    preds = {"a": ["c", "d"], "b": ["c", "x"]}
    gold = {"a": ["c", "d"], "b": ["c", "d"]}
    r = em.score_completion(preds, gold)
    assert r["exact_match"] == 0.5
    assert abs(r["token_accuracy"] - 0.75) < 1e-9

def test_anomaly_f1_and_auc():
    # IS_VALID convention: 1 valid, 0 invalid; positive class = invalid
    pred = {"a": (0, 0.1, "RULE_DEP_NO_CLEAN"), "b": (1, 0.9, ""),
            "c": (0, 0.2, "RULE_ETCH_NO_MASK"), "d": (1, 0.8, "")}
    gold = {"a": (0, "RULE_DEP_NO_CLEAN"), "b": (1, ""),
            "c": (0, "RULE_CMP_NO_DEP"), "d": (1, "")}
    r = em.score_anomaly(pred, gold)
    assert r["binary_accuracy"] == 1.0
    assert r["f1"] == 1.0
    assert r["rule_attribution_accuracy"] == 0.5  # 1 of 2 invalids matched
    assert 0.0 <= r["roc_auc"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_metrics.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `procseq/eval_metrics.py`**

```python
"""Self-scoring harness for Tasks 1-3 + OOD + logic probe. numpy-only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

# ---------- Task 1: next-step ----------
def score_nextstep(preds: dict[str, list[str]], gold: dict[str, str]) -> dict:
    ids = [i for i in gold if i in preds]
    top1 = top3 = top5 = 0.0
    mrr = 0.0
    for i in ids:
        ranked = preds[i]
        g = gold[i]
        rank = ranked.index(g) + 1 if g in ranked else None
        if rank == 1: top1 += 1
        if rank and rank <= 3: top3 += 1
        if rank and rank <= 5: top5 += 1
        mrr += (1.0 / rank) if rank else 0.0
    n = max(1, len(ids))
    return {"n": len(ids), "top1": top1/n, "top3": top3/n, "top5": top5/n, "mrr": mrr/n}

# ---------- Task 2: completion ----------
def _levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (a[i-1] != b[j-1]))
            prev = cur
    return dp[n]

def normalized_edit_distance(a, b):
    if not a and not b: return 1.0
    return 1.0 - _levenshtein(a, b) / max(len(a), len(b))

def _block_signature(steps):
    """Collapse consecutive duplicate category prefixes into ordered blocks."""
    sig = []
    for s in steps:
        cat = s.split()[0]
        if not sig or sig[-1] != cat:
            sig.append(cat)
    return sig

def score_completion(preds, gold):
    ids = [i for i in gold if i in preds]
    em = ned = tok = blk = 0.0
    for i in ids:
        p, g = preds[i], gold[i]
        em += 1.0 if p == g else 0.0
        ned += normalized_edit_distance(p, g)
        L = max(1, len(g))
        tok += sum(1 for k in range(min(len(p), len(g))) if p[k] == g[k]) / L
        blk += normalized_edit_distance(_block_signature(p), _block_signature(g))
    n = max(1, len(ids))
    return {"n": len(ids), "exact_match": em/n, "normalized_edit_distance": ned/n,
            "token_accuracy": tok/n, "block_accuracy": blk/n}

# ---------- Task 3: anomaly ----------
def _auc(scores, labels):
    """scores = P(valid); positive class = invalid (label 0). AUC for detecting invalid."""
    inv = np.array([1 - l for l in labels])  # 1 = invalid (positive)
    s = -np.array(scores)                     # higher => more likely invalid
    order = np.argsort(s)
    inv = inv[order]
    P = inv.sum(); N = len(inv) - P
    if P == 0 or N == 0: return 0.5
    ranks = np.arange(1, len(inv) + 1)
    auc = (ranks[inv == 1].sum() - P * (P + 1) / 2) / (P * N)
    return float(auc)

def score_anomaly(pred, gold):
    ids = [i for i in gold if i in pred]
    tp = fp = tn = fn = 0
    rule_hit = rule_tot = 0
    scores, labels = [], []
    for i in ids:
        pv, ps, pr = pred[i]
        gv, gr = gold[i]
        scores.append(ps); labels.append(gv)
        if gv == 0 and pv == 0: tp += 1
        elif gv == 1 and pv == 0: fp += 1
        elif gv == 1 and pv == 1: tn += 1
        elif gv == 0 and pv == 1: fn += 1
        if gv == 0:
            rule_tot += 1
            if pv == 0 and pr == gr: rule_hit += 1
    n = max(1, len(ids))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"n": len(ids), "binary_accuracy": (tp + tn) / n,
            "precision": prec, "recall": rec, "f1": f1,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "roc_auc": _auc(scores, labels),
            "rule_attribution_accuracy": rule_hit / max(1, rule_tot)}

# ---------- logic probe ----------
def logic_validity_rate(generated_full_sequences):
    """Fraction of generated full sequences with zero rule violations."""
    from procseq.grammar import validate_sequence
    ok = sum(1 for s in generated_full_sequences if not validate_sequence(s))
    return ok / max(1, len(generated_full_sequences))

# ---------- CLI ----------
def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["nextstep", "completion", "anomaly"], required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    gt = _read_csv(a.ground_truth); pr = _read_csv(a.predictions)
    if a.task == "nextstep":
        gold = {r["EXAMPLE_ID"]: r["NEXT_STEP"] for r in gt}
        preds = {r["EXAMPLE_ID"]: [r[f"RANK_{k}"] for k in range(1, 6)] for r in pr}
        res = score_nextstep(preds, gold)
    elif a.task == "completion":
        gold = {r["EXAMPLE_ID"]: r["SUFFIX"].split("|") for r in gt}
        preds = {r["EXAMPLE_ID"]: r["PREDICTED_SEQUENCE"].split("|") for r in pr}
        res = score_completion(preds, gold)
    else:
        gold = {r["EXAMPLE_ID"]: (int(r["IS_VALID"]), r["PREDICTED_RULE"]) for r in gt}
        preds = {r["EXAMPLE_ID"]: (int(r["IS_VALID"]), float(r.get("SCORE", 0.5) or 0.5),
                                   r.get("PREDICTED_RULE", "")) for r in pr}
        res = score_anomaly(preds, gold)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add procseq/eval_metrics.py tests/test_eval_metrics.py
git commit -m "feat: dependency-light eval harness (tasks 1-3 + logic probe)"
```

### Task 1.7: Baselines (n-gram + rule-oracle)

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/baselines.py`
- Test: `tracks/industrial-infineon/solution/tests/test_baselines.py`

(Perplexity baseline is added in Phase 2 once the decoder exists.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
from procseq import baselines

def test_ngram_predicts_seen_transition():
    seqs = [["A", "B", "C"], ["A", "B", "D"], ["A", "B", "C"]]
    ng = baselines.NgramModel(n=2).fit(seqs)
    top = ng.predict_next(["A", "B"], k=5)
    assert top[0] == "C"  # C seen twice after A,B vs D once

def test_rule_oracle_flags_invalid():
    seq = ["RECEIVE WAFER LOT", "SHIP LOT", "WAFER SORT TEST"]
    is_valid, rule = baselines.rule_oracle(seq)
    assert is_valid == 0 and rule == "RULE_SHIP_BEFORE_TEST"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baselines.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `procseq/baselines.py`**

```python
"""Statistical baselines: n-gram next-step + rule-oracle anomaly ceiling."""
from collections import Counter, defaultdict
from procseq.grammar import validate_sequence

class NgramModel:
    def __init__(self, n=3):
        self.n = n
        self.table: dict[tuple, Counter] = defaultdict(Counter)
        self.unigram = Counter()

    def fit(self, sequences):
        for seq in sequences:
            for s in seq:
                self.unigram[s] += 1
            for i in range(len(seq)):
                for k in range(1, self.n):
                    if i - k >= 0:
                        ctx = tuple(seq[i-k:i])
                        self.table[ctx][seq[i]] += 1
        return self

    def predict_next(self, prefix, k=5):
        for back in range(self.n - 1, 0, -1):
            ctx = tuple(prefix[-back:]) if back <= len(prefix) else None
            if ctx and self.table.get(ctx):
                return [s for s, _ in self.table[ctx].most_common(k)]
        return [s for s, _ in self.unigram.most_common(k)]

def rule_oracle(steps):
    """Ground-truth checker as an anomaly ceiling. Returns (is_valid, rule)."""
    v = validate_sequence(steps)
    if not v:
        return 1, ""
    return 0, v[0].rule
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baselines.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add procseq/baselines.py tests/test_baselines.py
git commit -m "feat: n-gram next-step baseline + rule-oracle anomaly ceiling"
```

### Task 1.8: Pipeline driver + Makefile smoke (data + baselines + metrics)

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/build_data.py`
- Create: `tracks/industrial-infineon/solution/Makefile`
- Test: manual smoke (no unit test; produces artifacts)

- [ ] **Step 1: Write `procseq/build_data.py`**

```python
"""End-to-end data prep: scale, split, build tokenizer + eval mirrors."""
import argparse
import random
from pathlib import Path
from procseq import data, make_eval
from procseq.tokenizer import build_tokenizer, DEFAULT_DIR
from procseq.grammar import FAMILIES, write_csv

ART = Path(__file__).resolve().parents[1] / "artifacts"

def run(n_per_family: int, seed: int, smoke: bool):
    ART.mkdir(exist_ok=True)
    rng = random.Random(seed)
    build_tokenizer(DEFAULT_DIR)
    eval_valid_rows, anomaly_rows = [], []
    for fam in FAMILIES:
        seqs = {f"{fam}_{i:05d}": s for i, s in
                enumerate(data.scale_family(fam, n_per_family, seed))}
        tr, va, te = data.split_ids(list(seqs), 0.1, 0.1, seed)
        (ART / "splits").mkdir(exist_ok=True)
        write_csv(ART / "splits" / f"{fam}_train.csv", [seqs[i] for i in tr])
        write_csv(ART / "splits" / f"{fam}_val.csv", [seqs[i] for i in va])
        write_csv(ART / "splits" / f"{fam}_test.csv", [seqs[i] for i in te])
        held = {i: seqs[i] for i in te}
        k = 5 if smoke else 100
        sample_ids = list(held)[:k]
        eval_valid_rows += make_eval.build_valid_rows(
            {i: held[i] for i in sample_ids}, fam, (0.6, 0.8), rng)
        nv, ni = (5, 3) if smoke else (200, 129)
        anomaly_rows += make_eval.build_anomaly_rows(held, fam, nv, ni, rng)
    make_eval.write_valid_files(eval_valid_rows,
        ART / "eval_input_valid.csv", ART / "eval_valid_groundtruth.csv")
    make_eval.write_anomaly_files(anomaly_rows,
        ART / "eval_input_anomaly.csv", ART / "eval_anomaly_labels.csv")
    print(f"Built data: {len(eval_valid_rows)} valid-eval rows, "
          f"{len(anomaly_rows)} anomaly rows -> {ART}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.n_per_family if not a.smoke else 20, a.seed, a.smoke)
```

- [ ] **Step 2: Write `Makefile`**

```makefile
# Use the local venv python if present (local smoke); override on the cluster, e.g. `make smoke PY=python`.
PY ?= $(shell [ -x ./.venv/bin/python ] && echo ./.venv/bin/python || echo python)
SOL := $(CURDIR)

.PHONY: test smoke data train-decoder train-encoder eval submit demo clean

test:
	$(PY) -m pytest -q

data:
	$(PY) -m procseq.build_data --n-per-family 2000 --seed 42

smoke:
	$(PY) -m pytest -q
	$(PY) -m procseq.build_data --smoke
	$(PY) -m procseq.train_decoder --config configs/smoke.yaml
	$(PY) -m procseq.train_encoder --config configs/smoke.yaml
	$(PY) -m procseq.infer --all --config configs/smoke.yaml
	$(PY) -m procseq.run_eval --config configs/smoke.yaml
	@echo "SMOKE OK"

train-decoder:
	accelerate launch -m procseq.train_decoder --config configs/leonardo_decoder.yaml

train-encoder:
	accelerate launch -m procseq.train_encoder --config configs/leonardo_encoder.yaml

eval:
	$(PY) -m procseq.run_eval --config configs/leonardo_decoder.yaml

demo:
	$(PY) -m procseq.demo --config configs/leonardo_decoder.yaml

clean:
	rm -rf artifacts/* runs/*
```

- [ ] **Step 3: Run the partial smoke (data only) to verify it produces artifacts**

Run: `cd tracks/industrial-infineon/solution && python -m procseq.build_data --smoke`
Expected: prints "Built data: ... rows", creates `artifacts/eval_input_valid.csv`, `artifacts/eval_input_anomaly.csv`, `artifacts/splits/*`, `artifacts/tokenizer/tokenizer.json`.

- [ ] **Step 4: Commit**

```bash
git add procseq/build_data.py Makefile
git commit -m "feat: data-build driver + Makefile (smoke wires later phases)"
```

---

## Phase 2 — Decoder (Tasks 1 & 2)

### Task 2.1: Config loader + decoder model builder

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/config.py`
- Create: `tracks/industrial-infineon/solution/procseq/models/decoder.py`
- Create: `tracks/industrial-infineon/solution/configs/smoke.yaml`
- Test: `tracks/industrial-infineon/solution/tests/test_decoder_build.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decoder_build.py
from procseq.tokenizer import build_tokenizer
from procseq.models.decoder import build_decoder

def test_build_decoder_tiny_has_correct_vocab():
    tok = build_tokenizer()
    model = build_decoder(size="tiny", tokenizer=tok)
    assert model.config.vocab_size == len(tok)
    assert model.config.hidden_size == 128
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decoder_build.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `procseq/config.py`**

```python
"""YAML config loader with dotted access."""
from pathlib import Path
import yaml

class Config(dict):
    __getattr__ = dict.get

def load_config(path) -> Config:
    data = yaml.safe_load(Path(path).read_text())
    return Config(data)
```

- [ ] **Step 4: Write `procseq/models/decoder.py`**

```python
"""From-scratch Llama-style causal LM sized by a named preset."""
from transformers import LlamaConfig, LlamaForCausalLM

SIZES = {
    "tiny":  dict(hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=4),
    "small": dict(hidden_size=256, intermediate_size=768, num_hidden_layers=6,
                  num_attention_heads=8, num_key_value_heads=8),
    "base":  dict(hidden_size=512, intermediate_size=1536, num_hidden_layers=8,
                  num_attention_heads=8, num_key_value_heads=8),
    "large": dict(hidden_size=768, intermediate_size=2304, num_hidden_layers=12,
                  num_attention_heads=12, num_key_value_heads=12),
}

def build_decoder(size, tokenizer, max_position_embeddings=256):
    p = SIZES[size]
    cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=max_position_embeddings,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
        **p,
    )
    return LlamaForCausalLM(cfg)
```

- [ ] **Step 5: Write `configs/smoke.yaml`**

```yaml
run_name: smoke
seed: 42
device: auto            # auto -> cuda if available else cpu/mps
precision: fp32         # smoke runs fp32 for CPU/MPS portability
decoder:
  size: tiny
  max_len: 256
  data_per_family: 20
  batch_size: 4
  lr: 0.003
  max_steps: 5
  eval_every: 5
encoder:
  size: tiny
  max_len: 256
  batch_size: 4
  lr: 0.003
  max_steps: 5
artifacts: artifacts
decoder_ckpt: artifacts/decoder
encoder_ckpt: artifacts/encoder
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_decoder_build.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add procseq/config.py procseq/models/decoder.py configs/smoke.yaml tests/test_decoder_build.py
git commit -m "feat: config loader + Llama-style decoder builder + smoke config"
```

### Task 2.2: CLM dataset + collator

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/datasets.py`
- Test: `tracks/industrial-infineon/solution/tests/test_datasets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_datasets.py
from procseq.tokenizer import build_tokenizer
from procseq.datasets import ClmDataset, clm_collate

def test_clm_dataset_item_has_input_ids_and_family():
    tok = build_tokenizer()
    seqs = [(["RECEIVE WAFER LOT", "SHIP LOT"], "MOSFET")]
    ds = ClmDataset(seqs, tok, max_len=16)
    item = ds[0]
    assert "input_ids" in item and item["input_ids"][0] == tok.bos_token_id

def test_clm_collate_pads_and_masks_labels():
    tok = build_tokenizer()
    seqs = [(["RECEIVE WAFER LOT", "SHIP LOT"], "MOSFET"),
            (["RECEIVE WAFER LOT", "THERMAL OXIDATION", "SHIP LOT"], "IC")]
    ds = ClmDataset(seqs, tok, max_len=16)
    batch = clm_collate([ds[0], ds[1]], pad_id=tok.pad_token_id)
    assert batch["input_ids"].shape[0] == 2
    # labels == -100 wherever input is pad
    import torch
    assert (batch["labels"][batch["input_ids"] == tok.pad_token_id] == -100).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_datasets.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `procseq/datasets.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_datasets.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add procseq/datasets.py tests/test_datasets.py
git commit -m "feat: CLM + classification datasets and collators"
```

### Task 2.3: Decoder training entrypoint (Accelerate)

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/train_decoder.py`
- Test: covered by smoke (`make smoke`)

- [ ] **Step 1: Write `procseq/train_decoder.py`**

```python
"""Train the decoder with Accelerate (DeepSpeed via accelerate config on cluster)."""
import argparse
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from accelerate import Accelerator
from accelerate.utils import set_seed
from procseq.config import load_config
from procseq.tokenizer import build_tokenizer, DEFAULT_DIR
from procseq.models.decoder import build_decoder
from procseq.datasets import ClmDataset, clm_collate
from procseq.data import scale_family, ucbs_weights
from procseq.grammar import FAMILIES

def _load_training_pairs(n_per_family, seed):
    pairs = []
    for fam in FAMILIES:
        for s in scale_family(fam, n_per_family, seed):
            pairs.append((s, fam))
    return pairs

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config); dc = cfg["decoder"]
    set_seed(cfg.get("seed", 42))
    acc = Accelerator(mixed_precision=("bf16" if cfg.get("precision")=="bf16" else "no"),
                      log_with="tensorboard", project_dir=cfg.get("artifacts", "artifacts"))
    acc.init_trackers(cfg.get("run_name", "decoder"))
    tok = build_tokenizer(DEFAULT_DIR)
    model = build_decoder(dc["size"], tok, dc.get("max_len", 256))
    pairs = _load_training_pairs(dc.get("data_per_family", 20), cfg.get("seed", 42))
    ds = ClmDataset(pairs, tok, dc.get("max_len", 256))
    weights = ucbs_weights([p[0] for p in pairs])
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    dl = DataLoader(ds, batch_size=dc.get("batch_size", 4), sampler=sampler,
                    collate_fn=partial(clm_collate, pad_id=tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=dc.get("lr", 3e-3))
    model, opt, dl = acc.prepare(model, opt, dl)
    model.train(); step = 0; max_steps = dc.get("max_steps", 5)
    while step < max_steps:
        for batch in dl:
            out = model(**batch); loss = out.loss
            acc.backward(loss); opt.step(); opt.zero_grad()
            acc.log({"train/loss": loss.item()}, step=step)
            step += 1
            if step >= max_steps: break
    out_dir = Path(cfg.get("decoder_ckpt", "artifacts/decoder"))
    acc.wait_for_everyone()
    unwrapped = acc.unwrap_model(model)
    unwrapped.save_pretrained(out_dir, save_function=acc.save)
    tok.save_pretrained(out_dir)
    acc.end_training()
    if acc.is_main_process:
        print(f"Saved decoder -> {out_dir}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a tiny train to verify it completes**

Run: `cd tracks/industrial-infineon/solution && python -m procseq.train_decoder --config configs/smoke.yaml`
Expected: prints "Saved decoder -> artifacts/decoder"; `artifacts/decoder/` contains `config.json`, model weights, tokenizer.

- [ ] **Step 3: Commit**

```bash
git add procseq/train_decoder.py
git commit -m "feat: Accelerate decoder training entrypoint (CLM + UCBS sampler)"
```

### Task 2.4: Inference for Tasks 1 & 2 + perplexity baseline

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/infer.py`
- Modify: `tracks/industrial-infineon/solution/procseq/baselines.py` (add `PerplexityAnomaly`)
- Test: `tracks/industrial-infineon/solution/tests/test_infer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infer.py
from procseq.tokenizer import build_tokenizer
from procseq.models.decoder import build_decoder
from procseq import infer

def test_nextstep_returns_5_valid_steps():
    tok = build_tokenizer()
    model = build_decoder("tiny", tok)
    partial = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"]
    ranked = infer.predict_next_step(model, tok, partial, "MOSFET", k=5)
    assert len(ranked) == 5
    assert all(isinstance(s, str) and "_" not in s for s in ranked)  # decoded to spaces

def test_completion_stops_and_returns_suffix():
    tok = build_tokenizer()
    model = build_decoder("tiny", tok)
    partial = ["RECEIVE WAFER LOT"]
    suffix = infer.complete_sequence(model, tok, partial, "IC", max_new=10)
    assert isinstance(suffix, list)
    assert len(suffix) <= 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_infer.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `procseq/infer.py`**

```python
"""Inference: produce the 3 submission CSVs from eval_input files."""
import argparse
import csv
from pathlib import Path
import torch
from procseq.config import load_config
from procseq.tokenizer import load_tokenizer, encode_sequence, decode_to_steps
from procseq.vocab import SPECIAL_TOKENS, token_to_step

@torch.no_grad()
def _next_logits(model, tok, steps, family):
    ids = encode_sequence(tok, steps, family=family, add_bos_eos=False)
    # prepend BOS + family handled inside encode when add_bos_eos True; here we add BOS+fam manually
    from procseq.vocab import FAMILY_TOKEN
    prefix = tok.encode(" ".join([tok.bos_token, FAMILY_TOKEN[family]]))
    ids = prefix + ids
    x = torch.tensor([ids], device=next(model.parameters()).device)
    out = model(input_ids=x)
    return out.logits[0, -1]

def predict_next_step(model, tok, partial_steps, family, k=5):
    model.eval()
    logits = _next_logits(model, tok, partial_steps, family)
    special_ids = {tok.convert_tokens_to_ids(s) for s in SPECIAL_TOKENS}
    order = torch.argsort(logits, descending=True).tolist()
    out = []
    for tid in order:
        if tid in special_ids:
            continue
        tokn = tok.convert_ids_to_tokens(tid)
        out.append(token_to_step(tokn))
        if len(out) == k:
            break
    return out

def complete_sequence(model, tok, partial_steps, family, max_new=200):
    model.eval()
    steps = list(partial_steps)
    eos = tok.eos_token_id
    from procseq.vocab import FAMILY_TOKEN
    base = tok.encode(" ".join([tok.bos_token, FAMILY_TOKEN[family]]))
    ids = base + encode_sequence(tok, partial_steps, family=None, add_bos_eos=False)
    dev = next(model.parameters()).device
    produced = []
    for _ in range(max_new):
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            nxt = int(model(input_ids=x).logits[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        tokn = tok.convert_ids_to_tokens(nxt)
        if tokn in SPECIAL_TOKENS:
            continue
        produced.append(token_to_step(tokn))
    return produced

def _load_decoder(ckpt):
    from transformers import LlamaForCausalLM
    return LlamaForCausalLM.from_pretrained(ckpt)

def run_task1(cfg):
    tok = load_tokenizer(Path(cfg["decoder_ckpt"]))
    model = _load_decoder(cfg["decoder_ckpt"])
    art = Path(cfg["artifacts"])
    rows_out = []
    with (art / "eval_input_valid.csv").open() as f:
        for r in csv.DictReader(f):
            steps = r["PARTIAL_SEQUENCE"].split("|")
            ranked = predict_next_step(model, tok, steps, r["FAMILY"], k=5)
            ranked += [""] * (5 - len(ranked))
            rows_out.append([r["EXAMPLE_ID"], *ranked[:5]])
    out = art / "submission_task1.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["EXAMPLE_ID","RANK_1","RANK_2","RANK_3","RANK_4","RANK_5"])
        w.writerows(rows_out)
    print(f"Task1 -> {out} ({len(rows_out)} rows)")

def run_task2(cfg):
    tok = load_tokenizer(Path(cfg["decoder_ckpt"]))
    model = _load_decoder(cfg["decoder_ckpt"])
    art = Path(cfg["artifacts"])
    rows_out = []
    with (art / "eval_input_valid.csv").open() as f:
        for r in csv.DictReader(f):
            steps = r["PARTIAL_SEQUENCE"].split("|")
            suffix = complete_sequence(model, tok, steps, r["FAMILY"],
                                       max_new=cfg["decoder"].get("max_len", 256))
            rows_out.append([r["EXAMPLE_ID"], "|".join(suffix)])
    out = art / "submission_task2.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["EXAMPLE_ID","PREDICTED_SEQUENCE"])
        w.writerows(rows_out)
    print(f"Task2 -> {out} ({len(rows_out)} rows)")

def run_task3(cfg):
    from procseq.infer_anomaly import run_anomaly
    run_anomaly(cfg)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--task", choices=["1","2","3"])
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.all or a.task == "1": run_task1(cfg)
    if a.all or a.task == "2": run_task2(cfg)
    if a.all or a.task == "3": run_task3(cfg)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add `PerplexityAnomaly` to `procseq/baselines.py`**

Append:

```python
import math
import torch

class PerplexityAnomaly:
    """Anomaly score = decoder NLL per token; threshold on validation."""
    def __init__(self, model, tokenizer):
        self.model = model.eval(); self.tok = tokenizer; self.threshold = None

    @torch.no_grad()
    def nll(self, steps, family):
        from procseq.tokenizer import encode_sequence
        ids = encode_sequence(self.tok, steps, family=family, add_bos_eos=True)
        x = torch.tensor([ids], device=next(self.model.parameters()).device)
        out = self.model(input_ids=x, labels=x)
        return float(out.loss)

    def fit_threshold(self, valid_examples, quantile=0.95):
        scores = sorted(self.nll(s, f) for s, f in valid_examples)
        idx = min(len(scores) - 1, int(quantile * len(scores)))
        self.threshold = scores[idx]
        return self

    def predict(self, steps, family):
        score = self.nll(steps, family)
        is_valid = 1 if (self.threshold is None or score <= self.threshold) else 0
        p_valid = 1.0 / (1.0 + math.exp(score - (self.threshold or score)))
        return is_valid, p_valid
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_infer.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add procseq/infer.py procseq/baselines.py tests/test_infer.py
git commit -m "feat: Task1/2 inference + submission CSVs + perplexity baseline"
```

---

## Phase 3 — Encoder (Task 3)

### Task 3.1: Encoder model builder (DeBERTa + 2 heads)

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/models/encoder.py`
- Test: `tracks/industrial-infineon/solution/tests/test_encoder_build.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_encoder_build.py
import torch
from procseq.tokenizer import build_tokenizer
from procseq.models.encoder import build_encoder
from procseq.grammar import RULE_IDS

def test_encoder_forward_shapes():
    tok = build_tokenizer()
    model = build_encoder("tiny", tok, n_rules=len(RULE_IDS))
    ids = torch.tensor([[tok.cls_token_id, 5, 6, tok.sep_token_id]])
    mask = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=mask)
    assert out["invalid_logit"].shape == (1,)
    assert out["rule_logits"].shape == (1, len(RULE_IDS))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_encoder_build.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `procseq/models/encoder.py`**

```python
"""From-scratch DeBERTa-v2-style encoder with binary + multi-label rule heads."""
import torch
import torch.nn as nn
from transformers import DebertaV2Config, DebertaV2Model

SIZES = {
    "tiny":  dict(hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                  num_attention_heads=4),
    "small": dict(hidden_size=256, intermediate_size=768, num_hidden_layers=6,
                  num_attention_heads=8),
    "base":  dict(hidden_size=512, intermediate_size=1536, num_hidden_layers=8,
                  num_attention_heads=8),
}

class ProcessAnomalyModel(nn.Module):
    def __init__(self, cfg, n_rules):
        super().__init__()
        self.encoder = DebertaV2Model(cfg)
        h = cfg.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.invalid_head = nn.Linear(h, 1)
        self.rule_head = nn.Linear(h, n_rules)

    def forward(self, input_ids, attention_mask=None, **_):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]   # [CLS]
        pooled = self.dropout(pooled)
        return {"invalid_logit": self.invalid_head(pooled).squeeze(-1),
                "rule_logits": self.rule_head(pooled)}

def build_encoder(size, tokenizer, n_rules, max_position_embeddings=256):
    p = SIZES[size]
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer), max_position_embeddings=max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id, relative_attention=True,
        pos_att_type=["p2c", "c2p"], **p,
    )
    return ProcessAnomalyModel(cfg, n_rules)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_encoder_build.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add procseq/models/encoder.py tests/test_encoder_build.py
git commit -m "feat: DeBERTa-style anomaly encoder with binary + rule heads"
```

### Task 3.2: Encoder training entrypoint

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/train_encoder.py`
- Create: `tracks/industrial-infineon/solution/procseq/anomaly_data.py`
- Test: covered by smoke

- [ ] **Step 1: Write `procseq/anomaly_data.py`**

```python
"""Build a balanced (valid, invalid+rule) training set for the encoder."""
import random
from procseq.data import scale_family
from procseq.anomaly_inject import inject_random
from procseq.grammar import FAMILIES, validate_sequence

def build_anomaly_training(n_per_family, seed):
    rng = random.Random(seed)
    items = []  # (steps, family, is_valid, rule)
    for fam in FAMILIES:
        seqs = scale_family(fam, n_per_family, seed)
        for s in seqs:
            if validate_sequence(s):
                continue  # keep the valid pool clean
            # (rare) skip already-invalid generated seqs
        for s in seqs:
            items.append((s, fam, 1, ""))                 # valid
            inv, rule = inject_random(s, rng)
            items.append((inv, fam, 0, rule))             # matched invalid
    rng.shuffle(items)
    return items
```

- [ ] **Step 2: Write `procseq/train_encoder.py`**

```python
"""Train the anomaly encoder (binary BCE + multi-label rule BCE)."""
import argparse
from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from procseq.config import load_config
from procseq.tokenizer import build_tokenizer, DEFAULT_DIR
from procseq.models.encoder import build_encoder
from procseq.datasets import ClsDataset, cls_collate
from procseq.anomaly_data import build_anomaly_training
from procseq.grammar import RULE_IDS

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(argv); cfg = load_config(a.config); ec = cfg["encoder"]
    set_seed(cfg.get("seed", 42))
    acc = Accelerator(mixed_precision=("bf16" if cfg.get("precision")=="bf16" else "no"),
                      log_with="tensorboard", project_dir=cfg.get("artifacts","artifacts"))
    acc.init_trackers(cfg.get("run_name","encoder") + "_enc")
    tok = build_tokenizer(DEFAULT_DIR)
    model = build_encoder(ec["size"], tok, n_rules=len(RULE_IDS), max_position_embeddings=ec.get("max_len",256))
    items = build_anomaly_training(ec.get("data_per_family", 20), cfg.get("seed", 42))
    ds = ClsDataset(items, tok, RULE_IDS, ec.get("max_len", 256))
    dl = DataLoader(ds, batch_size=ec.get("batch_size",4), shuffle=True,
                    collate_fn=partial(cls_collate, pad_id=tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=ec.get("lr", 3e-3))
    bce = nn.BCEWithLogitsLoss()
    model, opt, dl = acc.prepare(model, opt, dl)
    model.train(); step = 0; max_steps = ec.get("max_steps", 5)
    while step < max_steps:
        for b in dl:
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            loss = bce(out["invalid_logit"], b["invalid"]) + \
                   bce(out["rule_logits"], b["rules"])
            acc.backward(loss); opt.step(); opt.zero_grad()
            acc.log({"train/enc_loss": loss.item()}, step=step); step += 1
            if step >= max_steps: break
    out_dir = Path(cfg.get("encoder_ckpt","artifacts/encoder")); out_dir.mkdir(parents=True, exist_ok=True)
    acc.wait_for_everyone()
    acc.save(acc.unwrap_model(model).state_dict(), out_dir / "pytorch_model.bin")
    tok.save_pretrained(out_dir)
    acc.end_training()
    if acc.is_main_process: print(f"Saved encoder -> {out_dir}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run a tiny train to verify it completes**

Run: `cd tracks/industrial-infineon/solution && python -m procseq.build_data --smoke && python -m procseq.train_encoder --config configs/smoke.yaml`
Expected: prints "Saved encoder -> artifacts/encoder"; `artifacts/encoder/pytorch_model.bin` exists.

- [ ] **Step 4: Commit**

```bash
git add procseq/anomaly_data.py procseq/train_encoder.py
git commit -m "feat: anomaly encoder training (balanced injected dataset, BCE heads)"
```

### Task 3.3: Anomaly inference → Task 3 submission

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/infer_anomaly.py`
- Test: `tracks/industrial-infineon/solution/tests/test_infer_anomaly.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infer_anomaly.py
import torch
from procseq.tokenizer import build_tokenizer
from procseq.models.encoder import build_encoder
from procseq.grammar import RULE_IDS
from procseq.infer_anomaly import classify_sequence

def test_classify_returns_triple():
    tok = build_tokenizer()
    model = build_encoder("tiny", tok, n_rules=len(RULE_IDS))
    steps = ["RECEIVE WAFER LOT", "THERMAL OXIDATION", "SHIP LOT"]
    is_valid, score, rule = classify_sequence(model, tok, steps, "MOSFET", RULE_IDS)
    assert is_valid in (0, 1)
    assert 0.0 <= score <= 1.0
    assert rule in RULE_IDS or rule == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_infer_anomaly.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `procseq/infer_anomaly.py`**

```python
"""Task 3 inference: classify each anomaly-eval sequence -> submission CSV."""
import csv
from pathlib import Path
import torch
from procseq.tokenizer import load_tokenizer
from procseq.models.encoder import build_encoder
from procseq.vocab import FAMILY_TOKEN, step_to_token
from procseq.grammar import RULE_IDS

@torch.no_grad()
def classify_sequence(model, tok, steps, family, rule_ids):
    model.eval()
    text = " ".join([tok.cls_token, FAMILY_TOKEN[family]] +
                    [step_to_token(s) for s in steps] + [tok.sep_token])
    ids = torch.tensor([tok.encode(text)], device=next(model.parameters()).device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    p_invalid = torch.sigmoid(out["invalid_logit"])[0].item()
    is_valid = 0 if p_invalid >= 0.5 else 1
    score = 1.0 - p_invalid  # P(valid) for AUC
    rule = ""
    if is_valid == 0:
        rule = rule_ids[int(out["rule_logits"][0].argmax())]
    return is_valid, score, rule

def _load_encoder(ckpt, tok):
    model = build_encoder("base", tok, n_rules=len(RULE_IDS))  # size from state shape
    sd = torch.load(Path(ckpt) / "pytorch_model.bin", map_location="cpu")
    # infer size by trying presets until shapes match
    for size in ("tiny", "small", "base"):
        m = build_encoder(size, tok, n_rules=len(RULE_IDS))
        try:
            m.load_state_dict(sd); return m
        except Exception:
            continue
    raise RuntimeError("encoder size mismatch")

def run_anomaly(cfg):
    art = Path(cfg["artifacts"])
    tok = load_tokenizer(Path(cfg["encoder_ckpt"]))
    model = _load_encoder(cfg["encoder_ckpt"], tok)
    rows = []
    with (art / "eval_input_anomaly.csv").open() as f:
        for r in csv.DictReader(f):
            steps = r["SEQUENCE"].split("|")
            iv, sc, rule = classify_sequence(model, tok, steps, r["FAMILY"], RULE_IDS)
            rows.append([r["EXAMPLE_ID"], iv, f"{sc:.4f}", rule])
    out = art / "submission_task3.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["EXAMPLE_ID","IS_VALID","SCORE","PREDICTED_RULE"])
        w.writerows(rows)
    print(f"Task3 -> {out} ({len(rows)} rows)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_infer_anomaly.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add procseq/infer_anomaly.py tests/test_infer_anomaly.py
git commit -m "feat: Task3 anomaly inference + submission CSV"
```

### Task 3.4: Unified eval runner + complete smoke

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/run_eval.py`
- Test: full `make smoke`

- [ ] **Step 1: Write `procseq/run_eval.py`**

```python
"""Score all three submission files against the internal mirror ground truth."""
import argparse, csv, json
from pathlib import Path
from procseq.config import load_config
from procseq import eval_metrics as em

def _gt_nextstep(valid_gt_path):
    """Derive NEXT_STEP gold from full sequence + partial length per example."""
    gold = {}
    with open(valid_gt_path) as f:
        for r in csv.DictReader(f):
            gold[r["EXAMPLE_ID"]] = r["FULL_SEQUENCE"].split("|")
    return gold

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(argv); cfg = load_config(a.config); art = Path(cfg["artifacts"])
    results = {}
    # build next-step + completion gold from the mirror input+gt
    full = {}
    with open(art / "eval_valid_groundtruth.csv") as f:
        for r in csv.DictReader(f):
            full[r["EXAMPLE_ID"]] = r["FULL_SEQUENCE"].split("|")
    partial = {}
    with open(art / "eval_input_valid.csv") as f:
        for r in csv.DictReader(f):
            partial[r["EXAMPLE_ID"]] = r["PARTIAL_SEQUENCE"].split("|")
    # Task1 gold = the single next step after the cut
    ns_gold = {eid: full[eid][len(partial[eid])] for eid in partial
               if len(full[eid]) > len(partial[eid])}
    if (art / "submission_task1.csv").exists():
        preds = {}
        with open(art / "submission_task1.csv") as f:
            for r in csv.DictReader(f):
                preds[r["EXAMPLE_ID"]] = [r[f"RANK_{k}"] for k in range(1,6)]
        results["task1_nextstep"] = em.score_nextstep(preds, ns_gold)
    # Task2 gold = suffix after cut
    comp_gold = {eid: full[eid][len(partial[eid]):] for eid in partial}
    if (art / "submission_task2.csv").exists():
        preds = {}
        with open(art / "submission_task2.csv") as f:
            for r in csv.DictReader(f):
                preds[r["EXAMPLE_ID"]] = (r["PREDICTED_SEQUENCE"].split("|")
                                          if r["PREDICTED_SEQUENCE"] else [])
        results["task2_completion"] = em.score_completion(preds, comp_gold)
        # logic probe: are reconstructed FULL sequences rule-valid?
        recon = [partial[eid] + preds.get(eid, []) for eid in partial]
        results["task2_logic_validity"] = em.logic_validity_rate(recon)
    # Task3
    if (art / "submission_task3.csv").exists():
        gold = {}
        with open(art / "eval_anomaly_labels.csv") as f:
            for r in csv.DictReader(f):
                gold[r["EXAMPLE_ID"]] = (int(r["IS_VALID"]), r["PREDICTED_RULE"])
        preds = {}
        with open(art / "submission_task3.csv") as f:
            for r in csv.DictReader(f):
                preds[r["EXAMPLE_ID"]] = (int(r["IS_VALID"]),
                    float(r["SCORE"] or 0.5), r["PREDICTED_RULE"])
        results["task3_anomaly"] = em.score_anomaly(preds, gold)
    out = art / "metrics.json"; out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2)); print(f"-> {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full smoke**

Run: `cd tracks/industrial-infineon/solution && make smoke`
Expected: ends with `SMOKE OK`; `artifacts/metrics.json` has `task1_nextstep`, `task2_completion`, `task2_logic_validity`, `task3_anomaly`; all three `submission_task*.csv` exist with correct headers.

- [ ] **Step 3: Commit**

```bash
git add procseq/run_eval.py
git commit -m "feat: unified eval runner + green end-to-end smoke (all 3 tasks)"
```

---

## Phase 4 — Scaling & Leonardo

### Task 4.1: Cluster configs (Accelerate + DeepSpeed + Leonardo YAML)

**Files:**
- Create: `tracks/industrial-infineon/solution/configs/leonardo_decoder.yaml`
- Create: `tracks/industrial-infineon/solution/configs/leonardo_encoder.yaml`
- Create: `tracks/industrial-infineon/solution/configs/ds_zero2.json`
- Create: `tracks/industrial-infineon/solution/configs/accelerate.yaml`
- Create: `tracks/industrial-infineon/solution/configs/scaling_grid.yaml`

- [ ] **Step 1: Write `configs/leonardo_decoder.yaml`**

```yaml
run_name: leo_decoder_base
seed: 42
precision: bf16
decoder:
  size: base
  max_len: 256
  data_per_family: 5000
  batch_size: 64
  lr: 0.0006
  max_steps: 4000
  eval_every: 250
artifacts: artifacts
decoder_ckpt: artifacts/decoder_base
encoder_ckpt: artifacts/encoder_base
encoder:
  size: base
  max_len: 256
  data_per_family: 5000
  batch_size: 64
  lr: 0.0005
  max_steps: 3000
```

- [ ] **Step 2: Write `configs/leonardo_encoder.yaml`** (same as above but `run_name: leo_encoder_base`; identical `encoder`/`decoder` blocks so either entrypoint can read it).

```yaml
run_name: leo_encoder_base
seed: 42
precision: bf16
artifacts: artifacts
decoder_ckpt: artifacts/decoder_base
encoder_ckpt: artifacts/encoder_base
decoder:
  size: base
  max_len: 256
  data_per_family: 5000
  batch_size: 64
  lr: 0.0006
  max_steps: 4000
  eval_every: 250
encoder:
  size: base
  max_len: 256
  data_per_family: 5000
  batch_size: 64
  lr: 0.0005
  max_steps: 3000
```

- [ ] **Step 3: Write `configs/ds_zero2.json`**

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": "auto"
}
```

- [ ] **Step 4: Write `configs/accelerate.yaml`**

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: configs/ds_zero2.json
  zero3_init_flag: false
mixed_precision: bf16
use_cpu: false
num_machines: 1
num_processes: 4
machine_rank: 0
main_process_port: 29500
```

- [ ] **Step 5: Write `configs/scaling_grid.yaml`**

```yaml
# Cartesian grid for the scaling experiment. The sweep driver expands these.
sizes: [tiny, small, base, large]
data_per_family: [100, 1000, 5000, 20000]
seed: 42
max_steps: 4000
```

- [ ] **Step 6: Commit**

```bash
git add configs/leonardo_decoder.yaml configs/leonardo_encoder.yaml configs/ds_zero2.json configs/accelerate.yaml configs/scaling_grid.yaml
git commit -m "feat: Leonardo Accelerate+DeepSpeed configs + scaling grid"
```

### Task 4.2: SLURM templates

**Files:**
- Create: `tracks/industrial-infineon/solution/slurm/train_decoder.sbatch`
- Create: `tracks/industrial-infineon/solution/slurm/train_encoder.sbatch`
- Create: `tracks/industrial-infineon/solution/slurm/scaling_sweep.sbatch`

- [ ] **Step 1: Write `slurm/train_decoder.sbatch`**

```bash
#!/bin/bash
#SBATCH --job-name=procseq-dec
#SBATCH --account=__ACCOUNT__          # <-- fill: CINECA project account
#SBATCH --partition=boost_usr_prod     # Leonardo Booster
#SBATCH --qos=__QOS__                  # <-- fill if required
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/dec_%j.out

set -euo pipefail
module load profile/deeplrn || true
module load python cuda || true        # <-- adjust to available Leonardo modules
source "${VENV:-$HOME/procseq-venv}/bin/activate"

cd "${SOLUTION_DIR:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=8
CONFIG="${CONFIG:-configs/leonardo_decoder.yaml}"

srun accelerate launch \
  --config_file configs/accelerate.yaml \
  --num_processes ${SLURM_GPUS_ON_NODE:-4} \
  -m procseq.train_decoder --config "$CONFIG"
```

- [ ] **Step 2: Write `slurm/train_encoder.sbatch`** (identical preamble; swap last lines)

```bash
#!/bin/bash
#SBATCH --job-name=procseq-enc
#SBATCH --account=__ACCOUNT__
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=__QOS__
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/enc_%j.out

set -euo pipefail
module load profile/deeplrn || true
module load python cuda || true
source "${VENV:-$HOME/procseq-venv}/bin/activate"
cd "${SOLUTION_DIR:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=8
CONFIG="${CONFIG:-configs/leonardo_encoder.yaml}"
srun accelerate launch --config_file configs/accelerate.yaml \
  --num_processes ${SLURM_GPUS_ON_NODE:-4} \
  -m procseq.train_encoder --config "$CONFIG"
```

- [ ] **Step 3: Write `slurm/scaling_sweep.sbatch`** (array job over the grid)

```bash
#!/bin/bash
#SBATCH --job-name=procseq-sweep
#SBATCH --account=__ACCOUNT__
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=__QOS__
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --array=0-15
#SBATCH --output=logs/sweep_%A_%a.out

set -euo pipefail
module load python cuda || true
source "${VENV:-$HOME/procseq-venv}/bin/activate"
cd "${SOLUTION_DIR:-$SLURM_SUBMIT_DIR}"

SIZES=(tiny small base large)
DATA=(100 1000 5000 20000)
SIZE=${SIZES[$((SLURM_ARRAY_TASK_ID / 4))]}
DV=${DATA[$((SLURM_ARRAY_TASK_ID % 4))]}

python -m procseq.sweep_run --size "$SIZE" --data-per-family "$DV" \
  --seed 42 --out artifacts/sweep
```

- [ ] **Step 4: Commit**

```bash
mkdir -p logs
git add slurm/
git commit -m "feat: Leonardo SLURM templates (decoder, encoder, scaling array job)"
```

### Task 4.3: Scaling sweep driver + plot data

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/sweep_run.py`
- Test: smoke invocation with tiny size

- [ ] **Step 1: Write `procseq/sweep_run.py`**

```python
"""One grid cell of the scaling experiment: train tiny decoder, self-score, append JSONL."""
import argparse, json, shutil
from pathlib import Path
from procseq.config import Config
from procseq import train_decoder, infer, run_eval

# Files the decoder tasks (1+2) need for inference and scoring.
_EVAL_MIRROR_FILES = ["eval_input_valid.csv", "eval_valid_groundtruth.csv"]

def _stage_eval_mirrors(eval_from: Path, out: Path) -> None:
    """Copy the eval-mirror CSVs into the cell's artifacts dir so infer/run_eval
    (which read cfg.artifacts) can find them. Run build_data first to create them."""
    for name in _EVAL_MIRROR_FILES:
        src, dst = eval_from / name, out / name
        if dst.exists():
            continue
        if not src.exists():
            raise FileNotFoundError(
                f"{src} missing. Run `python -m procseq.build_data` (or --smoke) "
                f"first so the eval mirrors exist, or pass --eval-from <dir>."
            )
        shutil.copy(src, dst)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--data-per-family", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--out", default="artifacts/sweep")
    ap.add_argument("--eval-from", default="artifacts",
                    help="dir holding the eval mirrors built by build_data")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _stage_eval_mirrors(Path(a.eval_from), out)
    ckpt = out / f"decoder_{a.size}_{a.data_per_family}"
    cfg = Config({
        "run_name": f"sweep_{a.size}_{a.data_per_family}", "seed": a.seed,
        "precision": "bf16", "artifacts": str(out),
        "decoder_ckpt": str(ckpt), "encoder_ckpt": str(out / "encoder_unused"),
        "decoder": {"size": a.size, "max_len": 256,
                    "data_per_family": a.data_per_family, "batch_size": 64,
                    "lr": 6e-4, "max_steps": a.max_steps, "eval_every": 500},
    })
    cfg_path = out / f"cfg_{a.size}_{a.data_per_family}.yaml"
    import yaml; cfg_path.write_text(yaml.safe_dump(dict(cfg)))
    train_decoder.main(["--config", str(cfg_path)])
    # NOTE: requires eval_input files in cfg.artifacts; build_data must run first.
    infer.run_task1(cfg); infer.run_task2(cfg)
    run_eval.main(["--config", str(cfg_path)])
    metrics = json.loads((out / "metrics.json").read_text())
    rec = {"size": a.size, "data_per_family": a.data_per_family, **metrics}
    with (out / "sweep_results.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"sweep cell done: {a.size}/{a.data_per_family}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the cell runs locally (tiny)**

Run: `cd tracks/industrial-infineon/solution && python -m procseq.build_data --smoke && python -m procseq.sweep_run --size tiny --data-per-family 20 --max-steps 3 --out artifacts/sweep`
Expected: prints "sweep cell done: tiny/20"; `artifacts/sweep/sweep_results.jsonl` has one line.

- [ ] **Step 3: Commit**

```bash
git add procseq/sweep_run.py
git commit -m "feat: scaling-sweep cell driver (train+score+append jsonl)"
```

---

## Phase 5 — Demonstrator, dashboard, docs

### Task 5.1: Before/after demo + plots

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/demo.py`
- Test: smoke invocation

- [ ] **Step 1: Write `procseq/demo.py`**

```python
"""Side-by-side baseline vs trained outputs + metric/scaling plots."""
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from procseq.config import load_config
from procseq.tokenizer import load_tokenizer
from procseq import infer, baselines
from procseq.data import scale_family
from procseq.grammar import FAMILIES

def _ngram(seed):
    seqs = []
    for fam in FAMILIES:
        seqs += scale_family(fam, 0, seed)  # provided only
    return baselines.NgramModel(n=3).fit(seqs)

def make_prediction_examples(cfg, n=5):
    art = Path(cfg["artifacts"])
    tok = load_tokenizer(Path(cfg["decoder_ckpt"]))
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(cfg["decoder_ckpt"])
    ng = _ngram(cfg.get("seed", 42))
    rows = []
    with (art / "eval_input_valid.csv").open() as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= n: break
            steps = r["PARTIAL_SEQUENCE"].split("|")
            rows.append({
                "example": r["EXAMPLE_ID"], "family": r["FAMILY"],
                "prefix_tail": " -> ".join(steps[-3:]),
                "baseline_next": ng.predict_next(steps, k=3),
                "model_next": infer.predict_next_step(model, tok, steps, r["FAMILY"], k=3),
            })
    (art / "demo_examples.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))

def plot_metrics(cfg):
    art = Path(cfg["artifacts"])
    m = json.loads((art / "metrics.json").read_text())
    if "task1_nextstep" in m:
        t1 = m["task1_nextstep"]
        plt.figure()
        plt.bar(["top1","top3","top5","mrr"],
                [t1["top1"], t1["top3"], t1["top5"], t1["mrr"]])
        plt.title("Task 1 — next-step"); plt.ylim(0,1)
        plt.savefig(art / "plot_task1.png", dpi=120); plt.close()
    sweep = art / "sweep" / "sweep_results.jsonl"
    if sweep.exists():
        recs = [json.loads(l) for l in sweep.read_text().splitlines()]
        plt.figure()
        for size in sorted({r["size"] for r in recs}):
            pts = sorted([r for r in recs if r["size"]==size],
                         key=lambda r: r["data_per_family"])
            xs = [r["data_per_family"] for r in pts]
            ys = [r.get("task1_nextstep",{}).get("top1",0) for r in pts]
            plt.plot(xs, ys, marker="o", label=size)
        plt.xscale("log"); plt.xlabel("sequences/family"); plt.ylabel("Top-1")
        plt.title("Scaling"); plt.legend()
        plt.savefig(art / "plot_scaling.png", dpi=120); plt.close()
    print(f"plots -> {art}")

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(argv); cfg = load_config(a.config)
    make_prediction_examples(cfg); plot_metrics(cfg)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify demo runs after smoke**

Run: `cd tracks/industrial-infineon/solution && make smoke && python -m procseq.demo --config configs/smoke.yaml`
Expected: prints example JSON + "plots -> artifacts"; `artifacts/demo_examples.json`, `artifacts/plot_task1.png` exist.

- [ ] **Step 3: Commit**

```bash
git add procseq/demo.py
git commit -m "feat: before/after demo + metric and scaling plots"
```

### Task 5.2: Streamlit dashboard (README bonus)

**Files:**
- Create: `tracks/industrial-infineon/solution/dashboard/app.py`

- [ ] **Step 1: Write `dashboard/app.py`**

```python
"""Streamlit dashboard: metrics, plots, before/after examples. Run:
   streamlit run dashboard/app.py -- --artifacts artifacts
"""
import json, sys
from pathlib import Path
import streamlit as st

art = Path("artifacts")
for i, a in enumerate(sys.argv):
    if a == "--artifacts" and i + 1 < len(sys.argv):
        art = Path(sys.argv[i + 1])

st.set_page_config(page_title="Infineon Process-Logic", layout="wide")
st.title("Process-Logic Pipeline — Results")

mp = art / "metrics.json"
if mp.exists():
    st.subheader("Metrics")
    st.json(json.loads(mp.read_text()))
else:
    st.warning("No metrics.json yet — run `make eval`.")

c1, c2 = st.columns(2)
for col, name in [(c1, "plot_task1.png"), (c2, "plot_scaling.png")]:
    p = art / name
    if p.exists():
        col.image(str(p))

ex = art / "demo_examples.json"
if ex.exists():
    st.subheader("Baseline vs Trained — next step")
    for r in json.loads(ex.read_text()):
        st.markdown(f"**{r['example']}** ({r['family']}) — `…{r['prefix_tail']}`")
        a, b = st.columns(2)
        a.write({"baseline (n-gram)": r["baseline_next"]})
        b.write({"trained model": r["model_next"]})
```

- [ ] **Step 2: Verify it imports (no run needed in CI)**

Run: `cd tracks/industrial-infineon/solution && python -c "import ast; ast.parse(open('dashboard/app.py').read()); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: streamlit results dashboard (README bonus)"
```

### Task 5.3: README + Leonardo runbook + submission docs

**Files:**
- Create: `tracks/industrial-infineon/solution/README.md`
- Modify: `submission/SUBMISSION.md` (link the solution + how to reproduce)

- [ ] **Step 1: Write `tracks/industrial-infineon/solution/README.md`**

Include: overview, install (`pip install -r requirements.txt`), `make smoke` quickstart, the full pipeline commands (`make data`, `accelerate launch ... train_decoder`, `train_encoder`, `infer --all`, `run_eval`, `demo`), Leonardo runbook (edit `__ACCOUNT__`/`__QOS__`/module lines in `slurm/*.sbatch`, create venv, `sbatch slurm/train_decoder.sbatch`, `sbatch slurm/scaling_sweep.sbatch`), the architecture summary (two models, why from-scratch), the honest-eval framing (n-gram floor, perplexity baseline, rule-oracle ceiling, logic probe, leave-one-family-out OOD), and a results table placeholder filled after real runs.

- [ ] **Step 2: Fill `submission/SUBMISSION.md`**

Add a section pointing to `tracks/industrial-infineon/solution/`, the reproduce-in-one-command (`make smoke`), where the 3 submission CSVs land (`artifacts/submission_task{1,2,3}.csv`), and where plots/dashboard live.

- [ ] **Step 3: Commit**

```bash
git add tracks/industrial-infineon/solution/README.md submission/SUBMISSION.md
git commit -m "docs: solution README + Leonardo runbook + submission pointer"
```

---

## Phase 6 — Leave-one-family-out OOD probe

### Task 6.1: OOD train/eval driver

**Files:**
- Create: `tracks/industrial-infineon/solution/procseq/ood_probe.py`
- Test: smoke invocation

- [ ] **Step 1: Write `procseq/ood_probe.py`**

```python
"""Leave-one-family-out: train on 2 families, score Task1 on the held-out 3rd.
Reports ID vs OOD Top-1 and the drop Delta, self-simulating Task 4."""
import argparse, json
from pathlib import Path
import torch
from transformers import LlamaForCausalLM
from procseq.config import Config
from procseq.tokenizer import build_tokenizer, DEFAULT_DIR
from procseq.models.decoder import build_decoder
from procseq.datasets import ClmDataset, clm_collate
from procseq.data import scale_family
from procseq import train_decoder, infer, eval_metrics as em
from procseq.grammar import FAMILIES

def _score_family_top1(model, tok, family, n, seed):
    seqs = scale_family(family, n, seed)
    preds, gold = {}, {}
    for i, s in enumerate(seqs[:50]):
        cut = max(1, int(len(s) * 0.8))
        if cut >= len(s):
            continue
        eid = f"{family}_{i}"
        preds[eid] = infer.predict_next_step(model, tok, s[:cut], family, k=5)
        gold[eid] = s[cut]
    return em.score_nextstep(preds, gold)["top1"]

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True, choices=FAMILIES)
    ap.add_argument("--data-per-family", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/ood")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"decoder_no_{a.holdout}"
    cfg = Config({"run_name": f"ood_no_{a.holdout}", "seed": a.seed,
                  "precision": "bf16", "artifacts": str(out),
                  "decoder_ckpt": str(ckpt),
                  "decoder": {"size": "base", "max_len": 256,
                              "data_per_family": a.data_per_family, "batch_size": 64,
                              "lr": 6e-4, "max_steps": a.max_steps, "eval_every": 500},
                  "_ood_holdout": a.holdout})
    import yaml; cfgp = out / f"cfg_no_{a.holdout}.yaml"; cfgp.write_text(yaml.safe_dump(dict(cfg)))
    # train_decoder uses all families by default; monkeypatch loader to exclude holdout:
    orig = train_decoder._load_training_pairs
    def filtered(n, seed):
        return [(s, f) for (s, f) in orig(n, seed) if f != a.holdout]
    train_decoder._load_training_pairs = filtered
    try:
        train_decoder.main(["--config", str(cfgp)])
    finally:
        train_decoder._load_training_pairs = orig
    tok = build_tokenizer(DEFAULT_DIR)
    model = LlamaForCausalLM.from_pretrained(str(ckpt))
    id_fams = [f for f in FAMILIES if f != a.holdout]
    id_top1 = sum(_score_family_top1(model, tok, f, a.data_per_family, a.seed)
                  for f in id_fams) / len(id_fams)
    ood_top1 = _score_family_top1(model, tok, a.holdout, a.data_per_family, a.seed)
    rec = {"holdout": a.holdout, "id_top1": id_top1, "ood_top1": ood_top1,
           "delta": id_top1 - ood_top1}
    (out / f"ood_{a.holdout}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs tiny**

Run: `cd tracks/industrial-infineon/solution && python -m procseq.ood_probe --holdout IC --data-per-family 20 --max-steps 3 --out artifacts/ood`
Expected: prints a JSON with `id_top1`, `ood_top1`, `delta`; `artifacts/ood/ood_IC.json` exists.

- [ ] **Step 3: Commit**

```bash
git add procseq/ood_probe.py
git commit -m "feat: leave-one-family-out OOD probe (self-simulated Task 4)"
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** tokenizer (1.1–1.2), data/scaling/UCBS (1.3), anomaly injection (1.4),
  eval mirrors (1.5), eval_metrics all tasks + logic probe (1.6, 3.4), baselines incl.
  perplexity + oracle (1.7, 2.4), decoder Tasks 1+2 (2.1–2.4), encoder Task 3 (3.1–3.3),
  Accelerate+DeepSpeed+SLURM+scaling (4.1–4.3), demo+dashboard+docs (5.1–5.3), OOD probe (6.1).
- **Type/name consistency:** `validate_sequence`, `read_csv_sequences`, `write_csv`,
  `generate_dataset` come from `procseq.grammar`; `RULE_IDS`/`FAMILIES` defined there;
  `encode_sequence`/`decode_to_steps`/`load_tokenizer` in `tokenizer.py`; submission column
  names match `generation_rules.md` §5.3 exactly.
- **Known follow-ups for execution (not blockers):** the encoder `_load_encoder` size-probe
  in `infer_anomaly.py` brute-forces presets — fine for our 3 sizes; if more sizes are added,
  persist the size in the checkpoint dir. The `run_eval` Task-1 gold derivation assumes the
  partial cut index equals `len(partial)`, which holds because mirrors are built by slicing.
```
