"""Unit tests for MLMTokenizer — vocab loading, BERT token extension, compat check."""
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.tokenizer import MLMTokenizer


def _write_vocab(path: Path, tokens: list[str]) -> None:
    with open(path, "w") as f:
        json.dump({"id_to_step": tokens}, f)


class TestMLMTokenizerLoad:
    def test_from_id_to_step_adds_bert_tokens(self):
        base_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]", "STEP_A", "STEP_B"]
        tok = MLMTokenizer._from_id_to_step(base_tokens)
        assert "[MASK]" in tok.step_to_id
        assert "[CLS]" in tok.step_to_id
        assert "[SEP]" in tok.step_to_id
        # existing tokens must keep original IDs
        assert tok.step_to_id["[PAD]"] == 0
        assert tok.step_to_id["STEP_A"] == 7
        assert tok.step_to_id["STEP_B"] == 8

    def test_load_from_file(self):
        base = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]", "STEP_X"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "vocab.json"
            _write_vocab(p, base)
            tok = MLMTokenizer.load(p)
        assert tok.step_to_id["[PAD]"] == 0
        assert tok.step_to_id["STEP_X"] == 7
        assert "[MASK]" in tok.step_to_id
        assert "[CLS]" in tok.step_to_id
        assert "[SEP]" in tok.step_to_id

    def test_bert_tokens_not_duplicated_if_present(self):
        base = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]", "[MASK]", "[CLS]", "[SEP]"]
        tok = MLMTokenizer._from_id_to_step(base)
        # Should not add duplicates
        assert tok.id_to_step.count("[MASK]") == 1
        assert tok.id_to_step.count("[CLS]") == 1

    def test_save_and_reload(self):
        base = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]", "STEP_Z"]
        tok = MLMTokenizer._from_id_to_step(base)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "vocab.json"
            tok.save(p)
            reloaded = MLMTokenizer.load(p)
        assert reloaded.vocab_size == tok.vocab_size
        assert reloaded.step_to_id == tok.step_to_id


class TestMLMTokenizerEncode:
    def _tok(self) -> MLMTokenizer:
        steps = [f"STEP_{i}" for i in range(20)]
        specials = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]"]
        return MLMTokenizer._from_id_to_step(specials + steps)

    def test_encode_mlm_length(self):
        tok = self._tok()
        ids = tok.encode_mlm("IC", [f"STEP_{i}" for i in range(5)], max_len=20)
        assert len(ids) == 20

    def test_encode_mlm_starts_with_cls(self):
        tok = self._tok()
        ids = tok.encode_mlm("IC", ["STEP_0", "STEP_1"], max_len=10)
        assert ids[0] == tok.cls_id

    def test_encode_mlm_second_token_is_variant(self):
        tok = self._tok()
        ids = tok.encode_mlm("IGBT", ["STEP_0"], max_len=10)
        assert ids[1] == tok.variant_id("IGBT")

    def test_encode_mlm_ends_with_sep_then_pad(self):
        tok = self._tok()
        ids = tok.encode_mlm("IC", ["STEP_0"], max_len=10)
        # find sep position
        sep_pos = ids.index(tok.sep_id)
        # everything after SEP should be PAD
        assert all(x == tok.pad_id for x in ids[sep_pos + 1:])

    def test_encode_mlm_truncates_long_sequences(self):
        tok = self._tok()
        ids = tok.encode_mlm("IC", [f"STEP_{i}" for i in range(100)], max_len=10)
        assert len(ids) == 10
        assert ids[-1] == tok.sep_id


class TestCompatCheck:
    def test_compat_ok(self):
        base = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]", "STEP_A"]
        tok = MLMTokenizer._from_id_to_step(base)
        with tempfile.TemporaryDirectory() as d:
            p001 = Path(d) / "vocab001.json"
            _write_vocab(p001, base)
            tok.verify_compat(p001)  # should not raise

    def test_compat_fails_on_mismatch(self):
        base001 = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]", "STEP_A"]
        base002 = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]", "STEP_B"]  # STEP_A missing
        tok = MLMTokenizer._from_id_to_step(base002)
        with tempfile.TemporaryDirectory() as d:
            p001 = Path(d) / "vocab001.json"
            _write_vocab(p001, base001)
            with pytest.raises(ValueError, match="mismatch"):
                tok.verify_compat(p001)
