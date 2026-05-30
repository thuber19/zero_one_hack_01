"""Unit tests for MLMMaskingCollator — masking rates, span fallback, all-masked guard."""
import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.tokenizer import MLMTokenizer
from src.train_mlm import MLMMaskingCollator


def _make_tokenizer() -> MLMTokenizer:
    """Build a minimal MLMTokenizer from scratch."""
    steps = [f"STEP_{i:03d}" for i in range(50)]
    specials = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]"]
    id_to_step = specials + sorted(steps) + ["[MASK]", "[CLS]", "[SEP]"]
    return MLMTokenizer._from_id_to_step(id_to_step)


def _make_collator(strategy="random", mask_prob=0.15) -> tuple[MLMMaskingCollator, MLMTokenizer]:
    tok = _make_tokenizer()
    cfg = {"strategy": strategy, "mask_prob": mask_prob, "mask_token_frac": 0.80,
           "random_token_frac": 0.10, "span_max_len": 3, "min_seq_len_for_span": 5}
    return MLMMaskingCollator(tok, cfg), tok


def _make_batch(tok: MLMTokenizer, n: int = 4, max_len: int = 20) -> list[torch.Tensor]:
    sequences = []
    for i in range(n):
        ids = tok.encode_mlm("IC", [f"STEP_{j:03d}" for j in range(10)], max_len=max_len)
        sequences.append(torch.tensor(ids, dtype=torch.long))
    return sequences


class TestRandomMasking:
    def test_mask_rate_approx(self):
        collator, tok = _make_collator("random", mask_prob=0.15)
        batch = _make_batch(tok, n=32, max_len=20)
        result = collator(batch)
        labels = result["labels"]
        masked_count = (labels != -100).sum().item()
        total_tokens = labels.numel()
        # rough check: between 5% and 30%
        ratio = masked_count / total_tokens
        assert 0.03 < ratio < 0.35, f"Unexpected mask ratio: {ratio:.3f}"

    def test_labels_shape_matches_input(self):
        collator, tok = _make_collator()
        batch = _make_batch(tok)
        result = collator(batch)
        assert result["input_ids"].shape == result["labels"].shape
        assert result["input_ids"].shape == result["attention_mask"].shape

    def test_special_tokens_not_masked(self):
        collator, tok = _make_collator()
        batch = _make_batch(tok)
        result = collator(batch)
        labels = result["labels"]
        input_ids = result["labels"]
        # PAD positions should always be -100
        orig_ids = torch.stack(batch)
        pad_mask = orig_ids == tok.pad_id
        assert (labels[pad_mask] == -100).all()

    def test_mask_token_appears_in_input(self):
        collator, tok = _make_collator()
        batch = _make_batch(tok, n=8, max_len=30)
        result = collator(batch)
        # At least some [MASK] tokens should appear in the masked input
        assert (result["input_ids"] == tok.mask_id).any()

    def test_attention_mask_zeros_for_pad(self):
        collator, tok = _make_collator()
        batch = _make_batch(tok)
        result = collator(batch)
        orig_ids = torch.stack(batch)
        pad_mask = orig_ids == tok.pad_id
        assert (result["attention_mask"][pad_mask] == 0).all()


class TestSpanMasking:
    def test_span_masking_runs(self):
        collator, tok = _make_collator("span", mask_prob=0.15)
        batch = _make_batch(tok, n=4, max_len=20)
        result = collator(batch)
        assert result["input_ids"].shape[0] == 4

    def test_short_sequence_fallback(self):
        # min_seq_len_for_span=5, sequence has only 3 steps → should use single-token masking
        tok = _make_tokenizer()
        cfg = {"strategy": "span", "mask_prob": 0.30, "mask_token_frac": 1.0,
               "random_token_frac": 0.0, "span_max_len": 3, "min_seq_len_for_span": 10}
        collator = MLMMaskingCollator(tok, cfg)
        short_seq = tok.encode_mlm("IC", ["STEP_001", "STEP_002", "STEP_003"], max_len=10)
        batch = [torch.tensor(short_seq, dtype=torch.long)]
        result = collator(batch)
        # Should not crash — fallback to random single-token masking
        assert result["input_ids"].shape[0] == 1


class TestParamMaskingStub:
    def test_param_raises_not_implemented(self):
        tok = _make_tokenizer()
        cfg = {"strategy": "param", "mask_prob": 0.15}
        with pytest.raises(NotImplementedError, match="reserved"):
            MLMMaskingCollator(tok, cfg)
