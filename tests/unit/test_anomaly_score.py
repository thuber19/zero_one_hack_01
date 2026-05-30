"""Unit tests for anomaly scoring — per-step loss, z-score, threshold, OOD flag."""
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.infer import score_sequence
from src.eval.shared import compute_roc_auc, compute_precision_recall_f1


def _make_tokenizer():
    from src.tokenizer import MLMTokenizer
    steps = [f"STEP_{i:03d}" for i in range(20)]
    specials = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]"]
    return MLMTokenizer._from_id_to_step(specials + steps)


def _make_model(tokenizer):
    from src.model.bert_mlm import BertMLMEncoder, BertMLMConfig
    cfg = BertMLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_len=20,
        dropout=0.0,
        pad_id=tokenizer.pad_id,
    )
    return BertMLMEncoder(cfg)


class TestAnomalyScore:
    def setup_method(self):
        self.tok = _make_tokenizer()
        self.model = _make_model(self.tok)
        self.model.eval()
        self.threshold = {"p95_loss": 1.0, "p99_loss": 2.0, "ood_p99": 0.5}

    def test_score_sequence_returns_report(self):
        steps = [f"STEP_{i:03d}" for i in range(5)]
        report = score_sequence(
            self.model, self.tok, "IC", steps, self.threshold,
            seq_id="test_001", device=torch.device("cpu"), batch_scoring=True,
        )
        assert report.sequence_id == "test_001"
        assert len(report.per_step_raw_loss) == 20  # max_len
        assert len(report.per_step_zscore) == 20

    def test_per_step_loss_non_negative(self):
        steps = [f"STEP_{i:03d}" for i in range(8)]
        report = score_sequence(
            self.model, self.tok, "IC", steps, self.threshold,
            device=torch.device("cpu"), batch_scoring=True,
        )
        assert all(l >= 0.0 for l in report.per_step_raw_loss)

    def test_seq_score_max_equals_max_of_per_step(self):
        steps = [f"STEP_{i:03d}" for i in range(5)]
        report = score_sequence(
            self.model, self.tok, "MOSFET", steps, self.threshold,
            device=torch.device("cpu"),
        )
        assert abs(report.seq_score_max - max(report.per_step_raw_loss)) < 1e-6

    def test_is_anomalous_flag_consistent(self):
        steps = [f"STEP_{i:03d}" for i in range(5)]
        threshold = {"p95_loss": 0.0, "p99_loss": 0.0, "ood_p99": 9999.0}  # force anomaly
        report = score_sequence(
            self.model, self.tok, "IC", steps, threshold, device=torch.device("cpu")
        )
        if report.seq_score_max >= 0.0:
            assert report.is_anomalous  # any loss ≥ 0 should trigger with threshold=0

    def test_sequential_and_batch_scoring_close(self):
        steps = [f"STEP_{i:03d}" for i in range(5)]
        rep_batch = score_sequence(
            self.model, self.tok, "IC", steps, self.threshold,
            device=torch.device("cpu"), batch_scoring=True,
        )
        rep_seq = score_sequence(
            self.model, self.tok, "IC", steps, self.threshold,
            device=torch.device("cpu"), batch_scoring=False,
        )
        for l1, l2 in zip(rep_batch.per_step_raw_loss, rep_seq.per_step_raw_loss):
            assert abs(l1 - l2) < 1e-4, f"Batch vs sequential mismatch: {l1} vs {l2}"


class TestSharedMetrics:
    def test_roc_auc_perfect(self):
        scores = [0.9, 0.8, 0.3, 0.2]
        labels = [1, 1, 0, 0]
        auc = compute_roc_auc(scores, labels)
        assert abs(auc - 1.0) < 0.01

    def test_roc_auc_random(self):
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [1, 0, 1, 0]
        auc = compute_roc_auc(scores, labels)
        assert 0.0 <= auc <= 1.0

    def test_prf_at_threshold(self):
        scores = [0.9, 0.8, 0.3, 0.2]
        labels = [1, 1, 0, 0]
        result = compute_precision_recall_f1(scores, labels, threshold=0.5)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert abs(result["f1"] - 1.0) < 0.01

    def test_ties_at_threshold(self):
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [1, 0, 1, 0]
        result = compute_precision_recall_f1(scores, labels, threshold=0.5)
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0
