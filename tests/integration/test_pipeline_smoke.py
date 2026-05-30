"""End-to-end smoke test: tiny dataset, 1 epoch, CPU only. Safe on login node."""
import json
import sys
import tempfile
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.tokenizer import MLMTokenizer
from src.model.bert_mlm import BertMLMEncoder, BertMLMConfig
from src.train_mlm import FabMLMDataset, MLMMaskingCollator
from src.infer import score_sequence
from src.eval.shared import compute_roc_auc


def _make_tokenizer() -> MLMTokenizer:
    steps = [f"STEP_{i:03d}" for i in range(30)]
    specials = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[IC]", "[IGBT]", "[MOSFET]"]
    return MLMTokenizer._from_id_to_step(specials + steps)


def _make_tiny_model(tok: MLMTokenizer) -> BertMLMEncoder:
    cfg = BertMLMConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_len=15,
        dropout=0.0,
        pad_id=tok.pad_id,
    )
    return BertMLMEncoder(cfg)


class TestPipelineSmoke:
    def test_tokenize_encode_decode_roundtrip(self):
        tok = _make_tokenizer()
        steps = ["STEP_001", "STEP_002", "STEP_010"]
        ids = tok.encode_mlm("IC", steps, max_len=15)
        assert len(ids) == 15
        assert ids[0] == tok.cls_id

    def test_model_forward_pass(self):
        tok = _make_tokenizer()
        model = _make_tiny_model(tok)
        ids = tok.encode_mlm("IC", ["STEP_001", "STEP_002"], max_len=15)
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = (input_ids != tok.pad_id).long()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        assert logits.shape == (1, 15, tok.vocab_size)

    def test_collator_produces_valid_batch(self):
        tok = _make_tokenizer()
        masking_cfg = {"strategy": "random", "mask_prob": 0.15, "mask_token_frac": 0.80,
                       "random_token_frac": 0.10, "span_max_len": 3, "min_seq_len_for_span": 5}
        collator = MLMMaskingCollator(tok, masking_cfg)
        seqs = [
            torch.tensor(tok.encode_mlm("IC", [f"STEP_{i:03d}" for i in range(5)], max_len=15))
            for _ in range(4)
        ]
        batch = collator(seqs)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape == (4, 15)

    def test_one_epoch_training_step(self):
        import torch.nn.functional as F
        tok = _make_tokenizer()
        model = _make_tiny_model(tok)
        model.train()
        masking_cfg = {"strategy": "random", "mask_prob": 0.15, "mask_token_frac": 0.80,
                       "random_token_frac": 0.10, "span_max_len": 3, "min_seq_len_for_span": 5}
        collator = MLMMaskingCollator(tok, masking_cfg)
        seqs = [
            torch.tensor(tok.encode_mlm("IC", [f"STEP_{i:03d}" for i in range(5)], max_len=15))
            for _ in range(8)
        ]
        batch = collator(seqs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"]
        mask = labels != -100
        if mask.any():
            loss = F.cross_entropy(logits[mask], labels[mask])
            loss.backward()
            optimizer.step()
            assert loss.item() > 0

    def test_inference_produces_report(self):
        tok = _make_tokenizer()
        model = _make_tiny_model(tok)
        model.eval()
        threshold = {"p95_loss": 1.0, "p99_loss": 2.0, "ood_p99": 0.5}
        steps = [f"STEP_{i:03d}" for i in range(5)]
        report = score_sequence(model, tok, "IC", steps, threshold, device=torch.device("cpu"))
        assert report.seq_score_max >= 0.0
        assert isinstance(report.is_anomalous, bool)
        assert isinstance(report.anomalous_steps, list)

    def test_roc_auc_with_fake_scores(self):
        scores = [0.9, 0.85, 0.3, 0.1]
        labels = [1, 1, 0, 0]
        auc = compute_roc_auc(scores, labels)
        assert auc > 0.5

    def test_checkpoint_save_load(self):
        tok = _make_tokenizer()
        model = _make_tiny_model(tok)
        with tempfile.TemporaryDirectory() as d:
            ckpt_path = Path(d) / "checkpoint_test.pt"
            tok_path = Path(d) / "tokenizer.json"
            tok.save(tok_path)
            cfg_mock = {
                "model": {
                    "d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64,
                    "max_len": 15, "dropout": 0.0,
                },
                "seed": 42,
            }
            torch.save({
                "model": model.state_dict(),
                "optimizer": {},
                "epoch": 0,
                "step": 0,
                "best_val_acc": 0.5,
                "config": cfg_mock,
                "vocab_path": str(tok_path),
                "vocab_size": tok.vocab_size,
                "seed": 42,
            }, ckpt_path)
            from src.infer import load_model
            loaded_model, loaded_tok = load_model(ckpt_path, torch.device("cpu"))
        assert loaded_tok.vocab_size == tok.vocab_size
        assert loaded_model.num_params() == model.num_params()
