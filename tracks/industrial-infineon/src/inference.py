"""
Inference: generates submission CSVs from eval input CSVs.

Reads eval_input_valid.csv and eval_input_anomaly.csv (organizer format),
writes nextstep.csv, completion.csv, anomaly.csv (submission format).

Same code works for self-eval and official eval — just swap input files.

Usage:
    python src/inference.py --model-dir outputs --eval-valid eval_input_valid.csv --eval-anomaly eval_input_anomaly.csv --out-dir submissions
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SRC_DIR.parent
_DATA_DIR = _PROJECT_DIR / "data"
for _p in (str(_SRC_DIR), str(_DATA_DIR), str(_PROJECT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tokenizer import StepTokenizer, BOS_ID, EOS_ID, PAD_ID
from transformer_model import create_model as create_transformer
from lstm_model import create_lstm_model
from random_forest import StepCandidateForest
from block_classifier import get_valid_next_steps_by_block


class ProcessPredictor:
    """Combined inference engine: model + optional RF + optional physics."""

    def __init__(self, tokenizer, model, rf=None, refinery=None, device="cpu",
                 use_rf=True, use_physics=False):
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.rf = rf
        self.refinery = refinery
        self.device = device
        self.use_rf = use_rf and rf is not None and rf.is_fitted
        self.use_physics = use_physics and refinery is not None

    @classmethod
    def load(cls, model_dir: Path, device="cpu", use_rf=True, use_physics=False):
        """Load model + tokenizer + optional RF + optional physics from model_dir."""
        tokenizer = StepTokenizer.load(model_dir / "tokenizer.txt")

        # Auto-detect arch/size
        arch, model_size = "transformer", "small"
        history_path = model_dir / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                cfg = json.load(f).get("config", {})
            arch = cfg.get("arch", arch)
            model_size = cfg.get("model_size", model_size)

        if arch == "lstm":
            model = create_lstm_model(tokenizer.vocab_size, size=model_size)
        else:
            model = create_transformer(tokenizer.vocab_size, size=model_size)

        ckpt = model_dir / "best_model.pt"
        if not ckpt.exists():
            ckpt = model_dir / "best_transformer.pt"  # backwards compat
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        rf = None
        if use_rf:
            rf_path = model_dir / "random_forest.pkl"
            if rf_path.exists():
                rf = StepCandidateForest()
                rf.load(rf_path, tokenizer)

        refinery = None
        if use_physics:
            try:
                from refinery import PhysicsRefinery
                refinery = PhysicsRefinery(category_mode="soft")
                print("  Physics refinery loaded")
            except ImportError as e:
                print(f"  Physics refinery not available: {e}")

        return cls(tokenizer, model, rf, refinery, device, use_rf, use_physics)

    def _encode_partial(self, steps, family):
        ids = self.tokenizer.encode_sequence(steps, family)
        ids = ids[:-1]  # remove EOS
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn_mask = torch.ones_like(input_ids)
        return input_ids, attn_mask

    def _get_litho_level(self, steps):
        level = 0
        for s in steps:
            if s.startswith("ALIGN MASK LEVEL "):
                parts = s.split("ALIGN MASK LEVEL ")
                if len(parts) > 1 and parts[1].isdigit():
                    level = int(parts[1])
        return level

    # ── Task 1: Next-step prediction ──

    def predict_next_steps(self, steps, family, top_k=5):
        """Returns list of (step_name, probability) sorted desc."""
        input_ids, attn_mask = self._encode_partial(steps, family)
        probs = self.model.get_next_step_probs(input_ids, attn_mask)

        # Block-based filtering: only allow steps from current or next block
        if steps:
            valid_block_steps = get_valid_next_steps_by_block(steps)
            block_mask = torch.zeros(self.tokenizer.vocab_size, device=self.device)
            for step_name in valid_block_steps:
                tid = self.tokenizer.encode_step(step_name)
                if 0 <= tid < self.tokenizer.vocab_size:
                    block_mask[tid] = 1.0
            block_mask[EOS_ID] = 1.0
            probs = probs * block_mask
            s = probs.sum()
            if s > 0:
                probs = probs / s

        # RF candidate filtering
        if self.use_rf and steps:
            litho_level = self._get_litho_level(steps)
            mask = self.rf.get_candidate_mask(
                self.tokenizer.vocab_size, family,
                steps[-1], steps[-3:] if len(steps) >= 3 else steps,
                litho_level, len(steps) / 150.0,
            )
            mask_t = torch.tensor(mask, device=self.device)
            probs = probs * mask_t
            s = probs.sum()
            if s > 0:
                probs = probs / s

        pool = max(top_k, 15) if self.use_physics else top_k
        topk_probs, topk_ids = torch.topk(probs, min(pool, len(probs)))
        candidates = []
        for prob, tid in zip(topk_probs.cpu().tolist(), topk_ids.cpu().tolist()):
            name = self.tokenizer.id2token.get(tid, "[UNK]")
            if not (name.startswith("[") and name.endswith("]")):
                candidates.append((name, prob))

        if self.use_physics and self.refinery and candidates:
            names = [n for n, _ in candidates]
            prob_map = {n: p for n, p in candidates}
            reranked = self.refinery.rerank(steps, names, k=top_k)
            candidates = [(n, prob_map.get(n, 0.0)) for n in reranked]
        else:
            candidates = candidates[:top_k]

        # Pad if needed
        all_steps = [t for t in self.tokenizer.id2token.values()
                     if not (t.startswith("[") and t.endswith("]"))]
        have = {n for n, _ in candidates}
        for s in all_steps:
            if len(candidates) >= top_k:
                break
            if s not in have:
                candidates.append((s, 0.0))
                have.add(s)

        return candidates[:top_k]

    # ── Task 2: Sequence completion ──

    def complete_sequence(self, partial_steps, family, max_new_steps=80):
        """Returns only the NEW steps after the partial."""
        if self.use_physics and self.refinery:
            def score_fn(prefix):
                preds = self.predict_next_steps(prefix, family, top_k=15)
                return [name for name, _ in preds]
            return self.refinery.constrained_decode(partial_steps, score_fn, max_steps=max_new_steps)

        current = list(partial_steps)
        new_steps = []
        for _ in range(max_new_steps):
            preds = self.predict_next_steps(current, family, top_k=1)
            if not preds:
                break
            step, _ = preds[0]
            if step in ("[EOS]", "[PAD]", "[UNK]"):
                break
            if step == "SHIP LOT":
                new_steps.append(step)
                break
            new_steps.append(step)
            current.append(step)
        return new_steps

    # ── Task 3: Anomaly detection ──

    def detect_anomaly(self, steps, family):
        """Returns dict with is_valid, score, predicted_rule."""
        from generate_sequences import validate_sequence

        # Rule-based validator
        rule_violations = validate_sequence(steps)
        predicted_rule = rule_violations[0].rule if rule_violations else ""

        # Physics state machine
        physics_violations = []
        if self.use_physics:
            try:
                from physics.state_machine import validate_by_state_machine
                physics_violations = validate_by_state_machine(steps)
            except ImportError:
                pass

        has_violation = len(rule_violations) > 0 or len(physics_violations) > 0

        # Transformer loss
        ids = self.tokenizer.encode_sequence(steps, family)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn_mask = torch.ones_like(input_ids)
        avg_loss = self.model.sequence_loss(input_ids, attn_mask)

        if has_violation:
            is_valid = False
            score = 0.0
            if not predicted_rule and physics_violations:
                predicted_rule = str(physics_violations[0])
        else:
            score = max(0.0, 1.0 - avg_loss / 3.0)
            is_valid = score > 0.4

        return {
            "is_valid": is_valid,
            "score": round(score, 4),
            "predicted_rule": predicted_rule,
        }


# ── Submission generators ────────────────────────────────────────────────

def generate_submissions(predictor, eval_valid_csv, eval_anomaly_csv, out_dir):
    """Generate all 3 submission CSVs from eval input CSVs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Task 1 + 2: from eval_input_valid.csv
    if eval_valid_csv and Path(eval_valid_csv).exists():
        with open(eval_valid_csv, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        task1, task2 = [], []
        for i, row in enumerate(rows):
            eid = row["EXAMPLE_ID"].strip()
            family = row["FAMILY"].strip().lower()
            partial = [s.strip() for s in row["PARTIAL_SEQUENCE"].strip().split("|") if s.strip()]

            if (i + 1) % 50 == 0:
                print(f"  Processing {i+1}/{len(rows)}...")

            # Task 1
            preds = predictor.predict_next_steps(partial, family, top_k=5)
            task1.append({"EXAMPLE_ID": eid,
                          **{f"RANK_{j+1}": preds[j][0] for j in range(5)}})

            # Task 2
            completion = predictor.complete_sequence(partial, family)
            task2.append({"EXAMPLE_ID": eid,
                          "PREDICTED_SEQUENCE": "|".join(completion)})

        _write_csv(out_dir / "nextstep.csv",
                   ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"], task1)
        _write_csv(out_dir / "completion.csv",
                   ["EXAMPLE_ID", "PREDICTED_SEQUENCE"], task2)
        print(f"  nextstep.csv: {len(task1)} rows")
        print(f"  completion.csv: {len(task2)} rows")

    # Task 3: from eval_input_anomaly.csv
    if eval_anomaly_csv and Path(eval_anomaly_csv).exists():
        with open(eval_anomaly_csv, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        task3 = []
        for i, row in enumerate(rows):
            eid = row["EXAMPLE_ID"].strip()
            family = row["FAMILY"].strip().lower()
            sequence = [s.strip() for s in row["SEQUENCE"].strip().split("|") if s.strip()]

            if (i + 1) % 100 == 0:
                print(f"  Processing anomaly {i+1}/{len(rows)}...")

            result = predictor.detect_anomaly(sequence, family)
            task3.append({
                "EXAMPLE_ID": eid,
                "IS_VALID": 1 if result["is_valid"] else 0,
                "SCORE": result["score"],
                "PREDICTED_RULE": result["predicted_rule"],
            })

        _write_csv(out_dir / "anomaly.csv",
                   ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"], task3)
        print(f"  anomaly.csv: {len(task3)} rows")


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission CSVs")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory with model + tokenizer")
    parser.add_argument("--eval-valid", type=Path, default=None, help="eval_input_valid.csv")
    parser.add_argument("--eval-anomaly", type=Path, default=None, help="eval_input_anomaly.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output dir for submission CSVs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-rf", action="store_true")
    parser.add_argument("--physics", action="store_true")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.out_dir is None:
        args.out_dir = args.model_dir / "submissions"

    print("Loading model...")
    predictor = ProcessPredictor.load(
        args.model_dir, device=args.device,
        use_rf=not args.no_rf, use_physics=args.physics,
    )

    print("\nGenerating submissions...")
    generate_submissions(predictor, args.eval_valid, args.eval_anomaly, args.out_dir)
    print(f"\nDone. Submissions in: {args.out_dir}")
