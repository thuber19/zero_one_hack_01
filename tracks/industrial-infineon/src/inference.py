"""
Inference pipeline: combines Random Forest + Transformer for all 3 eval tasks.

Task 1: Next-step prediction (top-5)
Task 2: Sequence completion
Task 3: Anomaly detection
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tokenizer import StepTokenizer, FAMILY_TOKENS, BOS_ID, EOS_ID, PAD_ID
from transformer_model import create_model, ProcessTransformer
from random_forest import StepCandidateForest


class ProcessPredictor:
    """
    Combined RF + Transformer inference engine.

    The RF provides candidate sets (hard filter), the transformer ranks
    among candidates using full sequence context.
    """

    def __init__(
        self,
        tokenizer: StepTokenizer,
        model: ProcessTransformer,
        rf: StepCandidateForest,
        device: str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.rf = rf
        self.device = device

    @classmethod
    def load(cls, output_dir: Path, model_size: str = "small", arch: str = "transformer", device: str = "cpu"):
        """Load all components from saved outputs."""
        tokenizer = StepTokenizer.load(output_dir / "tokenizer.txt")

        # Auto-detect arch from training_history if available
        history_path = output_dir / "training_history.json"
        if arch == "transformer" and history_path.exists():
            import json
            with open(history_path) as f:
                config = json.load(f).get("config", {})
            arch = config.get("arch", "transformer")
            model_size = config.get("model_size", model_size)

        if arch == "lstm":
            from lstm_model import create_lstm_model
            model = create_lstm_model(tokenizer.vocab_size, size=model_size)
        else:
            model = create_model(tokenizer.vocab_size, size=model_size)

        model.load_state_dict(torch.load(output_dir / "best_transformer.pt", map_location=device, weights_only=True))
        rf = StepCandidateForest()
        rf.load(output_dir / "random_forest.pkl", tokenizer)
        return cls(tokenizer, model, rf, device)

    def _encode_partial(self, steps: list[str], family: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a partial sequence into model inputs."""
        ids = self.tokenizer.encode_sequence(steps, family)
        # Remove EOS since sequence is partial
        ids = ids[:-1]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn_mask = torch.ones_like(input_ids)
        return input_ids, attn_mask

    def _get_litho_level(self, steps: list[str]) -> int:
        """Extract current litho mask level from sequence."""
        level = 0
        for s in steps:
            if s.startswith("ALIGN MASK LEVEL "):
                parts = s.split("ALIGN MASK LEVEL ")
                if len(parts) > 1 and parts[1].isdigit():
                    level = int(parts[1])
        return level

    # ── Task 1: Next-step prediction ──────────────────────────────────────

    def predict_next_steps(
        self,
        steps: list[str],
        family: str,
        top_k: int = 5,
        use_rf_mask: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Predict the top-K most likely next steps.

        Returns list of (step_name, probability) sorted by probability desc.
        """
        input_ids, attn_mask = self._encode_partial(steps, family)

        # Get transformer probabilities
        probs = self.model.get_next_step_probs(input_ids, attn_mask)

        if use_rf_mask and self.rf.is_fitted:
            # Get RF candidate mask
            litho_level = self._get_litho_level(steps)
            position_frac = len(steps) / 150.0  # rough normalization
            mask = self.rf.get_candidate_mask(
                self.tokenizer.vocab_size,
                family, steps[-1], steps[-3:] if len(steps) >= 3 else steps,
                litho_level, position_frac,
            )
            mask_tensor = torch.tensor(mask, device=self.device)
            # Apply mask: zero out non-candidates, renormalize
            probs = probs * mask_tensor
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum

        # Get top-K
        topk_probs, topk_ids = torch.topk(probs, min(top_k, len(probs)))
        results = []
        for prob, tid in zip(topk_probs.cpu().tolist(), topk_ids.cpu().tolist()):
            step_name = self.tokenizer.id2token.get(tid, "[UNK]")
            results.append((step_name, prob))

        return results

    # ── Task 2: Sequence completion ───────────────────────────────────────

    def complete_sequence(
        self,
        partial_steps: list[str],
        family: str,
        max_new_steps: int = 80,
        use_rf_mask: bool = True,
    ) -> list[str]:
        """
        Autoregressively complete a partial sequence.

        Returns only the NEW steps (after the partial sequence).
        """
        current_steps = list(partial_steps)
        new_steps = []

        for _ in range(max_new_steps):
            predictions = self.predict_next_steps(
                current_steps, family, top_k=1, use_rf_mask=use_rf_mask
            )
            if not predictions:
                break

            next_step, prob = predictions[0]

            # Stop conditions
            if next_step in ("[EOS]", "[PAD]", "[UNK]"):
                break
            if next_step == "SHIP LOT":
                new_steps.append(next_step)
                break

            new_steps.append(next_step)
            current_steps.append(next_step)

        return new_steps

    # ── Task 3: Anomaly detection ─────────────────────────────────────────

    def detect_anomaly(
        self,
        steps: list[str],
        family: str,
    ) -> dict:
        """
        Score a sequence for anomalies using two signals:
        1. Transformer per-token loss (high loss = unusual)
        2. RF candidate violations (step not in candidate set)

        Returns dict with:
          - is_valid: bool
          - score: float (probability of being valid, 0-1)
          - predicted_rule: str or empty
          - details: dict with per-step breakdown
        """
        # Signal 1: Transformer loss
        ids = self.tokenizer.encode_sequence(steps, family)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn_mask = torch.ones_like(input_ids)
        avg_loss = self.model.sequence_loss(input_ids, attn_mask)

        # Signal 2: RF violations — check each transition
        rf_violations = []
        if self.rf.is_fitted:
            litho_level = 0
            for t in range(len(steps) - 1):
                if steps[t].startswith("ALIGN MASK LEVEL "):
                    parts = steps[t].split("ALIGN MASK LEVEL ")
                    if len(parts) > 1 and parts[1].isdigit():
                        litho_level = int(parts[1])

                prev = steps[max(0, t - 2):t]
                candidates = self.rf.get_candidates(
                    family, steps[t], prev, litho_level, t / len(steps)
                )
                actual_next_id = self.tokenizer.encode_step(steps[t + 1])
                if actual_next_id not in candidates:
                    rf_violations.append({
                        "position": t + 1,
                        "expected_candidates": [
                            self.tokenizer.id2token.get(c, "?") for c in candidates[:5]
                        ],
                        "actual": steps[t + 1],
                        "context": steps[max(0, t - 2):t + 2],
                    })

        # Signal 3: Rule-based validator (ground truth rules)
        from generate_sequences import validate_sequence
        rule_violations = validate_sequence(steps)
        has_rule_violation = len(rule_violations) > 0
        predicted_rule = rule_violations[0].rule if rule_violations else ""

        # Combine all signals
        # Rule validator is the strongest signal — if it fires, it's definitely invalid
        # RF violations and high loss are softer signals for unknown violations
        if has_rule_violation:
            is_valid = False
            combined_score = 0.0
        else:
            # Use loss + RF for cases the validator might miss
            loss_score = max(0.0, 1.0 - avg_loss / 3.0)
            rf_score = 1.0 if len(rf_violations) == 0 else max(0.0, 1.0 - len(rf_violations) * 0.2)
            combined_score = 0.5 * loss_score + 0.5 * rf_score
            is_valid = combined_score > 0.4

        return {
            "is_valid": is_valid,
            "score": round(combined_score, 4),
            "predicted_rule": predicted_rule,
            "avg_loss": avg_loss,
            "n_rf_violations": len(rf_violations),
            "n_rule_violations": len(rule_violations),
            "rf_violations": rf_violations[:5],
        }



# ── Submission file generators ────────────────────────────────────────────

def generate_task1_submission(
    predictor: ProcessPredictor,
    eval_csv: Path,
    output_csv: Path,
):
    """Generate Task 1 (next-step prediction) submission file."""
    with open(eval_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []
    for row in rows:
        example_id = row["EXAMPLE_ID"].strip()
        family = row["FAMILY"].strip().lower()
        partial = row["PARTIAL_SEQUENCE"].strip().split("|")
        partial = [s.strip() for s in partial if s.strip()]

        preds = predictor.predict_next_steps(partial, family, top_k=5)
        # Pad to 5 if needed
        while len(preds) < 5:
            preds.append(("[UNK]", 0.0))

        results.append({
            "EXAMPLE_ID": example_id,
            "RANK_1": preds[0][0],
            "RANK_2": preds[1][0],
            "RANK_3": preds[2][0],
            "RANK_4": preds[3][0],
            "RANK_5": preds[4][0],
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Task 1 submission: {len(results)} rows -> {output_csv}")


def generate_task2_submission(
    predictor: ProcessPredictor,
    eval_csv: Path,
    output_csv: Path,
):
    """Generate Task 2 (sequence completion) submission file."""
    with open(eval_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []
    for row in rows:
        example_id = row["EXAMPLE_ID"].strip()
        family = row["FAMILY"].strip().lower()
        partial = row["PARTIAL_SEQUENCE"].strip().split("|")
        partial = [s.strip() for s in partial if s.strip()]

        completion = predictor.complete_sequence(partial, family)
        pred_seq = "|".join(completion)

        results.append({
            "EXAMPLE_ID": example_id,
            "PREDICTED_SEQUENCE": pred_seq,
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Task 2 submission: {len(results)} rows -> {output_csv}")


def generate_task3_submission(
    predictor: ProcessPredictor,
    eval_csv: Path,
    output_csv: Path,
):
    """Generate Task 3 (anomaly detection) submission file."""
    with open(eval_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []
    for row in rows:
        example_id = row["EXAMPLE_ID"].strip()
        family = row["FAMILY"].strip().lower()
        sequence = row["SEQUENCE"].strip().split("|")
        sequence = [s.strip() for s in sequence if s.strip()]

        result = predictor.detect_anomaly(sequence, family)

        results.append({
            "EXAMPLE_ID": example_id,
            "IS_VALID": 1 if result["is_valid"] else 0,
            "SCORE": result["score"],
            "PREDICTED_RULE": result["predicted_rule"],
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Task 3 submission: {len(results)} rows -> {output_csv}")


def generate_all_submissions(
    output_dir: Path,
    eval_dir: Path,
    model_size: str = "small",
    device: str = "cpu",
):
    """Generate all 3 submission files."""
    print("Loading models...")
    predictor = ProcessPredictor.load(output_dir, model_size=model_size, device=device)

    submission_dir = output_dir / "submissions"
    submission_dir.mkdir(exist_ok=True)

    eval_valid = eval_dir / "eval_input_valid.csv"
    eval_anomaly = eval_dir / "eval_input_anomaly.csv"

    if eval_valid.exists():
        print("\n--- Task 1: Next-step prediction ---")
        generate_task1_submission(predictor, eval_valid, submission_dir / "task1_nextstep.csv")
        print("\n--- Task 2: Sequence completion ---")
        generate_task2_submission(predictor, eval_valid, submission_dir / "task2_completion.csv")
    else:
        print(f"  Eval file not found: {eval_valid}")

    if eval_anomaly.exists():
        print("\n--- Task 3: Anomaly detection ---")
        generate_task3_submission(predictor, eval_anomaly, submission_dir / "task3_anomaly.csv")
    else:
        print(f"  Eval file not found: {eval_anomaly}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent / "outputs")
    parser.add_argument("--eval-dir", type=Path, default=Path(__file__).parent.parent / "training_data")
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    generate_all_submissions(args.output_dir, args.eval_dir, args.model_size, args.device)
