"""
Inference pipeline: combines model + optional RF + optional physics refinery.

Task 1: Next-step prediction (top-5)
Task 2: Sequence completion
Task 3: Anomaly detection

Layers (all optional, stackable):
  - RF mask: statistical candidate filtering
  - Physics refinery: physical legality + category grammar + constrained decode
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tokenizer import StepTokenizer, FAMILY_TOKENS, BOS_ID, EOS_ID, PAD_ID
from transformer_model import create_model as create_transformer
from lstm_model import create_lstm_model
from random_forest import StepCandidateForest

# Make physics + refinery importable
_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "training_data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class ProcessPredictor:
    """
    Combined inference engine with optional layers:
      - RF mask (statistical candidate filtering)
      - Physics refinery (physical legality guard)
    """

    def __init__(
        self,
        tokenizer: StepTokenizer,
        model: torch.nn.Module,
        rf: StepCandidateForest | None = None,
        refinery=None,
        device: str = "cpu",
        use_rf: bool = True,
        use_physics: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.rf = rf
        self.refinery = refinery
        self.device = device
        self.use_rf = use_rf and rf is not None and rf.is_fitted
        self.use_physics = use_physics and refinery is not None

    @classmethod
    def load(
        cls,
        output_dir: Path,
        model_size: str = "small",
        arch: str = "transformer",
        device: str = "cpu",
        use_rf: bool = True,
        use_physics: bool = False,
    ):
        """Load all components from saved outputs."""
        import json
        tokenizer = StepTokenizer.load(output_dir / "tokenizer.txt")

        # Auto-detect arch/size from training_history
        history_path = output_dir / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                config = json.load(f).get("config", {})
            arch = config.get("arch", arch)
            model_size = config.get("model_size", model_size)

        if arch == "lstm":
            model = create_lstm_model(tokenizer.vocab_size, size=model_size)
        else:
            model = create_transformer(tokenizer.vocab_size, size=model_size)

        # Load best checkpoint (try new name first, fall back to old)
        ckpt = output_dir / "best_model.pt"
        if not ckpt.exists():
            ckpt = output_dir / "best_transformer.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        # Load RF if available and requested
        rf = None
        if use_rf:
            rf_path = output_dir / "random_forest.pkl"
            if rf_path.exists():
                rf = StepCandidateForest()
                rf.load(rf_path, tokenizer)

        # Load physics refinery if requested
        refinery = None
        if use_physics:
            try:
                from refinery import PhysicsRefinery
                refinery = PhysicsRefinery(category_mode="soft")
                print("  Physics refinery loaded")
            except ImportError as e:
                print(f"  Warning: physics refinery not available ({e})")

        return cls(tokenizer, model, rf, refinery, device, use_rf, use_physics)

    def _encode_partial(self, steps: list[str], family: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a partial sequence into model inputs."""
        ids = self.tokenizer.encode_sequence(steps, family)
        ids = ids[:-1]  # Remove EOS since sequence is partial
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn_mask = torch.ones_like(input_ids)
        return input_ids, attn_mask

    def _get_litho_level(self, steps: list[str]) -> int:
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
    ) -> list[tuple[str, float]]:
        """
        Predict the top-K most likely next steps.
        Applies RF mask and/or physics reranking based on config.
        """
        input_ids, attn_mask = self._encode_partial(steps, family)
        probs = self.model.get_next_step_probs(input_ids, attn_mask)

        # Layer 1: RF candidate mask
        if self.use_rf and steps:
            litho_level = self._get_litho_level(steps)
            position_frac = len(steps) / 150.0
            mask = self.rf.get_candidate_mask(
                self.tokenizer.vocab_size,
                family, steps[-1], steps[-3:] if len(steps) >= 3 else steps,
                litho_level, position_frac,
            )
            mask_tensor = torch.tensor(mask, device=self.device)
            probs = probs * mask_tensor
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum

        # Get a larger pool for physics reranking
        pool_size = max(top_k, 15) if self.use_physics else top_k
        topk_probs, topk_ids = torch.topk(probs, min(pool_size, len(probs)))
        candidates = []
        for prob, tid in zip(topk_probs.cpu().tolist(), topk_ids.cpu().tolist()):
            step_name = self.tokenizer.id2token.get(tid, "[UNK]")
            if step_name not in ("[PAD]", "[BOS]", "[EOS]", "[UNK]"):
                candidates.append((step_name, prob))

        # Layer 2: Physics reranking
        if self.use_physics and self.refinery and candidates:
            names = [n for n, _ in candidates]
            prob_map = {n: p for n, p in candidates}
            reranked = self.refinery.rerank(steps, names, k=top_k)
            candidates = [(n, prob_map.get(n, 0.0)) for n in reranked]
        else:
            candidates = candidates[:top_k]

        # Pad to top_k if needed
        while len(candidates) < top_k:
            candidates.append(("[UNK]", 0.0))

        return candidates

    # ── Task 2: Sequence completion ───────────────────────────────────────

    def complete_sequence(
        self,
        partial_steps: list[str],
        family: str,
        max_new_steps: int = 80,
    ) -> list[str]:
        """
        Autoregressively complete a partial sequence.
        If physics is enabled, uses constrained decoding for guaranteed valid output.
        """
        if self.use_physics and self.refinery:
            # Physics-constrained decoding
            def score_fn(prefix):
                preds = self.predict_next_steps(prefix, family, top_k=15)
                return [name for name, _ in preds]

            return self.refinery.constrained_decode(
                partial_steps, score_fn, max_steps=max_new_steps
            )

        # Standard autoregressive decoding
        current_steps = list(partial_steps)
        new_steps = []

        for _ in range(max_new_steps):
            predictions = self.predict_next_steps(current_steps, family, top_k=1)
            if not predictions:
                break

            next_step, prob = predictions[0]

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
        Score a sequence for anomalies using available signals:
        1. Rule-based validator (strongest, always used)
        2. Physics state machine (if enabled)
        3. Transformer per-token loss
        4. RF candidate violations (if enabled)
        """
        # Signal 1: Rule-based validator
        from generate_sequences import validate_sequence
        rule_violations = validate_sequence(steps)
        has_rule_violation = len(rule_violations) > 0
        predicted_rule = rule_violations[0].rule if rule_violations else ""

        # Signal 2: Physics state machine (if available)
        physics_violations = []
        if self.use_physics:
            try:
                from physics.state_machine import validate_by_state_machine
                physics_violations = validate_by_state_machine(steps)
                if not has_rule_violation and physics_violations:
                    has_rule_violation = True
                    predicted_rule = str(physics_violations[0])
            except ImportError:
                pass

        # Signal 3: Transformer loss
        ids = self.tokenizer.encode_sequence(steps, family)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn_mask = torch.ones_like(input_ids)
        avg_loss = self.model.sequence_loss(input_ids, attn_mask)

        # Signal 4: RF violations
        rf_violations = []
        if self.use_rf:
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
                    rf_violations.append({"position": t + 1, "actual": steps[t + 1]})

        # Combine signals
        if has_rule_violation:
            is_valid = False
            combined_score = 0.0
        else:
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
            "n_physics_violations": len(physics_violations),
        }


# ── Submission file generators ────────────────────────────────────────────

def generate_task1_submission(predictor, eval_csv: Path, output_csv: Path):
    with open(eval_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    results = []
    for row in rows:
        partial = [s.strip() for s in row["PARTIAL_SEQUENCE"].strip().split("|") if s.strip()]
        preds = predictor.predict_next_steps(partial, row["FAMILY"].strip().lower(), top_k=5)
        while len(preds) < 5:
            preds.append(("[UNK]", 0.0))
        results.append({"EXAMPLE_ID": row["EXAMPLE_ID"].strip(),
                         **{f"RANK_{i+1}": preds[i][0] for i in range(5)}})
    with open(output_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["EXAMPLE_ID","RANK_1","RANK_2","RANK_3","RANK_4","RANK_5"]).writeheader()
        csv.DictWriter(f, fieldnames=["EXAMPLE_ID","RANK_1","RANK_2","RANK_3","RANK_4","RANK_5"]).writerows(results)
    print(f"Task 1 submission: {len(results)} rows -> {output_csv}")


def generate_task2_submission(predictor, eval_csv: Path, output_csv: Path):
    with open(eval_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    results = []
    for row in rows:
        partial = [s.strip() for s in row["PARTIAL_SEQUENCE"].strip().split("|") if s.strip()]
        completion = predictor.complete_sequence(partial, row["FAMILY"].strip().lower())
        results.append({"EXAMPLE_ID": row["EXAMPLE_ID"].strip(),
                         "PREDICTED_SEQUENCE": "|".join(completion)})
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        w.writeheader(); w.writerows(results)
    print(f"Task 2 submission: {len(results)} rows -> {output_csv}")


def generate_task3_submission(predictor, eval_csv: Path, output_csv: Path):
    with open(eval_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    results = []
    for row in rows:
        sequence = [s.strip() for s in row["SEQUENCE"].strip().split("|") if s.strip()]
        result = predictor.detect_anomaly(sequence, row["FAMILY"].strip().lower())
        results.append({"EXAMPLE_ID": row["EXAMPLE_ID"].strip(),
                         "IS_VALID": 1 if result["is_valid"] else 0,
                         "SCORE": result["score"],
                         "PREDICTED_RULE": result["predicted_rule"]})
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID","IS_VALID","SCORE","PREDICTED_RULE"])
        w.writeheader(); w.writerows(results)
    print(f"Task 3 submission: {len(results)} rows -> {output_csv}")
