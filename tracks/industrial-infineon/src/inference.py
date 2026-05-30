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

# ── Physics integration (the merged symbolic harness) ──────────────────────
# Make the harness importable: it lives one level up from src/.
import sys as _sys
from pathlib import Path as _Path
_SUBROOT = _Path(__file__).resolve().parent.parent
for _p in (str(_SUBROOT), str(_SUBROOT / "training_data")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from physics.state_machine import validate_sequence_combined, unknown_tokens as _unknown_tokens

# SCORE polarity (ASSUMPTION A5): spec §5.3 says SCORE = P(valid), higher = more
# valid. If the real grader wants P(anomaly) instead, flip this ONE switch (or set
# env SCORE_IS_P_VALID=0) — every emitted SCORE inverts, no other change needed.
import os as _os
SCORE_IS_P_VALID = _os.environ.get("SCORE_IS_P_VALID", "1") != "0"
from refinery import PhysicsRefinery, _ranked as _refinery_ranked
import fix as _fix


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
        # Physics layer: model proposes, physics disposes. category_mode="off"
        # = pure physical legality (no learned-grammar dependency), safe for OOD.
        self.refinery = PhysicsRefinery(category_mode="off")
        # Real (non-special) step names, used to guarantee Task-1 ranks are
        # always valid step strings (never "[UNK]") even if a prediction pool is
        # short.
        self._real_steps = [t for t in tokenizer.id2token.values()
                            if not (t.startswith("[") and t.endswith("]"))]

    @classmethod
    def load(cls, output_dir: Path, model_size: str = "small", device: str = "cpu"):
        """Load all components from saved outputs.

        Robustness: reads the model size from model_config.json when present (so
        the architecture always matches the checkpoint — avoids a shape-mismatch
        crash), and tolerates a missing random_forest.pkl (inference then runs on
        transformer + physics alone instead of failing on a clean checkout)."""
        import json
        tokenizer = StepTokenizer.load(output_dir / "tokenizer.txt")
        cfg = output_dir / "model_config.json"
        if cfg.exists():
            try:
                model_size = json.loads(cfg.read_text()).get("model_size", model_size)
            except Exception:
                pass
        model = create_model(tokenizer.vocab_size, size=model_size)
        model.load_state_dict(torch.load(
            output_dir / "best_transformer.pt", map_location=device, weights_only=True))
        rf = StepCandidateForest()
        rf_path = output_dir / "random_forest.pkl"
        if rf_path.exists():
            rf.load(rf_path, tokenizer)
        else:
            # LOUD: capability loss (no RF candidate mask) must not be silent.
            print(f"[inference WARNING] no random_forest.pkl in {output_dir} — "
                  "running on transformer + physics only (RF candidate mask disabled).",
                  file=_sys.stderr)
        return cls(tokenizer, model, rf, device)

    def _encode_partial(self, steps: list[str], family: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a partial sequence into model inputs."""
        ids = self.tokenizer.encode_sequence(steps, family)
        # Remove EOS since sequence is partial
        ids = ids[:-1]
        # Robustness: never exceed the model's positional range. Keep [BOS]+family
        # token and the MOST RECENT steps (recency drives next-step prediction).
        maxlen = getattr(self.model, "max_seq_len", 200)
        if len(ids) > maxlen:
            ids = ids[:2] + ids[2:][-(maxlen - 2):]
        if not ids:                      # fully-empty input -> at least [BOS]
            ids = [1]
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
        use_physics: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Predict the top-K most likely next steps.

        Returns list of (step_name, probability) sorted by probability desc.
        """
        input_ids, attn_mask = self._encode_partial(steps, family)

        # Get transformer probabilities
        probs = self.model.get_next_step_probs(input_ids, attn_mask)

        if use_rf_mask and self.rf.is_fitted and steps:
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

        # Pull a larger candidate pool so physics can re-rank within it.
        pool = max(top_k, 15) if use_physics else top_k
        topk_probs, topk_ids = torch.topk(probs, min(pool, len(probs)))
        cand = []
        for prob, tid in zip(topk_probs.cpu().tolist(), topk_ids.cpu().tolist()):
            cand.append((self.tokenizer.id2token.get(tid, "[UNK]"), prob))

        if use_physics:
            # model proposes -> physics floats legal next-steps up (model order
            # preserved among legal ones), illegal demoted but never dropped.
            prob_of = {n: p for n, p in cand}
            names = [n for n, _ in cand
                     if not (n.startswith("[") and n.endswith("]"))]
            reranked = self.refinery.rerank(steps, names, k=top_k)
            results = [(n, prob_of.get(n, 0.0)) for n in reranked]
        else:
            results = cand[:top_k]

        # Guarantee top_k REAL step names (never "[UNK]"): if the pool was short
        # (e.g. an aggressive RF mask), fill with the next real vocab steps.
        if len(results) < top_k:
            have = {n for n, _ in results}
            for s in self._real_steps:
                if s not in have:
                    results.append((s, 0.0)); have.add(s)
                    if len(results) >= top_k:
                        break

        return results

    # ── Task 2: Sequence completion ───────────────────────────────────────

    def complete_sequence(
        self,
        partial_steps: list[str],
        family: str,
        max_new_steps: int = 80,
        use_rf_mask: bool = True,
        use_physics: bool = True,
    ) -> list[str]:
        """
        Autoregressively complete a partial sequence.

        With use_physics, the model is the proposal distribution and the physics
        refinery vetoes any step that would create a rule violation, detects
        loops, and guarantees clean termination — so the completion is ALWAYS
        physically valid, even on an unseen family.

        Returns only the NEW steps (after the partial sequence).
        """
        if use_physics:
            def score_fn(steps_so_far):
                preds = self.predict_next_steps(
                    steps_so_far, family, top_k=15,
                    use_rf_mask=use_rf_mask, use_physics=False,
                )
                return [n for n, _ in preds
                        if not (n.startswith("[") and n.endswith("]"))]
            return self.refinery.constrained_decode(
                partial_steps, score_fn, beam=15, max_steps=max_new_steps)

        # ── baseline (no physics): plain greedy, for ablation ──
        current_steps = list(partial_steps)
        new_steps = []
        for _ in range(max_new_steps):
            predictions = self.predict_next_steps(
                current_steps, family, top_k=1, use_rf_mask=use_rf_mask,
                use_physics=False,
            )
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
        use_physics: bool = True,
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
        # Signal 1: Transformer loss. Truncate to the model's positional range so
        # an over-long (or malformed) sequence cannot crash the loss head; the
        # rule validator below still sees the FULL sequence (it is stdlib, no limit).
        ids = self.tokenizer.encode_sequence(steps, family)
        maxlen = getattr(self.model, "max_seq_len", 200)
        if len(ids) > maxlen:
            ids = ids[:2] + ids[2:][-(maxlen - 2):]
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

        # Signal 3: physics rule validator. validate_sequence_combined is EXACT
        # for the three known families AND generalises to unseen vocabulary via
        # category reasoning — unlike the name-based checker — so it covers the
        # hidden 4th family (Task 4). This is the authoritative signal.
        if use_physics:
            rule_violations = validate_sequence_combined(steps)
        else:
            from generate_sequences import validate_sequence
            rule_violations = validate_sequence(steps)
        has_rule_violation = len(rule_violations) > 0
        predicted_rule = rule_violations[0].rule if rule_violations else ""

        # Explicit uncertainty (audit R2): if NO rule fired but the sequence
        # contains an UNKNOWN-classified token, a hidden violation cannot be ruled
        # out — flag INSUFFICIENT_INFORMATION instead of a silent confident pass.
        unknowns = _unknown_tokens(steps) if use_physics else []
        verdict = ("INVALID" if has_rule_violation
                   else "INSUFFICIENT_INFORMATION" if unknowns else "VALID")

        # The 10 rules ARE the definition of invalid, so the validator is
        # authoritative. The model loss / RF only modulate the SCORE (= P valid)
        # to separate the classes for ROC-AUC.
        if has_rule_violation:
            is_valid = False
            combined_score = 0.02
        else:
            loss_score = max(0.0, 1.0 - avg_loss / 3.0)
            rf_score = 1.0 if len(rf_violations) == 0 else max(0.0, 1.0 - len(rf_violations) * 0.2)
            combined_score = max(0.85, 0.5 * loss_score + 0.5 * rf_score)
            is_valid = True
            # No proof of invalidity, but UNKNOWN tokens => cap the confidence and
            # make the uncertainty visible (never a silent 1.0-ish pass on OOD).
            if verdict == "INSUFFICIENT_INFORMATION":
                combined_score = min(combined_score, 0.5)

        # Detect -> explain -> repair: attach the physical reason and concrete
        # fix for every violation (drives the demo and the engineer-facing UX).
        explanation = ""
        suggested_fixes = []
        if has_rule_violation and use_physics:
            findings = _fix.analyze(steps).get("findings", [])
            if findings:
                explanation = findings[0].why
                suggested_fixes = [f.fix_description for f in findings]

        emitted_score = combined_score if SCORE_IS_P_VALID else (1.0 - combined_score)
        return {
            "is_valid": is_valid,
            "verdict": verdict,                    # VALID / INVALID / INSUFFICIENT_INFORMATION
            "insufficient_information": verdict == "INSUFFICIENT_INFORMATION",
            "unknown_tokens": unknowns[:10],       # explicit: what we could not classify
            "score": round(emitted_score, 4),
            "predicted_rule": predicted_rule,
            "explanation": explanation,
            "suggested_fixes": suggested_fixes,
            "all_violations": sorted({v.rule for v in rule_violations}),
            "avg_loss": avg_loss,
            "n_rf_violations": len(rf_violations),
            "n_rule_violations": len(rule_violations),
            "rf_violations": rf_violations[:5],
        }



# ── Submission file generators ────────────────────────────────────────────

def _read_eval(eval_csv: Path, required: list[str]) -> list[dict]:
    """Read an eval CSV and FAIL LOUDLY if the schema doesn't match the spec —
    never silently produce empty / zero-filled submissions on a column mismatch."""
    if not Path(eval_csv).exists():
        raise FileNotFoundError(f"eval file not found: {eval_csv}")
    with open(eval_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(
                f"{Path(eval_csv).name}: missing required column(s) {missing}. "
                f"Found {cols}; spec requires {required}. Refusing to emit a "
                f"submission from an unrecognised schema (see ASSUMPTIONS.md A1/A2).")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{Path(eval_csv).name}: 0 data rows — refusing to emit an empty submission.")
    return rows


def generate_task1_submission(
    predictor: ProcessPredictor,
    eval_csv: Path,
    output_csv: Path,
):
    """Generate Task 1 (next-step prediction) submission file."""
    rows = _read_eval(eval_csv, ["EXAMPLE_ID", "FAMILY", "PARTIAL_SEQUENCE"])

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
    rows = _read_eval(eval_csv, ["EXAMPLE_ID", "FAMILY", "PARTIAL_SEQUENCE"])

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
    rows = _read_eval(eval_csv, ["EXAMPLE_ID", "FAMILY", "SEQUENCE"])

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
        generate_task1_submission(predictor, eval_valid, submission_dir / "nextstep.csv")
        print("\n--- Task 2: Sequence completion ---")
        generate_task2_submission(predictor, eval_valid, submission_dir / "completion.csv")
    else:
        print(f"  Eval file not found: {eval_valid}")

    if eval_anomaly.exists():
        print("\n--- Task 3: Anomaly detection ---")
        generate_task3_submission(predictor, eval_anomaly, submission_dir / "anomaly.csv")
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
