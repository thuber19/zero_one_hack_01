"""
Evaluation script: runs our model on held-out data and produces metrics + CSVs.

Can use either:
  - Our own held-out split (self-eval, always available)
  - The official eval files (eval_input_valid.csv, eval_input_anomaly.csv) once distributed

Usage:
    # Self-evaluation with held-out split
    python src/evaluate.py --self-eval --output-dir outputs

    # Official eval (once you have the files)
    python src/evaluate.py --eval-dir path/to/eval_files --output-dir outputs
"""

import argparse
import csv
import json
import os
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

TRAINING_DATA_DIR = Path(__file__).resolve().parent.parent / "training_data"
sys.path.insert(0, str(TRAINING_DATA_DIR))

from generate_sequences import generate_dataset, validate_sequence
from data_pipeline import prepare_all_data
from tokenizer import StepTokenizer, FAMILY_TOKENS, BOS_ID, EOS_ID, PAD_ID
from transformer_model import create_model
from random_forest import StepCandidateForest
from inference import ProcessPredictor


# ── Self-eval: create our own eval set ────────────────────────────────────

def create_self_eval_set(
    n_per_family: int = 50,
    seed: int = 99999,
) -> tuple[list[dict], list[dict]]:
    """
    Generate held-out sequences for self-evaluation.
    Returns (valid_eval_rows, anomaly_eval_rows).
    """
    rng = random.Random(seed)

    valid_rows = []
    anomaly_rows = []
    example_id = 0

    for family in ["mosfet", "igbt", "ic"]:
        # Generate fresh sequences (different seed than training)
        seqs = generate_dataset(family, n_per_family, seed=seed + hash(family) % 10000, validate=True)

        for seq in seqs:
            # Task 1 & 2: partial sequences at 60% and 80%
            for frac in [0.6, 0.8]:
                cut = int(len(seq) * frac)
                partial = seq[:cut]
                remaining = seq[cut:]
                example_id += 1
                valid_rows.append({
                    "EXAMPLE_ID": f"self_{example_id:04d}",
                    "FAMILY": family.upper(),
                    "COMPLETION_FRACTION": frac,
                    "PARTIAL_SEQUENCE": "|".join(partial),
                    "_full_sequence": seq,
                    "_remaining": remaining,
                    "_next_step": seq[cut] if cut < len(seq) else None,
                })

            # Task 3: valid sequence
            example_id += 1
            anomaly_rows.append({
                "EXAMPLE_ID": f"self_{example_id:04d}",
                "FAMILY": family.upper(),
                "SEQUENCE": "|".join(seq),
                "_is_valid": True,
                "_rule": "",
            })

        # Task 3: generate anomalous sequences by injecting violations
        for seq in seqs[:n_per_family // 2]:
            mutated, rule = inject_violation(seq, rng)
            if mutated is not None:
                example_id += 1
                anomaly_rows.append({
                    "EXAMPLE_ID": f"self_{example_id:04d}",
                    "FAMILY": family.upper(),
                    "SEQUENCE": "|".join(mutated),
                    "_is_valid": False,
                    "_rule": rule,
                })

    rng.shuffle(anomaly_rows)
    return valid_rows, anomaly_rows


def inject_violation(seq: list[str], rng: random.Random) -> tuple[list[str] | None, str]:
    """Inject a single rule violation into a sequence."""
    mutated = list(seq)

    # Try RULE_DEP_NO_CLEAN: remove a clean step before a deposition
    from generate_sequences import DEPOSITION_STEPS, CLEAN_STEPS
    for i, step in enumerate(mutated):
        if step in DEPOSITION_STEPS and i > 0:
            # Find the clean step before it and remove it
            for j in range(i - 1, max(0, i - 12) - 1, -1):
                if mutated[j] in CLEAN_STEPS:
                    removed = mutated.pop(j)
                    violations = validate_sequence(mutated)
                    if violations:
                        return mutated, violations[0].rule
                    # Didn't create a violation, undo
                    mutated.insert(j, removed)
                    break

    # Try RULE_ETCH_NO_MASK: remove DEVELOP PHOTORESIST before an etch
    from generate_sequences import ETCH_STEPS
    for i, step in enumerate(mutated):
        if step in ETCH_STEPS:
            for j in range(i - 1, max(0, i - 12) - 1, -1):
                if mutated[j] == "DEVELOP PHOTORESIST":
                    removed = mutated.pop(j)
                    violations = validate_sequence(mutated)
                    if violations:
                        return mutated, violations[0].rule
                    mutated.insert(j, removed)
                    break

    # Try swapping SHIP LOT before WAFER SORT TEST
    try:
        ship_idx = mutated.index("SHIP LOT")
        sort_idx = mutated.index("WAFER SORT TEST")
        if ship_idx > sort_idx:
            mutated[ship_idx], mutated[sort_idx] = mutated[sort_idx], mutated[ship_idx]
            violations = validate_sequence(mutated)
            if violations:
                return mutated, violations[0].rule
            mutated[ship_idx], mutated[sort_idx] = mutated[sort_idx], mutated[ship_idx]
    except ValueError:
        pass

    return None, ""


# ── Metrics computation ───────────────────────────────────────────────────

def compute_nextstep_metrics(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Compute Task 1 metrics: Top-1/3/5 accuracy and MRR."""
    gt_map = {r["EXAMPLE_ID"]: r["_next_step"] for r in ground_truth if r.get("_next_step")}

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    mrr_sum = 0.0
    total = 0

    for pred in predictions:
        eid = pred["EXAMPLE_ID"]
        if eid not in gt_map:
            continue
        true_step = gt_map[eid]
        ranks = [pred.get(f"RANK_{i}", "") for i in range(1, 6)]
        total += 1

        if true_step == ranks[0]:
            top1_correct += 1
        if true_step in ranks[:3]:
            top3_correct += 1
        if true_step in ranks[:5]:
            top5_correct += 1

        for i, r in enumerate(ranks):
            if r == true_step:
                mrr_sum += 1.0 / (i + 1)
                break

    n = max(total, 1)
    return {
        "top1_accuracy": top1_correct / n,
        "top3_accuracy": top3_correct / n,
        "top5_accuracy": top5_correct / n,
        "mrr": mrr_sum / n,
        "total": total,
    }


def compute_completion_metrics(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Compute Task 2 metrics: exact match, edit distance, token accuracy."""
    gt_map = {r["EXAMPLE_ID"]: r["_remaining"] for r in ground_truth}

    exact_match = 0
    edit_distances = []
    token_accuracies = []
    total = 0

    for pred in predictions:
        eid = pred["EXAMPLE_ID"]
        if eid not in gt_map:
            continue
        true_remaining = gt_map[eid]
        pred_steps = pred["PREDICTED_SEQUENCE"].split("|") if pred["PREDICTED_SEQUENCE"] else []
        pred_steps = [s.strip() for s in pred_steps if s.strip()]
        total += 1

        if pred_steps == true_remaining:
            exact_match += 1

        # Normalized edit distance
        ed = levenshtein(pred_steps, true_remaining)
        max_len = max(len(pred_steps), len(true_remaining), 1)
        edit_distances.append(ed / max_len)

        # Token accuracy (position-wise)
        matches = sum(1 for a, b in zip(pred_steps, true_remaining) if a == b)
        token_accuracies.append(matches / max(len(true_remaining), 1))

    n = max(total, 1)
    return {
        "exact_match_rate": exact_match / n,
        "normalized_edit_distance": np.mean(edit_distances) if edit_distances else 1.0,
        "token_accuracy": np.mean(token_accuracies) if token_accuracies else 0.0,
        "total": total,
    }


def compute_anomaly_metrics(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Compute Task 3 metrics: accuracy, precision, recall, F1, rule attribution."""
    gt_map = {r["EXAMPLE_ID"]: (r["_is_valid"], r["_rule"]) for r in ground_truth}

    tp = fp = tn = fn = 0
    rule_correct = 0
    rule_total = 0

    for pred in predictions:
        eid = pred["EXAMPLE_ID"]
        if eid not in gt_map:
            continue
        true_valid, true_rule = gt_map[eid]
        pred_valid = pred["IS_VALID"] == 1

        if true_valid and pred_valid:
            tn += 1
        elif true_valid and not pred_valid:
            fp += 1
        elif not true_valid and not pred_valid:
            tp += 1
            if true_rule and pred.get("PREDICTED_RULE") == true_rule:
                rule_correct += 1
            if true_rule:
                rule_total += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "binary_accuracy": (tp + tn) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rule_attribution_accuracy": rule_correct / max(rule_total, 1),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "total": total,
    }


def levenshtein(s1: list, s2: list) -> int:
    """Compute Levenshtein edit distance between two lists."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


# ── Main evaluation pipeline ─────────────────────────────────────────────

def run_baseline_eval(output_dir: Path, model_size: str = "small", device: str = "cpu"):
    """Run evaluation with an UNTRAINED model as baseline."""
    print("=== Baseline Evaluation (untrained model) ===\n")

    tokenizer = StepTokenizer.load(output_dir / "tokenizer.txt")
    untrained_model = create_model(tokenizer.vocab_size, size=model_size)
    rf = StepCandidateForest()
    rf.load(output_dir / "random_forest.pkl", tokenizer)
    baseline = ProcessPredictor(tokenizer, untrained_model, rf, device)

    valid_rows, anomaly_rows = create_self_eval_set(n_per_family=50)

    # Task 1 baseline
    print("--- Baseline Task 1: Next-step prediction ---")
    task1_preds = []
    for row in valid_rows:
        partial = row["PARTIAL_SEQUENCE"].split("|")
        family = row["FAMILY"].lower()
        preds = baseline.predict_next_steps(partial, family, top_k=5)
        while len(preds) < 5:
            preds.append(("[UNK]", 0.0))
        task1_preds.append({
            "EXAMPLE_ID": row["EXAMPLE_ID"],
            **{f"RANK_{i+1}": preds[i][0] for i in range(5)},
        })
    task1_metrics = compute_nextstep_metrics(task1_preds, valid_rows)
    print(f"  Top-1 Accuracy: {task1_metrics['top1_accuracy']:.4f}")
    print(f"  Top-3 Accuracy: {task1_metrics['top3_accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {task1_metrics['top5_accuracy']:.4f}")
    print(f"  MRR:            {task1_metrics['mrr']:.4f}")

    results_dir = output_dir / "eval_results"
    results_dir.mkdir(exist_ok=True)

    baseline_metrics = {"task1_nextstep_baseline": task1_metrics}
    with open(results_dir / "baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2, default=str)

    print(f"\nBaseline results saved to {results_dir}/baseline_metrics.json")
    return baseline_metrics


def run_self_eval(output_dir: Path, model_size: str = "small", device: str = "cpu"):
    """Run evaluation using self-generated held-out data."""
    print("=== Self-Evaluation ===\n")

    # Run baseline first
    baseline_metrics = run_baseline_eval(output_dir, model_size, device)

    # Load trained model
    print("\n=== Trained Model Evaluation ===\n")
    print("Loading models...")
    predictor = ProcessPredictor.load(output_dir, model_size=model_size, device=device)

    # Create eval set
    print("Generating held-out eval set...")
    valid_rows, anomaly_rows = create_self_eval_set(n_per_family=50)
    print(f"  Valid eval: {len(valid_rows)} rows")
    print(f"  Anomaly eval: {len(anomaly_rows)} rows")

    results_dir = output_dir / "eval_results"
    results_dir.mkdir(exist_ok=True)

    # ── Task 1: Next-step prediction ──
    print("\n--- Task 1: Next-step prediction ---")
    task1_preds = []
    for row in valid_rows:
        partial = row["PARTIAL_SEQUENCE"].split("|")
        family = row["FAMILY"].lower()
        preds = predictor.predict_next_steps(partial, family, top_k=5)
        while len(preds) < 5:
            preds.append(("[UNK]", 0.0))
        task1_preds.append({
            "EXAMPLE_ID": row["EXAMPLE_ID"],
            **{f"RANK_{i+1}": preds[i][0] for i in range(5)},
        })

    task1_metrics = compute_nextstep_metrics(task1_preds, valid_rows)
    print(f"  Top-1 Accuracy: {task1_metrics['top1_accuracy']:.4f}")
    print(f"  Top-3 Accuracy: {task1_metrics['top3_accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {task1_metrics['top5_accuracy']:.4f}")
    print(f"  MRR:            {task1_metrics['mrr']:.4f}")

    # ── Task 2: Sequence completion ──
    print("\n--- Task 2: Sequence completion ---")
    task2_preds = []
    for row in valid_rows:
        partial = row["PARTIAL_SEQUENCE"].split("|")
        family = row["FAMILY"].lower()
        completion = predictor.complete_sequence(partial, family)
        task2_preds.append({
            "EXAMPLE_ID": row["EXAMPLE_ID"],
            "PREDICTED_SEQUENCE": "|".join(completion),
        })

    task2_metrics = compute_completion_metrics(task2_preds, valid_rows)
    print(f"  Exact Match Rate:          {task2_metrics['exact_match_rate']:.4f}")
    print(f"  Normalized Edit Distance:  {task2_metrics['normalized_edit_distance']:.4f}")
    print(f"  Token Accuracy:            {task2_metrics['token_accuracy']:.4f}")

    # ── Task 3: Anomaly detection ──
    print("\n--- Task 3: Anomaly detection ---")
    task3_preds = []
    for row in anomaly_rows:
        sequence = row["SEQUENCE"].split("|")
        family = row["FAMILY"].lower()
        result = predictor.detect_anomaly(sequence, family)
        task3_preds.append({
            "EXAMPLE_ID": row["EXAMPLE_ID"],
            "IS_VALID": 1 if result["is_valid"] else 0,
            "SCORE": result["score"],
            "PREDICTED_RULE": result["predicted_rule"],
        })

    task3_metrics = compute_anomaly_metrics(task3_preds, anomaly_rows)
    print(f"  Binary Accuracy:           {task3_metrics['binary_accuracy']:.4f}")
    print(f"  Precision:                 {task3_metrics['precision']:.4f}")
    print(f"  Recall:                    {task3_metrics['recall']:.4f}")
    print(f"  F1:                        {task3_metrics['f1']:.4f}")
    print(f"  Rule Attribution Accuracy: {task3_metrics['rule_attribution_accuracy']:.4f}")

    # ── Save results ──
    all_metrics = {
        "task1_nextstep": task1_metrics,
        "task2_completion": {k: v for k, v in task2_metrics.items()},
        "task3_anomaly": {k: v for k, v in task3_metrics.items()},
    }
    with open(results_dir / "self_eval_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Save predictions as CSVs
    _write_csv(results_dir / "self_eval_task1.csv", task1_preds)
    _write_csv(results_dir / "self_eval_task2.csv", task2_preds)
    _write_csv(results_dir / "self_eval_task3.csv", task3_preds)

    print(f"\nResults saved to {results_dir}/")
    return all_metrics


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate process sequence models")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.environ.get("OUTPUT_DIR", "outputs")))
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--self-eval", action="store_true",
                        help="Run self-evaluation with held-out generated data")
    parser.add_argument("--eval-dir", type=Path, default=None,
                        help="Path to official eval files")
    args = parser.parse_args()

    if args.self_eval or args.eval_dir is None:
        run_self_eval(args.output_dir, args.model_size, args.device)
    else:
        from inference import generate_all_submissions
        generate_all_submissions(args.output_dir, args.eval_dir, args.model_size, args.device)


if __name__ == "__main__":
    main()
