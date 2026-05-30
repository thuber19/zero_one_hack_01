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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for physics.*

from physics.ontology import classify_step as _classify_step


def _block_seq(steps):
    """Collapse a step list to its category ('block') sequence."""
    out = []
    for s in steps:
        c = _classify_step(s)
        if not out or out[-1] != c:
            out.append(c)
    return out


def _roc_auc(pos_scores, neg_scores):
    """Mann-Whitney ROC-AUC; pos=invalid-class scores, neg=valid. None if a
    class is empty."""
    if not pos_scores or not neg_scores:
        return None
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0)
               for a in pos_scores for b in neg_scores)
    return wins / (len(pos_scores) * len(neg_scores))

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
    block_accuracies = []
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

        # Block-level accuracy: position-wise match on the category ("block")
        # sequence (consecutive duplicates collapsed).
        tb, pb = _block_seq(true_remaining), _block_seq(pred_steps)
        bm = sum(1 for a, b in zip(pb, tb) if a == b)
        block_accuracies.append(bm / max(len(tb), 1))

    n = max(total, 1)
    return {
        "exact_match_rate": exact_match / n,
        "normalized_edit_distance": float(np.mean(edit_distances)) if edit_distances else 1.0,
        "token_accuracy": float(np.mean(token_accuracies)) if token_accuracies else 0.0,
        "block_accuracy": float(np.mean(block_accuracies)) if block_accuracies else 0.0,
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

    # ROC-AUC: positive class = INVALID. SCORE is P(valid), so P(invalid)=1-SCORE.
    pos, neg = [], []   # pos = invalid-class scores, neg = valid-class scores
    for pred in predictions:
        eid = pred["EXAMPLE_ID"]
        if eid not in gt_map:
            continue
        true_valid, _ = gt_map[eid]
        try:
            p_invalid = 1.0 - float(pred.get("SCORE", 0.5))
        except (TypeError, ValueError):
            p_invalid = 0.5
        (neg if true_valid else pos).append(p_invalid)
    auc = _roc_auc(pos, neg)

    return {
        "binary_accuracy": (tp + tn) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
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

def run_self_eval(output_dir: Path, model_size: str = "small", device: str = "cpu"):
    """Run evaluation using self-generated held-out data."""
    print("=== Self-Evaluation ===\n")

    # Load model
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
    print(f"  Block-level Accuracy:      {task2_metrics['block_accuracy']:.4f}")

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
    if task3_metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC:                   {task3_metrics['roc_auc']:.4f}")

    # ── Per-family breakdown (spec requires it) ──
    print("\n--- Per-family breakdown ---")
    for fam in ("MOSFET", "IGBT", "IC"):
        vr = [r for r in valid_rows if r["FAMILY"] == fam]
        vids = {r["EXAMPLE_ID"] for r in vr}
        ar = [r for r in anomaly_rows if r["FAMILY"] == fam]
        aids = {r["EXAMPLE_ID"] for r in ar}
        m1 = compute_nextstep_metrics([p for p in task1_preds if p["EXAMPLE_ID"] in vids], vr)
        m2 = compute_completion_metrics([p for p in task2_preds if p["EXAMPLE_ID"] in vids], vr)
        m3 = compute_anomaly_metrics([p for p in task3_preds if p["EXAMPLE_ID"] in aids], ar)
        auc = m3.get("roc_auc")
        print(f"  {fam:<7} T1 Top1={m1['top1_accuracy']:.3f} Top5={m1['top5_accuracy']:.3f} "
              f"MRR={m1['mrr']:.3f} | T2 Exact={m2['exact_match_rate']:.3f} "
              f"Block={m2['block_accuracy']:.3f} EditD={m2['normalized_edit_distance']:.3f} | "
              f"T3 F1={m3['f1']:.3f} AUC={auc if auc is None else round(auc,3)}")

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
