"""
Baseline predictors for comparison against trained models.

Baselines:
  1. Random: uniform random step from vocabulary
  2. Frequency: most common next step from training data (bigram)
  3. Untrained model: random weights + RF masking

Usage:
    python src/baseline.py --model-dir outputs --eval-valid eval_input_valid.csv --eval-anomaly eval_input_anomaly.csv --out-dir outputs/baseline_submissions
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SRC_DIR.parent
_DATA_DIR = _PROJECT_DIR / "data"
for _p in (str(_SRC_DIR), str(_DATA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tokenizer import StepTokenizer
from data_pipeline import load_train_csv


def _read_eval_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


class FrequencyBaseline:
    """Predicts the most common next step given the current step (bigram model)."""

    def __init__(self, train_csv: Path):
        pairs = load_train_csv(train_csv)
        # Build bigram counts: current_step -> Counter of next steps
        self.bigrams: dict[str, Counter] = defaultdict(Counter)
        self.global_counts = Counter()

        for family, steps in pairs:
            for i in range(len(steps) - 1):
                self.bigrams[steps[i]][steps[i + 1]] += 1
                self.global_counts[steps[i + 1]] += 1

        # All step names for random fallback
        self.all_steps = list(self.global_counts.keys())

    def predict_next(self, partial: list[str], top_k: int = 5) -> list[str]:
        if partial:
            counts = self.bigrams.get(partial[-1])
            if counts:
                return [step for step, _ in counts.most_common(top_k)]
        # Fallback: global most common
        return [step for step, _ in self.global_counts.most_common(top_k)]

    def complete(self, partial: list[str], max_steps: int = 80) -> list[str]:
        current = list(partial)
        new_steps = []
        for _ in range(max_steps):
            preds = self.predict_next(current, top_k=1)
            if not preds:
                break
            step = preds[0]
            if step == "SHIP LOT":
                new_steps.append(step)
                break
            new_steps.append(step)
            current.append(step)
            # Prevent infinite loops
            if len(new_steps) > 3 and new_steps[-1] == new_steps[-2] == new_steps[-3]:
                break
        return new_steps


class RandomBaseline:
    """Predicts random steps from the vocabulary."""

    def __init__(self, tokenizer: StepTokenizer, seed: int = 42):
        self.rng = random.Random(seed)
        self.steps = [t for t in tokenizer.id2token.values()
                      if not (t.startswith("[") and t.endswith("]"))]

    def predict_next(self, partial: list[str], top_k: int = 5) -> list[str]:
        return self.rng.sample(self.steps, min(top_k, len(self.steps)))

    def complete(self, partial: list[str], max_steps: int = 80) -> list[str]:
        new_steps = []
        for _ in range(max_steps):
            step = self.rng.choice(self.steps)
            if step == "SHIP LOT":
                new_steps.append(step)
                break
            new_steps.append(step)
        return new_steps


def run_baseline(baseline, name, eval_valid_csv, eval_anomaly_csv, out_dir):
    """Run a baseline predictor and generate submission CSVs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {name} Baseline ===")

    # Task 1 + 2
    if eval_valid_csv and Path(eval_valid_csv).exists():
        rows = _read_eval_csv(eval_valid_csv)
        task1, task2 = [], []
        for row in rows:
            partial = [s.strip() for s in row["PARTIAL_SEQUENCE"].strip().split("|") if s.strip()]

            preds = baseline.predict_next(partial, top_k=5)
            while len(preds) < 5:
                preds.append("UNKNOWN")
            task1.append({"EXAMPLE_ID": row["EXAMPLE_ID"].strip(),
                          **{f"RANK_{j+1}": preds[j] for j in range(5)}})

            completion = baseline.complete(partial)
            task2.append({"EXAMPLE_ID": row["EXAMPLE_ID"].strip(),
                          "PREDICTED_SEQUENCE": "|".join(completion)})

        _write_csv(out_dir / "nextstep.csv",
                   ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"], task1)
        _write_csv(out_dir / "completion.csv",
                   ["EXAMPLE_ID", "PREDICTED_SEQUENCE"], task2)
        print(f"  nextstep.csv + completion.csv: {len(task1)} rows")

    # Task 3: baseline just predicts everything as valid
    if eval_anomaly_csv and Path(eval_anomaly_csv).exists():
        rows = _read_eval_csv(eval_anomaly_csv)
        task3 = [{"EXAMPLE_ID": row["EXAMPLE_ID"].strip(),
                  "IS_VALID": 1, "SCORE": 0.5, "PREDICTED_RULE": ""} for row in rows]
        _write_csv(out_dir / "anomaly.csv",
                   ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"], task3)
        print(f"  anomaly.csv: {len(task3)} rows")

    print(f"  Saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline predictors")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Dir with train_sequences.csv + tokenizer.txt")
    parser.add_argument("--eval-valid", type=Path, default=None)
    parser.add_argument("--eval-anomaly", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--baseline", choices=["random", "frequency", "both"], default="both")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.model_dir / "baselines"

    tokenizer = StepTokenizer.load(args.model_dir / "tokenizer.txt")

    if args.baseline in ("random", "both"):
        bl = RandomBaseline(tokenizer)
        run_baseline(bl, "Random", args.eval_valid, args.eval_anomaly,
                     args.out_dir / "random")

    if args.baseline in ("frequency", "both"):
        bl = FrequencyBaseline(args.model_dir / "train_sequences.csv")
        run_baseline(bl, "Frequency (bigram)", args.eval_valid, args.eval_anomaly,
                     args.out_dir / "frequency")


if __name__ == "__main__":
    main()
