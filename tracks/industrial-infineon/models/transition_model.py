"""
models/transition_model.py

Bigram + trigram Markov chain over process step sequences.

The model learns the empirical transition probabilities from training data
and uses them for:
  - Task 1: next-step prediction (top-K candidates)
  - Task 3: anomaly scoring (low log-probability ↔ suspicious sequence)
  - Task 2: proposal distribution for grammar-constrained beam search

Generalisation to unseen families (Task 4):
  When the model encounters a step it has not seen before, it falls back to
  the category-level transition distribution: P(cat_B | cat_A) learned from
  the same training data. Steps in those categories are then ranked by their
  unigram frequency. This means the model always produces a non-trivial
  prediction even for a completely novel step vocabulary.

No external dependencies: stdlib only (collections, math, csv, pickle).
"""

from __future__ import annotations

import csv
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# Make both the repo root and training_data importable regardless of cwd
_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE))                   # industrial-infineon/ (for physics.*)
sys.path.insert(0, str(_HERE / "training_data")) # for generate_sequences

from physics.ontology import classify_step


# ---------------------------------------------------------------------------
# TransitionModel
# ---------------------------------------------------------------------------

class TransitionModel:
    """
    Bigram / trigram language model over semiconductor process steps.

    Training data format: the dict returned by read_csv_sequences() —
        { sequence_id: [step_0, step_1, ...], ... }

    All sequences from all families should be passed together so the model
    learns cross-family transitions (e.g. the ILD block is shared).
    """

    def __init__(self) -> None:
        # Step-level counts
        self._bigram:  dict[str, Counter]             = defaultdict(Counter)
        self._trigram: dict[tuple[str, str], Counter] = defaultdict(Counter)
        self._unigram: Counter                        = Counter()

        # Category-level counts (fallback for unseen step names)
        self._cat_bigram:  dict[str, Counter] = defaultdict(Counter)
        self._cat_unigram: Counter            = Counter()

        # Vocabulary
        self._vocab: set[str] = set()
        self._vocab_size: int = 0

        # Flag
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, sequences: dict[str, list[str]]) -> None:
        """
        Fit the model on a dict of sequences.

        Parameters
        ----------
        sequences : dict[str, list[str]]
            {sequence_id: [step, ...]}  — from read_csv_sequences()
        """
        for seq in sequences.values():
            cats = [classify_step(s) for s in seq]
            self._unigram.update(seq)
            self._cat_unigram.update(cats)

            for i, step in enumerate(seq):
                self._vocab.add(step)
                if i > 0:
                    self._bigram[seq[i - 1]][step] += 1
                    self._cat_bigram[cats[i - 1]][cats[i]] += 1
                if i > 1:
                    self._trigram[(seq[i - 2], seq[i - 1])][step] += 1

        self._vocab_size = len(self._vocab)
        self._trained = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    # Linear-interpolation weight for the trigram component when both a trigram
    # and a bigram context are available (Jelinek–Mercer style backoff). The
    # bigram supplies a smoothed floor so the trigram is never overconfident
    # (no probability collapses to 1.0, no log-prob cliffs in scoring).
    _TRIGRAM_LAMBDA: float = 0.7

    def _bigram_laplace(self, prev: str) -> Optional[dict[str, float]]:
        """Laplace-smoothed P(next | prev) over the full vocabulary, or None."""
        counts = self._bigram.get(prev)
        if not counts:
            return None
        total = sum(counts.values())
        denom = total + self._vocab_size
        return {
            s: (counts.get(s, 0) + 1) / denom
            for s in self._vocab
        }

    def _step_probs(
        self,
        prev: Optional[str],
        prev_prev: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Return a probability distribution over next steps.

        Backoff chain:
          1. trigram ⊕ bigram   — interpolated (trigram sharpness + bigram floor)
          2. bigram (Laplace)   — when no usable trigram context
          3. category-level     — for unseen step names (Task 4 OOD)
          4. unigram            — last resort

        Every returned distribution assigns non-zero mass to all vocab items,
        which keeps sequence_log_prob() numerically stable.

        Parameters
        ----------
        prev      : last step in the partial sequence
        prev_prev : second-to-last step (for trigram context)
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        bigram = self._bigram_laplace(prev) if prev is not None else None

        # --- Trigram interpolated with the bigram floor ---
        if prev_prev is not None and prev is not None:
            tri = self._trigram.get((prev_prev, prev))
            if tri:
                tri_total = sum(tri.values())
                lam = self._TRIGRAM_LAMBDA
                if bigram is not None:
                    # bigram (Laplace) already spans the full vocab, so iterate it.
                    return {
                        s: lam * (tri.get(s, 0) / tri_total)
                           + (1.0 - lam) * bigram[s]
                        for s in self._vocab
                    }
                # No bigram context: smooth the trigram by itself.
                denom = tri_total + self._vocab_size
                return {
                    s: (tri.get(s, 0) + 1) / denom
                    for s in self._vocab
                }

        # --- Step-level bigram (Laplace smoothed) ---
        if bigram is not None:
            return bigram

        # --- Category-level fallback (for unknown step names in Task 4) ---
        prev_cat = classify_step(prev) if prev else "LOGISTICS"
        cat_counts = self._cat_bigram.get(prev_cat, Counter())

        if cat_counts:
            cat_total = sum(cat_counts.values())
            cat_probs = {c: n / cat_total for c, n in cat_counts.items()}

            # Weight each step by: P(category of step) * P(step | category)
            uni_total = sum(self._unigram.values())
            result: dict[str, float] = {}
            for s, uni_count in self._unigram.items():
                s_cat = classify_step(s)
                cat_p = cat_probs.get(s_cat, 1e-9)
                result[s] = cat_p * (uni_count / uni_total)
            return result

        # Last resort: uniform unigram
        uni_total = sum(self._unigram.values())
        return {s: c / uni_total for s, c in self._unigram.items()}

    def predict_top_k(
        self,
        partial_sequence: list[str],
        k: int = 5,
    ) -> list[str]:
        """
        Return the top-k most likely next steps for a partial sequence.

        Parameters
        ----------
        partial_sequence : list[str]
            Steps seen so far (non-empty).
        k : int
            Number of candidates to return.

        Returns
        -------
        list[str]
            Top-k step names, most likely first.
        """
        if not partial_sequence:
            # Very start of a sequence — RECEIVE WAFER LOT is always first;
            # fill remaining ranks by global step frequency (deterministic order).
            ranked = ["RECEIVE WAFER LOT"] + [
                s for s, _ in self._unigram.most_common() if s != "RECEIVE WAFER LOT"
            ]
            return ranked[:k]

        prev = partial_sequence[-1]
        prev_prev = partial_sequence[-2] if len(partial_sequence) > 1 else None
        probs = self._step_probs(prev, prev_prev)

        return sorted(probs, key=probs.__getitem__, reverse=True)[:k]

    def sequence_log_prob(self, steps: list[str]) -> float:
        """
        Compute log P(sequence) under the bigram model.

        A very negative value indicates a sequence whose transitions are
        rarely or never observed in training data — a useful anomaly signal.

        Returns the per-step mean log-probability (normalised by length) so
        that sequences of different lengths are comparable.
        """
        if len(steps) < 2:
            return 0.0

        log_p = 0.0
        for i in range(1, len(steps)):
            prev = steps[i - 1]
            prev_prev = steps[i - 2] if i > 1 else None
            probs = self._step_probs(prev, prev_prev)
            p = probs.get(steps[i], 1e-12)
            log_p += math.log(max(p, 1e-12))

        return log_p / (len(steps) - 1)   # per-step mean

    def score_anomaly(self, steps: list[str]) -> float:
        """
        Return an anomaly score in [0.0, 1.0]:
          0.0 — very likely invalid (low-probability sequence)
          1.0 — looks valid (high-probability transitions)

        Used as the SCORE column in Task 3 submissions.
        """
        per_step_log_p = self.sequence_log_prob(steps)
        # Empirical calibration: valid sequences cluster around -2 to -4 per step.
        # Map that range to [0.9, 0.5] via a soft sigmoid.
        # Values below -6 → near 0.0 (anomalous)
        # Values above -1 → near 1.0 (very confident valid)
        x = per_step_log_p + 3.0   # shift so "average valid" ≈ 0
        score = 1.0 / (1.0 + math.exp(-x))
        return round(float(score), 4)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Pickle the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        print(f"  Model saved -> {path}")

    @staticmethod
    def load(path: Path | str) -> "TransitionModel":
        """Load a pickled model from disk."""
        with Path(path).open("rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = (
            f"vocab={self._vocab_size}, "
            f"bigram_keys={len(self._bigram)}, "
            f"trigram_keys={len(self._trigram)}"
        )
        return f"TransitionModel({status})"


# ---------------------------------------------------------------------------
# Factory: build and optionally cache the model
# ---------------------------------------------------------------------------

def build_model(
    data_dir: Optional[Path | str] = None,
    cache_path: Optional[Path | str] = None,
    families: tuple[str, ...] = ("MOSFET", "IGBT", "IC"),
    force_rebuild: bool = False,
) -> TransitionModel:
    """
    Build a TransitionModel from the pre-generated training sequences.

    Parameters
    ----------
    data_dir    : directory containing *_variants.csv files.
                  Defaults to <repo_root>/training_data/
    cache_path  : if provided, the model is saved here after training and
                  loaded from here on subsequent calls (speeds up iteration).
    families    : which families to include in training.
    force_rebuild : ignore any cached model and retrain from scratch.

    Returns
    -------
    TransitionModel, ready to use.
    """
    data_dir = Path(data_dir) if data_dir else _HERE / "training_data"
    cache_path = Path(cache_path) if cache_path else None

    # Load from cache if available. Fail-safe: a corrupt/incompatible/foreign
    # cache must never crash the pipeline (and never let a bad pickle take over)
    # — on ANY load error we discard it and rebuild from the source CSVs.
    if cache_path and cache_path.exists() and not force_rebuild:
        try:
            print(f"  Loading cached model from {cache_path} …")
            m = TransitionModel.load(cache_path)
            if getattr(m, "_trained", False):
                return m
            raise ValueError("cache not a trained model")
        except Exception as e:
            print(f"  WARNING: ignoring unusable cache ({type(e).__name__}); rebuilding.")

    model = TransitionModel()

    try:
        from generate_sequences import read_csv_sequences
    except ImportError:
        raise ImportError(
            "Could not import generate_sequences. "
            "Run this script from the industrial-infineon/ directory."
        )

    total_seqs = 0
    for family in families:
        path = data_dir / f"{family}_variants.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {family}")
            continue
        print(f"  Loading {family} sequences from {path.name} …", end=" ")
        seqs = read_csv_sequences(path)
        model.train(seqs)
        total_seqs += len(seqs)
        print(f"{len(seqs)} sequences loaded.")

    print(f"  Training complete. {total_seqs} total sequences. {model}")

    if cache_path:
        model.save(cache_path)

    return model


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("Building transition model from training data …")
    m = build_model()
    print()

    # Test next-step prediction
    partial = [
        "RECEIVE WAFER LOT",
        "LOT IDENTIFICATION",
        "INITIAL WAFER INSPECTION",
        "MEASURE THICKNESS",
        "MEASURE SURFACE PARTICLES",
        "PRE CLEAN WAFER",
        "RCA CLEAN 1",
    ]
    print("Partial sequence:")
    for s in partial:
        print(f"  {s}")
    top5 = m.predict_top_k(partial, k=5)
    print(f"\nTop-5 predicted next steps:")
    for i, s in enumerate(top5, 1):
        print(f"  {i}. {s}")

    # Test anomaly scoring
    from generate_sequences import read_csv_sequences, validate_sequence
    valid_seqs = read_csv_sequences(
        Path(__file__).parent.parent / "training_data" / "MOSFET_variants.csv"
    )
    sample_valid = list(valid_seqs.values())[0]
    print(f"\nAnomaly score for a VALID sequence: {m.score_anomaly(sample_valid):.4f}")

    # Break a sequence
    broken = list(sample_valid)
    for i, s in enumerate(broken):
        if "CLEAN AFTER" in s:
            broken.pop(i)
            break
    print(f"Anomaly score for a BROKEN sequence: {m.score_anomaly(broken):.4f}")
