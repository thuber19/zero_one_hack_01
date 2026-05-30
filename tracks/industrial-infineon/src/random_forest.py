"""
Random forest for candidate step filtering.

Given the current context (family, current step, recent history, litho level,
position), predicts the set of plausible next steps. Used to mask the
transformer's output logits during inference.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_pipeline import extract_rf_features, build_transition_map
from tokenizer import StepTokenizer


class StepCandidateForest:
    """
    Wraps a RandomForestClassifier to produce candidate sets for next-step prediction.

    At inference time, we use predict_proba to get the top-K most likely next steps,
    then use those as a mask for the transformer's logits.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 20, top_k: int = 15):
        self.top_k = top_k
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        self.is_fitted = False
        self.classes_: np.ndarray | None = None
        # Fallback transition map for steps the forest hasn't seen
        self.transition_map: dict[str, set[str]] | None = None
        self.tokenizer: StepTokenizer | None = None

    def train(
        self,
        sequences: list[tuple[str, list[str]]],
        tokenizer: StepTokenizer,
        test_size: float = 0.1,
    ) -> dict[str, float]:
        """Train the forest and return accuracy metrics."""
        self.tokenizer = tokenizer
        self.transition_map = build_transition_map(sequences)

        X, y = extract_rf_features(sequences, tokenizer)
        print(f"RF training data: {X.shape[0]} transitions, {len(np.unique(y))} unique next-steps")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print("Training random forest...")
        self.clf.fit(X_train, y_train)
        self.is_fitted = True
        self.classes_ = self.clf.classes_

        # Evaluate
        train_acc = self.clf.score(X_train, y_train)
        test_acc = self.clf.score(X_test, y_test)

        # Top-K accuracy: does the true next step appear in top-K predictions?
        proba = self.clf.predict_proba(X_test)
        top_k_indices = np.argsort(proba, axis=1)[:, -self.top_k:]
        top_k_classes = self.classes_[top_k_indices]
        top_k_acc = np.mean([y_test[i] in top_k_classes[i] for i in range(len(y_test))])

        metrics = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            f"top_{self.top_k}_accuracy": top_k_acc,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        print(f"  Train acc: {train_acc:.4f}")
        print(f"  Test acc:  {test_acc:.4f}")
        print(f"  Top-{self.top_k} acc: {top_k_acc:.4f}")
        return metrics

    def get_candidates(
        self,
        family: str,
        current_step: str,
        prev_steps: list[str],
        litho_level: int,
        position_frac: float,
    ) -> list[int]:
        """
        Get top-K candidate token IDs for the next step.
        Returns list of token IDs sorted by RF probability (descending).
        """
        if not self.is_fitted or self.tokenizer is None:
            raise RuntimeError("Forest not trained yet. Call train() first.")

        from block_classifier import classify_step_block_id

        family_id_map = {"mosfet": 0, "igbt": 1, "ic": 2}
        fam_id = family_id_map[family.lower()]
        curr_id = self.tokenizer.encode_step(current_step)
        prev1 = self.tokenizer.encode_step(prev_steps[-1]) if len(prev_steps) >= 1 else 0
        prev2 = self.tokenizer.encode_step(prev_steps[-2]) if len(prev_steps) >= 2 else 0
        prev3 = self.tokenizer.encode_step(prev_steps[-3]) if len(prev_steps) >= 3 else 0
        block_id = classify_step_block_id(current_step)

        features = np.array([[fam_id, curr_id, prev1, prev2, prev3, litho_level, position_frac, block_id]])
        proba = self.clf.predict_proba(features)[0]

        # Get top-K
        top_indices = np.argsort(proba)[-self.top_k:][::-1]
        candidates = self.classes_[top_indices].tolist()

        # Also add transition-map fallback candidates
        key = f"{family.lower()}|{current_step}"
        if self.transition_map and key in self.transition_map:
            for step_name in self.transition_map[key]:
                tid = self.tokenizer.encode_step(step_name)
                if tid not in candidates:
                    candidates.append(tid)

        return candidates

    def get_candidate_mask(
        self,
        vocab_size: int,
        family: str,
        current_step: str,
        prev_steps: list[str],
        litho_level: int,
        position_frac: float,
    ) -> np.ndarray:
        """
        Get a binary mask of shape (vocab_size,) where 1 = allowed candidate.
        Used to mask transformer logits.
        """
        candidates = self.get_candidates(
            family, current_step, prev_steps, litho_level, position_frac
        )
        mask = np.zeros(vocab_size, dtype=np.float32)
        for c in candidates:
            if 0 <= c < vocab_size:
                mask[c] = 1.0
        # Always allow EOS
        mask[2] = 1.0  # EOS_ID
        return mask

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({
                "clf": self.clf,
                "top_k": self.top_k,
                "classes": self.classes_,
                "transition_map": self.transition_map,
                "is_fitted": self.is_fitted,
            }, f)
        print(f"  Saved RF model to {path}")

    def load(self, path: Path, tokenizer: StepTokenizer):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.clf = data["clf"]
        self.top_k = data["top_k"]
        self.classes_ = data["classes"]
        self.transition_map = data["transition_map"]
        self.is_fitted = data["is_fitted"]
        self.tokenizer = tokenizer
        print(f"  Loaded RF model from {path}")
