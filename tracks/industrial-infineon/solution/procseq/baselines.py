"""Statistical baselines: n-gram next-step + rule-oracle anomaly ceiling."""
from collections import Counter, defaultdict
from procseq.grammar import validate_sequence

class NgramModel:
    def __init__(self, n=3):
        self.n = n
        self.table: dict[tuple, Counter] = defaultdict(Counter)
        self.unigram = Counter()

    def fit(self, sequences):
        for seq in sequences:
            for s in seq:
                self.unigram[s] += 1
            for i in range(len(seq)):
                for k in range(1, self.n):
                    if i - k >= 0:
                        ctx = tuple(seq[i-k:i])
                        self.table[ctx][seq[i]] += 1
        return self

    def predict_next(self, prefix, k=5):
        for back in range(self.n - 1, 0, -1):
            ctx = tuple(prefix[-back:]) if back <= len(prefix) else None
            if ctx and self.table.get(ctx):
                return [s for s, _ in self.table[ctx].most_common(k)]
        return [s for s, _ in self.unigram.most_common(k)]

def rule_oracle(steps):
    """Ground-truth checker as an anomaly ceiling. Returns (is_valid, rule)."""
    v = validate_sequence(steps)
    if not v:
        return 1, ""
    return 0, v[0].rule

import math
import torch

class PerplexityAnomaly:
    """Anomaly score = decoder NLL per token; threshold on validation."""
    def __init__(self, model, tokenizer):
        self.model = model.eval(); self.tok = tokenizer; self.threshold = None

    @torch.no_grad()
    def nll(self, steps, family):
        from procseq.tokenizer import encode_sequence
        ids = encode_sequence(self.tok, steps, family=family, add_bos_eos=True)
        x = torch.tensor([ids], device=next(self.model.parameters()).device)
        out = self.model(input_ids=x, labels=x)
        return float(out.loss)

    def fit_threshold(self, valid_examples, quantile=0.95):
        scores = sorted(self.nll(s, f) for s, f in valid_examples)
        idx = min(len(scores) - 1, int(quantile * len(scores)))
        self.threshold = scores[idx]
        return self

    def predict(self, steps, family):
        score = self.nll(steps, family)
        is_valid = 1 if (self.threshold is None or score <= self.threshold) else 0
        p_valid = 1.0 / (1.0 + math.exp(score - (self.threshold or score)))
        return is_valid, p_valid
