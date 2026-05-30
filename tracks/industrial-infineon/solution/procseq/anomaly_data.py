"""Build a balanced (valid, invalid+rule) training set for the encoder."""
import random
from procseq.data import scale_family
from procseq.anomaly_inject import inject_random
from procseq.grammar import FAMILIES, validate_sequence

def build_anomaly_training(n_per_family, seed):
    rng = random.Random(seed)
    items = []  # (steps, family, is_valid, rule)
    for fam in FAMILIES:
        seqs = scale_family(fam, n_per_family, seed)
        for s in seqs:
            if validate_sequence(s):
                continue  # keep the valid pool clean
            # (rare) skip already-invalid generated seqs
        for s in seqs:
            items.append((s, fam, 1, ""))                 # valid
            inv, rule = inject_random(s, rng)
            items.append((inv, fam, 0, rule))             # matched invalid
    rng.shuffle(items)
    return items
