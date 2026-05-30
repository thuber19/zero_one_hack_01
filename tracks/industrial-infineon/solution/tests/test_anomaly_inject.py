# tests/test_anomaly_inject.py
import random
from procseq import anomaly_inject as ai
from procseq.data import load_provided
from procseq.grammar import validate_sequence, RULE_IDS

def _a_valid_mosfet():
    for s in load_provided("MOSFET").values():
        if not validate_sequence(s):
            return s
    raise AssertionError("no clean MOSFET sequence found")

def test_each_injector_triggers_its_rule():
    base = _a_valid_mosfet()
    rng = random.Random(0)
    fired_any = False
    for rule, fn in ai.INJECTORS.items():
        res = fn(base, rng)
        if res is None:
            continue  # not applicable to this base sequence
        fired_any = True
        rules = {v.rule for v in validate_sequence(res)}
        assert rule in rules, f"{rule} expected, got {rules}"
    assert fired_any

def test_inject_random_returns_labeled_invalid():
    base = _a_valid_mosfet()
    rng = random.Random(1)
    seq, rule = ai.inject_random(base, rng)
    assert rule in RULE_IDS
    assert rule in {v.rule for v in validate_sequence(seq)}
