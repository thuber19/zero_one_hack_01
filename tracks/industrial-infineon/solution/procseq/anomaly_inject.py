"""Inject each of the 10 process-logic violations into valid sequences.

Each injector returns a NEW perturbed list, or None if not applicable to the
given base sequence. All injectors are verified by validate_sequence in tests.
"""
import random
from procseq.grammar import validate_sequence

# Steps used as anchors (mirrors generate_sequences.py vocab sets).
_DEPOSITIONS = {"THERMAL OXIDATION", "DEPOSIT POLYSILICON", "DEPOSIT BARRIER METAL",
                "DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC",
                "DEPOSIT METAL 1", "DEPOSIT TOP METAL", "DEPOSIT PASSIVATION",
                "DEPOSIT PASSIVATION LAYER", "EPITAXIAL DEPOSITION"}
_CLEANS = {"PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS", "WAFER SURFACE CLEAN",
           "RCA CLEAN 1", "RCA CLEAN 2", "WET CLEAN RCA1", "WET CLEAN RCA2",
           "HF DIP", "OXIDE STRIP", "FRONTSIDE CLEAN", "BACKSIDE CLEAN",
           "CLEAN AFTER ETCH", "CLEAN AFTER OXIDE ETCH", "CLEAN AFTER POLY ETCH",
           "CLEAN AFTER VIA ETCH", "CLEAN AFTER METAL ETCH", "DRY WAFER"}
_ETCHES = {"OXIDE ETCH", "OXIDE ETCH DRY", "POLYSILICON ETCH", "POLYSILICON ETCH DRY",
           "VIA ETCH", "METAL ETCH", "METAL ETCH DRY", "FIELD OXIDE ETCH"}
_IMPLANTS = {"IMPLANT WELL", "IMPLANT SOURCE DRAIN", "IMPLANT LDD", "IMPLANT P BODY",
             "IMPLANT N BUFFER", "IMPLANT N-TYPE", "IMPLANT CHANNEL STOP"}
_CMP = {"CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC", "CMP METAL", "CMP VIA FILL"}
_METAL_ETCH = {"METAL ETCH", "METAL ETCH DRY"}

def _first_index(steps, predicate):
    for i, s in enumerate(steps):
        if predicate(s):
            return i
    return None

def _remove_preceding(steps, trigger_idx, targets, window):
    """Return a copy with all `targets` removed from the window before trigger."""
    lo = max(0, trigger_idx - window)
    return [s for i, s in enumerate(steps)
            if not (lo <= i < trigger_idx and s in targets)]

def inj_dep_no_clean(steps, rng):
    idx = _first_index(steps, lambda s: s in _DEPOSITIONS)
    if idx is None:
        return None
    return _remove_preceding(steps, idx, _CLEANS | {"RAPID THERMAL ANNEAL",
            "THERMAL OXIDATION", "GATE OXIDE PREP", "ANNEAL OXIDE", "EPITAXY ANNEAL"}, 12)

def inj_etch_no_mask(steps, rng):
    idx = _first_index(steps, lambda s: s in _ETCHES)
    if idx is None:
        return None
    return _remove_preceding(steps, idx, {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}, 12)

def inj_metal_etch_no_litho(steps, rng):
    idx = _first_index(steps, lambda s: s in _METAL_ETCH)
    if idx is None:
        return None
    # Remove DEVELOP PHOTORESIST / DEVELOP PAD WINDOW in the window before metal etch
    out = _remove_preceding(steps, idx,
        {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}, 15)
    # Recompute the metal-etch index after _remove_preceding shifted the list
    new_idx = _first_index(out, lambda s: s in _METAL_ETCH)
    if new_idx is None:
        new_idx = idx  # fallback (shouldn't happen)
    # Remove EXPOSE LITHO LEVEL steps that appear within 15 steps before the metal etch
    result = [s for i, s in enumerate(out)
              if not (s.startswith("EXPOSE LITHO LEVEL") and
                      i < new_idx and (new_idx - i) <= 15)]
    return result

def inj_litho_level_skip(steps, rng):
    # swap the order of two consecutive ALIGN MASK levels
    aligns = [i for i, s in enumerate(steps) if s.startswith("ALIGN MASK LEVEL")]
    if len(aligns) < 2:
        return None
    i, j = aligns[0], aligns[1]
    out = steps[:]
    out[i], out[j] = out[j], out[i]
    return out

def inj_implant_no_mask(steps, rng):
    idx = _first_index(steps, lambda s: s in _IMPLANTS)
    if idx is None:
        return None
    return _remove_preceding(steps, idx,
        {"OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW",
         "DEVELOP PHOTORESIST"}, 15)

def inj_cmp_no_dep(steps, rng):
    idx = _first_index(steps, lambda s: s in _CMP)
    if idx is None:
        return None
    return _remove_preceding(steps, idx,
        _DEPOSITIONS | {"FILL VIA METAL", "FILL VIA TUNGSTEN", "DEPOSIT METAL SEED",
                        "DEPOSIT TUNGSTEN SEED", "DEPOSIT BARRIER METAL"}, 6)

def _move_before(steps, src_pred, dst_pred):
    si = _first_index(steps, src_pred)
    di = _first_index(steps, dst_pred)
    if si is None or di is None or si <= di:
        return None
    out = steps[:]
    item = out.pop(si)
    out.insert(di, item)  # now item appears before dst
    return out

def inj_pad_open_before_dep(steps, rng):
    pad = lambda s: s in {"OPEN PAD WINDOW", "OPEN BOND PAD WINDOW", "PAD WINDOW LITHO"}
    dep = lambda s: s in {"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"}
    return _move_before(steps, pad, dep)

def inj_test_before_passivation(steps, rng):
    test = lambda s: s in {"PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST",
                           "LEAKAGE TEST", "SWITCHING TEST"}
    cure = lambda s: s == "CURE PASSIVATION"
    return _move_before(steps, test, cure)

def inj_ship_before_test(steps, rng):
    ship = lambda s: s == "SHIP LOT"
    sort = lambda s: s == "WAFER SORT TEST"
    return _move_before(steps, ship, sort)

def inj_backside_before_passivation(steps, rng):
    bsm = lambda s: s == "DEPOSIT BACKSIDE METAL"
    cure = lambda s: s == "CURE PASSIVATION"
    return _move_before(steps, bsm, cure)

INJECTORS = {
    "RULE_DEP_NO_CLEAN": inj_dep_no_clean,
    "RULE_ETCH_NO_MASK": inj_etch_no_mask,
    "RULE_METAL_ETCH_NO_LITHO": inj_metal_etch_no_litho,
    "RULE_LITHO_LEVEL_SKIP": inj_litho_level_skip,
    "RULE_IMPLANT_NO_MASK": inj_implant_no_mask,
    "RULE_CMP_NO_DEP": inj_cmp_no_dep,
    "RULE_PAD_OPEN_BEFORE_DEP": inj_pad_open_before_dep,
    "RULE_TEST_BEFORE_PASSIVATION": inj_test_before_passivation,
    "RULE_SHIP_BEFORE_TEST": inj_ship_before_test,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": inj_backside_before_passivation,
}

def inject_random(steps, rng):
    """Try injectors in random order; return (seq, rule) for the first that fires
    its intended rule. Falls back to ship-before-test (always applicable)."""
    rules = list(INJECTORS)
    rng.shuffle(rules)
    for rule in rules:
        res = INJECTORS[rule](steps, rng)
        if res is None:
            continue
        fired = {v.rule for v in validate_sequence(res)}
        if rule in fired:
            return res, rule
    # last resort
    res = inj_ship_before_test(steps, rng)
    return res, "RULE_SHIP_BEFORE_TEST"
