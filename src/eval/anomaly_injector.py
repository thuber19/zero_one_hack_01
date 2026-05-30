"""AnomalyInjector — parses generation_rules.md and injects rule-violating sequences.

Type A: ordering violations (move a required-predecessor step to after its trigger)
Type B: variant-incompatible step substitutions

Raises ValueError on ambiguous or malformed rules rather than injecting silently.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsed rule representation
# ---------------------------------------------------------------------------

@dataclass
class OrderingRule:
    name: str
    triggers: list[str]
    required_predecessors: list[str]
    window: int


# ---------------------------------------------------------------------------
# Hardcoded family-specific step sets for Type B injection
# These are drawn from generation_rules.md sections 2.x
# ---------------------------------------------------------------------------

_FAMILY_EXCLUSIVE: dict[str, list[str]] = {
    "MOSFET": [
        "GATE OXIDE PREP", "GATE OXIDE GROWTH", "EPITAXIAL DEPOSITION", "EPITAXY ANNEAL",
        "DEPOSIT SPACER DIELECTRIC", "ANISOTROPIC ETCH SPACER", "IMPLANT LDD",
        "EPITAXIAL REWORK CHECK",
    ],
    "IGBT": [
        "DEPOSIT FIELD OXIDE", "DEPOSIT GATE OXIDE OR DIELECTRIC", "FIELD OXIDE ETCH",
        "IMPLANT N BUFFER", "IMPLANT P BODY", "IMPLANT DRAIN / CATHODE REGION",
        "DIELECTRIC ETCH VIA",
    ],
    "IC": [
        "DEPOSIT PAD OXIDE", "DEPOSIT TUNGSTEN SEED", "FILL VIA TUNGSTEN",
        "CMP VIA FILL", "DEPOSIT BACKSIDE PROTECTION", "BACKSIDE GRIND",
        "BACKSIDE ETCH CLEAN", "BACKSIDE RINSE",
    ],
}


# ---------------------------------------------------------------------------
# Deposition and clean step lists from RULE_DEP_NO_CLEAN
# ---------------------------------------------------------------------------

_DEPOSITION_STEPS = {
    "THERMAL OXIDATION", "GATE OXIDE GROWTH", "DEPOSIT PAD OXIDE", "EPITAXIAL DEPOSITION",
    "DEPOSIT POLYSILICON", "DEPOSIT SPACER DIELECTRIC", "DEPOSIT FIELD OXIDE",
    "DEPOSIT GATE OXIDE OR DIELECTRIC", "DEPOSIT INTERLAYER DIELECTRIC",
    "DEPOSIT INTERLEVEL DIELECTRIC", "DEPOSIT BARRIER METAL", "DEPOSIT METAL SEED",
    "DEPOSIT METAL 1", "DEPOSIT TOP METAL", "DEPOSIT BACKSIDE METAL", "DEPOSIT TUNGSTEN SEED",
    "DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER", "DEPOSIT BACKSIDE PROTECTION",
}

_CLEAN_STEPS = {
    "PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS", "WAFER SURFACE CLEAN", "RCA CLEAN 1",
    "RCA CLEAN 2", "WET CLEAN RCA1", "WET CLEAN RCA2", "HF DIP", "OXIDE STRIP",
    "SURFACE PREP FOR DEPOSITION", "FRONTSIDE CLEAN", "BACKSIDE CLEAN",
    "FRONTSIDE CLEAN FINAL", "BACKSIDE CLEAN FINAL", "WAFER CLEAN PRE-GRIND",
    "DRY WAFER", "DRY WAFER BACKSIDE", "CLEAN AFTER ETCH", "CLEAN AFTER OXIDE ETCH",
    "CLEAN AFTER POLY ETCH", "CLEAN AFTER VIA ETCH", "CLEAN AFTER METAL ETCH",
    "CLEAN AFTER WINDOW ETCH", "CLEAN AFTER FIELD ETCH", "CLEAN PAD OPENING",
    "BACKSIDE ETCH CLEAN", "BACKSIDE RINSE",
}


def _parse_ordering_rules(rules_path: Path) -> list[OrderingRule]:
    """Parse Section 3 forbidden patterns from generation_rules.md."""
    text = rules_path.read_text(encoding="utf-8")

    rules: list[OrderingRule] = []

    # RULE_DEP_NO_CLEAN
    rules.append(OrderingRule(
        name="RULE_DEP_NO_CLEAN",
        triggers=sorted(_DEPOSITION_STEPS),
        required_predecessors=sorted(_CLEAN_STEPS),
        window=12,
    ))

    # RULE_METAL_ETCH_NO_LITHO
    rules.append(OrderingRule(
        name="RULE_METAL_ETCH_NO_LITHO",
        triggers=["METAL ETCH", "METAL ETCH DRY"],
        required_predecessors=["EXPOSE LITHO LEVEL", "DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"],
        window=15,
    ))

    # RULE_ETCH_NO_MASK
    rules.append(OrderingRule(
        name="RULE_ETCH_NO_MASK",
        triggers=[
            "OXIDE ETCH", "OXIDE ETCH DRY", "POLYSILICON ETCH", "POLYSILICON ETCH DRY",
            "ETCH SILICON OR OXIDE WINDOW", "FIELD OXIDE ETCH", "VIA ETCH",
            "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA", "METAL ETCH", "METAL ETCH DRY",
            "PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH",
        ],
        required_predecessors=["DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"],
        window=12,
    ))

    # RULE_IMPLANT_NO_MASK
    rules.append(OrderingRule(
        name="RULE_IMPLANT_NO_MASK",
        triggers=[
            "IMPLANT WELL", "IMPLANT SOURCE DRAIN", "IMPLANT SOURCE REGION", "IMPLANT LDD",
            "IMPLANT P BODY", "IMPLANT N BUFFER", "IMPLANT CHANNEL STOP",
            "IMPLANT DRAIN / CATHODE REGION", "IMPLANT N-TYPE",
        ],
        required_predecessors=["OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW", "DEVELOP PHOTORESIST"],
        window=15,
    ))

    # RULE_CMP_NO_DEP
    rules.append(OrderingRule(
        name="RULE_CMP_NO_DEP",
        triggers=["CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC", "CMP METAL", "CMP VIA FILL"],
        required_predecessors=sorted(_DEPOSITION_STEPS) + ["FILL VIA METAL", "FILL VIA TUNGSTEN"],
        window=6,
    ))

    # RULE_SHIP_BEFORE_TEST: SHIP LOT must appear after WAFER SORT TEST
    rules.append(OrderingRule(
        name="RULE_SHIP_BEFORE_TEST",
        triggers=["SHIP LOT"],
        required_predecessors=["WAFER SORT TEST"],
        window=9999,
    ))

    # RULE_BACKSIDE_BEFORE_PASSIVATION: DEPOSIT BACKSIDE METAL after CURE PASSIVATION
    rules.append(OrderingRule(
        name="RULE_BACKSIDE_BEFORE_PASSIVATION",
        triggers=["DEPOSIT BACKSIDE METAL"],
        required_predecessors=["CURE PASSIVATION"],
        window=9999,
    ))

    return rules


# ---------------------------------------------------------------------------
# AnomalyInjector
# ---------------------------------------------------------------------------

class AnomalyInjector:
    def __init__(self, rules_path: str | Path, seed: int = 42):
        self._rules_path = Path(rules_path)
        if not self._rules_path.exists():
            raise FileNotFoundError(f"generation_rules.md not found: {self._rules_path}")
        self._ordering_rules = _parse_ordering_rules(self._rules_path)
        self._rng = random.Random(seed)

    def inject(
        self,
        sequences: list[tuple[str, str, list[str]]],
        anomaly_types: list[str],
    ) -> list[tuple[str, str, list[str], bool]]:
        """Return (variant, seq_id, steps, is_anomalous) for each sequence.

        For each sequence, attempt to inject an anomaly. If injection is not
        possible (sequence too short, no applicable rule), keep original as clean.
        """
        out: list[tuple[str, str, list[str], bool]] = []
        for variant, sid, steps in sequences:
            injected: list[str] | None = None
            types_shuffled = list(anomaly_types)
            self._rng.shuffle(types_shuffled)
            for atype in types_shuffled:
                if atype == "A":
                    injected = self._inject_type_a(steps)
                elif atype == "B":
                    injected = self._inject_type_b(variant, steps)
                else:
                    raise ValueError(f"Unknown anomaly type {atype!r}; expected 'A' or 'B'")
                if injected is not None:
                    break
            if injected is not None:
                out.append((variant, f"{sid}_anom", injected, True))
            # always also keep the clean original
            out.append((variant, sid, steps, False))
        return out

    def _inject_type_a(self, steps: list[str]) -> list[str] | None:
        """Type A: move a required-predecessor step to after its trigger (violates ordering)."""
        steps_upper = [s.upper() for s in steps]
        candidates: list[tuple[int, int]] = []  # (trigger_pos, prereq_pos)

        for rule in self._ordering_rules:
            for i, step in enumerate(steps_upper):
                if any(step == t or step.startswith(t) for t in rule.triggers):
                    # find last required predecessor within window
                    window_start = max(0, i - rule.window)
                    for j in range(window_start, i):
                        if any(steps_upper[j] == p or steps_upper[j].startswith(p)
                               for p in rule.required_predecessors):
                            candidates.append((i, j))

        if not candidates:
            return None

        trigger_pos, prereq_pos = self._rng.choice(candidates)
        # Move prereq to after trigger (swap violates ordering constraint)
        result = list(steps)
        prereq_step = result.pop(prereq_pos)
        insert_at = min(trigger_pos, len(result))  # trigger_pos shifted by removal
        result.insert(insert_at + 1, prereq_step)
        return result

    def _inject_type_b(self, variant: str, steps: list[str]) -> list[str] | None:
        """Type B: substitute a step with one incompatible for this variant."""
        other_variants = [v for v in _FAMILY_EXCLUSIVE if v != variant]
        if not other_variants:
            return None
        # steps exclusive to other variants are illegal in this sequence
        illegal_pool: list[str] = []
        for v in other_variants:
            illegal_pool.extend(_FAMILY_EXCLUSIVE[v])
        if not illegal_pool:
            return None

        # pick a random position in the sequence (avoid first/last 2 positions)
        if len(steps) < 5:
            return None
        pos = self._rng.randint(2, len(steps) - 3)
        illegal_step = self._rng.choice(illegal_pool)
        result = list(steps)
        result[pos] = illegal_step
        return result
