#!/usr/bin/env python3
"""
reward.py — turn the physics verifier into a TRAINING SIGNAL (verifier-guided RL).

Everything else in this repo trains the model by IMITATION (copy valid
sequences, copy labelled anomalies). Imitation plateaus at "memorise the
distribution" — the exact failure mode the track is testing for. The way to
push the model's *intrinsic* awareness past that ceiling is reinforcement
learning against a verifier: the model GENERATES, the physics engine SCORES, and
the model is optimised to be physically correct.

We are in the ideal RL setting: the reward is exact (the engine is a perfect,
free, instant verifier), dense (per-step), and works on unseen vocabulary (it is
category-based). This module exposes that reward in the shapes a GRPO / PPO /
best-of-n loop needs. The colleagues' Leonardo training loop imports these.

stdlib + physics only.
"""

from __future__ import annotations

import sys
from copy import copy
from pathlib import Path

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

from physics.state_machine import WaferState, apply_step, validate_by_state_machine

_TERMINAL = "SHIP LOT"


# ---------------------------------------------------------------------------
# Core: incremental legality
# ---------------------------------------------------------------------------

def _replay(steps: list[str]) -> WaferState:
    st = WaferState()
    for s in steps:
        st, _ = apply_step(st, s)
    return st


def step_legal(state: WaferState, step: str) -> bool:
    """True if applying `step` to `state` introduces no violation."""
    _, viol = apply_step(copy(state), step)
    return not viol


def per_step_legality(steps: list[str], start_state: WaferState | None = None) -> list[int]:
    """1/0 legality for each step in order (each evaluated against the running
    state). This is the densest signal — good for token-level credit."""
    st = start_state if start_state is not None else WaferState()
    flags = []
    for s in steps:
        ns, viol = apply_step(st, s)
        flags.append(0 if viol else 1)
        st = ns
    return flags


# ---------------------------------------------------------------------------
# Sequence-level reward (Task 3 / generation correctness)
# ---------------------------------------------------------------------------

def validity_reward(steps: list[str]) -> float:
    """Dense validity in [0,1]: fraction of steps that introduced no violation.
    1.0 == fully physically valid. Smooth, so partial progress is rewarded."""
    if not steps:
        return 0.0
    flags = per_step_legality(steps)
    return sum(flags) / len(flags)


# ---------------------------------------------------------------------------
# Completion reward (Task 2) — the main RL target
# ---------------------------------------------------------------------------

def completion_reward(
    partial: list[str],
    completion: list[str],
    reference: list[str] | None = None,
    *,
    w_valid: float = 1.0,
    w_terminal: float = 0.15,
    w_ref: float = 0.25,
) -> float:
    """
    Reward a generated completion of `partial`.

    Components (weighted):
      * validity   — fraction of completion steps that introduce no violation,
                     evaluated against the state left by `partial`.
      * terminal   — bonus for ending at SHIP LOT with WAFER SORT TEST before it
                     (a properly finished flow), penalised for never terminating.
      * reference  — optional (1 - normalised edit distance) to a gold
                     completion, when one is available (e.g. teacher forcing /
                     curriculum). Omit for pure self-play.

    Returns a scalar; not clamped to [0,1] but bounded by the weights.
    """
    state = _replay(partial)
    flags = per_step_legality(completion, start_state=state)
    validity = (sum(flags) / len(flags)) if flags else 0.0

    full = list(partial) + list(completion)
    has_sort = any(s.upper() == "WAFER SORT TEST" for s in full)
    ends_ship = bool(completion) and completion[-1].upper() == _TERMINAL
    terminal = 1.0 if (ends_ship and has_sort) else (0.3 if ends_ship else 0.0)

    reward = w_valid * validity + w_terminal * terminal
    if reference is not None:
        denom = max(len(reference), len(completion), 1)
        ed = _edit_distance(completion, reference) / denom
        reward += w_ref * (1.0 - ed)
    return reward


# ---------------------------------------------------------------------------
# Next-step reward (Task 1)
# ---------------------------------------------------------------------------

def nextstep_reward(partial: list[str], predicted: str,
                    gold: str | None = None) -> float:
    """Reward a single predicted next step: legal (physics) gets 0.5, matching
    the gold next step (if provided) adds 0.5. Pure-legality signal when no gold
    is available (self-play)."""
    state = _replay(partial)
    r = 0.5 if step_legal(state, predicted) else 0.0
    if gold is not None and predicted == gold:
        r += 0.5
    return r


# ---------------------------------------------------------------------------
# GRPO group reward — score a batch of samples for one prompt
# ---------------------------------------------------------------------------

def grpo_completion_rewards(
    partial: list[str],
    samples: list[list[str]],
    reference: list[str] | None = None,
) -> list[float]:
    """Reward each sampled completion for the same prompt. GRPO uses the
    group-relative advantage (reward minus the group mean), so absolute scaling
    does not matter — only the ranking + spread, which the verifier provides
    exactly. Plug directly into a GRPO step."""
    return [completion_reward(partial, s, reference) for s in samples]


def attribution_reward(steps: list[str], predicted_rule: str) -> float:
    """For RL on the anomaly task: reward naming the rule the verifier finds.
    1.0 correct rule, 0.3 detected-invalid-but-wrong-rule, 0.0 missed; for a
    valid sequence, 1.0 iff predicted_rule is empty."""
    viol = validate_by_state_machine(steps)
    if not viol:
        return 1.0 if not predicted_rule else 0.0
    true_rule = viol[0].rule
    if predicted_rule == true_rule:
        return 1.0
    return 0.3 if predicted_rule else 0.0


# ---------------------------------------------------------------------------
# util
# ---------------------------------------------------------------------------

def _edit_distance(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    from generate_sequences import generate_sequence

    rng = random.Random(0)
    seq = generate_sequence("mosfet", rng)
    cut = int(len(seq) * 0.7)
    partial, gold = seq[:cut], seq[cut:]

    print("Verifier-as-reward demo (higher = more physically aware)")
    print("-" * 60)
    print(f"  reward(gold completion)            = "
          f"{completion_reward(partial, gold, gold):.3f}")

    # a broken completion: gold with all cleans removed
    broken = [s for s in gold if "CLEAN" not in s and s != "RAPID THERMAL ANNEAL"]
    print(f"  reward(clean-stripped completion)  = "
          f"{completion_reward(partial, broken, gold):.3f}")

    # a truncated completion that never ships
    print(f"  reward(truncated, no SHIP LOT)     = "
          f"{completion_reward(partial, gold[:10], gold):.3f}")

    # GRPO group: gold vs broken vs truncated -> group-relative advantage is clear
    grp = grpo_completion_rewards(partial, [gold, broken, gold[:10]], gold)
    mean = sum(grp) / len(grp)
    print(f"\n  GRPO group rewards   = {[round(x,3) for x in grp]}")
    print(f"  group-relative adv.  = {[round(x-mean,3) for x in grp]}")
    print("\n  -> the verifier ranks the physically-correct sample highest,")
    print("     with zero human labels. This is the RL signal for awareness.")
