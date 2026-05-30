#!/usr/bin/env python3
"""
refinery.py — Neuro-symbolic refinement layer ("model proposes, physics disposes").

This is the small ecosystem that sits ON TOP of any sequence model — a trained
LLM, a from-scratch decoder, our n-gram model, anything — and refines its raw
output so it is always physically legal, never loops, and always terminates.

The model is treated as a black-box *scorer*:

    score_fn(prefix: list[str]) -> list[str]            # ranked next-step candidates
                                or dict[str, float]     # token -> score

That is the ONLY thing the refinery needs from the model. Wrap an LLM by mapping
the prefix to a prompt and reading its next-token logits over the step
vocabulary; wrap a from-scratch model the same way; wrap our TransitionModel
with the one-line adapter at the bottom of this file. Everything else here is
model-independent.

What it gives you
-----------------
  rerank(prefix, ranked)        Task 1: drop physically-illegal candidates,
                                keep the model's order, optionally demote
                                category-inconsistent ones. The true next step
                                is almost always legal, so this lifts Top-1/MRR
                                "for free".
  constrained_decode(prefix, score_fn)
                                Task 2: greedy decode where the model proposes
                                and the physics state machine vetoes any step
                                that would create a violation; loop-detection +
                                guaranteed clean termination.
  guard(sequence)               Task 3 / safety: (is_valid, violations). Use as
                                the symbolic half of the anomaly ensemble, or to
                                certify that a generated completion is clean.

Why this is the right shape (per the design discussion)
-------------------------------------------------------
  * The hard constraint is expressed at the CATEGORY level (via the physics
    state machine), so it works on an unseen 4th-family vocabulary AND on a
    novel block structure — neither requires the exact step names.
  * It is open and local (no API), interpretable (every veto cites a rule), and
    it upgrades a weak model into a always-valid one without retraining.

Dependencies: physics/ only (stdlib otherwise). The __main__ demo additionally
uses models/ to show a real and a deliberately-weak scorer.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from physics.ontology import classify_step
from physics.state_machine import (
    WaferState,
    apply_step,
    validate_by_state_machine,
)

# A scorer returns either a ranked list of candidate steps or a {step: score} map.
ScoreResult = Union[list[str], dict[str, float]]
ScoreFn = Callable[[list[str]], ScoreResult]

_TERMINAL = "SHIP LOT"


def _ranked(result: ScoreResult) -> list[str]:
    """Normalise a scorer result to a ranked list (highest first)."""
    if isinstance(result, dict):
        return sorted(result, key=result.__getitem__, reverse=True)
    return list(result)


def _looks_stuck(generated: list[str], min_len: int = 8) -> bool:
    """True if the tail is a short-period repetition (model is cycling)."""
    if len(generated) < min_len:
        return False
    tail = generated[-min_len:]
    for period in (1, 2, 3, 4):
        if period * 2 > min_len:
            break
        if all(tail[i] == tail[i % period] for i in range(min_len)):
            return True
    return False


# ---------------------------------------------------------------------------
# Learned category grammar (optional soft signal)
# ---------------------------------------------------------------------------

def learn_category_grammar(sequences: Iterable[list[str]]) -> dict[str, set[str]]:
    """
    Learn which physical categories were ever observed to follow which, from
    training data. Used as a SOFT re-rank signal (demote transitions never seen
    in training) — never as a hard block, so it does not over-constrain a novel
    structure. Returns {category: {categories observed to follow it}}.
    """
    grammar: dict[str, set[str]] = defaultdict(set)
    for seq in sequences:
        cats = [classify_step(s) for s in seq]
        for a, b in zip(cats, cats[1:]):
            grammar[a].add(b)
    return dict(grammar)


# ---------------------------------------------------------------------------
# The refinery
# ---------------------------------------------------------------------------

class PhysicsRefinery:
    """
    Wraps a model scorer with physical legality, category grammar, and safe
    decoding. Stateless across calls except for the (optional) learned grammar.
    """

    def __init__(
        self,
        category_grammar: Optional[dict[str, set[str]]] = None,
        category_mode: str = "soft",      # "off" | "soft" | "hard"
    ) -> None:
        self.category_grammar = category_grammar
        assert category_mode in ("off", "soft", "hard")
        self.category_mode = category_mode

    # ── state helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def replay(prefix: list[str]) -> WaferState:
        """Replay a prefix through the state machine and return the end state."""
        st = WaferState()
        for s in prefix:
            st, _ = apply_step(st, s)
        return st

    @staticmethod
    def is_legal(state: WaferState, step: str) -> bool:
        """True if applying `step` to `state` creates no physical violation."""
        _, viol = apply_step(copy(state), step)
        return not viol

    def _category_ok(self, prev_step: Optional[str], candidate: str) -> bool:
        if self.category_mode == "off" or not self.category_grammar:
            return True
        prev_cat = classify_step(prev_step) if prev_step else None
        allowed = self.category_grammar.get(prev_cat)
        if not allowed:
            return True  # unseen context → don't restrict (preserve OOD novelty)
        return classify_step(candidate) in allowed

    # ── Task 1: rerank ────────────────────────────────────────────────────────
    def rerank(
        self,
        prefix: list[str],
        ranked: list[str],
        k: int = 5,
        state: Optional[WaferState] = None,
    ) -> list[str]:
        """
        Re-rank the model's candidate list: physically-legal candidates first
        (model order preserved among them), category-inconsistent legals demoted
        in 'soft' mode (or dropped in 'hard'), illegal candidates only as a last
        resort so the output is never empty.
        """
        st = state if state is not None else self.replay(prefix)
        prev = prefix[-1] if prefix else None

        legal_consistent: list[str] = []
        legal_other: list[str] = []
        illegal: list[str] = []
        for c in ranked:
            if not c:
                continue
            if self.is_legal(st, c):
                if self._category_ok(prev, c):
                    legal_consistent.append(c)
                elif self.category_mode == "hard":
                    illegal.append(c)
                else:
                    legal_other.append(c)
            else:
                illegal.append(c)

        out: list[str] = []
        for c in legal_consistent + legal_other + illegal:
            if c not in out:
                out.append(c)
        return out[:k]

    # ── Task 2: constrained decode ──────────────────────────────────────────────
    def constrained_decode(
        self,
        prefix: list[str],
        score_fn: ScoreFn,
        beam: int = 20,
        max_steps: int = 160,
    ) -> list[str]:
        """
        Extend `prefix` to completion. At each step the model proposes a ranked
        candidate set; the first physically-LEGAL one (scanning the full ranking,
        model order preserved) is chosen.

        VALIDITY GUARANTEE: we NEVER append a step that would create a violation.
        If the model proposes no legal continuation, we try to terminate legally
        (sort-test then ship, each only if legal); if even that is impossible we
        STOP and return the valid partial completion (honest fallback — better an
        incomplete-but-valid route than a "complete" illegal one, since the score
        is P(valid)). A final guard trims any trailing step that would leave the
        whole route invalid, so the returned COMPLETION never introduces a
        rule violation relative to a valid prefix.

        Returns the COMPLETION ONLY (steps after the prefix).
        """
        state = self.replay(prefix)
        current = list(prefix)
        generated: list[str] = []
        terminated = bool(current) and current[-1].upper() == _TERMINAL

        def _legal(st, step):
            ns, viol = apply_step(copy(st), step)
            return ns if not viol else None

        while len(generated) < max_steps and not terminated:
            ranked = _ranked(score_fn(current))
            if not ranked:
                break
            chosen = chosen_state = None
            for cand in ranked:                      # full ranking, not just beam
                ns = _legal(state, cand)
                if ns is not None:
                    chosen, chosen_state = cand, ns
                    break
            if chosen is None:
                break                                # no legal proposal -> stop (don't emit illegal)

            current.append(chosen)
            generated.append(chosen)
            state = chosen_state

            if chosen.upper() == _TERMINAL:
                terminated = True
                break
            if _looks_stuck(generated):
                break

        # Legal termination only: add sort-test / ship iff each keeps the route valid.
        if not terminated:
            for tail in ("WAFER SORT TEST", _TERMINAL):
                if tail == "WAFER SORT TEST" and any(s.upper() == tail for s in current):
                    continue
                ns = _legal(state, tail)
                if ns is not None:
                    current.append(tail)
                    generated.append(tail)
                    state = ns
                    if tail.upper() == _TERMINAL:
                        terminated = True

        # Final guarantee: trim any trailing step that leaves the whole route invalid.
        while generated and not self.guard(current)[0]:
            current.pop()
            generated.pop()
        return generated

    # ── Task 2: physics-vetoed BEAM search (better completions than greedy) ──────
    def beam_decode(
        self,
        prefix: list[str],
        score_fn,                       # steps -> list[(step_name, prob)]
        beam: int = 5,
        branch: int = 8,
        max_steps: int = 160,
        length_alpha: float = 0.7,
    ) -> list[str]:
        """Beam search with the SAME validity guarantee as constrained_decode: every
        expansion is physics-legal, beams terminate at SHIP LOT, and the returned
        completion never introduces a violation. Explores `beam` parallel legal
        paths (each expanded by its top `branch` legal candidates) and keeps the
        best by LENGTH-NORMALISED log-probability (length_alpha curbs the
        short-sequence bias) — this lowers edit distance vs greedy on Task 2.

        score_fn(steps) must return ranked (name, prob) pairs. Returns COMPLETION only.
        """
        import math
        start = self.replay(prefix)

        def _legal(st, step):
            ns, viol = apply_step(copy(st), step)
            return ns if not viol else None

        live = [([], start, 0.0)]       # (gen_steps, state, sum_logprob)
        done = []
        for _ in range(max_steps):
            if not live:
                break
            cands = []
            for gen, st, lp in live:
                preds = score_fn(prefix + gen)
                legal_found = False
                for name, prob in (preds or [])[:branch]:
                    ns = _legal(st, name)
                    if ns is None:
                        continue
                    legal_found = True
                    nlp = lp + math.log(max(float(prob), 1e-9))
                    ngen = gen + [name]
                    if name.upper() == _TERMINAL:
                        done.append((ngen, ns, nlp))
                    elif not _looks_stuck(ngen):
                        cands.append((ngen, ns, nlp))
                if not legal_found:
                    done.append((gen, st, lp))      # stuck legally -> finalize as-is
            if not cands:
                break
            cands.sort(key=lambda x: x[2] / (len(x[0]) ** length_alpha), reverse=True)
            live = cands[:beam]

        pool = done + live
        if not pool:
            return []
        pool.sort(key=lambda x: x[2] / (max(len(x[0]), 1) ** length_alpha), reverse=True)
        gen = list(pool[0][0])

        # legal termination + final validity trim (same guarantee as greedy path)
        current = list(prefix) + gen
        st = self.replay(current)
        if not (current and current[-1].upper() == _TERMINAL):
            for tail in ("WAFER SORT TEST", _TERMINAL):
                if tail == "WAFER SORT TEST" and any(s.upper() == tail for s in current):
                    continue
                ns, viol = apply_step(copy(st), tail)
                if not viol:
                    current.append(tail); gen.append(tail); st = ns
        while gen and not self.guard(current)[0]:
            current.pop(); gen.pop()
        return gen

    # ── Task 3 / safety: guard ──────────────────────────────────────────────────
    @staticmethod
    def guard(sequence: list[str]):
        """
        Certify a full sequence. Returns (is_valid: bool, violations: list).
        Use as the symbolic half of the anomaly ensemble, or to verify a
        generated completion before emitting it.
        """
        viol = validate_by_state_machine(sequence)
        return (len(viol) == 0, viol)


# ---------------------------------------------------------------------------
# Adapters — turn a concrete model into a score_fn
# ---------------------------------------------------------------------------

def transition_model_scorer(model, k: int = 25) -> ScoreFn:
    """Adapter for our TransitionModel: prefix -> ranked next steps."""
    return lambda prefix: model.predict_top_k(prefix, k=k)


def llm_scorer_example(generate_logits, vocab: list[str]) -> ScoreFn:
    """
    Template adapter for a trained LLM / from-scratch model.

    `generate_logits(prefix) -> dict[str, float]` is your model's next-token
    distribution over the step vocabulary (read from logits / softmax). The
    refinery handles legality, ranking and termination on top of it.
    """
    def _fn(prefix: list[str]) -> dict[str, float]:
        logits = generate_logits(prefix)              # {step: score}
        return {s: logits.get(s, float("-inf")) for s in vocab}
    return _fn


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    from generate_sequences import generate_sequence, read_csv_sequences
    from models.transition_model import build_model

    print("Loading model + learning category grammar …")
    model = build_model(
        data_dir=_REPO / "data",
        cache_path=_REPO / "models" / "transition_model.pkl",
    )
    all_seqs = []
    for fam in ("MOSFET", "IGBT", "IC"):
        p = _REPO / "data" / f"{fam}_variants.csv"
        if p.exists():
            all_seqs += list(read_csv_sequences(p).values())
    grammar = learn_category_grammar(all_seqs)
    refinery = PhysicsRefinery(category_grammar=grammar, category_mode="soft")
    vocab = sorted({s for seq in all_seqs for s in seq})

    rng = random.Random(0)
    seq = generate_sequence("mosfet", rng)
    cut = int(len(seq) * 0.7)
    prefix, truth = seq[:cut], seq[cut:]

    # 1) A real model, refined.
    print("\n[1] constrained_decode with the trained n-gram model")
    comp = refinery.constrained_decode(prefix, transition_model_scorer(model))
    ok, viol = refinery.guard(prefix + comp)
    print(f"    completion length={len(comp)} (truth={len(truth)})  "
          f"physically valid={ok}  violations={len(viol)}")

    # 2) A deliberately WEAK scorer (uniform random over vocab). The physics
    #    layer still forces a legal, terminating sequence — proving the refinery
    #    adds correctness on top of even a near-useless model.
    print("\n[2] constrained_decode with a WEAK (random) scorer")
    def weak_scorer(prefix, _vocab=vocab):
        cand = list(_vocab)
        rng.shuffle(cand)
        return cand[:25]
    comp2 = refinery.constrained_decode(prefix, weak_scorer)
    ok2, viol2 = refinery.guard(prefix + comp2)
    print(f"    completion length={len(comp2)}  physically valid={ok2}  "
          f"violations={len(viol2)}")

    # 3) rerank demo: legal candidates float to the top.
    print("\n[3] rerank — true next step should land at/near RANK_1")
    ranked = model.predict_top_k(prefix, k=8)
    refined = refinery.rerank(prefix, ranked, k=5)
    print(f"    true next step : {truth[0]}")
    print(f"    model top-5    : {ranked[:5]}")
    print(f"    refined top-5  : {refined}")

    # 4) guard demo on a known-bad sequence.
    print("\n[4] guard — flagging an invalid sequence")
    bad = [s for s in seq if "CLEAN" not in classify_step(s)]  # strip cleans
    okb, violb = refinery.guard(bad)
    print(f"    valid={okb}  violations={len(violb)}  "
          f"first={violb[0].rule if violb else None}")
