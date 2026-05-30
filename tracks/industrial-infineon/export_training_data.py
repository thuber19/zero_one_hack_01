#!/usr/bin/env python3
"""
export_data.py — turn the harness's knowledge into training corpora.

This is the bridge between everything we built and the two models being trained
on Leonardo (one from-scratch, one fine-tune). It converts the grammar, the
category ontology, the physical descriptions/parameters, and the rule knowledge
into ready-to-train files so the MODEL itself absorbs the process understanding
— not just token order.

Outputs (under --out-dir, default training_export/)
---------------------------------------------------
FROM-SCRATCH sequence model (OOD-robust via categories):
  lm_plain.txt            one sequence per line: "[FAMILY] step | step | ... | SHIP LOT"
  lm_factorized.jsonl     {family, steps[], categories[], roles[]} — parallel
                          category/role streams so the model can learn an
                          auxiliary "next-category" signal (transfers to a new
                          family whose step NAMES are unseen but whose CATEGORIES
                          are not).

FINE-TUNE LLM (physics-aware, the "understanding"):
  instruct_nextstep.jsonl   prompt: partial sequence -> completion's first step
  instruct_completion.jsonl prompt: partial (60/80%) -> remaining steps
  instruct_anomaly.jsonl    prompt: full sequence -> VALID / INVALID + RULE + WHY
  knowledge_cards.jsonl     per step: description, real fab parameters, category,
                            and the physical preconditions it must satisfy + why
                            — a knowledge-injection / continued-pretraining corpus

SHARED:
  anomaly_labeled.jsonl     {steps, is_valid, rule, explanation} valid + bad
  DATA_EXPORT_README.md     what each file is and how to train on it

Everything is verified: valid sequences pass the reference checker, bad ones are
reference-labelled, explanations come from the declarative knowledge base.
stdlib only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from generate_sequences import generate_sequence, validate_sequence
from physics.ontology import classify_step, STEP_CATEGORY
from physics import process_knowledge as K
from physics.state_machine import validate_by_state_machine
from physics import step_semantics as SEM
import bad_data_generator as BDG
import pseudo_family as PF

FAMILIES = ("mosfet", "igbt", "ic")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roles(step: str) -> list[str]:
    """Event-class roles a step plays (trigger and/or enabler) — the rule-
    relevant 'function' of the step, which is what transfers across families."""
    return [name for name in K.EVENT_CLASSES if K.step_in_event(step, name)]


def _step_requirements(step: str) -> list[dict]:
    """The physical preconditions THIS step imposes (from the KB), with reasons."""
    reqs = []
    for r in K.WINDOWED_RULES:
        if K.step_in_event(step, r.trigger):
            for enabler, window in r.requires:
                reqs.append({"rule": r.id, "needs": enabler,
                             "within_steps": window, "why": r.physical_reason})
    for r in K.ORDERING_RULES:
        if K.step_in_event(step, r.trigger):
            reqs.append({"rule": r.id, "needs_before": [f for f, _ in r.requires],
                         "why": r.physical_reason})
    return reqs


def _gen_valid(family: str, n: int, rng: random.Random) -> list[list[str]]:
    out, seen = [], set()
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        seq = generate_sequence(family, rng)
        if validate_sequence(seq):
            continue
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        out.append(seq)
    return out


def _explain(steps: list[str]) -> tuple[int, str, str]:
    """Return (is_valid, rule, explanation) for a sequence using the engine+KB."""
    viol = validate_by_state_machine(steps)
    if not viol:
        return 1, "", "All process-logic preconditions are satisfied."
    v = viol[0]
    site = SEM.describe(v.step_name)
    expl = f"{v.description} {v.physical_reason}"
    if site:
        expl += f" (step '{v.step_name}': {site})"
    return 0, v.rule, expl


def _writelines(path: Path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

def emit_from_scratch(valid_by_family, out: Path):
    plain, fact = [], []
    for fam, seqs in valid_by_family.items():
        for seq in seqs:
            plain.append(f"[{fam.upper()}] " + " | ".join(seq))
            fact.append({
                "family": fam.upper(),
                "steps": seq,
                "categories": [classify_step(s) for s in seq],
                "roles": [_roles(s) for s in seq],
            })
    _writelines(out / "lm_plain.txt", plain)
    _write_jsonl(out / "lm_factorized.jsonl", fact)
    return len(plain)


def emit_instructions(valid_by_family, out: Path, rng: random.Random):
    nextstep, completion = [], []
    for fam, seqs in valid_by_family.items():
        for seq in seqs:
            for frac in (0.6, 0.8):
                cut = max(1, min(len(seq) - 1, int(len(seq) * frac)))
                partial, rest = seq[:cut], seq[cut:]
                ctx = f"Process family: {fam.upper()}. Partial fab sequence:\n" \
                      + " -> ".join(partial)
                nextstep.append({
                    "instruction": "Predict the single most likely next "
                                   "process step.",
                    "input": ctx, "output": rest[0],
                })
                completion.append({
                    "instruction": "Complete the remaining process steps in "
                                   "order, pipe-separated.",
                    "input": ctx, "output": "|".join(rest),
                })
    _write_jsonl(out / "instruct_nextstep.jsonl", nextstep)
    _write_jsonl(out / "instruct_completion.jsonl", completion)
    return len(nextstep), len(completion)


def emit_anomaly(valid_by_family, bad_records, out: Path):
    rows = []
    for fam, seqs in valid_by_family.items():
        for seq in seqs:
            isv, rule, expl = _explain(seq)
            rows.append({"family": fam.upper(), "steps": seq,
                         "is_valid": isv, "rule": rule, "explanation": expl})
    for r in bad_records:
        isv, rule, expl = _explain(r["steps"])
        rows.append({"family": r["family"], "steps": r["steps"],
                     "is_valid": isv, "rule": rule or r["first_rule"],
                     "explanation": expl})

    # plain labeled file
    _write_jsonl(out / "anomaly_labeled.jsonl", rows)

    # instruction-style for the fine-tune model (teaches the WHY)
    instruct = []
    for r in rows:
        ctx = f"Process family: {r['family']}. Full fab sequence:\n" \
              + " -> ".join(r["steps"])
        if r["is_valid"]:
            ans = "VALID. " + r["explanation"]
        else:
            ans = f"INVALID. Rule: {r['rule']}. {r['explanation']}"
        instruct.append({
            "instruction": "Determine whether this semiconductor process "
                           "sequence is valid. If not, name the violated rule "
                           "and explain why it is physically impossible.",
            "input": ctx, "output": ans,
        })
    _write_jsonl(out / "instruct_anomaly.jsonl", instruct)
    return len(rows)


def emit_contrastive(valid_by_family, out: Path, rng: random.Random):
    """Minimal-pair contrastive data: a valid sequence and a NEAR-IDENTICAL
    invalid twin (one enabling step deleted). Training a model to separate these
    teaches the *decision boundary* — the single step that makes a route wrong —
    rather than coarse class statistics. Use for contrastive / preference
    (DPO-style) objectives."""
    from physics.process_knowledge import step_in_event
    pairs = []
    # enabler -> the trigger it serves and the rule it satisfies
    enabler_for = [("CLEAN_SURFACE", "DEPOSITION", "RULE_DEP_NO_CLEAN", 12),
                   ("DEVELOP", "PATTERNED_ETCH", "RULE_ETCH_NO_MASK", 12),
                   ("IMPLANT_OPENER", "IMPLANT", "RULE_IMPLANT_NO_MASK", 15)]
    for fam, seqs in valid_by_family.items():
        for seq in seqs:
            rng.shuffle(enabler_for)
            for enabler, trigger, rule, win in enabler_for:
                # find a trigger and an enabler within its window to delete
                tri = [i for i, s in enumerate(seq) if step_in_event(s, trigger)]
                if not tri:
                    continue
                t = rng.choice(tri)
                cand = [j for j in range(max(0, t - win), t)
                        if step_in_event(seq[j], enabler)]
                if not cand:
                    continue
                j = rng.choice(cand)
                twin = seq[:j] + seq[j + 1:]
                if {v.rule for v in validate_by_state_machine(twin)} == {rule}:
                    pairs.append({
                        "family": fam.upper(),
                        "valid": seq,
                        "invalid": twin,
                        "deleted_step": seq[j],
                        "rule": rule,
                        "note": f"removing '{seq[j]}' makes the later "
                                f"'{seq[t]}' violate {rule}",
                    })
                    break   # one clean minimal pair per sequence
    _write_jsonl(out / "contrastive_pairs.jsonl", pairs)
    return len(pairs)


def emit_window_edge(out: Path):
    """Window-EDGE contrastive pairs: for each windowed rule, a minimal sequence
    valid with the enabler exactly AT the window edge, paired with the same one
    step PAST the edge (invalid). Teaches the precise boundary. Engine-verified."""
    F = "MEASURE THICKNESS"  # benign filler (no precondition/effect)
    specs = [
        ("RULE_DEP_NO_CLEAN", 12, ["HF DIP"], "DEPOSIT POLYSILICON"),
        ("RULE_ETCH_NO_MASK", 12, ["DEVELOP PHOTORESIST"], "OXIDE ETCH"),
        ("RULE_IMPLANT_NO_MASK", 15, ["DEVELOP PHOTORESIST"], "IMPLANT WELL"),
        ("RULE_CMP_NO_DEP", 6, ["HF DIP", "DEPOSIT INTERLAYER DIELECTRIC"], "CMP DIELECTRIC"),
    ]
    rows = []
    for rule, w, head, trigger in specs:
        valid = head + [F] * (w - 1) + [trigger]     # enabler distance = w  -> valid
        invalid = head + [F] * w + [trigger]         # distance = w+1 -> invalid
        if (not validate_by_state_machine(valid)
                and {v.rule for v in validate_by_state_machine(invalid)} == {rule}):
            rows.append({"rule": rule, "window": w,
                         "valid_at_edge": valid, "invalid_past_edge": invalid,
                         "note": f"{trigger} needs {head[-1]} within {w} steps"})
    _write_jsonl(out / "window_edge_pairs.jsonl", rows)
    return len(rows)


def emit_knowledge_cards(out: Path):
    """One card per known step: what it is, real parameters, category, and the
    physical preconditions it imposes + why. The encoded 'understanding'."""
    cards = []
    for step in sorted(STEP_CATEGORY):
        sem = SEM.STEP_SEMANTICS.get(step)
        cards.append({
            "step": step,
            "category": classify_step(step),
            "roles": _roles(step),
            "description": SEM.describe(step),
            "fab_parameters": (sem.parameters[:3] if sem else []),
            "preconditions": _step_requirements(step),
        })
    _write_jsonl(out / "knowledge_cards.jsonl", cards)
    return len(cards)


_README = """# Training Data Export

Generated by `export_data.py` from the verified knowledge base. All
valid sequences pass the reference checker; all bad sequences are
reference-labelled; all explanations come from the declarative rule knowledge.

## From-scratch sequence model
- `lm_plain.txt` — one sequence per line, `[FAMILY]` conditioning token, steps
  joined by ` | `. Train a decoder LM with next-token objective. Treat each
  step string as one token (vocab ~120). Add BOS/EOS as your tokenizer needs.
- `lm_factorized.jsonl` — parallel `steps` / `categories` / `roles` streams.
  **Highest-leverage for OOD (Task 4):** add an auxiliary head that predicts the
  next *category* alongside the next step, or concatenate a category embedding
  to each step embedding. An unseen 4th-family step name still maps to a known
  category, so the category signal transfers where the token signal cannot.

## Fine-tune LLM
- `instruct_nextstep.jsonl`, `instruct_completion.jsonl` — instruction/response
  pairs for Tasks 1 & 2 (SFT format: instruction / input / output).
- `instruct_anomaly.jsonl` — Task 3 with the *physical reason* in the target,
  so the model is trained to explain WHY, not just classify.
- `knowledge_cards.jsonl` — per-step facts (description, real fab parameters,
  category, preconditions + why). Use for a short continued-pretraining pass or
  to seed a system prompt / retrieval store so the model knows the domain.

## Shared
- `anomaly_labeled.jsonl` — `{steps, is_valid, rule, explanation}` for any
  classifier head or eval.

## Notes
- Scale up with `--n-valid` and `--n-bad`. The grammar's combinatorial space is
  effectively unlimited (billions of valid sequences per family).
- After training, score with the official `eval_metrics.py` (or `run_all.py`),
  and wrap inference with `refinery.PhysicsRefinery` to guarantee physically
  valid, terminating outputs — including on the unknown 4th family.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="training_export", metavar="DIR")
    ap.add_argument("--n-valid", type=int, default=200,
                    help="valid sequences per family (default 200).")
    ap.add_argument("--n-bad", type=int, default=600,
                    help="total bad sequences for anomaly data (default 600).")
    ap.add_argument("--n-ood", type=int, default=200,
                    help="pseudo-family (novel-vocabulary) sequences for OOD "
                         "training/eval (default 200).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Exporting training data -> {out}/  "
          f"(n_valid={args.n_valid}/family, n_bad={args.n_bad})")

    print("  generating valid sequences …")
    valid_by_family = {fam: _gen_valid(fam, args.n_valid, rng) for fam in FAMILIES}

    print("  generating reference-labelled bad sequences …")
    bad, _ = BDG.build(per_combo=5, seed=args.seed, target_count=args.n_bad)

    print("  generating pseudo-family (OOD, novel-vocabulary) sequences …")
    ood = PF.generate_pseudo_valid(args.n_ood, random.Random(args.seed + 7))
    ood_by_tag: dict[str, list[list[str]]] = {}
    for tag, seq in ood:
        ood_by_tag.setdefault(f"OOD_{tag}", []).append(seq)

    print("  writing from-scratch corpora (ID families + OOD pseudo-families) …")
    # Train the from-scratch model on real AND novel-vocabulary sequences so it
    # must rely on categories — the key to surviving the unknown 4th family.
    n_lm = emit_from_scratch({**valid_by_family, **ood_by_tag}, out)

    # OOD anomaly split (self-estimate the ID->OOD drop the organizers measure).
    rng_o = random.Random(args.seed + 11)
    ood_rows = []
    for tag, seq in ood:
        ood_rows.append({"family": f"OOD_{tag}", "steps": seq, "is_valid": 1,
                         "rule": "", "explanation": "Valid sequence in a novel "
                         "(unseen-vocabulary) family; all preconditions hold."})
        inj = PF.inject_violation(seq, rng_o)
        if inj:
            broken, rule = inj
            isv, rl, expl = _explain(broken)
            ood_rows.append({"family": f"OOD_{tag}", "steps": broken,
                             "is_valid": isv, "rule": rl or rule, "explanation": expl})
    _write_jsonl(out / "anomaly_ood.jsonl", ood_rows)
    print("  writing instruction corpora …")
    n_ns, n_co = emit_instructions(valid_by_family, out, rng)
    print("  writing anomaly corpora …")
    n_an = emit_anomaly(valid_by_family, bad, out)
    print("  writing contrastive minimal pairs …")
    n_cp = emit_contrastive(valid_by_family, out, random.Random(args.seed + 3))
    print("  writing window-edge contrastive pairs …")
    n_we = emit_window_edge(out)
    print("  writing knowledge cards …")
    n_kc = emit_knowledge_cards(out)
    (out / "DATA_EXPORT_README.md").write_text(_README, encoding="utf-8")

    print("\nDone.")
    print(f"  lm_plain.txt / lm_factorized.jsonl : {n_lm} sequences")
    print(f"  instruct_nextstep.jsonl            : {n_ns}")
    print(f"  instruct_completion.jsonl          : {n_co}")
    print(f"  instruct_anomaly.jsonl             : {n_an}")
    print(f"  anomaly_labeled.jsonl              : {n_an}")
    print(f"  anomaly_ood.jsonl                  : {len(ood_rows)} (novel families)")
    print(f"  contrastive_pairs.jsonl            : {n_cp} minimal valid/invalid pairs")
    print(f"  window_edge_pairs.jsonl            : {n_we} window-boundary pairs")
    print(f"  knowledge_cards.jsonl              : {n_kc} steps")
    print(f"  -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
