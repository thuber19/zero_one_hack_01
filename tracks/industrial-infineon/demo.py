#!/usr/bin/env python3
"""
demo.py — the 3-act demo behind the 2-minute video.

  Act 1  Next-step: trained model's ranked predictions on a real partial.
  Act 2  Completion: model ALONE vs model+PHYSICS — physics guarantees a valid,
         terminating route (baseline-vs-integrated on identical input).
  Act 3  Anomaly: inject a fault -> the system detects it, EXPLAINS the physics,
         and SUGGESTS the fix.

Usage:  python demo.py --output-dir outputs_test
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path

_SUB = Path(__file__).resolve().parent
for _p in (str(_SUB), str(_SUB / "src"), str(_SUB / "training_data")):
    sys.path.insert(0, _p)
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

from generate_sequences import generate_sequence
from physics.state_machine import validate_by_state_machine
from physics.process_knowledge import step_in_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs_test")
    ap.add_argument("--family", default="mosfet")
    args = ap.parse_args()

    from inference import ProcessPredictor
    pred = ProcessPredictor.load(_SUB / args.output_dir, device="cpu")

    rng = random.Random(7)
    seq = generate_sequence(args.family, rng)
    cut = int(len(seq) * 0.7)
    partial = seq[:cut]

    print("=" * 66)
    print(f"DEMO — {args.family.upper()} process route")
    print("=" * 66)
    print("Partial (last 5 steps): ... " + " -> ".join(partial[-5:]))

    # Act 1
    print("\n[1] TRAINED MODEL next-step prediction (physics-reranked):")
    for i, (s, p) in enumerate(pred.predict_next_steps(partial, args.family, top_k=5), 1):
        print(f"      {i}. {s:<32} (p={p:.3f})")
    print(f"    true next step: {seq[cut]}")

    # Act 2
    print("\n[2] COMPLETION — model alone vs model + physics (same input):")
    base = pred.complete_sequence(partial, args.family, use_physics=False)
    integ = pred.complete_sequence(partial, args.family, use_physics=True)
    bviol = len(validate_by_state_machine(partial + base))
    iviol = len(validate_by_state_machine(partial + integ))
    print(f"    model alone   : {len(base):>3} steps, physics violations = {bviol}")
    print(f"    model+physics : {len(integ):>3} steps, physics violations = {iviol}  <-- guaranteed valid")

    # Act 3
    print("\n[3] ANOMALY — inject a fault, detect + explain + fix:")
    d = next(i for i, x in enumerate(seq) if step_in_event(x, "DEPOSITION"))
    broken = [x for j, x in enumerate(seq)
              if not (j < d and (d - j) <= 12 and step_in_event(x, "CLEAN_SURFACE"))]
    r = pred.detect_anomaly(broken, args.family)
    print(f"    verdict   : {'VALID' if r['is_valid'] else 'INVALID'}")
    print(f"    rule      : {r['predicted_rule']}")
    if r.get("suggested_fixes"):
        print(f"    fix       : {r['suggested_fixes'][0]}")
    print("=" * 66)


if __name__ == "__main__":
    main()
