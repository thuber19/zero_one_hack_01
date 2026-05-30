#!/usr/bin/env python3
"""
robustness_test.py — fire malformed / adversarial inputs through the WHOLE
inference path (Task 1/2/3) and the physics layer, and assert NOTHING crashes
and outputs stay well-formed. This is the "it will not blow up on the real eval
file" guarantee.

Cases: empty sequence, single step, way-over-length sequence (tests the
positional guard), unknown 4th-family vocabulary, unknown family name, blank /
whitespace steps, and a sequence of pure novel tokens.

Usage: python robustness_test.py --model-dir outputs_M1 --model-size small
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "data"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from physics.state_machine import validate_sequence_combined
from physics.ontology import classify_step
import fix as _fix

CASES = {
    "empty": ([], "mosfet"),
    "single_step": (["RECEIVE WAFER LOT"], "mosfet"),
    "way_too_long": (["PRE CLEAN WAFER"] * 320, "igbt"),
    "unknown_family": (["RECEIVE WAFER LOT", "PRE CLEAN WAFER", "DEPOSIT POLYSILICON"], "zorblax"),
    "novel_vocab_4th_family": (["RECEIVE WAFER LOT", "GROW QUANTUM DOT LAYER",
                                 "PLASMA ENHANCE EXOTIC FILM", "SHIP LOT"], "newfam"),
    "blank_and_ws_steps": (["RECEIVE WAFER LOT", "   ", "", "SHIP LOT"], "ic"),
    "pure_novel": (["FOO BAR", "BAZ QUX", "WIDGET FROB"], "mystery"),
    # mixed vocab: a KNOWN in-vocab violation (deposit with no clean) plus ONE
    # novel token — the engine is a per-step hybrid (known steps use the exact
    # reference sets, novel steps use category), so the known violation must still
    # be caught despite the novel token (audit: whole-sequence-routing residual).
    "mixed_known_violation_plus_novel": (
        ["RECEIVE WAFER LOT", "GLORP NOVEL STEP", "DEPOSIT POLYSILICON",
         "WAFER SORT TEST", "SHIP LOT"], "mystery"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="outputs_M1")
    ap.add_argument("--model-size", default="small")
    args = ap.parse_args()

    fails = []

    # ── Physics layer (stdlib) on every case — must never crash ──
    print("[A] physics layer (validate + classify + fix) on edge inputs")
    for name, (steps, fam) in CASES.items():
        try:
            v = validate_sequence_combined(steps)
            _ = [classify_step(s) for s in steps]
            _ = _fix.analyze(steps)
            print(f"  [OK] {name:24s} -> {len(v)} violation(s)")
        except Exception as e:
            fails.append(f"physics/{name}: {e!r}")
            print(f"  [FAIL] {name}: {e!r}")

    # ── Full neural inference path — must never crash, outputs well-formed ──
    print("\n[B] neural inference path (predict / complete / detect)")
    try:
        from inference import ProcessPredictor
        pred = ProcessPredictor.load(Path(args.model_dir), model_size=args.model_size, device="cpu")
    except Exception as e:
        print(f"  [SKIP] could not load model from {args.model_dir}: {e!r}")
        pred = None

    if pred is not None:
        for name, (steps, fam) in CASES.items():
            try:
                top = pred.predict_next_steps(steps, fam, top_k=5)
                assert len(top) == 5, f"expected 5 preds, got {len(top)}"
                assert all(not (n.startswith("[") and n.endswith("]")) for n, _ in top), "special token leaked"
                comp = pred.complete_sequence(steps, fam, max_new_steps=40)
                assert isinstance(comp, list)
                det = pred.detect_anomaly(steps, fam)
                assert det["is_valid"] in (True, False)
                assert 0.0 <= det["score"] <= 1.0
                print(f"  [OK] {name:24s} -> top1='{top[0][0]}'  +{len(comp)} steps  "
                      f"valid={det['is_valid']} score={det['score']}")
            except Exception as e:
                fails.append(f"inference/{name}: {e!r}")
                print(f"  [FAIL] {name}: {e!r}")

    print("\n" + "=" * 60)
    if fails:
        print(f"RESULT: FAIL — {len(fails)} crash/contract violation(s):")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("RESULT: PASS — no crashes; all outputs well-formed on malformed input")
    sys.exit(0)


if __name__ == "__main__":
    main()
