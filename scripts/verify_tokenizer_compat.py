#!/usr/bin/env python3
"""Verify that Spec 001 and Spec 002 vocabularies are compatible.

Every token from Spec 001 must appear at the same ID in Spec 002 vocab.
Spec 002 may have additional tokens (e.g. [CLS], [SEP], [MASK]) appended at the end.

Usage:
  python scripts/verify_tokenizer_compat.py \\
      --vocab001 $WORK/artifacts/001/vocab.json \\
      --vocab002 $WORK/artifacts/002/vocab.json

Exits 0 on success, 1 with detailed diff on failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab001", required=True, help="Path to Spec 001 vocab.json")
    ap.add_argument("--vocab002", required=True, help="Path to Spec 002 vocab.json")
    args = ap.parse_args()

    p001 = Path(args.vocab001)
    p002 = Path(args.vocab002)

    if not p001.exists():
        print(f"ERROR: Spec 001 vocab not found: {p001}", file=sys.stderr)
        return 1
    if not p002.exists():
        print(f"ERROR: Spec 002 vocab not found: {p002}", file=sys.stderr)
        return 1

    with open(p001) as f:
        d001 = json.load(f)
    with open(p002) as f:
        d002 = json.load(f)

    id_to_step_001 = d001.get("id_to_step", d001.get("id_to_token", []))
    id_to_step_002 = d002.get("id_to_step", d002.get("id_to_token", []))
    step_to_id_002 = {s: i for i, s in enumerate(id_to_step_002)}

    mismatches = []
    for i, tok in enumerate(id_to_step_001):
        got = step_to_id_002.get(tok)
        if got != i:
            mismatches.append((i, tok, got))

    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatches between Spec 001 and Spec 002 vocab:")
        for i, tok, got in mismatches[:30]:
            print(f"  Spec001 id={i} token={tok!r} → Spec002 id={got!r}")
        if len(mismatches) > 30:
            print(f"  ... and {len(mismatches) - 30} more")
        return 1

    extra = id_to_step_002[len(id_to_step_001):]
    print(f"OK: All {len(id_to_step_001)} Spec 001 tokens verified at correct IDs.")
    if extra:
        print(f"  Spec 002 extends Spec 001 vocab with {len(extra)} additional tokens: {extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
