#!/usr/bin/env python3
"""
validate_submission.py — verify the three submission CSVs conform EXACTLY to the
organizers' spec (generation_rules.md §5.3) BEFORE they are sent for scoring.

Catches the silent killers: wrong/missing columns, missing or extra EXAMPLE_IDs,
fewer than 5 ranks, [UNK]/[PAD] leaking into predictions, IS_VALID not in {0,1},
SCORE outside [0,1], PREDICTED_RULE present on a valid row or naming an unknown
rule, completion repeating the partial it was supposed to continue.

Usage:
  python validate_submission.py --submission-dir outputs/submissions --eval-dir <eval_files>
  (eval-dir optional; if given, row counts and EXAMPLE_IDs are cross-checked)

Exit code 0 = submission is spec-clean; 1 = problems found (all printed).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "data"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

VALID_RULES = {
    "RULE_DEP_NO_CLEAN", "RULE_METAL_ETCH_NO_LITHO", "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP", "RULE_IMPLANT_NO_MASK", "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP", "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST", "RULE_BACKSIDE_BEFORE_PASSIVATION",
}


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, reader.fieldnames


def _ids_from_eval(path, col="EXAMPLE_ID"):
    if not path or not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [r[col].strip() for r in csv.DictReader(f)]


def _check_header(errs, name, got, want):
    got = [c.strip() for c in (got or [])]
    if got != want:
        errs.append(f"[{name}] header mismatch:\n     got : {got}\n     want: {want}")


def _is_special(tok):
    return tok.startswith("[") and tok.endswith("]")


def check_task1(errs, path, eval_ids):
    rows, hdr = _read(path)
    _check_header(errs, "nextstep", hdr, ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
    seen = []
    for i, r in enumerate(rows):
        seen.append(r.get("EXAMPLE_ID", "").strip())
        ranks = [r.get(f"RANK_{k}", "").strip() for k in range(1, 6)]
        if any(not x for x in ranks):
            errs.append(f"[nextstep] row {i} ({r.get('EXAMPLE_ID')}): empty rank(s) {ranks}")
        if any(_is_special(x) for x in ranks if x):
            errs.append(f"[nextstep] row {i} ({r.get('EXAMPLE_ID')}): special token leaked {ranks}")
        if len({x for x in ranks if x}) < len([x for x in ranks if x]):
            errs.append(f"[nextstep] row {i} ({r.get('EXAMPLE_ID')}): duplicate ranks {ranks}")
    _check_ids(errs, "nextstep", seen, eval_ids)


def check_task2(errs, path, eval_ids, partials=None):
    rows, hdr = _read(path)
    _check_header(errs, "completion", hdr, ["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
    seen = []
    for i, r in enumerate(rows):
        eid = r.get("EXAMPLE_ID", "").strip()
        seen.append(eid)
        seq = [s for s in r.get("PREDICTED_SEQUENCE", "").split("|") if s.strip()]
        if not seq:
            errs.append(f"[completion] row {i} ({eid}): empty PREDICTED_SEQUENCE")
        if any(_is_special(s) for s in seq):
            errs.append(f"[completion] row {i} ({eid}): special token leaked")
        # spec: must NOT repeat the partial it continues
        if partials and eid in partials and seq[:len(partials[eid])] == partials[eid]:
            errs.append(f"[completion] row {i} ({eid}): repeats the PARTIAL_SEQUENCE (should predict only AFTER the cut)")
    _check_ids(errs, "completion", seen, eval_ids)


def check_task3(errs, path, eval_ids):
    rows, hdr = _read(path)
    _check_header(errs, "anomaly", hdr, ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
    seen = []
    n_score = 0
    for i, r in enumerate(rows):
        eid = r.get("EXAMPLE_ID", "").strip()
        seen.append(eid)
        iv = r.get("IS_VALID", "").strip()
        if iv not in ("0", "1"):
            errs.append(f"[anomaly] row {i} ({eid}): IS_VALID must be 0 or 1, got '{iv}'")
        sc = r.get("SCORE", "").strip()
        if sc:
            n_score += 1
            try:
                f = float(sc)
                if not (0.0 <= f <= 1.0):
                    errs.append(f"[anomaly] row {i} ({eid}): SCORE {f} outside [0,1]")
            except ValueError:
                errs.append(f"[anomaly] row {i} ({eid}): SCORE not a float: '{sc}'")
        rule = r.get("PREDICTED_RULE", "").strip()
        if iv == "1" and rule:
            errs.append(f"[anomaly] row {i} ({eid}): valid row should have empty PREDICTED_RULE, got '{rule}'")
        if iv == "0" and rule and rule not in VALID_RULES:
            errs.append(f"[anomaly] row {i} ({eid}): unknown PREDICTED_RULE '{rule}'")
    # SCORE is OPTIONAL per spec §5.3, so an empty SCORE is not an error — but if
    # EVERY row omits it, ROC-AUC cannot be computed by the grader. Warn (non-fatal).
    if rows and n_score == 0:
        print("  [WARN] anomaly.csv has NO SCORE values — spec-legal, but the grader"
              " cannot compute ROC-AUC. Emit P(valid) in SCORE to enable it.")
    _check_ids(errs, "anomaly", seen, eval_ids)


def _check_ids(errs, name, seen, eval_ids):
    if len(seen) != len(set(seen)):
        dups = {x for x in seen if seen.count(x) > 1}
        errs.append(f"[{name}] duplicate EXAMPLE_IDs: {sorted(dups)[:5]}")
    if eval_ids is not None:
        miss = set(eval_ids) - set(seen)
        extra = set(seen) - set(eval_ids)
        if miss:
            errs.append(f"[{name}] MISSING {len(miss)} EXAMPLE_IDs (e.g. {sorted(miss)[:3]})")
        if extra:
            errs.append(f"[{name}] EXTRA {len(extra)} EXAMPLE_IDs (e.g. {sorted(extra)[:3]})")
        if len(seen) != len(eval_ids):
            errs.append(f"[{name}] row count {len(seen)} != eval rows {len(eval_ids)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", default="outputs/submissions")
    ap.add_argument("--eval-dir", default=None)
    args = ap.parse_args()

    sub = Path(args.submission_dir)
    errs = []
    valid_ids = anomaly_ids = None
    partials = None
    if args.eval_dir:
        ev = Path(args.eval_dir)
        valid_ids = _ids_from_eval(ev / "eval_input_valid.csv")
        anomaly_ids = _ids_from_eval(ev / "eval_input_anomaly.csv")
        # build partials map for the repeat check
        vp = ev / "eval_input_valid.csv"
        if vp.exists():
            partials = {}
            with open(vp, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    partials[r["EXAMPLE_ID"].strip()] = [
                        s for s in r["PARTIAL_SEQUENCE"].split("|") if s.strip()]

    checks = [
        ("nextstep.csv", lambda: check_task1(errs, sub / "nextstep.csv", valid_ids)),
        ("completion.csv", lambda: check_task2(errs, sub / "completion.csv", valid_ids, partials)),
        ("anomaly.csv", lambda: check_task3(errs, sub / "anomaly.csv", anomaly_ids)),
    ]
    for fname, fn in checks:
        p = sub / fname
        if not p.exists():
            errs.append(f"[{fname}] MISSING file: {p}")
            continue
        try:
            fn()
        except Exception as e:
            errs.append(f"[{fname}] crashed during validation: {e!r}")

    print(f"Submission dir: {sub}")
    if errs:
        print(f"\n{len(errs)} PROBLEM(S) FOUND:\n")
        for e in errs:
            print("  - " + e)
        print("\nRESULT: FAIL — fix the above before submitting.")
        sys.exit(1)
    print("\nRESULT: PASS — all three files conform to the spec.")
    sys.exit(0)


if __name__ == "__main__":
    main()
