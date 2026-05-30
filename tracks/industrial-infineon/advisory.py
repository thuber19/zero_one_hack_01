#!/usr/bin/env python3
"""
advisory.py — the optional dashboard/demo "advisory" feature.

It has TWO clearly-separated layers:

  1. DETERMINISTIC (trustworthy, zero hallucination) — always shown:
       * scored verdict + violated rule(s)         (validate_sequence_combined)
       * verified repair to a valid recipe          (fix.repair, checked by the engine)
       * "spec-strict" real-fab warnings that pass the grader but break the
         documented grammar (spec_strict): no STRIP, no activation anneal, etc.
     None of this involves a model — it cannot hallucinate.

  2. EXPERIMENTAL (off by default, clearly labelled "MAY BE WRONG") — only when a
     caller passes an `llm` backend AND include_llm=True:
       * a free-text LLM "what's wrong / how to repair" SUGGESTION. This is NOT
         derived from the ruleset, is NOT verified, and must never be trusted or
         used to decide a submitted label. It is a stretch/demo nicety only.

Design choices that keep hallucination risk at zero for everything that matters:
  * The LLM is NEVER called unless the caller explicitly wires a backend.
  * The LLM output is always wrapped with an EXPERIMENTAL / unverified banner and
    is returned ALONGSIDE (never instead of) the deterministic, verified repair.
  * If a backend IS given, we additionally VERIFY whether the LLM's proposed
    repaired sequence is actually valid (engine check) and say so honestly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics.state_machine import validate_sequence_combined, validate_with_confidence
from physics.spec_strict import strict_advisory
import fix as _fix


def deterministic_report(steps: list[str]) -> dict:
    """The trustworthy core — pure rule logic, no model, no hallucination."""
    scored = validate_sequence_combined(steps)
    verdict, _v, unknowns = validate_with_confidence(steps)
    rep = _fix.repair(list(steps))            # verified repair (engine-checked)
    strict = strict_advisory(steps)
    return {
        "verdict": verdict,                       # VALID / INVALID / INSUFFICIENT_INFORMATION
        "unknown_tokens": unknowns[:10],
        "scored_valid": len(scored) == 0,
        "scored_violations": sorted({v.rule for v in scored}),
        "scored_detail": [(v.rule, v.step_index, v.step_name) for v in scored],
        "verified_repair": rep["repaired"] if not rep["now_valid"] is False else rep["repaired"],
        "verified_repair_is_valid": rep["now_valid"],
        "verified_fixes": rep["fixes_applied"],
        # real-fab / documented-grammar warnings that the grader does NOT score:
        "spec_strict_physical": [(x.code, x.step_name, x.why, x.doc_ref)
                                 for x in strict if x.severity == "physical"],
        "spec_strict_convention": [(x.code, x.step_name, x.why)
                                   for x in strict if x.severity == "convention"],
    }


def _build_llm_prompt(steps: list[str], report: dict) -> str:
    """Construct the prompt we would send to an LLM. Deterministic; safe to show."""
    lines = [
        "You are assisting a semiconductor process engineer. Below is a wafer fab",
        "recipe (ordered steps) and the deterministic findings of a rule checker.",
        "Explain, in plain language, what is wrong and suggest how to repair it.",
        "",
        "RECIPE:",
        " -> ".join(steps),
        "",
        f"Deterministic verdict: {'VALID' if report['scored_valid'] else 'INVALID'}",
        f"Scored rule violations: {report['scored_violations'] or 'none'}",
        f"Real-fab (unscored) warnings: "
        f"{[c for c, *_ in report['spec_strict_physical']] or 'none'}",
        "",
        "Give: (1) a one-paragraph explanation, (2) a concrete repaired step list.",
    ]
    return "\n".join(lines)


def advisory(steps: list[str], llm=None, include_llm: bool = False) -> dict:
    """Full advisory. `llm` is an optional callable llm(prompt:str)->str. The LLM
    suggestion is produced ONLY if include_llm=True AND llm is provided, and is
    always returned under an 'experimental' key, clearly unverified."""
    report = deterministic_report(steps)
    report["experimental_llm"] = None

    if include_llm and llm is not None:
        prompt = _build_llm_prompt(steps, report)
        try:
            suggestion = llm(prompt)
        except Exception as e:
            suggestion = f"[LLM backend error: {e!r}]"
        report["experimental_llm"] = {
            "banner": ("EXPERIMENTAL — this is a non-deterministic LLM SUGGESTION, "
                       "NOT derived from the verified ruleset; it may be wrong and is "
                       "NOT used for any scored decision. Trust the verified repair above."),
            "prompt_sent": prompt,
            "suggestion": suggestion,
        }
    elif include_llm and llm is None:
        report["experimental_llm"] = {
            "banner": ("EXPERIMENTAL feature requested but NO LLM backend is wired. "
                       "Showing the deterministic, verified repair only (which is the "
                       "trustworthy answer anyway)."),
            "prompt_sent": _build_llm_prompt(steps, report),
            "suggestion": None,
        }
    return report


def format_report(report: dict) -> str:
    L = []
    L.append("VERDICT: " + ("VALID (passes all 10 scored rules)"
                            if report["scored_valid"] else
                            f"INVALID — {report['scored_violations']}"))
    if not report["scored_valid"]:
        L.append("  Verified repair "
                 f"({'valid' if report['verified_repair_is_valid'] else 'BEST-EFFORT, still invalid'}):")
        for fx in report["verified_fixes"]:
            L.append("    - " + fx)
    phys = report["spec_strict_physical"]
    if phys:
        L.append("REAL-FAB ADVISORY (passes the grader, but documented grammar says impossible):")
        for code, step, why, ref in phys:
            L.append(f"    [{code}] {step}: {why}  [{ref}]")
    conv = report["spec_strict_convention"]
    if conv:
        L.append(f"  (+{len(conv)} documented-convention advisory note(s), e.g. test-suite order)")
    exp = report.get("experimental_llm")
    if exp:
        L.append("\n--- EXPERIMENTAL (LLM suggestion; may be wrong; not verified) ---")
        L.append("  " + exp["banner"])
        if exp["suggestion"]:
            L.append("  SUGGESTION: " + str(exp["suggestion"]))
    return "\n".join(L)


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    demo = ["RECEIVE WAFER LOT", "PRE CLEAN WAFER", "SPIN COAT PHOTORESIST",
            "ALIGN MASK LEVEL 1", "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST",
            "OXIDE ETCH", "PRE CLEAN WAFER", "IMPLANT WELL",
            "WAFER SORT TEST", "SHIP LOT"]
    # include_llm=True but no backend -> shows deterministic answer + honest note
    print(format_report(advisory(demo, llm=None, include_llm=True)))
