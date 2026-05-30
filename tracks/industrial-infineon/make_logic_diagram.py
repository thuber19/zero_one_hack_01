#!/usr/bin/env python3
"""
make_logic_diagram.py — judge-facing artifact that EXPLAINS the process logic.

Writes:
  knowledge/PROCESS_LOGIC.md  — the canonical flow + a Mermaid dependency graph
                                (renders on GitHub) + the plain-English WHY for
                                every rule (straight from the knowledge base).
  knowledge/logic_graph.png   — a clean "what enables what" diagram for slides.

This turns the symbolic knowledge base into something a non-expert judge can
look at and immediately understand the logic the model is being taught.
"""
from __future__ import annotations
import sys
from pathlib import Path

_SUB = Path(__file__).resolve().parent
sys.path.insert(0, str(_SUB)); sys.path.insert(0, str(_SUB / "training_data"))
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

from physics import process_knowledge as K

OUT = _SUB / "knowledge"; OUT.mkdir(exist_ok=True)


def write_markdown():
    L = ["# Process Logic — what the model must learn\n",
         "_Auto-generated from `physics/process_knowledge.py`. This is the "
         "process understanding the system encodes and teaches._\n",
         "## The canonical fabrication flow\n"]
    for i, (phase, desc) in enumerate(K.PROCESS_FLOW, 1):
        L.append(f"{i}. **{phase}** — {desc}")
    L.append("\n## The dependency graph (each arrow = a rule)\n")
    L.append("```mermaid")
    L.append(K.to_mermaid())
    L.append("```\n")
    L.append("## Why each rule exists (the physics)\n")
    seen = set()
    for e in K.causal_edges():
        if e["rule"] in seen:
            continue
        seen.add(e["rule"])
        within = f" within {e['window']} steps" if e["window"] else " (ordering)"
        L.append(f"- **{e['rule']}** — `{e['to']}` needs `{e['from']}`{within}. "
                 f"{e['why']}")
    L.append(f"\n- **{K.LITHO_RULE['id']}** — {K.LITHO_RULE['plain']} "
             f"{K.LITHO_RULE['physical_reason']}")
    (OUT / "PROCESS_LOGIC.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT / 'PROCESS_LOGIC.md'}")


def write_png():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except Exception as e:
        print(f"(skipping PNG — matplotlib unavailable: {e})")
        return

    edges = K.causal_edges()
    enablers = sorted({e["from"] for e in edges})
    triggers = sorted({e["to"] for e in edges})
    yL = {n: i for i, n in enumerate(enablers)}
    yR = {n: i for i, n in enumerate(triggers)}
    n = max(len(enablers), len(triggers))

    fig, ax = plt.subplots(figsize=(11, 0.7 * n + 1.5))
    for name, y in yL.items():
        ax.text(0.02, y, name, ha="left", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe", ec="#2563eb"))
    for name, y in yR.items():
        ax.text(0.98, y, name, ha="right", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#dc2626"))
    for e in edges:
        y0, y1 = yL[e["from"]], yR[e["to"]]
        ax.annotate("", xy=(0.74, y1), xytext=(0.26, y0),
                    arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.2,
                                    connectionstyle="arc3,rad=0.05"))
        lbl = e["rule"].replace("RULE_", "") + (f"≤{e['window']}" if e["window"] else "")
        ax.text(0.5, (y0 + y1) / 2, lbl, fontsize=6, ha="center", color="#374151",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    ax.text(0.02, n, "ENABLING / MILESTONE steps", fontsize=10, weight="bold", color="#2563eb")
    ax.text(0.98, n, "TRIGGERING operations", fontsize=10, weight="bold", color="#dc2626", ha="right")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.7, n + 0.5); ax.axis("off")
    ax.set_title("Process-logic dependencies (model proposes → physics requires)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "logic_graph.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'logic_graph.png'}")


if __name__ == "__main__":
    write_markdown()
    write_png()
