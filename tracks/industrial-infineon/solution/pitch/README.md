# ProcSeq — 3-minute pitch deck

Two artifacts, same content & theme (Infineon dark-tech, 16:9, 7 slides):

| File | What | How to use |
|------|------|------------|
| `index.html` | Self-contained, self-explanatory web deck | Open in any browser. **↑ ↓ / Space / ← →** to move; **Home/End** to jump. Fullscreen (F11) to present. |
| `ProcSeq_Pitch.pptx` | **Editable** PowerPoint (native text boxes, shapes, table, speaker notes) | Open in PowerPoint / Keynote / Google Slides. Edit any text directly. |
| `build_pptx.py` | Regenerates the `.pptx` | `.pptxvenv/bin/python build_pptx.py` |

## The 7 slides (~25s each → ~3 min)
1. Title · 2. Problem (process logic) · 3. Insight (recipe = sentence) ·
4. **Hybrid** (model proposes, physics disposes) · 5. The 3 tasks, tackled ·
6. Results (real numbers, official scorer) · 7. Why us (honest evaluation).

Speaker notes with per-slide timings are embedded in the PPTX (View → Notes).

## Export the HTML to PDF (optional)
Open `index.html` → **Print** → *Save as PDF* → paper **Landscape**, margins **None**
(the print CSS sizes each slide to 1280×720, one slide per page).

## Regenerate the editable PPTX
```bash
cd pitch
python3 -m venv .pptxvenv && .pptxvenv/bin/pip install python-pptx   # one-time
.pptxvenv/bin/python build_pptx.py                                    # -> ProcSeq_Pitch.pptx
```
(Homebrew's Python is PEP-668 "externally managed", so a venv is required — a global
`pip install python-pptx` is blocked.)

## Editing content
`index.html` is the design source; `build_pptx.py` mirrors it with native PowerPoint
shapes. To change wording, edit **both** (they're kept in sync by hand), or just edit
the `.pptx` directly in PowerPoint if you only need the deliverable.

## Notes on the numbers
All metrics are the **real, official-scorer** results from the completed ProcSeq
Leonardo run (`procseq_base_d20000_s16000`), held-out self-eval — see
[`../artifacts/metrics.json`](../artifacts/metrics.json) and the root
[`REPORT.md`](../../../../REPORT.md). Slide 6 states the provenance + honest limits
(learned anomaly encoder is weak; the physics hybrid carries Task 3).
