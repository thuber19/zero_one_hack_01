"""Streamlit dashboard: metrics, plots, before/after examples. Run:
   streamlit run dashboard/app.py -- --artifacts artifacts
"""
import json, sys
from pathlib import Path
import streamlit as st

art = Path("artifacts")
for i, a in enumerate(sys.argv):
    if a == "--artifacts" and i + 1 < len(sys.argv):
        art = Path(sys.argv[i + 1])

st.set_page_config(page_title="Infineon Process-Logic", layout="wide")
st.title("Process-Logic Pipeline — Results")

mp = art / "metrics.json"
if mp.exists():
    st.subheader("Metrics")
    st.json(json.loads(mp.read_text()))
else:
    st.warning("No metrics.json yet — run `make eval`.")

c1, c2 = st.columns(2)
for col, name in [(c1, "plot_task1.png"), (c2, "plot_scaling.png")]:
    p = art / name
    if p.exists():
        col.image(str(p))

ex = art / "demo_examples.json"
if ex.exists():
    st.subheader("Baseline vs Trained — next step")
    for r in json.loads(ex.read_text()):
        st.markdown(f"**{r['example']}** ({r['family']}) — `…{r['prefix_tail']}`")
        a, b = st.columns(2)
        a.write({"baseline (n-gram)": r["baseline_next"]})
        b.write({"trained model": r["model_next"]})
