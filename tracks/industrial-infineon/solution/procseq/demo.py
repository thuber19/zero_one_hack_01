"""Side-by-side baseline vs trained outputs + metric/scaling plots."""
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from procseq.config import load_config
from procseq.tokenizer import load_tokenizer
from procseq import infer, baselines
from procseq.data import scale_family
from procseq.grammar import FAMILIES

def _ngram(seed):
    seqs = []
    for fam in FAMILIES:
        seqs += scale_family(fam, 0, seed)  # provided only
    return baselines.NgramModel(n=3).fit(seqs)

def make_prediction_examples(cfg, n=5):
    art = Path(cfg["artifacts"])
    tok = load_tokenizer(Path(cfg["decoder_ckpt"]))
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(cfg["decoder_ckpt"])
    ng = _ngram(cfg.get("seed", 42))
    rows = []
    with (art / "eval_input_valid.csv").open() as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= n: break
            steps = r["PARTIAL_SEQUENCE"].split("|")
            rows.append({
                "example": r["EXAMPLE_ID"], "family": r["FAMILY"],
                "prefix_tail": " -> ".join(steps[-3:]),
                "baseline_next": ng.predict_next(steps, k=3),
                "model_next": infer.predict_next_step(model, tok, steps, r["FAMILY"], k=3),
            })
    (art / "demo_examples.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))

def plot_metrics(cfg):
    art = Path(cfg["artifacts"])
    m = json.loads((art / "metrics.json").read_text())
    if "task1_nextstep" in m:
        t1 = m["task1_nextstep"]
        plt.figure()
        plt.bar(["top1","top3","top5","mrr"],
                [t1["top1"], t1["top3"], t1["top5"], t1["mrr"]])
        plt.title("Task 1 — next-step"); plt.ylim(0,1)
        plt.savefig(art / "plot_task1.png", dpi=120); plt.close()
    sweep = art / "sweep" / "sweep_results.jsonl"
    if sweep.exists():
        recs = [json.loads(l) for l in sweep.read_text().splitlines()]
        plt.figure()
        for size in sorted({r["size"] for r in recs}):
            pts = sorted([r for r in recs if r["size"]==size],
                         key=lambda r: r["data_per_family"])
            xs = [r["data_per_family"] for r in pts]
            ys = [r.get("task1_nextstep",{}).get("top1",0) for r in pts]
            plt.plot(xs, ys, marker="o", label=size)
        plt.xscale("log"); plt.xlabel("sequences/family"); plt.ylabel("Top-1")
        plt.title("Scaling"); plt.legend()
        plt.savefig(art / "plot_scaling.png", dpi=120); plt.close()
    print(f"plots -> {art}")

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(argv); cfg = load_config(a.config)
    make_prediction_examples(cfg); plot_metrics(cfg)

if __name__ == "__main__":
    main()
