"""One grid cell of the scaling experiment: train tiny decoder, self-score, append JSONL."""
import argparse, json, shutil
from pathlib import Path
from procseq.config import Config
from procseq import train_decoder, infer, run_eval

# Files the decoder tasks (1+2) need for inference and scoring.
_EVAL_MIRROR_FILES = ["eval_input_valid.csv", "eval_valid_groundtruth.csv"]

def _stage_eval_mirrors(eval_from: Path, out: Path) -> None:
    """Copy the eval-mirror CSVs into the cell's artifacts dir so infer/run_eval
    (which read cfg.artifacts) can find them. Run build_data first to create them."""
    for name in _EVAL_MIRROR_FILES:
        src, dst = eval_from / name, out / name
        if dst.exists():
            continue
        if not src.exists():
            raise FileNotFoundError(
                f"{src} missing. Run `python -m procseq.build_data` (or --smoke) "
                f"first so the eval mirrors exist, or pass --eval-from <dir>."
            )
        shutil.copy(src, dst)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--data-per-family", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--out", default="artifacts/sweep")
    ap.add_argument("--eval-from", default="artifacts",
                    help="dir holding the eval mirrors built by build_data")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _stage_eval_mirrors(Path(a.eval_from), out)
    ckpt = out / f"decoder_{a.size}_{a.data_per_family}"
    cfg = Config({
        "run_name": f"sweep_{a.size}_{a.data_per_family}", "seed": a.seed,
        "precision": "bf16", "artifacts": str(out),
        "decoder_ckpt": str(ckpt), "encoder_ckpt": str(out / "encoder_unused"),
        "decoder": {"size": a.size, "max_len": 256,
                    "data_per_family": a.data_per_family, "batch_size": 64,
                    "lr": 6e-4, "max_steps": a.max_steps, "eval_every": 500},
    })
    cfg_path = out / f"cfg_{a.size}_{a.data_per_family}.yaml"
    import yaml; cfg_path.write_text(yaml.safe_dump(dict(cfg)))
    train_decoder.main(["--config", str(cfg_path)])
    # NOTE: requires eval_input files in cfg.artifacts; build_data must run first.
    infer.run_task1(cfg); infer.run_task2(cfg)
    run_eval.main(["--config", str(cfg_path)])
    metrics = json.loads((out / "metrics.json").read_text())
    rec = {"size": a.size, "data_per_family": a.data_per_family, **metrics}
    with (out / "sweep_results.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"sweep cell done: {a.size}/{a.data_per_family}")

if __name__ == "__main__":
    main()
