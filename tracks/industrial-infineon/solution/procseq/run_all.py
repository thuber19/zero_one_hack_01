"""One entrypoint for the WHOLE procseq pipeline, so a single call gets everything
ready: build data -> train decoder -> train encoder -> infer all 3 tasks (pure +
physics hybrids) -> self-eval -> OFFICIAL scores -> real submissions.

    python -m procseq.run_all --config configs/leonardo_full.yaml      # full
    python -m procseq.run_all --config <cfg> --skip-train              # inference only
    python -m procseq.run_all --config configs/smoke.yaml --smoke      # tiny local test

Training is run as a subprocess per model (clean Accelerator state each time);
everything else runs in-process. Honors PROCSEQ_ARTIFACTS / config `artifacts`.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from procseq.config import load_config


def _train(module, cfg_path):
    print(f"\n===== TRAIN {module} =====", flush=True)
    r = subprocess.run([sys.executable, "-m", module, "--config", cfg_path])
    if r.returncode != 0:
        raise SystemExit(f"{module} failed (exit {r.returncode})")


def _visible_gpus():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        return [g for g in cvd.split(",") if g.strip() != ""]
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _train_parallel(cfg_path, art):
    """Train decoder + encoder CONCURRENTLY, splitting the visible GPUs between
    them (4 GPUs -> 2 each via DDP; 2 -> 1 each; 0/1 -> shared). Each model uses
    `accelerate launch` across its assigned GPUs. Distinct ports avoid clashes;
    each logs to its own file so the streams don't garble."""
    gpus = _visible_gpus()
    if len(gpus) >= 2:
        half = len(gpus) // 2
        dec_gpus, enc_gpus = gpus[:half], gpus[half:]
    else:
        dec_gpus = enc_gpus = gpus[:1]   # 0 or 1 GPU -> both share
    accelerate = str(Path(sys.executable).parent / "accelerate")
    specs = [("procseq.train_decoder", "decoder", dec_gpus, 29500),
             ("procseq.train_encoder", "encoder", enc_gpus, 29501)]
    procs = []
    for module, name, g, port in specs:
        env = dict(os.environ)
        if g:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(g)
        nproc = max(1, len(g))
        cmd = [accelerate, "launch", "--num_processes", str(nproc),
               "--mixed_precision", "bf16", "--main_process_port", str(port),
               "-m", module, "--config", cfg_path]
        logf = open(Path(art) / f"train_{name}.log", "w")
        print(f"===== TRAIN {name}: {nproc} GPU(s) [{env.get('CUDA_VISIBLE_DEVICES', 'cpu/mps')}]"
              f" -> {art}/train_{name}.log =====", flush=True)
        procs.append((subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf, name))
    failed = False
    for p, logf, name in procs:
        rc = p.wait(); logf.close()
        print(f"  {name} training exit={rc}", flush=True)
        if rc != 0:
            failed = True
            try:
                print(f"--- tail {name} log ---\n" + (Path(art) / f"train_{name}.log").read_text()[-1800:], flush=True)
            except Exception:
                pass
    if failed:
        raise SystemExit("parallel training failed (see train_*.log)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-train", action="store_true", help="models already trained")
    ap.add_argument("--parallel-train", action="store_true",
                    help="train decoder + encoder concurrently, splitting available GPUs")
    ap.add_argument("--no-real", action="store_true", help="skip the real-eval submissions")
    ap.add_argument("--smoke", action="store_true", help="tiny data build (local sanity)")
    a = ap.parse_args(argv)
    cfg_path = a.config
    cfg = load_config(cfg_path)
    art = cfg.get("artifacts", "artifacts")
    t0 = time.time()

    # 1) shared data + tokenizer (built ONCE here so concurrent trainers don't race)
    print("===== BUILD DATA =====", flush=True)
    from procseq import build_data
    n = cfg.get("decoder", {}).get("data_per_family", 5000)
    build_data.run(n_per_family=n, seed=cfg.get("seed", 42), smoke=a.smoke)

    # 2) train BOTH models (concurrently if --parallel-train, else sequentially)
    if not a.skip_train:
        if a.parallel_train:
            _train_parallel(cfg_path, art)
        else:
            _train("procseq.train_decoder", cfg_path)
            _train("procseq.train_encoder", cfg_path)

    # 3) inference + self-eval on our labelled mirrors (pure + physics hybrids)
    print("\n===== INFERENCE + SELF-EVAL (mirrors) =====", flush=True)
    from procseq import infer, infer_hybrid, infer_anomaly_hybrid, run_eval, score_official
    infer.run_task1(cfg); infer.run_task2(cfg)
    try:
        infer.run_task3(cfg)
    except Exception as e:
        print(f"  Task 3 skipped (no encoder checkpoint): {e}", flush=True)
    run_eval.main(["--config", cfg_path])

    # 4) real submissions from the organizer-format eval files
    if not a.no_real:
        print("\n===== REAL SUBMISSIONS =====", flush=True)
        infer.run_task1(cfg, real=True); infer.run_task2(cfg, real=True)
        try:
            infer.run_task3(cfg, real=True)
        except Exception as e:
            print(f"  Task 3 real skipped (no encoder checkpoint): {e}", flush=True)

        # Rename to final submission format: nextstep.csv, completion.csv, anomaly.csv
        art_p = Path(art)
        for src_name, dst_name in [
            ("submission_task1_real.csv", "nextstep.csv"),
            ("submission_task2_real.csv", "completion.csv"),
            ("submission_task3_real.csv", "anomaly.csv"),
        ]:
            src_f = art_p / src_name
            dst_f = art_p / dst_name
            if src_f.exists():
                import shutil
                shutil.copy2(src_f, dst_f)
                print(f"  {src_name} -> {dst_name}", flush=True)
        print(f"\nFinal submission files in: {art}", flush=True)

    # 5) hybrid: physics rerank for task 1+3 (fast), skip task 2 beam decode (slow)
    try:
        from procseq import infer_hybrid, infer_anomaly_hybrid
        print("\n===== HYBRID (tasks 1+3 only, skipping task 2 beam-decode) =====", flush=True)
        infer_hybrid.run_hybrid_task1_only(cfg)
        infer_anomaly_hybrid.run(cfg)
        if not a.no_real:
            infer_hybrid.run_hybrid_task1_only(cfg, real=True)
            infer_anomaly_hybrid.run(cfg, real=True)
    except Exception as e:
        print(f"  Hybrid step failed (non-fatal): {e}", flush=True)

    print(f"\n===== DONE in {time.time()-t0:.0f}s =====", flush=True)
    print(f"artifacts: {cfg.get('artifacts')}", flush=True)


if __name__ == "__main__":
    main()
