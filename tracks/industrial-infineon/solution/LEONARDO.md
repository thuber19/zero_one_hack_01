# Running procseq on Leonardo (CINECA)

SLURM config is pre-filled for the hackathon: partition `boost_usr_prod`,
account `EUHPC_D30_031`, reservation `s_tra_ncc` (mirrors the team's `jobs/run.sh`).
Runs are **single A100 + plain Accelerate** (no DeepSpeed) — the models are small.

## One-time setup (login node)
```bash
# from your clone of the repo
cd zero_one_hack_01/tracks/industrial-infineon/solution
bash setup_leonardo.sh        # creates ~/procseq-venv and pip-installs the stack
```
If `setup_leonardo.sh` can't find a Python module, run `module avail python`,
edit the `module load` line, and re-run. (pixi's python at `~/.pixi/bin` also works.)

## Launch the runs (login node → compute)
```bash
cd zero_one_hack_01/tracks/industrial-infineon/solution
sbatch slurm/train_decoder.sbatch     # Tasks 1+2: build data -> train -> infer -> score
sbatch slurm/train_encoder.sbatch     # Task 3: anomaly + rule-attr + contrastive
squeue --me                           # watch
```
Each job builds data + tokenizer, trains, writes `artifacts/submission_task{1,2,3}.csv`
+ `artifacts/metrics.json`, and prints progress to `slurm-procseq-*-<jobid>.out`.

## Scaling sweep (optional, after a decoder run)
```bash
sbatch slurm/scaling_sweep.sbatch     # 16-cell size x data grid, 1 A100 per cell
# results accumulate in artifacts/sweep/sweep_results.jsonl
```

## Knobs
- `CONFIG=configs/leonardo_decoder.yaml` (default) — edit `size`/`max_steps`/`data_per_family`.
- `PROCSEQ_CANON=0 sbatch ...` — A/B canonicalization off vs on.
- `encoder.contrastive.enabled` in `configs/leonardo_encoder.yaml` — A/B the contrastive term.
- `decoder.constrained_decode` — grammar-veto decoding on/off.

## Notes
- Outputs land in `solution/artifacts/` next to the code; move to `$SCRATCH` for big sweeps.
- Compute nodes have no internet — all installs happen on the login node in setup.
- `make smoke` reproduces the whole flow locally in <1 min for sanity before submitting.
