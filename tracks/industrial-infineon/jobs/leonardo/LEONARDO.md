# Leonardo (CINECA) — secure connect + train, the hackathon way

Source of truth: `hackathon_leonardo_introduction.pdf` (authoritative for the
hackathon) + the AI-AT HPC Onboarding Kit ch.5/ch.6 (https://ai-at.eu/hpc-onboarding/).
Where they differ, the **hackathon PDF wins** (noted inline).

## 0. Security model (best practice)
- **Key-based SSH** (set up locally already): a dedicated `~/.ssh/leonardo_ed25519`
  keypair + a `Host leonardo` block in `~/.ssh/config`. The account **password is
  used exactly once** (to install the public key) and is **never stored in the repo**.
- `creds.txt` (your `user:a08trd14` + password) lives **outside** the repo and is
  gitignored — never commit it.
- The compute-node **proxy password** in the PDF is a *shared, rotating* hackathon
  secret — keep it in the Slurm script only, do not publish it outside the team.
- No 2FA for the hackathon (per the PDF); the general CINECA flow uses 2FA + Step CA.

## 1. One-time: install your public key on Leonardo
From this machine (you'll type the Leonardo password **once**):
```bash
# Windows PowerShell/Git-bash. Login nodes: login01/02/05/07-ext.leonardo.cineca.it
ssh-copy-id -i ~/.ssh/leonardo_ed25519.pub leonardo        # if ssh-copy-id available
# --- or manual (works everywhere) ---
cat ~/.ssh/leonardo_ed25519.pub | ssh a08trd14@login01-ext.leonardo.cineca.it \
  "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```
Then verify password-less login:
```bash
ssh leonardo "hostname; whoami; echo OK"
```

## 2. One-time: set up the project on a LOGIN node (has internet)
```bash
ssh leonardo
bash <(curl -fsSL https://raw.githubusercontent.com/thuber19/zero_one_hack_01/mina/physics-integration/tracks/industrial-infineon/jobs/leonardo/setup_leonardo.sh)
# (or: git clone the repo to $SCRATCH yourself, then run jobs/leonardo/setup_leonardo.sh)
```
What it does: clones the repo to **`$SCRATCH`** (big-files area — the PDF says use
`$SCRATCH`, **NOT** `$WORK`/`$FAST`, during the hackathon), points caches off the
50 GB `$HOME`, installs **pixi**, and builds the env (python 3.11 + torch/cuda12 +
numpy + scikit-learn). Run heavy login-node steps under
`srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash`
(login nodes kill processes after 10 min CPU).

## 3. Train + benchmark (GPU compute node, via Slurm)
```bash
ssh leonardo
cd $SCRATCH/zero_one_hack_01/tracks/industrial-infineon
sbatch jobs/leonardo/train.slurm           # 1 GPU, reservation s_tra_ncc, <=4h
squeue --me                                 # watch it
tail -f slurm-<jobid>.out                    # follow logs
```
The job: generate held-out data → SFT (`--aux-category --unk-dropout`) → optional
GRPO → inference → **score with the official `data/eval_metrics.py`** (real T1/T2/T3
numbers). Edit `--model-size`/`--epochs`/GPUs at the top of the script.

## 4. Produce the ACTUAL submission (organizers' real eval files)
After a model is trained (`$RUN/best_model.pt`), run inference on the organizers'
real, unlabeled eval files (they live in `data/`):
```bash
cd $SCRATCH/zero_one_hack_01/tracks/industrial-infineon
pixi run python src/inference.py --output-dir "$RUN" --eval-dir data --model-size medium
# -> $RUN/submissions/{nextstep,completion,anomaly}.csv  (submit these)
pixi run python validate_submission.py --submission-dir "$RUN/submissions"   # format check
```
(The organizers' `eval_input_*.csv` are **unlabeled** — you can't self-score them;
`eval_metrics.py` only gives numbers on the held-out split from step 3.)

## 5. Get results back to your machine
```bash
# from THIS machine (small files via login node; big via the data-mover node)
scp leonardo:'$SCRATCH/run_<jobid>/submissions/*.csv' ./outputs_leonardo/
scp leonardo-data:'$SCRATCH/run_<jobid>/best_model.pt' ./outputs_leonardo/   # large files
```

## Quick reference (from the PDF)
| Thing | Value |
|---|---|
| Login nodes | `login01/02/05/07-ext.leonardo.cineca.it` (no 2FA) |
| Partition / reservation | `boost_usr_prod` / `s_tra_ncc` (1 node/team) |
| GPUs | up to 4/node; `--mem=120GB×gpus`, `--cpus-per-task=8×gpus`, `--ntasks-per-node=1` |
| Walltime | up to `24:00:00` |
| Storage | `$HOME` 50GB (code) · `$SCRATCH` big files (40-day purge) · `$PUBLIC` 50GB · **avoid `$WORK`/`$FAST`** |
| Login CPU limit | 10 min → use `srun --partition=lrd_all_serial ... --pty bash` |
| Internet | login nodes: yes · compute nodes: no (proxy for low-bandwidth only) |
| Slurm | `sbatch`/`squeue --me`/`scancel <id>`/`tail -f slurm-<id>.out`/`saldo -b` |
