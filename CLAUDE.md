# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Context

This is the working repository for **Zero One Hack_01**, a 36-hour hackathon hosted at AI Factory Austria. We are on the **Industrial AI track (Infineon)**: train and benchmark sequence models on semiconductor process flows.

Compute is provided by **CINECA's Leonardo supercomputer** (Bologna, Italy). We currently operate from a **login node** — compute-heavy work must be submitted via SLURM, not run interactively.

---

## Leonardo HPC: Critical Rules

**Login node restrictions** — the login node permits only:
- File management, data transfer (`rsync`, `scp`, `wget`)
- Code editing, compilation, environment setup
- Submitting and monitoring jobs
- Short test runs (< 10 min CPU time, no GPU usage)

**Never run training, heavy data preprocessing, or long loops on the login node.** This degrades the system for all users and can result in disqualification.

---

## Storage System

| Area | Env Var | Quota | Backup | Purge | Use for |
|------|---------|-------|--------|-------|---------|
| Home | `$HOME` | 50 GB | None active | Never | Scripts, env configs |
| Work (project-shared) | `$WORK` | 1 TB | None | 6 mo post-project | Datasets, checkpoints |
| Fast (SSD, project-shared) | `$FAST` | 1 TB | None | 6 mo post-project | I/O-intensive training data |
| Scratch (user) | `$SCRATCH` | ~20 TB | None | **40 days** | Large temp files |
| Node-local tmp | `$TMPDIR` | varies | — | Job end | Per-job staging |
| Login-node local | `/scratch_local` | 14 TB shared | — | — | Staging only |

**Important:** `$HOME` backup is currently **not active** on Leonardo. Store anything important in the git repo or `$WORK`.

**Lustre filesystem rules:**
- Avoid `ls -l`, `find`, `du` on large directories — they hammer metadata. Use `lfs find` and `lfs quota` instead.
- Check usage: `cindata` (overview) or `cinQuota` (detailed)
- Aggregate small files into `.tar` archives or HDF5/NetCDF; never store thousands of individual small files on Lustre.
- Do not use `touch` to extend $SCRATCH file timestamps — this is monitored and causes account restrictions.

---

## SLURM Job Submission

### Booster partition (GPU — A100)

Each Booster node: **4× A100 64 GB GPUs**, 32 CPU cores, 512 GB RAM, NVLink 3.0.

| QOS | Max nodes | Walltime | Use |
|-----|-----------|----------|-----|
| `boost_qos_dbg` | 8 | 30 min | Quick functional test |
| `boost_usr_prod` | 64 | 24 h | Standard training runs |
| `boost_qos_bprod` | 256 | 24 h | Large-scale runs |
| `boost_qos_lprod` | 8 | 4 days | Long jobs |

### Example single-node GPU job script

```bash
#!/bin/bash
#SBATCH --job-name=infineon_train
#SBATCH --account=<your_account>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load cuda/12.2
module load openmpi

# Stage data to node-local storage if needed
# cp $WORK/data/... $TMPDIR/

python train.py
```

### Useful SLURM commands

```bash
squeue -u $USER              # view your jobs
scancel <jobid>              # cancel a job
scontrol show job <jobid>    # job details
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed  # accounting
sinfo -p boost_usr_prod      # partition status
```

### Debug QOS for fast iteration

Always prototype with `boost_qos_dbg` (30-min limit, quick queue) before submitting production runs.

---

## Software Environment

### Module system

```bash
module avail            # list available modules
module load cuda/12.2
module load python/3.11  # check exact name with module avail
module list             # show loaded modules
module purge            # clear all modules
```

### Conda / virtual environments

Create environments in `$WORK` (not `$HOME`) to avoid 50 GB quota issues:

```bash
conda create --prefix $WORK/envs/myenv python=3.11
conda activate $WORK/envs/myenv
```

Or with pip venv:

```bash
python -m venv $WORK/envs/myenv
source $WORK/envs/myenv/bin/activate
```

Install heavy packages (PyTorch, etc.) from within a job or on the login node (no GPU needed for install).

---

## Hackathon Track: Industrial AI (Infineon)

**Goal:** Train and benchmark sequence models on semiconductor process flows. The evaluation question is whether the model learns real process logic or just memorizes.

Submission deadline: **Sunday 10:00** via Tally form (link in `#announcements` on Discord).

Required submission materials:
1. Team name
2. Repository URL
3. Slides (PDF)
4. Demo video (max 2 min)

Judging criteria (from `judging/rubrics.md`): working artifact, reproducibility, honest evaluation, visible reasoning.

---

## SSH Access

```bash
ssh <username>@login.leonardo.cineca.it   # 2FA required every login
```

Two-factor authentication (Step client + certificate) is mandatory.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->
