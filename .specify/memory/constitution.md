<!--
SYNC IMPACT REPORT
==================
Version change: (unversioned template) → 1.0.0
Bump rationale: MINOR — first substantive population of all principles and sections from template.

Modified principles:
  [PRINCIPLE_1_NAME] → I. Experimental Breadth with Honest Baselines
  [PRINCIPLE_2_NAME] → II. HPC-First Execution (NON-NEGOTIABLE)
  [PRINCIPLE_3_NAME] → III. Reproducibility (NON-NEGOTIABLE)
  [PRINCIPLE_4_NAME] → IV. Honest Evaluation
  [PRINCIPLE_5_NAME] → V. Rapid Iteration via Debug QOS
  [PRINCIPLE_6_NAME] → VI. Storage Discipline
  [PRINCIPLE_7_NAME] → VII. Submission Readiness

Added sections:
  - Model Architecture Constraints (Section 2)
  - HPC Development Workflow (Section 3)

Removed sections: none

Templates requiring updates:
  ✅ .specify/memory/constitution.md — this file
  ✅ .specify/templates/plan-template.md — Constitution Check gate language is
     template-generic and works as-is; no changes required
  ✅ .specify/templates/spec-template.md — no principle-specific references; no
     changes required
  ✅ .specify/templates/tasks-template.md — no principle-specific references; no
     changes required

Deferred TODOs:
  - SLURM account name (<your_account>) is not captured in this constitution;
    must be supplied per job script from team credentials.
-->

# Zero One Hack_01 Constitution

## Core Principles

### I. Experimental Breadth with Honest Baselines

Every research direction MUST include at minimum one simple baseline (LSTM or
rule-based heuristic) before advancing to more complex architectures. Approaches
to be explored in priority order:

1. **LSTM** — sequence baseline, establishes minimum performance floor
2. **Transformer** — attention-based sequence model for process flow learning
3. **BERT-style** — masked-language-model pre-training adapted for anomaly detection
4. **Rule-Augmented / Constrained Decoding** — inject domain knowledge as hard
   constraints or soft biases during inference

No approach MUST be discarded without a recorded comparison against the baseline.
Complexity is only justified when a simpler model demonstrably fails to capture
the target process logic.

### II. HPC-First Execution (NON-NEGOTIABLE)

All GPU training, heavy data preprocessing, and long inference loops MUST be
submitted as SLURM jobs on Leonardo's Booster partition. They MUST NOT run
interactively on the login node.

- Login node is reserved for: file management, code editing, environment setup,
  job submission and monitoring, and test runs under 10 min CPU / zero GPU.
- Every training script MUST have a corresponding `.sbatch` file under
  `scripts/jobs/`.
- SLURM job scripts MUST specify account, partition, QOS, node count, GPU count,
  and walltime explicitly — no implicit defaults.

Violation of this principle risks account suspension and hackathon disqualification.

### III. Reproducibility (NON-NEGOTIABLE)

Every experiment MUST be fully reproducible by any team member from a clean
checkout. This requires:

- Random seeds fixed and logged (`--seed` argument or config field).
- Full environment captured (`pip freeze > requirements.txt` or `conda env export`
  stored in `$WORK/envs/` and committed as a lockfile in the repo).
- Hyperparameters stored in a config file (YAML/JSON), never hardcoded.
- Model checkpoints saved to `$WORK/checkpoints/<run-id>/` with the config file
  alongside.
- Experiment results (metrics, loss curves) logged to a structured format
  (CSV or JSON) under `results/`.

A result that cannot be reproduced from the repo + config is not a valid result.

### IV. Honest Evaluation

Models MUST be evaluated to distinguish genuine process logic learning from
memorization. Required evaluation practices:

- Train/validation/test splits MUST be fixed before any model sees data; test
  split is touched exactly once (final evaluation only).
- Evaluation MUST report at minimum: sequence accuracy, step-level F1, and a
  memorization diagnostic (e.g., performance on novel process orderings or held-
  out process families not seen during training).
- Leaderboard or headline metrics MUST NOT be cherry-picked from validation runs
  — only the held-out test evaluation counts.
- All evaluation code MUST be version-controlled and runnable independently of
  training code.

### V. Rapid Iteration via Debug QOS

All new training scripts MUST be validated with `boost_qos_dbg` (30-min limit,
fast queue) before submitting to production QOS. This gate catches environment
errors, data-loading failures, and shape bugs cheaply.

- Debug run MUST complete at least 1 full forward+backward pass and log a loss
  value before the production job is submitted.
- Production job script MUST reference which debug run validated it (comment in
  the `.sbatch` file: `# Validated by debug job <JOBID>`).

### VI. Storage Discipline

Storage areas MUST be used for their designated purposes:

| What | Where |
|------|-------|
| Scripts, configs, code | `$HOME` / git repo |
| Datasets, model checkpoints | `$WORK` or `$FAST` |
| Large temp/intermediate outputs | `$SCRATCH` (purged after 40 days) |
| Per-job staging | `$TMPDIR` (purged at job end) |

Lustre filesystem rules (violations degrade the system for all users):
- MUST use `lfs find` / `lfs quota` instead of `find` / `du` on large dirs.
- MUST aggregate small files into `.tar`, HDF5, or NetCDF — never thousands of
  individual small files on Lustre.
- MUST NOT use `touch` to extend `$SCRATCH` file timestamps.
- Conda / venv environments MUST live in `$WORK/envs/`, never in `$HOME`.

### VII. Submission Readiness

The `main` branch MUST remain in a state that can be submitted at any time before
the Sunday 10:00 deadline. Required at all times on `main`:

- `README.md` with setup instructions and how to reproduce the best result.
- At least one trained checkpoint committed or referenced (path in `$WORK`).
- Slides PDF under `docs/slides.pdf` (may be a draft until final).
- Demo video reference or placeholder under `docs/demo_link.txt`.

Feature/experiment branches MAY be in any state; only `main` is subject to this
constraint.

## Model Architecture Constraints

Architectural choices MUST be grounded in the process-flow domain:

- Input representations MUST encode process step identity and ordering; raw
  string tokenization alone is insufficient.
- BERT-style anomaly detection MUST define a clear notion of "normal" sequence
  derived from training data statistics, not from researcher intuition.
- Rule-Augmented / Constrained Decoding constraints MUST be derived from
  documented semiconductor process rules, not invented ad hoc. Source of rules
  MUST be cited in the relevant spec or research doc.
- Model size MUST be justified against the available A100 GPU memory (64 GB per
  GPU, 4 GPUs per node). Batch size and sequence length MUST fit within a single
  node unless multi-node training is explicitly planned.
- Transfer learning from pre-trained checkpoints (e.g., BERT, GPT-2) is
  encouraged but MUST be noted in the evaluation to avoid conflating pre-training
  data with domain generalization.

## HPC Development Workflow

All development MUST follow this order to protect login node stability and
experiment validity:

1. **Local logic check** — unit-test data loading and model forward pass on CPU
   with tiny synthetic data (runs on login node, < 1 min).
2. **Debug SLURM run** — `boost_qos_dbg`, single node, real data subset, confirm
   loss decreases for ≥ 1 epoch.
3. **Production SLURM run** — appropriate QOS selected based on expected walltime.
4. **Result logging** — metrics written to `results/`, checkpoint saved to
   `$WORK/checkpoints/`.
5. **Evaluation** — run eval script against held-out test set; record in
   `results/eval_<model>_<date>.json`.
6. **Commit** — config, job script, and results summary committed to `main` or
   feature branch.

Code review is not mandatory given the hackathon timeline, but any merge to
`main` that modifies evaluation logic or data splits MUST be reviewed by at least
one other team member before merging.

## Governance

This constitution supersedes all other development practices for Zero One Hack_01.
In case of conflict between this document and any other guideline, this document
takes precedence.

**Amendment procedure**: Any team member may propose an amendment by editing this
file and incrementing the version. MAJOR bump for removing or redefining a
principle; MINOR for adding a new principle or section; PATCH for clarifications
and wording. Amendments during the hackathon require verbal agreement from the
full team (no formal PR process required given time constraints).

**Compliance**: All SLURM job submissions, model commits, and evaluation reports
MUST be checked against Principles II, III, IV, and VI before merge to `main`.
The Constitution Check section in `plan.md` (generated by `/speckit-plan`) is
the canonical gate checklist for each feature.

**Runtime guidance**: See `CLAUDE.md` for Leonardo HPC operational details,
storage quotas, SLURM commands, and environment setup procedures.

**Version**: 1.0.0 | **Ratified**: 2026-05-30 | **Last Amended**: 2026-05-30
