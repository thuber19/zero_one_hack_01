#!/bin/bash
#SBATCH --job-name=procseq-dec
#SBATCH --account=__ACCOUNT__          # <-- fill: CINECA project account
#SBATCH --partition=boost_usr_prod     # Leonardo Booster
#SBATCH --qos=__QOS__                  # <-- fill if required
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/dec_%j.out

set -euo pipefail
module load profile/deeplrn || true
module load python cuda || true        # <-- adjust to available Leonardo modules
source "${VENV:-$HOME/procseq-venv}/bin/activate"

cd "${SOLUTION_DIR:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=8
CONFIG="${CONFIG:-configs/leonardo_decoder.yaml}"

srun accelerate launch \
  --config_file configs/accelerate.yaml \
  --num_processes ${SLURM_GPUS_ON_NODE:-4} \
  -m procseq.train_decoder --config "$CONFIG"
