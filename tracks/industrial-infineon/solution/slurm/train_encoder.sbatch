#!/bin/bash
#SBATCH --job-name=procseq-enc
#SBATCH --account=__ACCOUNT__
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=__QOS__
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/enc_%j.out

set -euo pipefail
module load profile/deeplrn || true
module load python cuda || true
source "${VENV:-$HOME/procseq-venv}/bin/activate"
cd "${SOLUTION_DIR:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=8
CONFIG="${CONFIG:-configs/leonardo_encoder.yaml}"
srun accelerate launch --config_file configs/accelerate.yaml \
  --num_processes ${SLURM_GPUS_ON_NODE:-4} \
  -m procseq.train_encoder --config "$CONFIG"
