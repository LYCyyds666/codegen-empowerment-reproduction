#!/bin/bash
#SBATCH --job-name=codegen-smoke
#SBATCH --account=hhu49
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/hhu49/97e911/logs/codegen-smoke-%j.out
#SBATCH --error=/scratch/hhu49/97e911/logs/codegen-smoke-%j.err

set -euo pipefail

module load gcc/8.5.0
module load anaconda3/2024.02-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/hhu49/97e911/codegen_env

cd /scratch/hhu49/97e911/codegen_empowerment

export CODEGEN_ROOT=/scratch/hhu49/97e911/codegen_root
export HF_HOME=/scratch/hhu49/97e911/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PYTHON=/scratch/hhu49/97e911/codegen_env/bin/python

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: ${PYTHON}"

echo "===== Create smoke dataset ====="
"$PYTHON" make_smoke_dataset.py

echo "===== Test two GPUs ====="
srun \
  --ntasks=1 \
  --gres=gpu:2 \
  --export=ALL \
  "$PYTHON" check_two_gpus.py

echo "===== Start two-GPU smoke training ====="
srun \
  --ntasks=1 \
  --gres=gpu:2 \
  --export=ALL \
  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    -m codegen.training.finetune_agent \
    --config-name finetune_agent_smoke

echo "===== Smoke training completed ====="
