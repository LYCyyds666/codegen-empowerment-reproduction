#!/bin/bash
#SBATCH --job-name=codegen-smoke
#SBATCH --account=hhu49
#SBATCH --partition=nvl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=60
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
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# 让 CUDA 和 NVML 使用相同的 PCI Bus 排序。
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Job ID: ${SLURM_JOB_ID}"
echo "Batch node: $(hostname)"
echo "Batch CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

echo "===== Creating smoke dataset ====="

# 生成测试数据不需要 GPU。
python make_smoke_dataset.py

echo "===== GPU job-step test ====="

# GPU 相关程序放到明确申请两张 GPU 的 job step 中。
srun \
  --ntasks=1 \
  --gres=gpu:2 \
  bash -lc '
    echo "Step node: $(hostname)"
    echo "Step CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    nvidia-smi -L

    python - <<'"'"'PY'"'"'
import os
import torch

print("PyTorch:", torch.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())

count = torch.cuda.device_count()
print("Visible GPU count:", count)

assert torch.cuda.is_available(), "CUDA is unavailable"
assert count == 2, f"Expected 2 GPUs, but CUDA sees {count}"

for index in range(count):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

x = torch.ones(1, device="cuda:0")
y = torch.ones(1, device="cuda:1")

print("GPU 0 tensor:", x)
print("GPU 1 tensor:", y)
print("Two-GPU CUDA test passed.")
PY
  '

echo "===== Starting smoke training ====="

# srun 先创建拥有两张 GPU 的 job step；
# torchrun 再在这个 step 内启动两个训练进程。
srun \
  --ntasks=1 \
  --gres=gpu:2 \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    -m codegen.training.finetune_agent \
    --config-name finetune_agent_smoke

echo "===== Smoke training completed ====="
