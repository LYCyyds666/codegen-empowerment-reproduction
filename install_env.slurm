#!/bin/bash
#SBATCH --job-name=codegen-env
#SBATCH --account=hhu49
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/hhu49/97e911/codegen_env_install-%j.out
#SBATCH --error=/scratch/hhu49/97e911/codegen_env_install-%j.err

module load gcc/8.5.0
module load anaconda3/2024.02-1

cd /scratch/hhu49/97e911/codegen_empowerment

conda env update \
  --prefix /scratch/hhu49/97e911/codegen_env \
  -f environment.yml \
  --prune
