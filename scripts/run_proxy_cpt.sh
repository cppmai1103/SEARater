#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a6000
#SBATCH --time=08:00:00
#SBATCH --job-name=sea-rater-proxy-cpt
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="sea_rater_env"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENVIRONMENT_NAME}

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

: "${HF_TOKEN:?HF_TOKEN is not set}"

nvidia-smi

python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

# Steps 7-8 (weight combinations + selected proxy data) must already exist:
#   python scripts/generate_weight_combinations.py
#   python scripts/select_weighted_corpus.py
python scripts/build_validation_set.py
python scripts/run_proxy_cpt.py --device cuda
python scripts/select_best_weight.py
