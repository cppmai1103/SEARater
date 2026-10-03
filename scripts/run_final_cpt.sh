#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a6000
#SBATCH --time=08:00:00
#SBATCH --job-name=sea-rater-final-cpt
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

# Section 11: build the final 50M-token (12.5M/language) dataset from data/best_weight.json
python scripts/build_final_cpt_dataset.py --baseline best_weighted
# Section 12: LoRA continue-pretrain Qwen2.5-1.5B Base on that dataset -> models/qwen1.5b_cpt_bestweighted/
python scripts/run_final_cpt.py --baseline best_weighted --device cuda

# Section 7/11/12 other baselines -- uncomment the pair for whichever baseline
# you want to add to the comparison
# python scripts/build_final_cpt_dataset.py --baseline random
# python scripts/run_final_cpt.py --baseline random --device cuda
# python scripts/build_final_cpt_dataset.py --baseline clean_heuristic
# python scripts/run_final_cpt.py --baseline clean_heuristic --device cuda
# python scripts/build_final_cpt_dataset.py --baseline equal_average
# python scripts/run_final_cpt.py --baseline equal_average --device cuda

# Section 13: compare Original Qwen2.5-1.5B Base vs each trained baseline above -> data/evaluation_results.json
python scripts/evaluate_models.py --baselines best_weighted --device cuda
