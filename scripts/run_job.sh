#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-h100 # Partition is a queue for jobs
#SBATCH --time=10:00:00         # Time limit for the job
#SBATCH --job-name=test   # Name of your job
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1               # Number of nodes you want to run your process on
#SBATCH --ntasks-per-node=8     # Number of CPU cores
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH --qos=normal

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="test_env"

module load Anaconda3
# You need this to be able to use the 'conda' command. I am not satisfied with this solution, I'll try to find a way so that you won't have to set it in the future.
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh


# if the environment does not exist, create it
if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENVIRONMENT_NAME}
 
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate matplotlib numpy seaborn pandas scikit-learn venn
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment}"

nvidia-smi 

# --limit 1
# --data-types medical,law,general,linguistic
python 02_extract_scores.py --data-types medical,law,economics,general,linguistic
python 03_select_top_neurons.py --data-types medical,law,economics,general,linguistic
python 04_analyze_plots.py --data-types medical,law,economics,general,linguistic
