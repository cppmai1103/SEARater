#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=1:00:00
#SBATCH --job-name=test
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=6GB
#SBATCH --qos=normal

# --time bumped from the old 08:00:00: this run now covers 8 languages
# (was 4), 26 weight combos (was 16), and LoRA proxy CPT (was full FT) --
# meaningfully more work than the time limit that used to fit the old
# pipeline. 24h is a guess, not a measurement -- check your partition's
# actual max allowed time and adjust. If the job times out partway, `set -e`
# below means it stops cleanly rather than skipping ahead on missing data;
# just re-submit after commenting out whatever already finished (check
# data/proxy_results.csv row count, data/candidate_corpus/selected/, etc.
# to see how far it got).

set -e  # stop on the first failing command instead of silently running every later step on bad/missing data

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="sea_rater_env"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENVIRONMENT_NAME}

# pip install -r requirements.txt

pip install myanmartools PyICU

# export HF_TOKEN=<your_hf_token>

python encode.py