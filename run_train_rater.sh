#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=24:00:00
#SBATCH --job-name=test
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
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

pip install torch --index-url https://download.pytorch.org/whl/cu124
# pip install -r requirements.txt

pip install myanmartools PyICU

# export HF_TOKEN=<your_hf_token>

nvidia-smi

python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

# # Section 3-4: split the 8-language human annotations, embed them with the
# # frozen encoder, and train the rater heads on all 8 languages -> models/rater_heads.pt
# # (the existing checkpoint was trained on only 4 languages -- retrain so the
# # rater has actually seen ms/tl/my/lo human labels before scoring them)
# python scripts/split_human_annotations.py
# python scripts/embed_splits.py --device cuda
# python scripts/train_rater.py --epochs 50 --lr 1e-3 --device cuda

# # Section 5: build + score the candidate corpus for all 8 languages -> data/candidate_corpus/
# # (vi/id/th/km's existing corpus files predate the cheap-feature fields in
# # sea_rater/heuristics.py -- rebuilding backfills those, which clean_heuristic
# # needs; ms/tl/my/lo have no candidate corpus at all yet)
# python scripts/build_candidate_corpus.py --pool-size 100000
# python scripts/score_candidate_corpus.py --device cuda

# # Section 8: materialize the fixed 26 interpretable weight vectors -> data/weight_combinations.json
# python scripts/generate_weight_combinations.py
# # Section 7: rank the scored corpus by each weight combo and select top docs per language
# # up to the proxy token budget -> data/candidate_corpus/selected/{weight_id}/{lang}.jsonl
# python scripts/select_weighted_corpus.py

# # Section 9: build the fixed held-out validation set (must not overlap with candidate training data)
# python scripts/build_validation_set.py
# # run_proxy_cpt.py APPENDS to data/proxy_results.csv rather than overwriting it, and the
# # existing file has 16 rows from the old 4-language/full-FT regime -- appending 26 fresh
# # 8-language/LoRA rows on top would leave duplicate weight_ids from two different regimes
# # in the same file. Move the old file aside (kept, not deleted) so this run starts clean.
# if [ -f data/proxy_results.csv ]; then
#   mv data/proxy_results.csv "data/proxy_results.csv.bak-4lang-fullft-$(date +%Y%m%d%H%M%S)"
# fi
# # Section 9: LoRA continue-pretrain a fresh Qwen2.5-0.5B on each of the 26 weight combos'
# # selected data, evaluate all 26 on the same 8-language validation set -> data/proxy_results.csv
# python scripts/run_proxy_cpt.py --device cuda
# # Section 10: pick the weight combo with the lowest macro validation loss -> data/best_weight.json
# python scripts/select_best_weight.py

# # Section 11: build the final 100M-token (12.5M/language) dataset from data/best_weight.json
# python scripts/build_final_cpt_dataset.py --baseline best_weighted
# # Section 12: LoRA continue-pretrain Qwen2.5-1.5B Base on that dataset -> models/qwen1.5b_cpt_bestweighted/
# python scripts/run_final_cpt.py --baseline best_weighted --device cuda

# Section 7/11/12 other baselines -- uncomment the pair for whichever
# baseline you want to add to the comparison (each builds its own dataset
# dir + models/qwen1.5b_cpt_<baseline>/, independent of best_weighted above)
# python scripts/build_final_cpt_dataset.py --baseline random
# python scripts/run_final_cpt.py --baseline random --device cuda
# python scripts/build_final_cpt_dataset.py --baseline clean_heuristic
# python scripts/run_final_cpt.py --baseline clean_heuristic --device cuda
# python scripts/build_final_cpt_dataset.py --baseline equal_average
# python scripts/run_final_cpt.py --baseline equal_average --device cuda

# Section 13: compare Original Qwen2.5-1.5B Base vs each trained baseline above -> data/evaluation_results.json
#
# Step 1: build the local eval data once (needs HF Hub access / the `datasets`
# library, no GPU) -- reused by every evaluate_models.py invocation below, so
# re-running eval never re-downloads/re-tokenizes sib200/belebele/mmlu.
# --mmlu-samples-per-subject 10: every one of mmlu's 57 subjects keeps up to 10
# questions (570 total, evenly spread) instead of the full 14,042-question test set --
# a fast proxy for the general-ability check rather than a precise number.
# --num-fewshot 5: matches mmlu's real "dev" split size (5 examples/subject, the
# standard MMLU few-shot setup) -- sib200's 5 exemplars come from its own train
# split, belebele has no separate split so its 5 are held out from the front of
# each language's test split instead (evaluated doc count per language drops by 5).
# Set to 0 for pure 0-shot.
# python scripts/build_downstream_eval_data.py --mmlu-samples-per-subject 10 --num-fewshot 5

# Step 2: score every model against it, one --task per invocation (a fresh process
# each time) rather than one process doing perplexity+sib200+belebele+mmlu back to
# back -- every real OOM hit while building this pipeline happened deep inside one
# task's own batch loop (see docs/09_evaluation.md's "Known OOM failure modes"), so
# a fresh process per task means a crash in one task can never inherit memory
# pressure from another, and losing one doesn't lose the others' already-written
# results (--output-path is merged, not overwritten, across these 4 calls).
# All 4 baselines in one call per task (not 4 separate calls per task): each call
# already evaluates "base" once and merges every baseline's result for that task
# into the same file, so splitting further by baseline would just add redundant
# "base" re-evaluations without adding any more memory isolation.

# perplexity sib200 mmlu
# for task in belebele; do
#   python scripts/evaluate_models.py --baselines best_weighted random clean_heuristic equal_average \
#       --task "$task" --device cuda --eval-batch-size 2 --downstream-batch-size 2
# done

# export HF_TOKEN=<your_hf_token>   # needs write scope -- the one in run_train_rater.sh may be read-only
# python3 scripts/push_to_hub.py --baselines best_weighted random clean_heuristic equal_average --push-rater-heads

# python3 scripts/evaluate_models.py --task belebele --eval-batch-size 1 --downstream-batch-size 1 \
#     --baselines best_weighted random clean_heuristic equal_average --device cuda


python data_preprocessing/encode.py --device cuda