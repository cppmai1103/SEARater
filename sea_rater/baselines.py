"""
pipeline.md Section 7/11-13 baseline naming, shared by build_final_cpt_dataset.py,
run_final_cpt.py, and evaluate_models.py so each baseline's final-CPT dataset
directory, trained LoRA adapter directory, and evaluation key all agree --
adding a new baseline means adding one entry here, not editing three scripts.
Dependency-free (no torch) so it can be imported from any of them.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_CORPUS_DIR = REPO_ROOT / "data" / "candidate_corpus"
MODELS_DIR = REPO_ROOT / "models"

BASELINE_CHOICES = ["random", "clean_heuristic", "equal_average", "best_weighted"]

# "best_weighted" keeps the original directory names (no baseline suffix)
# since that's the pilot's main method and already has real data/runs on disk
# under those paths; the other baselines get their own sibling directories.
DATASET_DIR_BY_BASELINE = {
    "best_weighted": CANDIDATE_CORPUS_DIR / "final_cpt_dataset",
    "random": CANDIDATE_CORPUS_DIR / "final_cpt_dataset_random",
    "clean_heuristic": CANDIDATE_CORPUS_DIR / "final_cpt_dataset_cleanheuristic",
    "equal_average": CANDIDATE_CORPUS_DIR / "final_cpt_dataset_equalaverage",
}

MODEL_DIR_BY_BASELINE = {
    "best_weighted": MODELS_DIR / "qwen1.5b_cpt_bestweighted",
    "random": MODELS_DIR / "qwen1.5b_cpt_random",
    "clean_heuristic": MODELS_DIR / "qwen1.5b_cpt_cleanheuristic",
    "equal_average": MODELS_DIR / "qwen1.5b_cpt_equalaverage",
}
