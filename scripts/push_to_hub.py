"""
Push trained SEA-Rater artifacts to Hugging Face Hub.

Uploads run_final_cpt.py's LoRA adapters (models/qwen1.5b_cpt_<baseline>/, one
PEFT adapter directory per baseline) and/or train_rater.py's quality-rater
checkpoint (models/rater_heads.pt) to the caller's Hugging Face account, one
repo per artifact. Only the small LoRA adapter weights are uploaded, never the
Qwen2.5-1.5B base model itself.

Requires HF_TOKEN in the environment with *write* access. The HF_TOKEN already
exported in run_train_rater.sh was only ever used to download gated models, so
it may be read-only -- generate a write-scoped token at
https://huggingface.co/settings/tokens if `create_repo`/`upload_*` reject it.

Usage:
    export HF_TOKEN=hf_...  # needs write scope
    python3 scripts/push_to_hub.py --baselines best_weighted random clean_heuristic equal_average --push-rater-heads
    python3 scripts/push_to_hub.py --baselines best_weighted  # just one adapter
    python3 scripts/push_to_hub.py --push-rater-heads --public  # public instead of private
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.baselines import BASELINE_CHOICES, MODEL_DIR_BY_BASELINE

# Same value as sea_rater.cpt.FINAL_MODEL_NAME, duplicated rather than imported
# so this upload-only script never needs torch installed (cpt.py imports it at
# module level for the training/eval loops, which this script never touches).
FINAL_MODEL_NAME = "Qwen/Qwen2.5-1.5B"

REPO_ROOT = Path(__file__).resolve().parent.parent
RATER_HEADS_PATH = REPO_ROOT / "models" / "rater_heads.pt"
RATER_REPORT_PATH = REPO_ROOT / "models" / "rater_test_report.json"

ADAPTER_MODEL_CARD = """---
base_model: {base_model}
library_name: peft
tags:
- sea-rater
- lora
- continued-pretraining
---

# SEA-Rater: {baseline} LoRA adapter

LoRA adapter continue-pretraining `{base_model}` on SEA-Rater's `{baseline}`
baseline data across 8 Southeast Asian languages (Vietnamese, Indonesian,
Thai, Khmer, Malay, Filipino, Burmese, Lao). Part of the SEA-Rater pilot
(Meta-rater / JQL style multilingual data-quality rating for LLM pretraining).

Load with:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```
"""

RATER_HEADS_MODEL_CARD = """---
tags:
- sea-rater
- quality-rater
---

# SEA-Rater: quality rater heads

5 independent MLP heads (`QualityRaterHeads`, `sea_rater/rater.py`) predicting
`educational_value` / `reasoning` / `professionalism` / `cleanliness` /
`cultural_nuance` for documents in 8 Southeast Asian languages, trained on
frozen `intfloat/multilingual-e5-large` embeddings. See `rater_test_report.json`
in this repo for the full evaluation report (macro Spearman, per-language and
per-dimension breakdown).

Not a standalone `transformers` model -- load the raw state dict with
`sea_rater.rater.QualityRaterHeads` from the
[SEA-Rater repo](https://github.com/cppmai1103/SEA-Rater).
"""


def push_adapter(api, username, baseline, private):
    adapter_dir = MODEL_DIR_BY_BASELINE[baseline]
    if not adapter_dir.exists():
        print(f"[{baseline}] no adapter at {adapter_dir}, skipping")
        return
    repo_id = f"{username}/sea-rater-qwen1.5b-cpt-{baseline.replace('_', '-')}"
    print(f"[{baseline}] pushing {adapter_dir} -> {repo_id} (private={private}) ...")
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(adapter_dir))
    card = ADAPTER_MODEL_CARD.format(base_model=FINAL_MODEL_NAME, baseline=baseline, repo_id=repo_id)
    api.upload_file(repo_id=repo_id, path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md")
    print(f"[{baseline}] done -> https://huggingface.co/{repo_id}")


def push_rater_heads(api, username, private):
    if not RATER_HEADS_PATH.exists():
        print(f"no rater heads checkpoint at {RATER_HEADS_PATH}, skipping")
        return
    repo_id = f"{username}/sea-rater-quality-heads"
    print(f"pushing {RATER_HEADS_PATH} -> {repo_id} (private={private}) ...")
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_file(repo_id=repo_id, path_or_fileobj=str(RATER_HEADS_PATH), path_in_repo="rater_heads.pt")
    if RATER_REPORT_PATH.exists():
        api.upload_file(
            repo_id=repo_id, path_or_fileobj=str(RATER_REPORT_PATH), path_in_repo="rater_test_report.json"
        )
    api.upload_file(
        repo_id=repo_id, path_or_fileobj=RATER_HEADS_MODEL_CARD.encode("utf-8"), path_in_repo="README.md"
    )
    print(f"done -> https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", nargs="+", choices=BASELINE_CHOICES, default=[])
    parser.add_argument("--push-rater-heads", action="store_true")
    parser.add_argument(
        "--username", default=None, help="HF namespace to push under; defaults to the token's own account"
    )
    parser.add_argument("--public", action="store_true", help="make repos public instead of private")
    args = parser.parse_args()

    if not args.baselines and not args.push_rater_heads:
        raise SystemExit("nothing to push -- pass --baselines and/or --push-rater-heads")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set in the environment -- export a write-scoped token first")

    api = HfApi(token=token)
    username = args.username or api.whoami()["name"]
    private = not args.public
    print(f"pushing as '{username}', private={private}")

    for baseline in args.baselines:
        push_adapter(api, username, baseline, private)
    if args.push_rater_heads:
        push_rater_heads(api, username, private)


if __name__ == "__main__":
    main()
