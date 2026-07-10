"""
pipeline.md Section 13.2 - build and cache the SEA downstream-eval datasets
as local JSONL files, decoupled from model evaluation (evaluate_models.py).

Splitting data prep out from scoring means:
- evaluate_models.py no longer touches the Hugging Face Hub or the
  `datasets` library at eval time -- it just reads local JSONL, so
  re-running eval (e.g. one baseline at a time, in a fresh process each
  time, for GPU-memory isolation -- see docs/09_evaluation.md) never
  re-downloads or re-tokenizes the same data.
- The exact evaluated set (which docs, which k-shot exemplars) is pinned to
  disk and inspectable/diffable, rather than implicitly whatever the live
  HF dataset happens to return at the moment each job runs.

Produces the same {"context", "choices", "choice_texts", "gold"} doc shape
evaluate_models.py's evaluate_multiple_choice expects, already MCQ-formatted
(format_mcq_context) with any --num-fewshot prefix baked in -- evaluate_models.py
does zero data preparation, only scoring.

Output layout:
    data/downstream_eval_data/sib200/{lang}.jsonl     -- one file per language
    data/downstream_eval_data/belebele/{lang}.jsonl   -- one file per language
    data/downstream_eval_data/mmlu.jsonl              -- one combined file,
                                                          every row tagged
                                                          with "subject"

Usage:
    python3 scripts/build_downstream_eval_data.py
    python3 scripts/build_downstream_eval_data.py --num-fewshot 5 --mmlu-samples-per-subject 10
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea_rater.languages import LANGUAGE_HF_CONFIGS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "downstream_eval_data"

# sib200 and belebele are real FLORES-200 benchmarks, and FLORES-200 has no
# separate Filipino code -- FineWeb2's "fil_Latn" (LANGUAGE_HF_CONFIGS) isn't
# a valid config for either dataset, confirmed via both datasets' real
# config lists ("BuilderConfig 'fil_Latn' not found" on a real run, and
# neither dataset's config list contains it); both use "tgl_Latn" (Tagalog)
# instead. LANGUAGE_HF_CONFIGS itself is left untouched since FineWeb2's
# "fil_Latn" is correct for corpus building -- only downstream eval needs
# the substitution.
DOWNSTREAM_HF_CONFIGS = dict(LANGUAGE_HF_CONFIGS)
DOWNSTREAM_HF_CONFIGS["tl"] = "tgl_Latn"

SIB200_CATEGORIES = [
    "science/technology", "travel", "politics", "sports", "health",
    "entertainment", "geography",
]

# Up to 8 lettered options -- more than either task needs (sib200: 7,
# belebele: 4) -- used to build real multiple-choice prompts ("A. ...\nB.
# ...\nAnswer:") instead of scoring each choice's raw text as a bare
# continuation. Plain continuation-scoring has a length bias: a short wrong
# answer's total log-likelihood can beat a longer correct one purely
# because it has fewer tokens to "clear" (see docs/09_evaluation.md for a
# worked example). Letter-choice format makes every candidate a single
# token, removing that bias -- at the cost of requiring the model to
# actually follow the lettered-option convention 0-shot.
MCQ_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def format_mcq_context(passage_and_question, choice_texts):
    """Build a "<passage_and_question>\nA. <choice>\nB. <choice>\n...\n
    Select the one correct letter for the answer.\nAnswer:" prompt listing
    every choice inline under a letter, with an explicit instruction to
    answer with a single letter -- a 0-shot base model has no other signal
    that "Answer:" means "reply with exactly one of the letters above"
    rather than, say, continuing the passage. Returns just the context
    string; callers score " A", " B", ... as the continuations
    (MCQ_LETTERS[: len(choice_texts)]), not the raw choice text."""
    lines = [passage_and_question]
    for letter, choice in zip(MCQ_LETTERS, choice_texts):
        lines.append(f"{letter}. {choice}")
    lines.append("Select the one correct letter for the answer.")
    lines.append("Answer:")
    return "\n".join(lines)


def build_fewshot_prefix(exemplar_docs):
    """Turn a list of {"context", "choices", "gold"} docs (same shape
    format_mcq_context/load_*_docs produce) into one k-shot prefix string:
    each exemplar's full MCQ prompt with the correct letter filled in right
    after "Answer:", separated by a blank line -- standard in-context
    few-shot formatting. Prepend the result to an eval doc's own context.
    Empty string if `exemplar_docs` is empty (0-shot, the default)."""
    if not exemplar_docs:
        return ""
    blocks = [f"{doc['context']} {doc['choices'][doc['gold']]}" for doc in exemplar_docs]
    return "\n\n".join(blocks) + "\n\n"


def load_sib200_docs(lang_hf_config, num_fewshot=0):
    from datasets import load_dataset

    def build_doc(row):
        return {
            "context": format_mcq_context(
                f"{row['text']}\nQuestion: What is the topic of this sentence?", SIB200_CATEGORIES
            ),
            "choices": MCQ_LETTERS[: len(SIB200_CATEGORIES)],
            "choice_texts": SIB200_CATEGORIES,
            "gold": SIB200_CATEGORIES.index(row["category"]),
        }

    fewshot_prefix = ""
    if num_fewshot > 0:
        train_rows = load_dataset("Davlan/sib200", lang_hf_config, split="train")
        exemplars = [build_doc(r) for r in train_rows.select(range(min(num_fewshot, len(train_rows))))]
        fewshot_prefix = build_fewshot_prefix(exemplars)

    rows = load_dataset("Davlan/sib200", lang_hf_config, split="test")
    docs = []
    for row in rows:
        doc = build_doc(row)
        doc["context"] = fewshot_prefix + doc["context"]
        docs.append(doc)
    return docs


def load_belebele_docs(lang_hf_config, num_fewshot=0):
    """belebele has only a "test" split (no train/dev) -- so, unlike
    sib200/mmlu, few-shot exemplars can't come from a separate held-out
    split without inventing one. Instead the first `num_fewshot` rows of
    the language's own test set are reserved as exemplars and excluded from
    evaluation (evaluated doc count per language shrinks by `num_fewshot`
    when > 0), so no exemplar is ever also scored as an eval doc."""
    from datasets import load_dataset

    def build_doc(row):
        choice_texts = [row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]]
        return {
            "context": format_mcq_context(f"{row['flores_passage']}\nQuestion: {row['question']}", choice_texts),
            "choices": MCQ_LETTERS[: len(choice_texts)],
            "choice_texts": choice_texts,
            "gold": int(row["correct_answer_num"]) - 1,
        }

    rows = load_dataset("facebook/belebele", lang_hf_config, split="test")
    num_fewshot = min(num_fewshot, len(rows) - 1) if num_fewshot > 0 else 0
    fewshot_prefix = build_fewshot_prefix([build_doc(r) for r in rows.select(range(num_fewshot))])
    eval_rows = rows.select(range(num_fewshot, len(rows)))

    docs = []
    for row in eval_rows:
        doc = build_doc(row)
        doc["context"] = fewshot_prefix + doc["context"]
        docs.append(doc)
    return docs


def load_mmlu_docs(limit=None, samples_per_subject=None, num_fewshot=0):
    """cais/mmlu, config "all" (every subject pooled), "test" split --
    English-only, used as a general-ability check (pipeline.md Section
    13.3: does continued pretraining on SEA data hurt general knowledge?),
    not part of the per-language SEA rollup sib200/belebele get. Schema
    confirmed via HF's datasets-server API: `question` (str), `subject`
    (str), `choices` (list[str], always 4), `answer` (int index into
    choices). Bucketed by `subject` (57 of them) rather than language.

    `num_fewshot`: exemplars come from the real "dev" split (cais/mmlu's
    dedicated few-shot split, 5 rows/subject by design -- the standard MMLU
    setup) rather than test, so no leakage; clamped to whatever's actually
    available for that subject if more is requested than exists.

    `samples_per_subject`: cap on questions *per subject*, applied after
    grouping by subject -- unlike `limit`, every one of the 57 subjects
    still gets up to this many questions instead of the first N rows in
    HF's natural (subject-grouped) row order silently skipping whichever
    subjects come later. Takes priority over `limit` if both are set.

    `limit`: flat cap on total questions (first N after HF's natural row
    order, which is grouped by subject) -- the full test split is 14,042
    questions x 4 choices = ~56K scored pairs per model, meaningfully more
    compute than sib200+belebele combined; pass e.g. --mmlu-limit 2000 for
    a cheaper (but subject-lopsided) proxy run, or prefer
    --mmlu-samples-per-subject for an even one."""
    from datasets import load_dataset

    def build_doc(row):
        return {
            "context": format_mcq_context(row["question"], row["choices"]),
            "choices": MCQ_LETTERS[: len(row["choices"])],
            "choice_texts": row["choices"],
            "gold": row["answer"],
        }

    fewshot_prefix_by_subject = {}
    if num_fewshot > 0:
        dev_by_subject = {}
        for row in load_dataset("cais/mmlu", "all", split="dev"):
            dev_by_subject.setdefault(row["subject"], []).append(row)
        for subject, subject_rows in dev_by_subject.items():
            exemplars = [build_doc(r) for r in subject_rows[:num_fewshot]]
            fewshot_prefix_by_subject[subject] = build_fewshot_prefix(exemplars)

    rows = load_dataset("cais/mmlu", "all", split="test")
    if samples_per_subject is not None:
        selected_indices = []
        count_by_subject = {}
        for i, row in enumerate(rows):
            seen = count_by_subject.get(row["subject"], 0)
            if seen < samples_per_subject:
                selected_indices.append(i)
                count_by_subject[row["subject"]] = seen + 1
        rows = rows.select(selected_indices)
    elif limit is not None:
        rows = rows.select(range(min(limit, len(rows))))

    docs_by_subject = {}
    for row in rows:
        doc = build_doc(row)
        doc["context"] = fewshot_prefix_by_subject.get(row["subject"], "") + doc["context"]
        docs_by_subject.setdefault(row["subject"], []).append(doc)
    return docs_by_subject


def available_language_configs(dataset_path, hf_configs):
    """Check every one of our languages' hf_config against the dataset's
    real, available config list *before* attempting to load any of them --
    lm-eval-style crashes only surface a single missing config at a time
    ("BuilderConfig 'fil_Latn' not found"), so a language that isn't in the
    benchmark at all (e.g. no separate Filipino code in FLORES-200) would
    otherwise take down the whole run instead of just being skipped.
    Returns {lang: hf_config} filtered down to the ones that actually
    exist; anything missing is reported once, up front, with the real
    available list so the cause is obvious."""
    from datasets import get_dataset_config_names

    available = set(get_dataset_config_names(dataset_path))
    valid, missing = {}, {}
    for lang, hf_config in hf_configs.items():
        if hf_config in available:
            valid[lang] = hf_config
        else:
            missing[lang] = hf_config
    if missing:
        tqdm.write(
            f"[{dataset_path}] no config for language(s) {missing} -- skipping them "
            f"for this dataset. Available configs: {sorted(available)}"
        )
    return valid


def save_docs(out_dir, bucket, docs):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bucket}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return out_path


def save_docs_combined(out_path, docs_by_bucket, bucket_label):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for bucket, docs in docs_by_bucket.items():
            for doc in docs:
                row = dict(doc)
                row[bucket_label] = bucket
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--num-fewshot", type=int, default=0,
        help="k-shot in-context examples prepended to every prompt (default 0 = 0-shot). sib200 sources "
             "exemplars from its train split, mmlu from its dev split (max 5/subject, the standard MMLU "
             "few-shot split); belebele has no separate split, so its exemplars are held out from its own "
             "test set instead (evaluated doc count per language shrinks by this amount).",
    )
    parser.add_argument(
        "--mmlu-limit", type=int, default=None,
        help="flat cap on total mmlu questions, first N in HF's subject-grouped row order (can skip later "
             "subjects entirely). Ignored if --mmlu-samples-per-subject is set. Default: full 14,042-question "
             "test set.",
    )
    parser.add_argument(
        "--mmlu-samples-per-subject", type=int, default=None,
        help="cap on mmlu questions PER subject (every one of the 57 subjects keeps up to this many, unlike "
             "--mmlu-limit) -- e.g. --mmlu-samples-per-subject 10 for a fast, evenly-sampled proxy run",
    )
    args = parser.parse_args()

    sib200_langs = available_language_configs("Davlan/sib200", DOWNSTREAM_HF_CONFIGS)
    for lang, hf_config in tqdm(sib200_langs.items(), desc="sib200", unit="lang"):
        docs = load_sib200_docs(hf_config, num_fewshot=args.num_fewshot)
        out_path = save_docs(args.output_dir / "sib200", lang, docs)
        tqdm.write(f"[sib200][{lang}] {len(docs)} docs -> {out_path}")

    belebele_langs = available_language_configs("facebook/belebele", DOWNSTREAM_HF_CONFIGS)
    for lang, hf_config in tqdm(belebele_langs.items(), desc="belebele", unit="lang"):
        docs = load_belebele_docs(hf_config, num_fewshot=args.num_fewshot)
        out_path = save_docs(args.output_dir / "belebele", lang, docs)
        tqdm.write(f"[belebele][{lang}] {len(docs)} docs -> {out_path}")

    print("loading mmlu ...")
    mmlu_docs_by_subject = load_mmlu_docs(
        limit=args.mmlu_limit, samples_per_subject=args.mmlu_samples_per_subject, num_fewshot=args.num_fewshot
    )
    mmlu_out_path = save_docs_combined(args.output_dir / "mmlu.jsonl", mmlu_docs_by_subject, "subject")
    total_mmlu = sum(len(docs) for docs in mmlu_docs_by_subject.values())
    print(f"[mmlu] {total_mmlu} docs across {len(mmlu_docs_by_subject)} subjects -> {mmlu_out_path}")

    print(f"\nWrote downstream eval data -> {args.output_dir}")


if __name__ == "__main__":
    main()
