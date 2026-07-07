# Step 5 — Building and Scoring the Candidate Corpus

Code: [`scripts/build_candidate_corpus.py`](../scripts/build_candidate_corpus.py),
[`scripts/score_candidate_corpus.py`](../scripts/score_candidate_corpus.py)
Pipeline reference: `pipeline.md` Section 5 ("Score Candidate Corpus")

## Goal

Two scripts, run in sequence:

1. **`build_candidate_corpus.py`** — assemble the large, unlabeled pool of
   documents: 100K documents per language, 800K total across the 8 pilot
   languages, 80% from FineWeb2's heuristically-clean split and 20% from
   its `_removed` split.
2. **`score_candidate_corpus.py`** — run the trained rater (Step 4) over
   that pool and predict all 5 quality dimensions for every document,
   producing `scored_corpus.jsonl`, the input the Section 8 weighted-combination
   search will rank and select from.

Deliberately minimal: pipeline.md leaves length/lang bucketing and batch
assignment unimplemented for this pilot (those exist in
`data/human_annotation/pipeline_revise.ipynb`'s pool-builder, used for the
earlier *human annotation* sampling, but this pilot has no per-batch
annotation round to stratify for). The one prefilter this script *does*
keep from that notebook is the bad-word hard-reject (see step 3 below) —
Basic Prefiltering's language-score/repetition-score thresholds are
deliberately not applied here (see `pipeline.md`'s "Basic Prefiltering"
section); those only gate the `clean_heuristic` baseline later, not the
shared candidate corpus.

## Part A — `build_candidate_corpus.py`: step-by-step through the script

### 1. Configuration constants

```python
HF_DATASET = "HuggingFaceFW/fineweb-2"
# from sea_rater/languages.py, shared by every script that loops per-language
LANGUAGE_HF_CONFIGS = {"vi": "vie_Latn", "id": "ind_Latn", "th": "tha_Thai", "km": "khm_Khmr",
                       "ms": "zsm_Latn", "tl": "fil_Latn", "my": "mya_Mymr", "lo": "lao_Laoo"}
```

Same 8 pilot languages and FineWeb2 config-name mapping used throughout the
pipeline. Each language's clean split lives at HF config `{name}`, and its
heuristically-filtered-out documents live at `{name}_removed` — this is the
same dataset structure `pipeline_revise.ipynb` streams from.

### 2. `stream_docs(hf_config, target_size, source_label)`

The core streaming primitive, one call per (language, source) pair:

1. Build an `hf://datasets/HuggingFaceFW/fineweb-2/data/{hf_config}/train`
   path and open it with `datatrove.pipeline.readers.ParquetReader` — this
   streams parquet shards directly from the Hugging Face Hub without
   downloading the whole dataset first.
2. Iterate documents, keeping `doc_id`, `text`, `char_len`,
   `language_score` (from FineWeb2's own metadata), and the given
   `source_label` (`"clean"` or `"removed"`).
3. Stop as soon as `target_size` documents have been collected.
4. If the stream runs dry before reaching `target_size` (rare — only
   possible for a low-resource language/split with fewer available docs),
   print a warning rather than failing.

No length filter is applied here — every streamed document is kept
regardless of size (an earlier version filtered to `[500, 4000]` chars,
matching the notebook; that was deliberately removed for this corpus so it
isn't biased against the length distribution the rater will actually see
downstream).

### 3. `build_language_corpus(lang, hf_config, pool_size, clean_ratio)`

For one language:

1. Split `pool_size` into `clean_size = round(pool_size * clean_ratio)` and
   `removed_size = pool_size - clean_size` (defaults: 100,000 total →
   80,000 clean / 20,000 removed, an 80/20 split).
2. Stream both sets via `stream_docs`.
3. Tag every row with the 2-letter `language` code, and compute Section 5's
   cheap prefilter features (see below) — stored as metadata only, they
   don't drop anything here.
4. Deduplicate by `doc_id` across the two streams (in practice these come
   from disjoint FineWeb2 splits, so duplicates aren't expected, but this
   guards against it rather than assuming).
5. **Bad-word hard-reject** (`contains_badword`): tokenizes each
   document's text on non-alphanumeric runs, lowercases, and drops the
   document if any token hits datatrove's built-in `banned_words.txt`
   (~400 English adult/profanity terms) — the exact same check
   `data/human_annotation/pipeline_revise.ipynb` uses for its own adult-content
   filter, reused here rather than reimplemented. This is the *only* hard
   filter applied to the shared candidate corpus (see pipeline.md's "Basic
   Prefiltering").
6. Print a one-line summary: total kept, how many from each source, how
   many duplicates dropped, how many bad-word docs dropped.

### 4. `main()`

CLI-driven orchestration:

```bash
python3 scripts/build_candidate_corpus.py --pool-size 100000
python3 scripts/build_candidate_corpus.py --languages vi --pool-size 1000  # quick test
```

Loops over `--languages` (all 8 by default), calls
`build_language_corpus` for each, and writes
`data/candidate_corpus/{lang}.jsonl` — one JSON object per line:

```json
{
  "doc_id": "<urn:uuid:...>",
  "text": "...",
  "char_len": 1234,
  "language_score": 0.98,
  "source": "clean",
  "language": "vi",
  "repetition_score": 0.04,
  "target_script_ratio": 0.99,
  "stopword_ratio": 0.31,
  "symbol_ratio": 0.02,
  "numeric_ratio": 0.01,
  "is_latin": 1
}
```

### 3.5 Cheap prefilter features (`sea_rater/heuristics.py`)

Section 5's "Cheap prefilter features" list needs no GPU and no rater, so
`build_language_corpus` computes them here — the cheapest point in the
pipeline to compute them once, rather than recomputing on every scoring or
selection run:

- `repetition_score`: max of duplicate-line ratio and duplicate
  word-trigram ratio — catches both repeated boilerplate lines and
  repetitive prose within one block of text.
- `target_script_ratio`: fraction of alphabetic characters inside the
  language's expected Unicode script range (Latin for vi/id/ms/tl, Thai
  block for th, Khmer block for km, Myanmar block for my, Lao block for
  lo) — flags script-mismatched or garbled/mis-encoded documents. Stands
  in for pipeline.md's `script_integrity`.
- `stopword_ratio`: function-word density against a small hand-picked
  per-language stopword list. Vietnamese/Indonesian/Malay/Filipino are
  whitespace-segmented so this counts real word tokens; Thai/Khmer/
  Burmese/Lao aren't, so it falls back to counting stopword substring
  occurrences per 20 characters as a rough proxy — not a real tokenizer,
  deliberately approximate.
- `symbol_ratio` / `numeric_ratio`: fraction of characters that are
  punctuation/symbols or digits, respectively.
- `is_latin`: 1 if at least half the document's alphabetic characters are
  Latin script, else 0 — a cheap sanity flag (e.g. catches Thai/Khmer
  documents that are actually mostly untranslated Latin boilerplate).

`language_score` (FineWeb2's own langid confidence, already streamed in
`stream_docs`) stands in for pipeline.md's `langid_conf`.

These fields are what `sea_rater/selection.py`'s `select_clean_heuristic`
(Section 7 Baseline 2, see `docs/08_final_cpt.md`) filters on.

### What a real run looks like

Verified first with a small live pull against the actual FineWeb2 dataset
(`--languages vi --pool-size 20`): got exactly 16 clean / 4 removed (an
exact 80/20 split via `round()`), 0 duplicates, correct schema.

A full-scale cluster job has since completed this part for real, back when
the pilot covered 4 languages and used the current `--pool-size 100000`
default: all 4 landed at exactly 100,000 lines each in
`data/candidate_corpus/{lang}.jsonl` (400,000 documents total, confirmed
by actually counting lines on disk), confirming the streaming, ratio, and
dedup logic hold up at scale, not just in the small test. Now that the
pilot covers 8 languages, the same run needs to be repeated for the 4 new
ones (`ms`, `tl`, `my`, `lo`) to bring the candidate corpus up to the full
800K-document target.

This part of the job is I/O-bound (streaming from the Hub, no GPU needed)
rather than compute-bound, so unlike `train_rater.py` or
`score_candidate_corpus.py` it doesn't strictly need the cluster's GPU
partition; `run_train_rater.sh` bundles it in anyway for convenience (one
job that builds the corpus, then immediately scores it and continues on to
later steps on the same allocation) rather than necessity.

## Part B — `score_candidate_corpus.py`: step-by-step through the script

Takes the 100K-document-per-language pool from Part A and predicts all 5
quality dimensions for every document, using the exact same frozen encoder
and trained heads as Step 4 — no retraining, no new model.

### 1. `load_rater(checkpoint_path, device)`

Loads `models/rater_heads.pt` (written by `train_rater.py`): reconstructs a
`QualityRaterHeads` with the checkpoint's saved `input_dim`/`hidden_dim`,
loads its `state_dict`, moves it to `device`, and sets `.eval()`.

### 2. `load_corpus(lang)`

Reads `data/candidate_corpus/{lang}.jsonl` (Part A's output) into a list of
dicts — straightforward JSONL loading, no filtering.

### 3. `score_embeddings(model, embeddings, device, batch_size)`

Runs the rater heads over already-computed embeddings in batches (default
256), under `torch.no_grad()`, and concatenates each dimension's
predictions back into one flat list per dimension — mirrors
`train_rater.py`'s `predict_all`, but without needing ground-truth labels
alongside, since this corpus is unlabeled by design.

### 4. `main()`

For each language:

1. Load its `.jsonl` corpus file (Part A's output).
2. Embed every document's `text` with `embed_texts` (encoder pass).
3. Score those embeddings with the loaded rater heads (heads pass).
4. Write `data/candidate_corpus/scored/{lang}.jsonl`.

#### How one document actually gets scored: encoder pass, then heads pass

The scoring happens in two clearly separate passes over the *whole*
language's corpus — not interleaved document-by-document — because the
encoder is frozen and heavy while the heads are tiny and cheap. Tracing
what happens to a single document `text`:

**Encoder pass** (`embed_texts` in `sea_rater/encoder.py`, called once per
language on all 100K texts in batches of `--embed-batch-size`, default 32):

```python
prefixed  = "passage: " + text                     # E5's document prefix
tokens    = tokenizer(prefixed, truncation=True,     # cut at 512 tokens
                       max_length=512, padding=True)
outputs   = encoder(**tokens)                        # frozen multilingual-e5-large forward pass
pooled    = mean_pool(outputs.last_hidden_state,     # average token vectors,
                       tokens["attention_mask"])      #   ignoring padding
embedding = normalize(pooled, p=2, dim=-1)            # unit-length vector, shape (1024,)
```

This step is identical to what happens to the human-labeled documents in
`embed_splits.py` (Step 4) — same model, same prefix, same pooling — which
is the whole point: the heads were trained on embeddings produced this
exact way, so they see the same kind of input at inference time.

**Heads pass** (`score_embeddings`, calling `QualityRaterHeads.forward` in
`sea_rater/rater.py`, on batches of already-computed embeddings, default
256 per batch — much larger than the encoder's batch size, since this part
is just 5 small matrix multiplies, not a transformer forward pass):

```python
scores = {}
for dimension, head in model.heads.items():   # 5 independent heads
    scores[dimension] = head(embedding)        # Linear(1024→256) → ReLU → Linear(256→1)
# scores == {"educational_value": 2.87, "reasoning": 1.94, "professionalism": 2.10,
#            "cleanliness": 3.42, "cultural_nuance": 2.55}
```

Each of the 5 heads independently reads the *same* 1024-dim embedding and
outputs its own single number — there's no interaction between dimensions
at scoring time, matching how they were trained (Step 4: 5 separate losses
averaged only for the optimizer, never a shared output layer). So
`educational_value`'s prediction has no way to be influenced by, say, the
`cleanliness` head's weights.

Write `data/candidate_corpus/scored/{lang}.jsonl`, one line per document:

```json
{
  "doc_id": "<urn:uuid:...>",
  "language": "vi",
  "source": "clean",
  "educational_value": 2.87,
  "reasoning": 1.94,
  "professionalism": 2.10,
  "cleanliness": 3.42,
  "cultural_nuance": 2.55
}
```

Deliberately excludes the raw `text` field — with up to 100K rows per
language, keeping only `doc_id` plus the 5 scores keeps this file small;
anything needing the text can join back to `data/candidate_corpus/{lang}.jsonl`
by `doc_id`.

### CLI

```bash
python3 scripts/score_candidate_corpus.py --device cuda
python3 scripts/score_candidate_corpus.py --languages vi --device cuda  # one language
```

Overridable: `--rater-checkpoint`, `--model-name`, `--max-length`,
`--embed-batch-size`, `--score-batch-size`, `--output-dir`.

### Status

As of the last check, Part A (the corpus build) had completed for the
original 4 languages, and Part B (scoring) had produced output for those
same 4. Both parts still need to run for the 4 new languages (`ms`, `tl`,
`my`, `lo`) added when the pilot expanded to 8 languages. Worth checking
the job's status/logs before assuming a scored corpus is ready to use for
all 8.

## Re-running

```bash
# Part A only:
python3 scripts/build_candidate_corpus.py --pool-size 100000

# Part B only (needs an existing corpus + trained rater checkpoint):
python3 scripts/score_candidate_corpus.py --device cuda

# both, plus every later step, as one cluster job (uncomment the relevant lines):
sbatch run_train_rater.sh
```

Output lands in `data/candidate_corpus/{lang}.jsonl` and
`data/candidate_corpus/scored/{lang}.jsonl` (both gitignored, under `/data`).
