# Step 2 — Splitting the Human-Labeled Data

Code: [`scripts/split_human_annotations.py`](../scripts/split_human_annotations.py)
Pipeline reference: `pipeline.md` Section 3 ("Human Annotation Data")

## Goal

Turn the raw per-language annotation CSVs into three clean, balanced
JSONL files (`human_train.jsonl`, `human_dev.jsonl`, `human_test.jsonl`)
that the rater-training step consumes.

## Input data

Each of `data/human_annotation/{vie_Latn,ind_Latn,tha_Thai,khm_Khmr,zsm_Latn,fil_Latn,mya_Mymr,lao_Laoo}.csv`
has 940 rows with these columns:

```text
text, char_len, language_score, source, len_bucket, lang_bucket,
batch_id, id, educational_value, reasoning, professionalism,
cleanliness, cultural_nuances, justification
```

The 5 score columns are integers 0–5 (some dimensions don't use the full
range in the raw data, e.g. `professionalism` is only ever 1–4).

## What the split balances

Two things, exactly as requested:

1. **Language** — each of the 8 pilot languages is split independently
   with the same fixed target sizes, so language balance is exact by
   construction (no stratification needed for this one).
2. **Score distribution of all 5 dimensions** — the 0–5 histogram of
   `educational_value`, `reasoning`, `professionalism`, `cleanliness`, and
   `cultural_nuances` is each kept proportionally balanced across
   train/dev/test.

Balancing 5 ordinal variables at once rules out a single combined
stratification key (`source × len_bucket × ...`): joining 5 dimensions
each with up to 6 levels gives up to 6^5 ≈ 7,776 possible cells, almost all
empty with only 940 documents per language. Instead the script uses
**iterative stratification** (Sechidis et al., 2011), which balances each
dimension's marginal distribution directly, without needing every
combination of all 5 to be populated.

## Step-by-step through the script

### 1. Configuration constants

```python
LANGUAGE_FILES = {code: f"human_annotation/{hf_config}.csv"
                   for code, hf_config in LANGUAGE_HF_CONFIGS.items()}
# {"vi": "human_annotation/vie_Latn.csv", "id": "human_annotation/ind_Latn.csv",
#  "th": "human_annotation/tha_Thai.csv", "km": "human_annotation/khm_Khmr.csv",
#  "ms": "human_annotation/zsm_Latn.csv", "tl": "human_annotation/fil_Latn.csv",
#  "my": "human_annotation/mya_Mymr.csv", "lo": "human_annotation/lao_Laoo.csv"}
DIMENSIONS = [... 5 dimension column names ...]
SPLIT_SIZES = {"train": 720, "dev": 90, "test": 90, "drop": 40}
SEED = 42
```

Derived from the shared `LANGUAGE_HF_CONFIGS` (`sea_rater/languages.py`)
rather than hardcoded, since every script that loops per-language now
imports the same 8-language list from there. Note the filenames have no
leading `/` — `Path("data") / "/human_annotation/x.csv"` would silently
resolve to the absolute path `/human_annotation/x.csv` instead of
`data/human_annotation/x.csv` (a real bug an earlier version of this dict
had, caught while extending it to 8 languages).

`SPLIT_SIZES` sums to exactly 940 — the full row count per language file.
Folding the "keep 900 of 940" downsampling into the split itself (as a 4th
bucket, `drop`, which just gets discarded) means there's only one
balancing pass instead of two.

### 2. `row_labels(row)`

```python
def row_labels(row):
    return [(dim, row[dim]) for dim in DIMENSIONS]
```

Each document contributes exactly 5 labels — one `(dimension, score)` pair
per quality dimension. These are the labels iterative stratification
balances across splits.

### 3. `load_rows(csv_path, lang)`

Unchanged from before: reads a CSV with `encoding="utf-8-sig"` (strips a
BOM) and a raised `csv.field_size_limit`, since some `text` fields contain
embedded newlines long enough to exceed Python's default limit. Tags every
row with its 2-letter language code.

### 4. `iterative_stratified_split(rows, label_fn, split_sizes, rng)`

The core balancing algorithm, run once per language:

1. Collect, for every `(dimension, score)` label, the set of row indices
   that carry it.
2. For each label, compute a **desired remaining count per split**,
   proportional to that split's target size (e.g. a label held by 100 rows
   wants 80/10/10 of them in train/dev/test).
3. Repeatedly:
   - Pick the label with the **fewest remaining unassigned rows** — rare
     labels (like `cleanliness=0`, or `professionalism=4`) are settled
     first, while there's still enough flexibility left to place them
     correctly; common labels can absorb whatever's left over at the end.
   - For each unassigned row carrying that label (in random order), assign
     it to whichever open split currently wants that label most (highest
     remaining desired count; ties broken by which split has the most
     overall remaining capacity, then randomly).
   - Assigning a row decrements the desired count for **all 5** of its
     labels at once, not just the one being processed, since one document
     satisfies all 5 of its dimension labels simultaneously.
4. Continue until every row is assigned. Because every split's
   `remaining_target` hits exactly 0 when it's full, the final split sizes
   are exact (720/90/90/40), not approximate.

### 5. `to_record(row)`

Unchanged: builds the final JSON schema, strips the CSV's raw
`<urn:uuid:...>` id down to a clean `doc_id`, and renames
`cultural_nuances` → `cultural_nuance`.

### 6. `report(name, rows)`

Now prints the full 0–5 histogram for **all 5 dimensions** per split (not
just one), so an imbalance in any dimension is visible immediately.

### 7. `main()`

For each language: load its 940 rows, run
`iterative_stratified_split(..., SPLIT_SIZES, rng)`, keep the `train`/
`dev`/`test` buckets and discard `drop`. Accumulate across the 8
languages, shuffle each combined split once more, and write
`data/splits/human_{train,dev,test}.jsonl`.

## What an actual run produced

```text
human_train.jsonl: 5760 docs (720/language x 8 languages)
  educational_value: {0: 733, 1: 1978, 2: 1463, 3: 1149, 4: 407, 5: 30}
  reasoning:         {0: 968, 1: 2336, 2: 1852, 3: 572, 4: 32}
  professionalism:   {1: 1897, 2: 2879, 3: 866, 4: 117, 5: 1}
  cleanliness:       {0: 196, 1: 777, 2: 948, 3: 886, 4: 1003, 5: 1950}
  cultural_nuances:  {0: 281, 1: 843, 2: 804, 3: 1307, 4: 1482, 5: 1043}

human_dev.jsonl: 720 docs (90/language x 8 languages)
  educational_value: {0: 93, 1: 246, 2: 187, 3: 140, 4: 51, 5: 3}
  reasoning:         {0: 122, 1: 292, 2: 231, 3: 71, 4: 4}
  professionalism:   {1: 240, 2: 358, 3: 108, 4: 14}
  cleanliness:       {0: 26, 1: 99, 2: 119, 3: 111, 4: 125, 5: 240}
  cultural_nuances:  {0: 35, 1: 101, 2: 102, 3: 166, 4: 185, 5: 131}

human_test.jsonl: 720 docs (90/language x 8 languages)
  educational_value: {0: 89, 1: 249, 2: 181, 3: 145, 4: 52, 5: 4}
  reasoning:         {0: 117, 1: 299, 2: 229, 3: 71, 4: 4}
  professionalism:   {1: 237, 2: 358, 3: 110, 4: 15}
  cleanliness:       {0: 25, 1: 95, 2: 122, 3: 112, 4: 125, 5: 241}
  cultural_nuances:  {0: 29, 1: 105, 2: 100, 3: 165, 4: 192, 5: 129}
```

Every dimension now lands close to the exact 80/10/10 split ratio at every
score level — e.g. `cleanliness=5`: 1950/240/241; `professionalism=4`:
117/14/15; `reasoning=4`: 32/4/4 — and language stays exactly 720/90/90 per
language, as before.

## Re-running

```bash
python3 scripts/split_human_annotations.py
```

`SEED = 42` makes the assignment (including the random tie-breaks inside
`iterative_stratified_split`) fully reproducible.
