# Step 2 — Splitting the Human-Labeled Data

Code: [`scripts/split_human_annotations.py`](../scripts/split_human_annotations.py)
Pipeline reference: `pipeline.md` Section 3 ("Human Annotation Data")

## Goal

Turn the raw per-language annotation CSVs into three clean, balanced
JSONL files (`human_train.jsonl`, `human_dev.jsonl`, `human_test.jsonl`)
that the rater-training step consumes.

## Input data

Each of `data/{vie_Latn,ind_Latn,tha_Thai,khm_Khmr}.csv` has 940 rows with
these columns:

```text
text, char_len, language_score, source, len_bucket, lang_bucket,
batch_id, id, educational_value, reasoning, professionalism,
cleanliness, cultural_nuances, justification
```

The 5 score columns are integers 0–5 (some dimensions don't use the full
range in the raw data, e.g. `professionalism` is only ever 1–4).

## What the split balances

Two things, exactly as requested:

1. **Language** — each of the 4 pilot languages is split independently
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
LANGUAGE_FILES = {"vi": "vie_Latn.csv", "id": "ind_Latn.csv",
                   "th": "tha_Thai.csv", "km": "khm_Khmr.csv"}
DIMENSIONS = [... 5 dimension column names ...]
SPLIT_SIZES = {"train": 720, "dev": 90, "test": 90, "drop": 40}
SEED = 42
```

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
`dev`/`test` buckets and discard `drop`. Accumulate across the 4
languages, shuffle each combined split once more, and write
`data/splits/human_{train,dev,test}.jsonl`.

## What an actual run produced

```text
human_train.jsonl: 2880 docs (720/language)
  educational_value: {0: 442, 1: 1021, 2: 621, 3: 545, 4: 234, 5: 17}
  reasoning:         {0: 595, 1: 1157, 2: 824, 3: 287, 4: 17}
  professionalism:   {1: 970, 2: 1404, 3: 429, 4: 76, 5: 1}
  cleanliness:       {0: 126, 1: 432, 2: 483, 3: 457, 4: 495, 5: 887}
  cultural_nuances:  {0: 206, 1: 356, 2: 473, 3: 716, 4: 683, 5: 446}

human_dev.jsonl: 360 docs (90/language)
  educational_value: {0: 56, 1: 127, 2: 76, 3: 69, 4: 30, 5: 2}
  reasoning:         {0: 76, 1: 142, 2: 104, 3: 35, 4: 3}
  professionalism:   {1: 123, 2: 173, 3: 54, 4: 10}
  cleanliness:       {0: 16, 1: 56, 2: 61, 3: 57, 4: 61, 5: 109}
  cultural_nuances:  {0: 26, 1: 44, 2: 60, 3: 89, 4: 85, 5: 56}

human_test.jsonl: 360 docs (90/language)
  educational_value: {0: 52, 1: 129, 2: 79, 3: 68, 4: 30, 5: 2}
  reasoning:         {0: 72, 1: 147, 2: 103, 3: 36, 4: 2}
  professionalism:   {1: 122, 2: 174, 3: 54, 4: 10}
  cleanliness:       {0: 17, 1: 51, 2: 64, 3: 58, 4: 61, 5: 109}
  cultural_nuances:  {0: 20, 1: 45, 2: 58, 3: 91, 4: 91, 5: 55}
```

Every dimension now lands close to the exact 80/10/10 split ratio at every
score level — e.g. `cleanliness=5`: 887/109/109; `professionalism=4`:
76/10/10; `reasoning=4`: 17/3/2 — and language stays exactly 720/90/90 per
language, as before.

## Re-running

```bash
python3 scripts/split_human_annotations.py
```

`SEED = 42` makes the assignment (including the random tie-breaks inside
`iterative_stratified_split`) fully reproducible.
