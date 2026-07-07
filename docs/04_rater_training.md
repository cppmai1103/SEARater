# Step 3 — Training the Human-Only Quality Rater

Code: [`sea_rater/encoder.py`](../sea_rater/encoder.py),
[`sea_rater/rater.py`](../sea_rater/rater.py),
[`scripts/embed_splits.py`](../scripts/embed_splits.py),
[`scripts/train_rater.py`](../scripts/train_rater.py),
[`scripts/plot_rater_report.py`](../scripts/plot_rater_report.py)
Pipeline reference: `pipeline.md` Section 4 ("Human-Only Quality Rater")

## Goal

Train something that predicts the 5 quality dimensions
(`educational_value`, `reasoning`, `professionalism`, `cleanliness`,
`cultural_nuance`) for any document in any of the 8 pilot languages, using
only the 900-per-language human labels from Step 2.

## Why two scripts instead of one

The encoder (`multilingual-e5-large`) is **frozen** — it never gets
gradient updates. That means its output for a given document never
changes, so there's no reason to re-run it every training epoch. The
pipeline splits into:

1. `embed_splits.py` — run the frozen encoder **once** over
   train/dev/test, cache the resulting vectors to disk.
2. `train_rater.py` — train only the small MLP heads on those cached
   vectors, which is fast enough to iterate on (no GPU-heavy forward pass
   through a 560M-parameter encoder needed per epoch).

This also sets up the encoding code (`sea_rater/encoder.py`) to be reused
as-is in Section 5, when the same frozen backbone scores the full
500k–1M document candidate corpus.

## `sea_rater/encoder.py`

- `load_encoder(model_name, device)`: loads `intfloat/multilingual-e5-large`
  via `AutoTokenizer`/`AutoModel`, moves it to `device`, sets `.eval()`, and
  freezes every parameter (`requires_grad = False`).
- `_mean_pool(last_hidden_state, attention_mask)`: E5's recommended pooling
  — average the token embeddings, but only over real tokens (mask out
  padding) rather than a naive mean over the whole padded sequence.
- `embed_texts(texts, tokenizer, model, ...)`: the actual embedding loop.
  For each batch:
  1. Prepend `"passage: "` to every document — E5 models were trained with
     an instruction prefix that distinguishes queries from documents, and
     we're only ever embedding documents here.
  2. Tokenize with padding/truncation at `max_length=512` (E5-large's
     position-embedding limit).
  3. Forward pass under `torch.no_grad()`.
  4. Mean-pool, then L2-normalize (E5 embeddings are trained for
     cosine-similarity use).
  Returns one `(N, 1024)` tensor for all `N` input documents.

## `scripts/embed_splits.py`

For each of `human_train.jsonl` / `human_dev.jsonl` / `human_test.jsonl`:
1. Load the JSONL rows from Step 2.
2. Call `embed_texts` on all their `text` fields.
3. Bundle `{embeddings, labels (one tensor per dimension), languages,
   doc_ids, model_name}` and save it with `torch.save` to
   `data/splits/embeddings/{split}.pt`.

This is the only step in the whole rater-training pipeline that needs the
big encoder loaded — everything downstream reads the small cached `.pt`
files instead.

## `sea_rater/rater.py`

- `QualityRaterHeads`: a `nn.ModuleDict` of 5 independent heads, one per
  dimension. Each head is a small `Linear(1024→256) → ReLU → Linear(256→1)`
  MLP. They are kept as **separate** heads (not one shared output layer)
  specifically so each dimension's score stays interpretable and can later
  be weighted independently in the Section 8 weighted-combination search —
  training them together doesn't merge them into one rater.
- `huber_multi_loss(predictions, targets, delta)`: computes a Huber loss
  per dimension, then averages the 5 losses into one scalar. Huber (rather
  than plain MSE) is used because it's more robust to the occasional noisy
  or disputed human label — it behaves like MSE near zero error but like
  MAE for large errors, so a handful of mislabeled documents don't dominate
  the gradient.

## `scripts/train_rater.py`

### Data plumbing

- `make_loader(bundle, batch_size, shuffle)`: wraps the cached embeddings
  and 5 label tensors in a `TensorDataset`/`DataLoader`.
- `unpack_batch`: splits a loader batch back into `(embeddings, targets
  dict)`.
- `predict_all` / `evaluate`: run the current model over an entire split
  and compute the full metrics report the pipeline doc asks for:
  - Spearman correlation per `(language, dimension)` pair
    (`scipy.stats.spearmanr` between true labels and predicted scores).
  - `macro_average_spearman`: mean Spearman across all
    `language × dimension` cells — the primary metric.
  - `worst_language_spearman`: the lowest per-language average Spearman —
    a check that the rater isn't only working well on the easiest language.
  - `mae_by_dimension`: mean absolute error per dimension, averaged across
    languages.
  - `score_distribution_by_language`: a histogram (bincount 0–5) per
    language/dimension, for spotting things like a head collapsing to
    always predict the same score.

### Training loop (`train(args)`)

1. Load the 3 cached embedding bundles.
2. Build `QualityRaterHeads` sized to the embedding dimension (1024) and an
   `AdamW` optimizer — since the heads are the only `nn.Module` with
   trainable parameters, the encoder is never touched.
3. Each epoch: iterate training batches, compute `huber_multi_loss`,
   backprop, step. Then evaluate on `dev` and check `macro_average_spearman`.
4. Keep an in-memory copy of the best-so-far head weights
   (`best_state`); stop early if `patience` epochs pass with no
   improvement (default 5).
5. Reload the best weights, save them to `models/rater_heads.pt`, run the
   same `evaluate()` on the held-out `test` split, and write the full
   metrics to `models/rater_test_report.json`.

### CLI

```bash
python3 scripts/embed_splits.py --device cuda
python3 scripts/train_rater.py --epochs 50 --lr 1e-3 --device cuda
```

Defaults: hidden dim 256, batch size 64, Huber `delta=1.0`, patience 5,
seed 42. All overridable via flags.

## What an actual run produced

Two real cluster runs now exist. The first (`--epochs 1`, RTX A6000, only
4 of the pilot's languages, before the 8-language expansion) was a smoke
test and is superseded. The current `models/rater_heads.pt` /
`models/rater_test_report.json` come from a full run (`sbatch
run_train_rater.sh`, `torch 2.6.0+cu124`) with the actual `--epochs 50`
default, early stopping never triggered before hitting 50 (`--patience 5`
never saw 5 consecutive non-improving epochs), and all 8 languages:

```text
macro_average_spearman:   0.788
worst_language_spearman:  0.740   (Burmese)

Per-language Spearman:
              edu    reasoning  prof   clean  cultural   avg
  Indonesian  0.808  0.859      0.704  0.719  0.817      0.781
  Khmer       0.851  0.835      0.806  0.765  0.910      0.833
  Lao         0.818  0.886      0.732  0.535  0.793      0.753
  Malay       0.792  0.799      0.821  0.624  0.730      0.753
  Burmese     0.720  0.737      0.755  0.684  0.804      0.740
  Thai        0.839  0.833      0.817  0.837  0.826      0.831
  Filipino    0.794  0.769      0.789  0.770  0.840      0.792
  Vietnamese  0.852  0.865      0.826  0.739  0.819      0.820

MAE by dimension: edu 0.51, reasoning 0.41, professionalism 0.34,
                  cleanliness 0.80, cultural_nuance 0.62
```

Takeaways from this full run, across all 8 languages:
- 0.788 macro Spearman comfortably clears JQL's ~0.75 benchmark mentioned
  in the project proposal, despite covering 4 more (and generally
  lower-resource) languages than that comparison point.
- `reasoning` and `educational_value` are again the strongest, most
  consistent dimensions — same pattern as the earlier smoke test, and
  matches Meta-rater's finding that Educational Value and Reasoning are
  its two highest-weighted raters.
- `cleanliness` is still the weakest dimension by MAE (0.80, vs 0.34-0.62
  for the others) and the most language-inconsistent by Spearman
  (0.535-0.837) — consistent with cleanliness being described in the
  SEA-Rater proposal as harder to judge cross-lingually and
  lowest-priority among the kept dimensions. Notably it's no longer
  uniformly weak: Thai's cleanliness Spearman (0.837) is now its
  *strongest* dimension, while Lao's (0.535) is its weakest — the
  difficulty seems language-specific rather than a property of the
  dimension itself.
- Burmese is the new worst-performing language (0.740 average) but still
  well above the old smoke test's worst case (Thai at 0.517) — no
  language is dragging the macro average down badly.

## `scripts/plot_rater_report.py`

A companion visualization script, not wired into `run_train_rater.sh` —
run manually against `train_rater.py`'s JSON report:

```bash
python3 scripts/plot_rater_report.py
# or against a specific report/output location:
python3 scripts/plot_rater_report.py --report models/rater_test_report.json --output-dir models/figures
```

Produces 3 PNGs in `models/figures/`:
- `spearman_heatmap.png`: language x dimension grid, colored by Spearman
  correlation, titled with the macro-average and worst-language numbers.
- `mae_by_dimension.png`: bar chart of `mae_by_dimension`.
- `score_distributions.png`: one grouped bar chart per dimension, comparing
  each language's 0–5 score histogram.

Pure matplotlib/numpy over the already-written JSON report — no GPU, no
torch, safe to run on the dev server once `rater_test_report.json` exists.

## Re-running

```bash
sbatch run_train_rater.sh
```

Outputs land in `models/rater_heads.pt` (checkpoint) and
`models/rater_test_report.json` (full metrics), both gitignored.
