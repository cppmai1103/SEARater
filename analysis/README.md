# Analysis

Code + generated results for the pilot's current run. Each script reads
straight from `data/` (no GPU/model needed — pure CSV/JSON) and writes its
output alongside itself:

| script | reads | writes | needs |
|---|---|---|---|
| `proxy_cpt_ranking.py` | `data/proxy_results.csv`, `data/weight_combinations.json`, `data/best_weight.json` | `proxy_cpt_ranking.md` | stdlib only |
| `final_eval_summary.py` | `data/evaluation_results.json` | `final_eval_summary.md` | stdlib only |
| `plot_downstream.py` | `data/evaluation_results.json` | `sib200_delta_{accuracy,token_normalized_prob_correct}.png`, `belebele_delta_*.png`, `mmlu_delta_*.png` | matplotlib/numpy |

Re-run any of these any time its source data changes (e.g. once belebele
finishes — see [`../docs/09_evaluation.md`](../docs/09_evaluation.md)):

```bash
python3 analysis/proxy_cpt_ranking.py
python3 analysis/final_eval_summary.py

# plot_downstream.py needs matplotlib/numpy, which aren't on the bare
# python3 on this dev server -- use the conda env that has them:
/Utilisateurs/pchau/.conda/envs/sea_rater_env/bin/python3 analysis/plot_downstream.py
```

## Plots

Every plot shows **delta vs. `base`** (baseline value − base value), not
raw scores — `base` is dropped from the chart entirely and shown only as
the zero line, since the actual question is how much each baseline moved
and in which direction, not their absolute scores (those are in the tables
above). Generated for both metrics (`accuracy` and
`token_normalized_prob_correct`):

- [`sib200_delta_accuracy.png`](sib200_delta_accuracy.png) /
  [`sib200_delta_token_normalized_prob_correct.png`](sib200_delta_token_normalized_prob_correct.png)
  — grouped bar, Δ by language. `equal_average` (green) is negative on
  *every* language — its -27% macro regression is a broad effect, not one
  bad language dragging the mean down. `best_weighted` (blue) is actually
  **negative on 4 of 8 languages** (id, th, tl, vi) and only positive on
  the other 4 — its near-zero macro improvement (+1.01%) is masking a
  genuinely mixed, not uniformly-small, effect. `random` (aqua) is
  positive almost everywhere and by the widest margins.
- [`mmlu_delta_accuracy.png`](mmlu_delta_accuracy.png) /
  [`mmlu_delta_token_normalized_prob_correct.png`](mmlu_delta_token_normalized_prob_correct.png)
  — mmlu is treated as one aggregate set here, not broken out per-subject
  (57 subjects isn't a useful plot axis for a general-ability check): one
  bar per baseline, height = macro delta vs base, same headline-number
  treatment as sib200/belebele. All four bars are negative and similarly
  sized on `accuracy` (-0.047 to -0.053) — forgetting is roughly uniform
  across weighting strategies. On `token_normalized_prob_correct`,
  `best_weighted` is the smallest-magnitude bar (least forgetting) even
  though it's unremarkable on `accuracy` — the continuous metric separates
  the baselines here where the binary one doesn't.
- `belebele_delta_*.png` isn't generated yet — belebele still has no data
  (see "Status" below).

## Status as of 2026-07-10

- **Proxy CPT search** (Section 9-10): all 26 weight combinations complete,
  8 languages — see [`proxy_cpt_ranking.md`](proxy_cpt_ranking.md). (Note:
  `docs/07_proxy_cpt.md` currently describes this data as stale/4-language
  with W17-26 missing — that's out of date; `data/proxy_results.csv`
  already has all 26 rows across all 8 languages, matching
  `data/best_weight.json`.)
- **Final eval** (Section 13): perplexity, sib200, mmlu complete for base +
  all 4 baselines; belebele still pending (OOM, see
  `docs/09_evaluation.md`'s "Known OOM failure modes" #5) — see
  [`final_eval_summary.md`](final_eval_summary.md).

## The headline finding

The proxy search's winner, **W11 (Cultural-heavy)**, is only **rank 4 of
26** by macro validation loss, sitting inside a **16-way tie** (every one
of those 16 combos lands within `select_best_weight.py`'s 1% tolerance of
the single lowest macro_loss, `W04`/Cleanliness-only at 2.1644 vs. W11's
2.1741). W11 won that tie only on the secondary worst-language-loss
tiebreak, not the primary ranking metric — with 16 candidates statistically
indistinguishable on macro loss, the proxy search's "signal" for which
weighting is best is weak at this scale (small 0.5B proxy model, ~16M
training tokens).

That weak signal shows up directly downstream: at the real 1.5B scale,
`best_weighted` (W11) is **not** the strongest of the four trained
baselines on either metric measured so far —

- **Perplexity**: `random` improves 26.75% vs. base; `best_weighted` only
  22.02% (the *worst* of the four).
- **sib200**: `random` improves 21.81%; `best_weighted` barely moves the
  needle at +1.01%.

This directly contradicts `pipeline.md` Section 14's Claim 3 ("the best
proxy-selected weighted data improves ... compared with random"). Given
the 16-way tie at the proxy stage, that's not entirely surprising — the
proxy search didn't have enough separation between candidates to reliably
pick a winner, and whatever separation it did find didn't transfer to the
1.5B/full-token-budget final run. Worth revisiting once belebele
completes, but two of three metrics already point the same direction.
