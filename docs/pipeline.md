# Human-Only CoRAL-CPT Pilot Pipeline
## 1. Pilot Goal

This pilot tests whether a **human-only multilingual quality rater** can select better Southeast Asian (SEA) continued-pretraining data than simple baselines.

The pilot uses only:

- Human annotations already available: **940 documents per language** from folder /data/*.csv
- A frozen rater backbone: **multilingual-e5-large**
- A small proxy CPT model: **Qwen2.5-0.5B Base**
- A final CPT model: **Qwen2.5-1.5B Base**
- **16 candidate score-weight combinations**
- **No LightGBM / meta-regressor for now**
- Continued pretraining only, no train-from-scratch final LM

Main research question:

> Can a human-only SEA-aware quality rater select better continued-pretraining data than random, clean heuristic, educational-value-only, or equal-average filtering?

---

## 2. Pilot Scope

### Languages

Use **4 languages** for the first clean pilot:

1. Vietnamese
2. Indonesian
3. Thai
4. Khmer

Rationale:

- Vietnamese and Indonesian: Latin-script, relatively higher-resource
- Thai: non-segmented script
- Khmer: lower-resource and more script/noise-sensitive

After the pilot works, expand to 8 SEA languages.

---

## 3. Human Annotation Data

### Available data

You already have:

```text
940 annotated documents per language in /data/*.csv folder 
```

For 4 pilot languages:

```text
900 docs/language × 4 languages = 3,600 human-labeled documents
```

### Dimensions

Use 5 quality dimensions:

1. Educational Value
2. Reasoning
3. Professionalism
4. Cleanliness
5. Cultural Nuance

### Label scale

Use the existing 0-5 scale:

```text
0 = unusable / no value
1 = very poor
2 = weak
3 = acceptable
4 = good
5 = excellent
```

### Train/dev/test split

Split each language separately:

| Split | Documents per language | Total for 4 languages |
|---|---:|---:|
| Train | 720 | 2,880 |
| Dev | 90 | 360 |
| Test | 90 | 360 |
| Total | 900 | 3,600 |

The split should be stratified by:

- Language
- Score distribution
- Document length
- Clean/noisy examples (there are two subset: _clean, _removed in the data)

### Output files

```text
human_train.jsonl
human_dev.jsonl
human_test.jsonl
```

Example row:

```json
{
  "doc_id": "th_000001",
  "language": "th",
  "text": "...",
  "educational_value": 4,
  "reasoning": 3,
  "professionalism": 4,
  "cleanliness": 5,
  "cultural_nuance": 3
}
```

---

## 4. Human-Only Quality Rater

### Model decision

Use:

```text
Backbone: multilingual-e5-large
Encoder: frozen
Trainable part: 5 separate MLP regression heads
```

The encoder is **not updated** in the pilot.

### Rater architecture

```text
Input document
    ↓
multilingual-e5-large encoder, frozen
    ↓
pooled document embedding
    ↓
5 separate MLP heads:
    1. Educational Value head
    2. Reasoning head
    3. Professionalism head
    4. Cleanliness head
    5. Cultural Nuance head
```

### Why one shared frozen encoder?

The encoder creates a strong multilingual document representation. Since the pilot has only 900 labeled documents per language, freezing the encoder reduces overfitting and makes training cheaper and easier to debug.

### Why 5 heads instead of one overall score?

The goal is to keep the quality dimensions separate and interpretable.

The model outputs:

```text
educational_value_score
reasoning_score
professionalism_score
cleanliness_score
cultural_nuance_score
```

These separate scores are later used in the weighted data-selection search.

### Loss function

Each head has its own loss:

```text
loss_edu
loss_reasoning
loss_professionalism
loss_cleanliness
loss_cultural
```

The total training loss is the mean:

```text
total_loss = mean(
    loss_edu,
    loss_reasoning,
    loss_professionalism,
    loss_cleanliness,
    loss_cultural
)
```

This does **not** merge the five raters into one rater. It only gives the optimizer one scalar objective.

### Optimizer

Use AdamW.

Important:

```text
Optimizer updates only the 5 MLP heads.
The multilingual-e5-large encoder stays frozen.
```

### Training target

Use regression on the 0-5 human labels.

Recommended loss per head:

```text
Huber loss
```

MSE is also acceptable, but Huber loss is more robust to noisy human labels.

### Rater evaluation

Evaluate on the held-out human test set.

Report:

- Spearman correlation per language and dimension
- Macro-average Spearman across languages
- Mean absolute error per dimension
- Worst-language Spearman
- Score distribution by language

Primary metric:

```text
Macro-average Spearman across languages and dimensions
```

---

## 5. Score Candidate Corpus

### Candidate corpus size

For the clean pilot:

```text
100K candidate documents per language
```

For 4 languages:

```text
400K candidate documents total
```

### Inputs

For each candidate document:

```text
doc_id
language
text
source/domain if available
```

### Model scoring

Use the human-only quality rater to predict:

```text
educational_value_score
reasoning_score
professionalism_score
cleanliness_score
cultural_nuance_score
```

<!-- ### Cheap prefilter features

Compute only simple cheap filters for the pilot:

1. Language ID confidence (language_score attributes in the dataset)
2. Document length in tokens (character_length)
3. Repetition / boilerplate score 
4. Script or encoding integrity, especially for Thai and Khmer -->

### Output

```text
scored_corpus.jsonl
```

<!-- Example row:

```json
{
  "doc_id": "th_000001",
  "language": "th",
  "text": "...",
  "edu_score": 0.73,
  "reasoning_score": 0.42,
  "professional_score": 0.68,
  "clean_score": 0.91,
  "cultural_score": 0.55,
  "langid_conf": 0.98,
  "length_tokens": 612,
  "repetition_score": 0.08,
  "script_integrity": 1
}
``` -->

---

<!-- ## Basic Prefiltering (Not implement this for now)

Before learned selection, apply a simple safety prefilter.

Recommended pilot thresholds:

```text
langid_conf >= 0.80
100 <= length_tokens <= 4096
repetition_score <= 0.20
script_integrity == pass
```

These thresholds can be tuned per language after inspecting score distributions.

Purpose:

- Remove obvious wrong-language documents
- Remove extremely short or extremely long noisy documents
- Remove repetitive boilerplate/spam
- Remove encoding/script corruption

This prefilter is applied before all selection methods, including random, to make comparisons fair. -->

---

## 7. Selection Baselines



<!-- ### Baseline 1: Random

Randomly sample documents after the basic prefilter.

Purpose:

```text
Tests whether any quality-based filtering is better than random SEA continued pretraining.
```

### Baseline 2: Clean-only heuristic

Use only cheap filters and stricter cleanliness thresholds.

Example:

```text
langid_conf >= 0.90
150 <= length_tokens <= 3072
repetition_score <= 0.10
script_integrity == pass
```

Purpose:

```text
Tests whether cheap heuristic cleaning is already enough.
```

### Baseline 3: Educational-value only

Select top documents by:

```text
educational_value_score
```

Purpose:

```text
Tests whether the strongest single quality dimension is enough.
```

### Baseline 4: Equal-average rater

Select top documents by:

```text
score = mean(
    educational_value_score,
    reasoning_score,
    professionalism_score,
    cleanliness_score,
    cultural_nuance_score
)
```

Purpose:

```text
Tests whether simple equal weighting is enough.
``` --> Not implement these baseline for now 

### Baseline: model before continue pretraining

### Main method: Best weighted combination

Use proxy CPT to choose the best weighted combination of the five rater scores.

General form:

```text
score =
    w1 * educational_value_score
  + w2 * reasoning_score
  + w3 * professionalism_score
  + w4 * cleanliness_score
  + w5 * cultural_nuance_score
```

where:

```text
w1 + w2 + w3 + w4 + w5 = 1
w_i >= 0
```

Purpose:

```text
Tests whether proxy-selected quality weighting improves CPT data selection.
```

---

## 8. Generate 16 Candidate Weight Combinations

For the pilot, use **16 combinations** only.

Include fixed interpretable weights:

| ID | Edu | Reasoning | Professionalism | Cleanliness | Cultural | Meaning |
|---|---:|---:|---:|---:|---:|---|
| W01 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | Edu only |
| W02 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | Reasoning only |
| W03 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | Professionalism only |
| W04 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | Cleanliness only |
| W05 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | Cultural only |
| W06 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | Equal average |
| W07 | 0.40 | 0.20 | 0.20 | 0.10 | 0.10 | Edu-heavy |
| W08 | 0.25 | 0.35 | 0.20 | 0.10 | 0.10 | Reasoning-heavy |
| W09 | 0.25 | 0.15 | 0.35 | 0.15 | 0.10 | Professionalism-heavy |
| W10 | 0.25 | 0.10 | 0.10 | 0.40 | 0.15 | Cleanliness-heavy |
| W11 | 0.25 | 0.10 | 0.10 | 0.15 | 0.40 | Cultural-heavy |
| W12 | 0.35 | 0.30 | 0.20 | 0.10 | 0.05 | Edu + reasoning |
| W13 | 0.35 | 0.10 | 0.30 | 0.15 | 0.10 | Edu + professionalism |
| W14 | 0.30 | 0.15 | 0.15 | 0.30 | 0.10 | Edu + cleanliness |
| W15 | 0.30 | 0.15 | 0.15 | 0.10 | 0.30 | Edu + cultural |
| W16 | 0.20 | 0.30 | 0.25 | 0.15 | 0.10 | Reasoning + professionalism |

No LightGBM is used in the pilot. The best weight is chosen directly from proxy CPT validation results.

---

## 9. Proxy Continued Pretraining

### Purpose

Use cheap continued pretraining to estimate which quality-weight combination gives the best training data.

### Proxy model

```text
Qwen2.5-0.5B Base
```

### Why this proxy model?

- Same family as the final model
- Same or highly compatible tokenizer behavior
- Much cheaper than Qwen2.5-1.5B
- Strong multilingual base
- Practical for running 16 proxy experiments

### Proxy data size

For each of the 16 weight combinations:

```text
8M tokens total
```

For 4 languages:

```text
2M tokens per language
```

### Proxy runs

```text
16 weight combinations × 50M tokens each
```

Each run starts from the same checkpoint:

```text
Qwen2.5-0.5B Base
```

### Control variables

Keep the same across all proxy runs:

- Base checkpoint
- Token budget
- Language balance
- Sequence length
- Learning rate schedule
- Batch size
- Training steps
- Validation set

Only change:

```text
Data selected by the weight combination
```

### Validation set

Create a fixed held-out validation set:

```text
1M tokens per language
```

For 4 languages:

```text
4M validation tokens total
```

This validation set must not overlap with candidate training data.

### Proxy metrics

For each proxy run, compute:

```text
validation_loss_vietnamese
validation_loss_indonesian
validation_loss_thai
validation_loss_khmer
macro_validation_loss
worst_language_validation_loss
```

Main proxy selection metric:

```text
macro_validation_loss
```

Tie-breaker:

```text
worst_language_validation_loss
```

### Output

```text
proxy_results.csv
```

Columns:

```text
weight_id
w_edu
w_reasoning
w_professionalism
w_cleanliness
w_cultural
loss_vi
loss_id
loss_th
loss_km
macro_loss
worst_language_loss
```

---

## 10. Select Final Weight Combination

Choose the weight combination with the lowest macro validation loss:

```text
best_weight = argmin(macro_validation_loss)
```

If two weights are very close, choose the one with better worst-language loss.

No LightGBM is used at this stage.

Output:

```text
best_weight.json
```

---

## 11. Final Continued Pretraining Datasets

### Final data size

For the pilot:

```text
50M tokens total
```

For 4 languages:

```text
12.5M tokens per language
```

### Final datasets

<!-- Build 5 final CPT datasets:

1. Random
2. Clean-only heuristic
3. Educational-value only
4. Equal-average rater
5. Best weighted combination from proxy CPT -->

Build 2 final CPT datasets:

1. Orginal model (no continue pretraining)
2. Best weighted combination from proxy CPT

Each dataset has exactly:

```text
250M tokens total
62.5M tokens per language
```

<!-- ### Output -->

<!-- ```text
D_random_1B
D_clean_heuristic_1B
D_edu_only_1B
D_equal_average_1B
D_best_weighted_1B
``` -->



---

## 12. Final Continued Pretraining

### Final model

```text
Qwen2.5-1.5B Base
```

### Training type

Continued pretraining only.

No final train-from-scratch LM in the pilot.

### Final CPT runs

<!-- Train 5 models:

```text
Qwen2.5-1.5B-CPT-Random
Qwen2.5-1.5B-CPT-CleanHeuristic
Qwen2.5-1.5B-CPT-EduOnly
Qwen2.5-1.5B-CPT-EqualAverage
Qwen2.5-1.5B-CPT-BestWeighted
``` -->

Train a model: Qwen2.5-1.5B-CPT-BestWeighted with LoRa 

### Control variables

Keep the same across all final CPT runs:

- Base checkpoint
- Token budget: 250M tokens
- Language balance: 62.5M tokens/language
- Sequence length
- Batch size
- Learning rate schedule
- Number of steps
- Evaluation checkpoints

Only change:

```text
Selected CPT dataset
```

---

## 13. Evaluation

Evaluate the following models:

<!-- ```text
Original Qwen2.5-1.5B Base
Qwen2.5-1.5B-CPT-Random
Qwen2.5-1.5B-CPT-CleanHeuristic
Qwen2.5-1.5B-CPT-EduOnly
Qwen2.5-1.5B-CPT-EqualAverage
Qwen2.5-1.5B-CPT-BestWeighted
``` -->

```text
Original Qwen2.5-1.5B Base
Qwen2.5-1.5B-CPT-BestWeighted
```

### 13.1 Held-out language modeling evaluation

Use the fixed validation set:

```text
10M tokens per language
40M tokens total
```

Report:

- Per-language perplexity
- Macro-average perplexity
- Worst-language perplexity
- Relative improvement over Random CPT
- Relative improvement over original base model

Primary metric:

```text
Macro-average perplexity reduction
```

### 13.2 SEA downstream evaluation: lm-evaluation-harness

Use 2 available SEA/multilingual benchmarks for the pilot.
- topic classification: https://huggingface.co/datasets/Davlan/sib200 
- MCQ: https://huggingface.co/datasets/facebook/belebele

Report:

- Accuracy or F1
- Per-language score
- Macro-average score
- Worst-language score

<!-- ### 13.3 Forgetting/general ability evaluation

Because this is continued pretraining, check whether general ability drops.

Minimum forgetting check:

- English validation perplexity
- English ARC or HellaSwag small evaluation
- Optional: English MMLU subset

Compare:

```text
Original Qwen2.5-1.5B Base
CPT-Random
CPT-BestWeighted
```

Goal:

```text
SEA performance improves while English/general performance does not collapse.
```

--- -->

## 14. Main Pilot Claims

The pilot can support these claims if results are positive:

1. A human-only multilingual quality rater trained from 900 annotated documents per language can produce useful quality scores for SEA data filtering.
2. Proxy continued pretraining with Qwen2.5-0.5B can identify better quality-weight combinations.
3. The best proxy-selected weighted data improves Qwen2.5-1.5B continued pretraining compared with random, clean heuristic, educational-value-only, and equal-average baselines.
4. Macro-language validation loss is a practical objective for multilingual SEA data selection.

---

## 15. What Is Explicitly Not Included in This Pilot

The pilot does **not** include:

- LLM weak labels
- LightGBM meta-regressor
- 128/256 weight combinations
- Train-from-scratch final LM
- Full 8-10 language setting
- Full-scale 4B-10B token continued pretraining
- Fine-tuning-heavy downstream evaluation

These can be added after the pilot is validated.

---

## 16. Clean Pilot Summary

```text
1. Use 4 languages: Vietnamese, Indonesian, Thai, Khmer.
2. Use 900 human-labeled documents per language.
3. Split labels into 720 train, 90 dev, 90 test per language.
4. Train human-only rater:
   - multilingual-e5-large frozen encoder
   - 5 separate MLP regression heads
   - mean of 5 dimension losses
   - optimizer updates only heads
5. Score 100K candidate docs per language.
<!-- 6. Apply basic prefilters: LangID, length, repetition, script integrity. -->
7. Create 16 candidate weighted combinations over 5 quality scores.
8. Run proxy CPT with Qwen2.5-0.5B Base:
   - 8M tokens per run
   - choose best weight by macro validation loss
9. Build 5 final 50M-token CPT datasets:
   - Random
   - Clean heuristic
   - Educational-value only
   - Equal average
   - Best weighted
10. Continue pretrain Qwen2.5-1.5B Base on each final dataset.
11. Evaluate perplexity, downstream SEA tasks, and forgetting.
```
