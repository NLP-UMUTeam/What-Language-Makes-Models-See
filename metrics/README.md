# Metrics and paper-result generation

This directory contains the analysis scripts used to transform the multilingual JSONL predictions into the CSV metrics, statistical tests, figures, and LaTeX tables reported in the paper.

## Directory contents

| File | Purpose |
|---|---|
| `compute_metrics.py` | Main evaluation pipeline. Harmonises JSONL outputs and computes validity, answer distributions, demographic nJSD, cross-lingual agreement, distribution shifts, modality shifts, ordinal rank statistics, bootstrap confidence intervals, Stuart–Maxwell tests, Cramér's V, and paired permutation tests. |
| `create_validity_table.py` | Builds the LaTeX table containing valid-response rates and dominant answers. |
| `create_cross_lingual_table.py` | Builds the main cross-lingual consistency table. |
| `create_demographic_njsd_table.py` | Builds the vision–language versus text-only demographic nJSD table. |
| `plot_modality_comparison.py` | Produces the modality-comparison figure and LaTeX table, including visual support for the text-only answer. |
| `create_appendix_metric_tables.py` | Generates the additional appendix tables from all model-specific metric directories. |
| `create_justification_hotwords.py` | Extracts multilingual lexical cues from justifications and generates CSV and LaTeX hotword tables. |
| `configs/custom_stopwords.json` | Additional task-specific stopwords used in the lexical analysis. |
| `configs/results_manifest.example.json` | Example manifest describing the raw JSONL outputs. Paths must be adapted locally. |
| `configs/metrics_dir_manifest.example.json` | Example manifest describing model-specific metric directories and display order. |

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

Chinese lexical segmentation additionally uses `jieba`, which is included in the analysis requirements.

## 1. Prepare a results manifest

`compute_metrics.py` expects a JSON list. Each entry must contain:

```json
{
  "path": "../results_english/qwen_3_32b_visbias_all_tasks_english",
  "language": "en",
  "condition": "visual",
  "model_name": "qwen3_vl_32b"
}
```

Supported language identifiers are `en`, `es`, and `zh`. The two experimental conditions are `visual` and `text_only`.

For the paper, metrics were computed separately for each model. A model-specific manifest should therefore contain the six corresponding files: three prompt languages multiplied by the two input conditions. Relative paths are resolved from the manifest location.

## 2. Compute metrics

```bash
python metrics/compute_metrics.py \
  --manifest metrics/configs/qwen3_vl_32b_results.json \
  --output_dir results/metrics/qwen3_vl_32b \
  --bootstrap_samples 10000 \
  --permutation_samples 10000 \
  --seed 42
```

The command produces:

```text
analysis_summary.json
harmonised_records.csv
response_validity.csv
answer_distributions.csv
mean_predicted_ranks.csv
demographic_njsd_by_occupation.csv
demographic_njsd_macro.csv
rank_gaps_by_occupation.csv
rank_gaps_macro.csv
cross_lingual_pairwise_consistency.csv
cross_lingual_three_way_consistency.csv
cross_lingual_distribution_shift_detail.csv
cross_lingual_distribution_shift_macro.csv
bias_delta_vs_english.csv
visual_text_modality_shift_detail.csv
visual_text_modality_shift_macro.csv
visual_text_ordinal_rank_differences.csv
visual_minus_text_demographic_njsd.csv
language_demographic_cramers_v.csv
stuart_maxwell_tests.csv
cross_lingual_agreement_bootstrap_ci.csv
visual_text_paired_permutation_ordinal.csv
visual_distributions_aggregated_by_combination.csv
```

Invalid model outputs contribute to the validity analysis but are excluded from the remaining metrics. The default generalized nJSD normalization uses the attainable maximum, `log(min(G, K))`, where `G` is the number of compared groups and `K` is the number of answer categories.

## 3. Generate the main-paper tables

### Response validity and dominant answers

```bash
python metrics/create_validity_table.py \
  --validity results/metrics/qwen3_vl_32b/response_validity.csv \
  --distributions results/metrics/qwen3_vl_32b/answer_distributions.csv \
  --output paper_tables/validity_answers.tex
```

The `--model` option may be repeated to restrict the output to selected models.

### Cross-lingual consistency

```bash
python metrics/create_cross_lingual_table.py \
  --metrics_dir results/metrics/qwen3_vl_32b \
  --condition visual \
  --percent \
  --output paper_tables/cross_lingual_consistency.tex
```

### Demographic differentiation

```bash
python metrics/create_demographic_njsd_table.py \
  --metrics_dir results/metrics/qwen3_vl_32b \
  --partition intersection \
  --output paper_tables/demographic_njsd.tex
```

### Vision–language versus text-only comparison

First create a metrics-directory manifest such as:

```json
[
  {
    "metrics_dir": "../../results/metrics/qwen3_vl_32b",
    "model_name": "qwen3_vl_32b",
    "display_name": "Qwen3-VL-32B",
    "size": "32B",
    "order": 1
  }
]
```

Then run:

```bash
python metrics/plot_modality_comparison.py \
  --manifest metrics/configs/metrics_dir_manifest.json \
  --figure_output paper_figures/modality_comparison.pdf \
  --latex_output paper_tables/modality_comparison.tex \
  --show_values
```

If no precomputed visual-support CSV is supplied, the script derives it from `harmonised_records.csv` and writes both detailed and macro support files into each model's metric directory.

## 4. Generate appendix metric tables

After all model-specific metric directories have been computed, list them in a metrics-directory manifest and run:

```bash
python metrics/create_appendix_metric_tables.py \
  --manifest metrics/configs/metrics_dir_manifest.json \
  --output_dir paper_tables/appendix \
  --partition intersection
```

Generated files include ordinal demographic gaps, cross-lingual distribution shifts, bootstrap confidence intervals, Stuart–Maxwell tests, ordinal modality comparisons, and language–demographic association tables. The script also writes `appendix_additional_metrics.tex`, which includes the generated tables.

## 5. Generate the lexical analysis

The lexical-analysis manifest uses the same raw-result entries as the main metrics manifest, but should also include model metadata when several models are combined:

```json
{
  "path": "../results/qwen3_english.jsonl",
  "language": "en",
  "condition": "visual",
  "model_name": "qwen3_vl_32b",
  "display_name": "Qwen3-VL-32B",
  "order": 1
}
```

Run:

```bash
python metrics/create_justification_hotwords.py \
  --manifest metrics/configs/all_results_manifest.json \
  --custom_stopwords metrics/configs/custom_stopwords.json \
  --output_dir paper_tables/hotwords \
  --normalization per_1000 \
  --model_table_top_n 7 \
  --min_frequency 3 \
  --highlight_min_models 3
```

The script preserves the prompt language as the table row while detecting the language actually used in each justification. When a model responds predominantly in another language—for example, English under a Chinese prompt—the original generated terms are retained without translation and the corresponding cell receives a dagger note.

Important outputs are:

```text
hotword_frequencies.csv
hotword_input_summary.csv
hotwords_visual_tasks_table.tex
hotwords_text_only_tasks_table.tex
hotwords_appendix_master.tex
```

## Reproducibility notes

- Inference uses deterministic decoding (`do_sample=False`).
- Bootstrap and permutation analyses use 10,000 samples and seed 42 by default.
- Text-only inputs are evaluated once per unique occupation–demographic combination.
- Visual predictions are aggregated at occupation–demographic cell level before modality comparisons.
- Holm–Bonferroni correction is applied within the hypothesis families defined by the scripts.
- Paths in the example manifests are placeholders and must be adapted to the local result layout.
