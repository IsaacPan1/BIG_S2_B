# Award B: Autonomous Data Analysis Pipeline

This is an autonomous data analysis pipeline built on Claude Code that handles tabular forecasting and regression problems. Given an unknown dataset placed in `data/`, it produces `submission.csv` and `report.pdf` within 2 hours without human intervention.

## Quick Start

1. Place the dataset in `data/` along with `DATA_DESCRIPTION.md`
2. Open Claude Code: `claude --dangerously-skip-permissions`
3. Prompt: `Do the data analysis.`
4. Wait for the pipeline to complete (typically 5–15 minutes depending on dataset size)
5. Find `submission.csv` and `report.pdf` at the repo root

## Expected Inputs

Place the following in `data/` before running:

- `DATA_DESCRIPTION.md` — describes the problem, target variable, and evaluation metric
- Training and validation data files
- `sample_submission.csv` — shows the expected output format (column names, row count)
- Image files if applicable (detected and noted in the schema report, but not used as model features)

The pipeline auto-detects one of three file layout conventions:

- **Split convention**: `covariates_train.csv` + `target_train.csv` + `covariates_val.csv`
- **Combined convention**: `train.csv` with embedded target + `val_features.csv`
- **Kaggle subfolder convention**: `data/train/*.csv` + `data/test/covariates.csv`

## Expected Outputs

- `submission.csv` — predictions at the repo root, matching `sample_submission.csv` format
- `report.pdf` — multi-section methodology report at the repo root, including feature importance charts and a limitations section

## Architecture

The pipeline runs seven sub-agents in sequence, each in its own context window. A marker file confirms each sub-agent was invoked (not inlined).

| Sub-agent | Role |
|-----------|------|
| `schema_analyst` | Discovers dataset structure and problem type |
| `feature_engineer` | Generates features adapted to schema and time granularity |
| `modeler` | Adaptive ensemble selection (LightGBM + XGBoost + Ridge) based on problem type and dataset size; each family tuned with Optuna and seed-aggregated; final predictions are median across ensemble |
| `validator` | Independent audit of CV honesty using purged walk-forward |
| `critic` | 5-check quality review with optional retune feedback loop |
| `submission_writer` | Validates format and writes `submission.csv` |
| `report_writer` | Assembles `report.pdf` |

## Adaptive Model Selection

The modeler picks its ensemble based on `problem_type` and training set size read from `profile.json`:

| Condition | Ensemble |
|-----------|----------|
| panel_forecasting, n_train ≥ 1,000 | LightGBM + XGBoost + Ridge |
| panel_forecasting, n_train < 1,000 | LightGBM + Ridge |
| tabular_regression, n_train ≥ 1,000 | LightGBM + XGBoost + Ridge |
| tabular_regression, n_train < 1,000 | LightGBM + Ridge |

Ridge predictions pass sanity checks (range and bias tests) before inclusion — excluded if systematically biased or out-of-range. Final prediction is the median across families that survive selection.

## Distribution-Shift-Aware Features

`feature_engineer` tests each covariate for distribution shift between training and validation using the Kolmogorov-Smirnov statistic. Covariates with KS > 0.15 receive five additional derived features: z-score normalization, rolling z-score, percentile rank, group-level deviation, and time interaction. These supplement standard features rather than replace them. Datasets with no detected shift are unaffected.

## Constraints

- CPU only (no GPU)
- No network access during analysis
- 2-hour wall-clock budget
- 1M token budget
- Three-family ensemble (LightGBM + XGBoost + Ridge); CatBoost not included due to Windows installation complexity

## Known Limitations

- Walk-forward MAE may underestimate out-of-sample error when the validation period has a different distribution than training
- Image data is detected but not used as model features
- Pipeline assumes one of the three documented file conventions; unusual structures may not be auto-detected
- On datasets with severe distribution shift between training and validation periods, internal CV estimates may still underestimate true generalization error even with shift-aware features applied

The `report.pdf` produced during analysis contains detailed methodology and results for the specific run.
