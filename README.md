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
- Image files if applicable (see Known Limitations for how they are handled)

The pipeline auto-detects one of three file layout conventions:

- **Split convention**: `covariates_train.csv` + `target_train.csv` + `covariates_val.csv`
- **Combined convention**: `train.csv` with embedded target + `val_features.csv`
- **Kaggle subfolder convention**: `data/train/*.csv` + `data/test/covariates.csv`

Optionally, a JSON codebook file in `data/` can map opaque time-column IDs to calendar dates — see [Codebook-Based Date Features](#codebook-based-date-features) below.

## Expected Outputs

- `submission.csv` — predictions at the repo root, matching `sample_submission.csv` format
- `report.pdf` — multi-section methodology report at the repo root, including feature importance charts and a limitations section

## Architecture

The pipeline runs seven sub-agents in sequence, each in its own context window. A marker file (`reports/{agent}_was_here.txt`) confirms each sub-agent was invoked and did not have its work inlined by the orchestrator.

| Sub-agent | Role |
|-----------|------|
| `schema_analyst` | Runs `tools/profile_data.py` to discover dataset structure, problem type, KS-tested shifts, and optional time codebook; writes `reports/profile.json` and `reports/schema_analysis.md` |
| `feature_engineer` | Runs `tools/feature_engineering.py` to generate features adapted to schema, detected time granularity, distribution shift (KS-based features), and adversarial validation (train-vs-val shift detection with sample weighting) |
| `modeler` | Executes inline Python: adaptive ensemble selection based on problem type and dataset size; adversarial sample weights from feature_engineer applied to all model families; each family tuned with Optuna (15 trials) and 5-seed aggregation; final prediction is median or weighted average across surviving families |
| `validator` | Runs `tools/validate.py` for an independent strict CV audit using purged walk-forward; diagnostic only — never blocks submission |
| `critic` | Runs `tools/run_critic.py`: 5-check quality review with optional retune feedback to the modeler |
| `submission_writer` | Runs `tools/build_submission.py` to validate format and write `submission.csv` |
| `report_writer` | Runs `tools/generate_report.py` to assemble `report.pdf` |

## Adaptive Model Selection

The modeler selects its ensemble from `profile.json` based on `problem_type` and training set size. Each tree family is tuned with 15 Optuna trials and predictions are aggregated across 5 random seeds before ensembling.

| Condition | Ensemble |
|-----------|----------|
| panel_forecasting or tabular_regression, n_train ≥ 1,000 | LightGBM + XGBoost + Ridge |
| panel_forecasting or tabular_regression, n_train < 1,000 | LightGBM + Ridge |
| classification (any size) | LightGBM only |

Ridge predictions pass two sanity checks before inclusion: predicted values must not exceed 5× the training maximum, and predicted mean must not deviate more than 100% from the training mean. A Ridge that fails either check is excluded from the ensemble; LightGBM always remains as the last-resort fallback.

**Shift-weighted ensembling with competence check.** When any covariate's KS statistic exceeds 0.40, the modeler considers weighting Ridge 1.5× in the final ensemble — Ridge's conservative extrapolation hedges against tree models overfitting shifted regions. The weight is applied only when Ridge is competitive: Ridge OOF MAE must be within 1.5× of the best family's OOF MAE. If Ridge is substantially worse, the ensemble falls back to equal-weight median to avoid pulling predictions toward the weaker family. The decision, OOF comparison, and applied weight are logged in `model_results.json` under `adaptive_choice`.

**Ridge competence check for inclusion.** Beyond the weighting decision, Ridge is also excluded from the ensemble entirely when its OOF MAE exceeds 2.0× the best tree family's OOF MAE. This prevents Ridge from pulling the median toward less accurate predictions when its model fit is substantially worse than the tree families. The decision is logged in `model_results.json` under `adaptive_choice.ridge_excluded_reason`.

**Time safeguards.** If total elapsed time exceeds 20 minutes when XGBoost would start, XGBoost is skipped. If two families have completed and elapsed time exceeds 30 minutes when Ridge would start, Ridge is skipped.

## Distribution-Shift-Aware Features

`feature_engineering.py` tests each numeric covariate for distribution shift between training and validation using the Kolmogorov-Smirnov statistic. Covariates with KS > 0.15 receive five additional derived features: z-score normalization (using training mean/std), rolling 4-period z-score, percentile rank within the training distribution, group-level deviation from training mean, and a covariate × normalized-time interaction. All statistics are derived from training rows only so there is no leakage. Datasets with no detected shift are unaffected.

## Time Granularity Detection

`feature_engineering.py` infers temporal resolution from the median inter-observation step within groups. For datetime time columns (stored internally as hours since a fixed epoch), the thresholds are: < 2 hours → hourly, < 48 hours → daily, < 216 hours → weekly, otherwise monthly. Integer time columns are assumed weekly.

The detected granularity drives additional seasonality features beyond the base annual-cycle sin/cos harmonics:

| Granularity | Extra features added |
|-------------|----------------------|
| **Hourly** | hour-of-day sin/cos/sin2/cos2 (24-h cycle); day-of-week sin/cos (7-day cycle); per-(group × hour-of-day) and per-(group × day-of-week) training mean and std of the target |
| **Daily** | day-of-week sin/cos; month-of-year sin/cos; week-of-year sin/cos |
| **Monthly** | month-of-year sin/cos; calendar quarter (1–4) |
| **Weekly** | base annual-cycle sin/cos/sin2/cos2 only (no extra features) |

For panel data with detected time granularity, val lag features that look back into the validation period are imputed using a cycle-aware method rather than the last known training value. The imputed value uses the mean target for the matching group at the same point in the seasonal cycle: matching hour-of-day for hourly data, day-of-week for daily data, week-of-year for weekly data, and month-of-year for monthly data. For monthly granularity with opaque time IDs (like Award A's period_id hashes), the pipeline first resolves IDs to calendar dates via the codebook, then computes the median date difference across unique time periods to confirm monthly cadence before applying month-of-year cycle imputation. This preserves seasonal pattern information far ahead of the training window, addressing the staleness that occurs when later val periods are many steps removed from training end. The method falls back to last known training value when seasonal position cannot be determined or when fewer than 10 percent of training rows resolve through the codebook.

## Adversarial Validation

`feature_engineering.py` trains a binary LightGBM classifier (5-fold StratifiedKFold, 100 estimators) to distinguish training rows (label 1) from validation rows (label 0). The classifier's OOF AUC measures multivariate covariate shift that per-column KS tests may miss.

**Activation conditions**: at least 500 combined rows, at least 100 training rows, at least 100 validation rows, and at least 3 numeric covariate columns. Datasets that do not meet all four conditions skip adversarial validation silently.

**AUC < 0.55**: no meaningful shift detected; uniform sample weights are used.  
**AUC ≥ 0.55**: each training row receives weight `w = clip(1 − P(is_train), 0.1, 10.0)`, then normalized so the mean weight equals 1.0. Rows that "look like" validation rows receive higher weight, nudging the model toward the actual prediction distribution.

The weights are stored as an `adversarial_weights` column in `data/features_train.parquet`. The modeler reads this column and passes it as `sample_weight` to all three model families — LightGBM OOF folds, LightGBM full-data retraining, XGBoost Optuna objective, XGBoost final fit, and Ridge. The top five shift-revealing features (by classifier importance), the AUC, and whether weights were applied are logged in `features.json` under `adversarial_validation` and propagated to `model_results.json` under `adaptive_choice.adversarial_validation`.

## Codebook-Based Date Features

For datasets where the time column uses opaque identifiers (e.g. base-64 hashes like `"uTjgI1Sv"`) instead of readable dates, `schema_analyst` looks for a JSON codebook file in `data/` mapping those IDs to calendar dates. The following filenames are checked in order: `period_id_codebook.json`, `period_codebook.json`, `period_id.json`, `dates.json`, `time_codebook.json`.

When a valid codebook is found, `profile.json` gains a `time_codebook` field that includes the filename, number of entries, detected mapping direction (`id_to_date` or `date_to_id`), and up to five sample pairs. `feature_engineering.py` then adds three calendar features: `month_of_year` (1–12), `quarter_of_year` (1–4), and `is_quarter_start` (1 if month ∈ {1, 4, 7, 10}). Rows with unmapped time values are imputed with the training-set median month. If more than 10% of training rows are unmapped, date features are skipped entirely and a warning is logged. When no codebook is present, this block is a silent no-op that leaves all other features unchanged.

## Constraints

- CPU only (no GPU)
- No network access or external data during analysis
- 2-hour wall-clock budget; 1M token budget (input + output, all agents combined)
- Three-family ensemble (LightGBM + XGBoost + Ridge); CatBoost is not included

## Known Limitations

- Walk-forward MAE may underestimate out-of-sample error when the validation period has a different distribution than training, even with shift-aware features applied. The validator's purged walk-forward provides a second estimate, but it may also be optimistic under severe shift.
- **Image handling differs by problem type.** For cross-sectional datasets (no time column), `feature_engineering.py` attempts to extract six grayscale summary statistics per image (mean intensity, std, center mean, corner mean, contrast, bright fraction) using PIL; these are included in the model. For panel (time-series) datasets, image features are not extracted. Note: `schema_analysis.md` states that image features are not used — this is accurate for panel problems but not for cross-sectional ones.
- Pipeline assumes one of three documented file conventions; unusual layouts may not be auto-detected and will fall back to heuristics that could misclassify train/val files.
- Stacking and target encoding are not used, to avoid leakage risks that arise when hierarchical outcome categories overlap across folds.
- Lag features compound error on long forecasts: when the validation horizon extends many periods ahead, lag imputation falls back to the last known training value per group, which degrades accuracy as horizon length increases.
- Smart lag imputation reduces but does not eliminate the gap between internal CV metrics and real test MAE. Verified on retail (weekly): real test MAE improved from 9.27 to 9.05 (2.4% reduction). Verified on Award A (monthly with codebook): smart imputation activated correctly with 7344 val lag cells filled via month-of-year cycle averages and zero fallbacks. In both cases internal CV remained essentially unchanged, demonstrating the imputation primarily helps test-time predictions rather than training-time evaluation.
- On datasets with severe distribution shift, internal CV estimates may still underestimate true generalization error even after shift-aware features are applied, because the KS-weighted ensemble adjustments are bounded and cannot fully correct for extreme covariate drift.

The `report.pdf` produced during analysis contains detailed methodology and results for the specific run.
