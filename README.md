# Autonomous Data Analysis Pipeline

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
| `modeler` | Executes inline Python: adaptive ensemble selection based on problem type and dataset size; adversarial sample weights from feature_engineer applied to all model families; each family tuned with Optuna (15 trials) with intra-run boundary recentering; 5-seed aggregation; final prediction is dynamically reweighted across surviving families based on OOF performance; ordinal targets receive post-processing rounding |
| `validator` | Runs `tools/validate.py` for an independent strict CV audit using purged walk-forward; diagnostic only — never blocks submission |
| `critic` | Runs `tools/run_critic.py`: 5-check quality review with optional retune feedback to the modeler |
| `submission_writer` | Runs `tools/build_submission.py` to validate format and write `submission.csv` |
| `report_writer` | Runs `tools/generate_report.py` to assemble `report.pdf` |

## Problem Subtype Detection

The pipeline detects a problem subtype from the target column characteristics, not just the broad problem type. This determines which ensemble path is used.

The subtypes are:

- **continuous_regression**: target is float or has many unique values, evaluated by regression metrics
- **ordinal_regression**: target is integer with few consecutive values that have natural order (e.g., count of days, severity scores, ratings 1-5). These are evaluated by MAE like regression, not by classification metrics
- **panel_forecasting**: time-indexed regression with group structure
- **binary_classification**: target has exactly 2 unique values
- **multiclass_classification**: target has 3-50 unique values that appear unordered (string categories or integer codes without meaningful order)

Subtype is inferred from:
- Number of unique target values
- Data type (int vs float vs string)
- Value range and consecutive structure
- `DATA_DESCRIPTION.md` keywords (days, score, count, rating, severity, level, stage suggest ordinal)

When subtype is unclear between ordinal_regression and multiclass_classification (e.g., integer target with 5-15 unique values), the pipeline defaults to ordinal_regression for MAE-evaluable behavior.

## Ensemble Path Mapping

Each problem subtype maps to an ensemble training path:

| Problem Subtype | Ensemble Path | Families | Loss |
|---|---|---|---|
| continuous_regression | full_regression_ensemble | LGB + XGB + CatBoost + Ridge | MAE |
| ordinal_regression | full_regression_ensemble | LGB + XGB + CatBoost + Ridge | MAE |
| panel_forecasting | panel_forecasting_path | LGB + XGB + CatBoost + Ridge | MAE |
| binary_classification | classification_fallback | LGB only | LogLoss |
| multiclass_classification | classification_fallback | LGB only | LogLoss |

Ordinal regression uses the regression path (not classification) because:
- The target has natural ordering, so distance between values matters
- MAE penalizes by distance, treating predicting 5 vs 6 as small error and 5 vs 14 as large error
- Classification cross-entropy would treat both errors equally, losing the ordinal structure
- This matches how MAE-graded competitions evaluate such targets

**Ordinal post-processing.** After ensembling, predictions for `ordinal_regression` targets are rounded to the nearest valid integer class observed in training. The rounding threshold is optimized on OOF predictions: the modeler tries a small grid of offsets (–0.5 to +0.5 in steps of 0.1) and selects the offset that minimises OOF MAE before clipping to the training min/max. This converts continuous ensemble outputs to valid ordinal integers without sacrificing the distance-aware MAE objective. The chosen offset is logged in `model_results.json` under `ordinal_rounding`.

## Reflective Hyperparameter Loop

The modeler performs intra-run hyperparameter reflection when Optuna's search hits a parameter boundary — for example, when the best trial lands on the maximum `num_leaves` or minimum `learning_rate`. In that case, the search space is recentered around the boundary value and a second Optuna pass runs within the same agent invocation. This exhausts the search space locally before declaring a family's best configuration, without consuming the single critic-triggered retune cycle reserved for cross-family changes. Recentering is logged in `model_results.json` under `optuna_recentering`.

**Dynamic ensemble reweighting.** Rather than a fixed equal-weight median, the final ensemble weights each surviving family inversely proportional to its OOF MAE (softmax over negative MAEs). Families with substantially worse OOF performance contribute less without being fully excluded. The applied weights are logged under `adaptive_choice.ensemble_weights`.

## Adaptive Model Selection

The modeler selects its ensemble from `profile.json` based on `problem_type` and training set size. Each tree family is tuned with 15 Optuna trials (with boundary recentering when applicable) and predictions are aggregated across 5 random seeds before ensembling.

| Condition | Ensemble |
|-----------|----------|
| panel_forecasting or tabular_regression, n_train ≥ 1,000 | LightGBM + XGBoost + CatBoost (conditional) + Ridge |
| panel_forecasting or tabular_regression, n_train < 1,000 | LightGBM + Ridge |
| classification (any size) | LightGBM only |

Ridge predictions pass two sanity checks before inclusion: predicted values must not exceed 5× the training maximum, and predicted mean must not deviate more than 100% from the training mean. A Ridge that fails either check is excluded from the ensemble; LightGBM always remains as the last-resort fallback.

**Shift-weighted ensembling with competence check.** When any covariate's KS statistic exceeds 0.40, the modeler considers weighting Ridge 1.5× in the final ensemble — Ridge's conservative extrapolation hedges against tree models overfitting shifted regions. The weight is applied only when Ridge is competitive: Ridge OOF MAE must be within 1.5× of the best family's OOF MAE. If Ridge is substantially worse, the ensemble falls back to equal-weight median to avoid pulling predictions toward the weaker family. The decision, OOF comparison, and applied weight are logged in `model_results.json` under `adaptive_choice`.

**Ridge competence check for inclusion.** Beyond the weighting decision, Ridge is also excluded from the ensemble entirely when its OOF MAE exceeds 1.5× the best tree family's OOF MAE. This prevents Ridge from pulling the median toward less accurate predictions when its model fit is substantially worse than the tree families. When excluded, the ensemble becomes LightGBM + XGBoost only. The decision is logged in `model_results.json` under `adaptive_choice.ridge_excluded_reason`.

**Time safeguards.** If total elapsed time exceeds 20 minutes when XGBoost would start, XGBoost is skipped. If two families have completed and elapsed time exceeds 30 minutes when Ridge would start, Ridge is skipped.

## Conditional CatBoost Addition

CatBoost is conditionally added as a fourth tree family when the data and time budget permit. CatBoost's symmetric oblivious trees and ordered boosting provide different inductive biases from LightGBM and XGBoost, offering ensemble diversity beyond what two leaf-wise boosters can capture.

Activation conditions require all of the following:

- catboost can be imported without errors (graceful fallback if not installed)
- n_train is at least 500 (ordered boosting requires sufficient data)
- Pipeline elapsed time is under 40 minutes when CatBoost would start (preserves budget for Ridge, validator, critic, submission_writer, and report_writer)
- Problem type is panel_forecasting, tabular_regression, or classification

When activated, CatBoost runs with 10 Optuna trials (fewer than LightGBM and XGBoost's 15 trials to manage compute), 5-seed multi-seed aggregation, and MAE loss for regression or Logloss for classification. Categorical features are passed via cat_features as column indices.

The same competence check applied to Ridge is applied to CatBoost for ensemble inclusion: CatBoost is excluded if its OOF MAE exceeds 1.5x the best tree family's OOF MAE. When included, CatBoost participates in the median or weighted ensemble alongside other surviving families.

If catboost cannot be imported or any activation condition fails, the pipeline runs with the existing 3-family ensemble (LightGBM + XGBoost + Ridge) with no interruption. All decisions are logged in model_results.json under adaptive_choice.catboost_decision.

## Distribution-Shift-Aware Features

`feature_engineering.py` tests each numeric covariate for distribution shift between training and validation using the Kolmogorov-Smirnov statistic. Covariates with KS > 0.15 receive five additional derived features: z-score normalization (using training mean/std), rolling 4-period z-score, percentile rank within the training distribution, group-level deviation from training mean, and a covariate × normalized-time interaction. All statistics are derived from training rows only so there is no leakage. Datasets with no detected shift are unaffected.

## Expanded Covariate Feature Engineering

Beyond the base lag, rolling, and group-baseline features for the target column, the pipeline generates six additional families of features derived from numeric covariates. These activate for panel forecasting when numeric covariates are present, `time_col` is identified, and sufficient training history is available. All families derive statistics from training rows only and map them onto the full frame to prevent leakage.

**Extended rolling windows** for each numeric covariate add rolling mean at windows 8 and 13 and rolling std at windows 4, 8, and 13, beyond the existing 4-period rolling mean. Windows are skipped if they exceed half of `min_periods_per_group`.

**Covariate ratios** are generated for numeric covariates that share a name prefix (split on underscore). For example, two covariates sharing a common prefix generate a ratio feature named `{prefix}_{col_a}_div_{prefix}_{col_b}`. Ratios use safe division and are capped to handle outliers. Limited to 10 ratio features per dataset.

**Group-level covariate aggregates** compute per-group historical mean, per-group recent 4-period mean, and per-group drift (recent minus historical) for each numeric covariate.

**Slope features** compute the linear regression slope of the target on time over recent 6 and 12 observations per group. For validation rows, the slope from the latest training period in that group is used. Skipped when `min_periods_per_group` is below 12.

**Centered covariates** are each numeric covariate minus its overall training mean, capturing centered deviation from cohort average.

**Entropy features** compute Shannon entropy of normalized values within covariate prefix groups when at least 2 covariates share a prefix. Captures concentration versus diffusion of related signals.

## Time Granularity Detection

`feature_engineering.py` infers temporal resolution from the median inter-observation step within groups. For datetime time columns (stored internally as hours since a fixed epoch), the thresholds are: < 2 hours → hourly, < 48 hours → daily, < 216 hours → weekly, otherwise monthly. Integer time columns are assumed weekly.

The detected granularity drives additional seasonality features beyond the base annual-cycle sin/cos harmonics:

| Granularity | Extra features added |
|-------------|----------------------|
| **Hourly** | hour-of-day sin/cos/sin2/cos2 (24-h cycle); day-of-week sin/cos (7-day cycle); per-(group × hour-of-day) and per-(group × day-of-week) training mean and std of the target |
| **Daily** | day-of-week sin/cos; month-of-year sin/cos; week-of-year sin/cos |
| **Monthly** | month-of-year sin/cos; calendar quarter (1–4) |
| **Weekly** | base annual-cycle sin/cos/sin2/cos2 only (no extra features) |

For panel data with detected time granularity, val lag features that look back into the validation period are imputed using a cycle-aware method rather than the last known training value. The imputed value uses the mean target for the matching group at the same point in the seasonal cycle: matching hour-of-day for hourly data, day-of-week for daily data, week-of-year for weekly data, and month-of-year for monthly data. For monthly granularity with opaque time IDs, the pipeline first resolves IDs to calendar dates via the codebook, then computes the median date difference across unique time periods to confirm monthly cadence before applying month-of-year cycle imputation. This preserves seasonal pattern information far ahead of the training window, addressing the staleness that occurs when later val periods are many steps removed from training end. The method falls back to last known training value when seasonal position cannot be determined or when fewer than 10 percent of training rows resolve through the codebook.

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
- Up to four-family ensemble (LightGBM + XGBoost + Ridge always, CatBoost conditionally)

## Known Limitations

- Walk-forward MAE may underestimate out-of-sample error when the validation period has a different distribution than training, even with shift-aware features applied. The validator's purged walk-forward provides a second estimate, but it may also be optimistic under severe shift.
- **Image embedding features.** For datasets with image sidecar files (PNG, JPG, JPEG), the pipeline extracts 23 hand-crafted spatial features per image using only PIL and numpy. This works for both panel and cross-sectional problems via automatic filename pattern detection.

  Activation conditions: image files present in any data subdirectory, PIL importable, at least 10 images.

  For panel data, the pipeline detects filename patterns like `{group_col}_{time_col}.png` via regex, parses linkage column values, and matches each image to corresponding panel rows. For cross-sectional data, direct row-to-image mapping is used.

  The 23 features per image cover:
  - Intensity statistics: mean, std, 99th/1st percentiles, median, IQR
  - Quadrant features: mean and std per 2×2 quadrant grid (8 features)
  - Center vs edge contrast: center mean, edge mean, center/edge ratio
  - Brightness distribution: bright/dark pixel fractions and spatial spread
  - Color features (when RGB present): inter-channel variance, intensity range

  After extraction, the pipeline computes Pearson correlation between each image feature and the target on training rows. The maximum absolute correlation is logged as a diagnostic. Features are added to the model regardless of correlation strength, letting tree-based feature importance determine ultimate usage.

  Graceful failure: missing PIL, no images detected, individual image load failures, or feature extraction errors are all caught and logged. The pipeline continues with standard features in any failure case.

  When image files are successfully matched, all 23 features are passed to the model regardless of their individual correlation with the target; tree-based feature importance determines which features are used in practice.

- Image feature extraction uses hand-crafted spatial statistics rather than deep learning embeddings. For domains where semantic content matters (object recognition, complex scenes), pre-trained CNN embeddings would extract richer features but require GPU and significant model weight downloads. The current approach trades some signal richness for CPU compatibility and zero external dependencies beyond PIL.
- Pipeline assumes one of three documented file conventions; unusual layouts may not be auto-detected and will fall back to heuristics that could misclassify train/val files.
- Stacking and target encoding are not used, to avoid leakage risks that arise when hierarchical outcome categories overlap across folds.
- Lag features compound error on long forecasts: when the validation horizon extends many periods ahead, lag imputation falls back to the last known training value per group, which degrades accuracy as horizon length increases.
- Smart lag imputation reduces but does not eliminate the gap between internal CV metrics and real test MAE. The imputation typically improves walk-forward predictions with minimal effect on internal CV metrics, since it primarily helps test-time predictions rather than training-time evaluation.
- On datasets with severe distribution shift, internal CV estimates may still underestimate true generalization error even after shift-aware features are applied, because the KS-weighted ensemble adjustments are bounded and cannot fully correct for extreme covariate drift.
- Expanded covariate features add substantial feature count, often doubling or tripling the feature set on datasets with many numeric covariates. Runtime increases proportionally during feature engineering and training. The benefit varies by dataset: datasets where existing features already capture the predictive signal show minimal change, while datasets with rich covariate information show meaningful improvement on strict CV.

- CatBoost adds 1-3 minutes to pipeline runtime when activated. On datasets where CatBoost provides similar OOF performance to LightGBM and XGBoost, the median ensemble may show minimal improvement because the three tree families converge on similar predictions. Real benefit varies by dataset characteristics.

- True multiclass classification (unordered categories like species or diagnosis codes) uses LightGBM-only path. Multi-family classification ensembling would require logloss-based competence checks and probability aggregation, which are not implemented. Ordinal regression targets (integer counts/scores with natural order) use the full regression ensemble correctly and are not affected by this limitation.

The `report.pdf` produced during analysis contains detailed methodology and results for the specific run.
