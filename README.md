# Autonomous Data Analysis Pipeline

This is an autonomous data analysis pipeline built on Claude Code that handles tabular forecasting and regression problems. Given an unknown dataset placed in `data/`, it produces `submission.csv` and `report.pdf` within 2 hours without human intervention.

## Quick Start

1. Place the dataset in `data/` along with `DATA_DESCRIPTION.md`
2. Open Claude Code: `claude --dangerously-skip-permissions`
3. Prompt: `Do the data analysis.`
4. Wait for the pipeline to complete. Full modeler runs (5-fold nested CV with Optuna and 5-seed final ensemble) typically take ~25–45 minutes; schema, feature engineering, validator, critic, and reporting add roughly another 10–20 minutes on top.
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
| `modeler` | Runs `tools/run_modeler.py`. CatBoost is the sole predictor; a Ridge baseline runs alongside as a linear diagnostic only. Nested CV is driven by the frozen `reports/cv_plan.json`: each outer fold builds a fresh per-fold `adaptive_pipeline` (impute → scale → per-fold `TargetEncoderCV`) that is fit on training rows only, then runs Optuna inside the fold. Trials are bounded by a per-fold time deadline derived from a tuning budget (default 25 min total); folds whose budget is exhausted fit CatBoost with default hyperparameters and still contribute their MAE. Final predictions aggregate a 5-seed CatBoost ensemble. The recursive-vs-static-imputation method is selected empirically on a walk-forward holdout. Ordinal targets receive post-processing rounding. A `--debug` flag caps iterations, trials, and seeds for fast iteration — debug OOF is not a valid score. |
| `validator` | Runs `tools/validate.py` for an independent strict CV audit using purged walk-forward; diagnostic only — never blocks submission |
| `critic` | Runs `tools/run_critic.py`: 5-check quality review with optional retune feedback to the modeler |
| `submission_writer` | Runs `tools/build_submission.py` to validate format and write `submission.csv` |
| `report_writer` | Runs `tools/generate_report.py` to assemble `report.pdf` |

## Knowledge Graph

The pipeline writes an observability-only knowledge graph to `reports/knowledge_graph.json` via `tools/kg.py`. It is a **derived, append-only, observability-only log** — written by sub-agents at key points using `kg_set_stage`, `kg_append_event`, and `kg_record_rejected`. No agent reads it to make decisions, and it has no effect on the analysis or the submission.

The graph records stage transitions (which agent ran and when), per-stage events (inferred problem type, resource counts, Optuna trial outcomes, hypothesis events), and rejected hypotheses from each stage. Its purpose is a run audit trail for transparency and post-hoc debugging — diagnostic tooling only, not a controller.

## Per-Seed OOF Diagnostic

The modeler writes an observability-only artifact at `reports/oof/oof_per_seed.csv`. It is a **derived, observability-only file** — no agent reads it, and the aggregated OOF in `reports/oof_predictions.csv`, the decision metric used by the modeler / validator / critic, and the final submission are all unaffected by its existence.

It is produced during the outer-fold scoring step. Each outer fold already trains a small CatBoost ensemble across `OUTER_FOLD_FINAL_SEEDS` and aggregates their predictions on the held-out outer-val rows; the diagnostic simply persists those per-seed prediction arrays by slot index. Slot `k` corresponds to seed `OUTER_FOLD_FINAL_SEEDS[k]`; uncovered slots stay `NaN`. **Nothing is added, reordered, reseeded, or refit** — the artifact stores arrays the scoring step has already computed.

The schema is **one row per covered OOF observation**, matching the row set in `reports/oof_predictions.csv`. Identifier columns are resolved generically from `profile.json` (no hardcoded names): a `row_index` join-key into the training frame, the group columns, the time column (when one is identified), `fold_id` (the outer-fold index), `y_true` (the held-out raw target), and one `pred_seed_{k}` column per outer-fold seed.

**Integrity check.** Immediately after the file is written, the modeler rebuilds the OOF predictions row-wise by applying the **same aggregation callable the run used** — `np.mean` by default, or `np.median` when the critic-triggered `median_seed_aggregation` retune path is active — over the `pred_seed_*` columns (NaN-aware for partially populated slots). The resulting per-row aggregate must reproduce the in-memory `oof_preds` array to within `1e-9`; if `max |rowwise_agg − oof_preds|` exceeds that tolerance the modeler raises an `AssertionError` and the run aborts. A companion assertion enforces that the per-seed row count matches the covered OOF row count exactly.

Under `--debug` only a single outer-fold seed is present, so the per-seed variance view is degenerate and not meaningful — mirroring the existing rule that debug OOF is not a valid score.

The pointer block is logged in `model_results.json` under `oof_per_seed` with `seeds`, `n_seeds`, `n_rows`, `per_fold_seeds`, `agg` (the aggregation callable's name as actually used this run), `path`, and `debug`.

## Cross-Validation Strategy

CV is treated as a frozen contract authored exactly once per run. `schema_analyst` invokes `tools/scheme_analysis.py`, the **only** module permitted to write `reports/cv_plan.json`. The plan selects a CV scheme from the inferred problem type — grouped or ungrouped time series default to `TimeSeriesExpanding`, tabular IID falls into `GroupKFold` / `StratifiedKFold` / `KFold` depending on whether group columns are present and whether the target is discrete — and is marked `frozen: true`. No downstream agent may modify it; the critic can request at most one replan by emitting `CV_INVALID`, in which case the orchestrator deletes `cv_plan.json` and re-invokes `schema_analyst` once.

`tools/cv_engine.CVEngine` materialises the plan into deterministic `(train_idx, valid_idx)` folds and enforces leakage invariants (no train/valid index overlap; for time-series schemes, `max(train_time) + gap ≤ min(valid_time)`; for `GroupKFold`, disjoint groups). The same `CVEngine` is consumed by the modeler — its outer evaluation folds come directly from this plan, not from any modeler-internal split.

**End-anchored expanding window.** For `TimeSeriesExpanding` (the default time-series scheme), the **last** fold's validation block ends at the final time index `T` — mirroring the train → test boundary the held-out submission will face. Earlier folds step backward by `valid_size`; training is always `[0:train_end]`, so the train window expands as fold index advances. `TimeSeriesSliding` follows the same end-anchored pattern with a fixed-width training window.

**Rank-resolved time axis.** When the time column is an opaque identifier resolved by a codebook, `cv_engine.attach_period_rank` materialises a dense integer `__period_rank__` column from `profile['period_rank_info']['id_to_rank']` and uses it as the ordering axis for all CV math; the raw hash column remains as the join / submission key. This is the only place `id_to_rank` lookups happen. A missing or NaN rank during this step raises rather than silently flowing into the fold math.

**Drift gate (expanding-vs-sliding).** `scheme_analysis.py` defaults to expanding and switches to sliding **only** when a recent-vs-full drift diagnostic shows the train→val shift is **recency-reducible**. For each shared numeric, non-window covariate (seasonality and time-index columns are excluded by name), it computes the standardised mean shift `|μ_train − μ_val| / pooled_std` twice — using the full training window and using only the last `RECENT_PERIODS = 14` periods — and aggregates `mean_improvement`, `rel = mean_improvement / mean_dist_full`, and `frac_improved` (share of features whose per-feature improvement exceeds 0.05). The sliding branch is taken **only** when all three affirmative gates hold simultaneously: `frac_improved ≥ 0.60` (primary), `rel ≥ 0.25` (secondary), and `n_features_scanned ≥ 12` (evidence-breadth floor). Every fallback path — no val frame, no time axis, fewer than `RECENT_PERIODS` of history, no rank, no shared numeric covariates — returns expanding. The diagnostic uses validation **covariates** (provided by the competition); the validation target is never read.

## Cross-Validation Decision Record

This section records the CV design as an explicit decision record — options considered,
choice made, evidence that drove the choice, and what rejecting the alternative would
have cost. Every claim is grounded in `reports/cv_plan.json`, `reports/features.json`,
and `reports/validator_review.json` written by the most recent pipeline run. No claim
is invented; the record is re-derived from those artifacts at each run.

### Decision 1 — Problem Type

**Candidates:** `plain_regression`, `univariate_time_series`, `panel_forecasting`

| Candidate | Outcome | Reason |
|-----------|---------|--------|
| `plain_regression` | REJECTED | Treats every row as IID. During cross-validation, training folds contain rows from future periods used to predict past periods — temporal leakage that produces an optimistically biased estimate and fails at submission time because the model receives no time-ordered signal. |
| `univariate_time_series` | REJECTED | One model per group. Cross-group patterns — shared seasonal cycles, shared covariate effects, correlated trend changes — are discarded. Per-group signal-to-noise is lower because training history is partitioned by group instead of pooled across groups. |
| `panel_forecasting` | **SELECTED** | Panel structure confirmed: both a time column (regular monthly cadence; `cv_plan.json: time_column`) and multiple group columns (`cv_plan.json: group_columns`) are detected. External numeric covariates are keyed on `(group, time)`. Cross-group signal is available and the temporal ordering is respected. |

**Evidence from artifacts:** regular time cadence detected (median step ≈ 744 h,
step CV < 0.03, `features.json: time_granularity`); group × time structure present;
problem type recorded as `grouped_time_series` / `panel_forecasting` in
`cv_plan.json: problem_type`.

### Decision 2 — CV Scheme

**Candidates:** random k-fold, single holdout, expanding window, sliding window

**Choice:** `TimeSeriesExpanding` — frozen in `reports/cv_plan.json`:
`cv_type = "TimeSeriesExpanding"`, `n_splits = 5`, `valid_size = 7`, `gap = 0`,
`shuffle = false`.

**Why expanding window for a forecasting target:**
Random k-fold is invalid for ordered data: observations from future periods flow into
training folds that predict past periods, leaking the answer and producing estimates
that are meaningless for forecasting evaluation. Expanding window respects the arrow
of time — training always uses data from `[0 : fold_end]`, validation is the
immediately following `valid_size` periods, and the **last** fold ends at the final
training period T, mirroring the actual train → submission boundary.

**Expanding vs. sliding gate (`cv_plan.json → cv_selection_reason`):**
The pipeline runs a data-driven diagnostic before freezing the scheme. Sliding window
(fixed recent training window) is chosen **only** when three conditions hold
simultaneously:

1. `frac_improved >= 0.60` (primary gate) — majority of covariates have smaller
   train-vs-val distribution distance when the training window is restricted to the
   most recent periods.
2. `rel >= 0.25` (secondary gate) — mean improvement is material relative to total
   shift magnitude.
3. `n_features_scanned >= 12` (evidence-breadth floor) — enough features to make
   the conclusion robust.

In the recorded run (`cv_plan.json: cv_selection_reason.drift_metrics`):
`frac_improved = 0.556 < 0.60` — the primary gate did not pass; expanding window
retained. Sliding would have discarded older training data without demonstrated
shift-reduction benefit.

**Embargo / purge gap:**
The validator's independent strict re-audit uses a 2-period embargo
(`validator_review.json: strict_cv_scheme`). The embargo drops observations adjacent
to the train/val fold boundary from both sides, preventing leakage from lag features
whose lookback window spans the boundary — a form of leakage a gap-free split does
not address.

**Nested tuning:**
Hyperparameter search runs **inside** each outer fold using `inner_folds = 3` inner
splits (`model_results.json: nested_cv.inner_folds`). The Optuna objective is the
inner-fold MAE, computed only on inner-fold training rows. The outer-fold validation
rows are never seen during the search, so the outer-fold MAE is an honest estimate of
the performance achieved by the configuration the pipeline would have selected without
access to the held-out data — not the best achievable configuration in hindsight.

**Cost of rejected schemes:**

| Rejected scheme | What the cost would have been |
|-----------------|-------------------------------|
| Random k-fold | Produces a lower (optimistic) CV MAE by training on future data to predict past. The estimate is meaningless for an ordered forecasting target. |
| Single holdout | High-variance estimate dependent on which single period was held out; no evidence about how performance scales with training size; no fold-ensemble variance reduction. |
| Sliding window (no affirmative gate) | Discards older training data whose patterns may remain valid; no demonstrated shift-reduction benefit in this run; would have required sliding without evidence. |

### Decision 3 — Why the OOF Is Honest, and Its Limit

**Why it is honest within the training distribution:**

The OOF is constructed so that each row's prediction is made by a model that has never
seen that row during training or hyperparameter selection:

- The expanding-window split ensures every outer-fold validation row is strictly later
  than every outer-fold training row in the same fold.
- Nested tuning (inner folds inside each outer fold) ensures no hyperparameter is
  selected using outer-fold validation rows.
- The validator's independent purged walk-forward re-audit provides a second estimate
  with an explicit embargo. Its gap-attribution block
  (`validator_review.json → gap_attribution`) classifies the gap between modeler CV
  MAE and strict validator MAE. A classification of `CV_SCHEME` means the gap is
  **structural pessimism** from smaller training windows in early folds — not model
  overfit. A `monotone_score = 1.0` means every fold pair shows MAE improving in the
  expected direction as the training window grows.

**The limit — distribution shift:**

Adversarial validation trains a binary classifier to distinguish training rows from
validation rows using only covariate features. The resulting AUC
(`features.json → adversarial_validation.auc_train_vs_val`) measures how reliably the
two populations can be separated by a linear or tree-based model.

When AUC is near 1.0, the populations are nearly perfectly separable — the model will
be evaluated on data that looks systematically different from what it trained on. This
is a general mechanism, not a dataset-specific quirk: any domain where economic
conditions, search behavior, or environmental signals shift substantially between the
training and evaluation windows will produce a high adversarial AUC.

**The OOF MAE is a within-training-distribution estimate.** When adversarial AUC is
high, the true test error exceeds OOF by an amount that no within-training CV
technique can quantify, because the correction would require knowing the test
distribution — which is not available at training time. The pipeline partially
compensates by applying adversarial sample weights (rows resembling the validation
distribution receive higher training weight), but this is a bounded heuristic, not an
exact correction.

The pipeline reports OOF MAE as a within-distribution estimate only. It is **not**
presented as a prediction of leaderboard position.

### Decision 4 — What This Framework Does Not Claim

| Claim | Why it is withheld |
|-------|-------------------|
| OOF MAE ≈ leaderboard score | Leaderboard is out-of-sample; OOF is within-training-distribution. Distribution shift (adversarial AUC) drives a gap that training-side CV cannot quantify. The pipeline reports no expected-test figure. |
| The model predicts within-cell variation | The model predicts the expected value at each cell (group × period). Within-cell deviation is noise-floor variance from the model's perspective — not predictable signal. Targets dominated by noise at the cell level are effectively noise-floored regardless of modeling effort. |
| All categories contribute equally to the scored metric | When scoring uses MAE and category means differ substantially, high-magnitude categories dominate the score even when per-category error rates are uniform. The metric is a size-weighted aggregate, not a uniform one. |
| More Optuna trials guarantee better generalization | Optuna minimizes a CV-estimated objective. If the CV estimate carries residual optimism (e.g., from distribution shift), further tuning toward that objective can increase overfit rather than improve test performance. |
| The submission covers the full training distribution | Training uses all available overdose categories; only a subset of those categories appear in the submission template. The scored metric reflects performance only on the submission-covered subset. |

---

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

LightGBM and XGBoost appear in the CV-plan's `modeler_contract.interface_only_backends` slot but are **interface-only stubs that raise `NotImplementedError`** — they are not trained or scored. Active families are CatBoost (predictor) and Ridge (diagnostic only).

| Problem Subtype | Path | Active families (predictor / diagnostic) | Loss |
|---|---|---|---|
| continuous_regression | regression | CatBoost / Ridge | MAE |
| ordinal_regression | regression | CatBoost / Ridge | MAE |
| panel_forecasting | panel forecasting (with recursive lag option) | CatBoost / Ridge | MAE |
| binary_classification | classification | CatBoost classifier / Ridge | LogLoss |
| multiclass_classification | classification | CatBoost classifier / Ridge | LogLoss |

Ordinal regression uses the regression path (not classification) because:
- The target has natural ordering, so distance between values matters
- MAE penalizes by distance, treating predicting 5 vs 6 as small error and 5 vs 14 as large error
- Classification cross-entropy would treat both errors equally, losing the ordinal structure
- This matches how MAE-graded competitions evaluate such targets

**Ordinal post-processing.** After ensembling, predictions for `ordinal_regression` targets are rounded to the nearest valid integer class observed in training. The rounding threshold is optimized on OOF predictions: the modeler tries a small grid of offsets (–0.5 to +0.5 in steps of 0.1) and selects the offset that minimises OOF MAE before clipping to the training min/max. This converts continuous ensemble outputs to valid ordinal integers without sacrificing the distance-aware MAE objective. The chosen offset is logged in `model_results.json` under `ordinal_rounding`.

## Model Selection

The modeler uses a single predictor family: **CatBoost**. **Ridge** is trained as a linear diagnostic baseline but its predictions are never written to the submission. All multi-family ensemble, blend-gate, and family-competence logic was intentionally removed in favour of a leaner, leak-proof pipeline.

| Family   | Role                       | In submission? |
|----------|----------------------------|----------------|
| CatBoost | Sole predictor             | Yes            |
| Ridge    | Linear diagnostic baseline | No             |

**Hyperparameter tuning.** CatBoost is tuned with Optuna inside each outer fold of the nested CV; final predictions are aggregated across 5 random seeds (3 seeds inside the outer-fold scoring step — see [Per-Seed OOF Diagnostic](#per-seed-oof-diagnostic) for the observability-only artifact that exposes those per-seed arrays). Trial count is not fixed — each outer fold gets up to `OPTUNA_N_TRIALS` (8 in full mode) bounded by a per-fold time deadline derived from a global tuning budget (default 25 minutes total). The per-fold timeout is the remaining budget divided by the number of folds left, so later folds adapt to time already spent. If the budget is exhausted before an outer fold begins tuning, that fold **skips Optuna**, fits CatBoost with default hyperparameters, and still contributes its MAE to the OOF estimate — no fold is silently dropped for being late. `--debug` drops this to 1 trial per fold, a 60-second budget, and a 1-seed ensemble; debug OOF is not a valid score.

**Ridge diagnostic.** Ridge is fit on the walk-forward holdout with alpha selected via probe split from `{0.01, 0.1, 1.0, 10.0, 100.0}`, then used for two diagnostic outputs in `model_results.json`: a linear-baseline OOF MAE to compare against CatBoost, and the top-10 absolute coefficients as a linear-importance signal. Ridge does not participate in `predictions.csv`.

## Target Transform Selection

The modeler exposes a `--transform` flag with choices `{auto, none, log1p, sqrt}`; the default is `auto`. Under `auto`, the modeler runs a data-driven A/B over the three forward maps `{none, sqrt, log1p}` on the same walk-forward probe split used elsewhere in the run. For each candidate a CatBoost probe is fit on the probe-training portion with identical pipeline transform and probe hyperparameters — only the forward map of the target changes — and its predictions on the held-out probe portion are **inverse-transformed back to raw scale before the metric is computed**. The decision metric is the scored walk-forward MAE in raw target space, so candidates are always compared in the metric's own space rather than in any transformed space. The argmin across `candidates_mae` is selected as the run's transform.

**Selection is structural and measured, not hardcoded** to any one transform or dataset: the same A/B always runs and any candidate may win on any dataset. Target skewness is computed via `scipy.stats.skew` and **logged as context** for the decision, but it no longer drives the choice — optimising for de-skewing is not the same as optimising the scored metric. The explicit choices `--transform=none|log1p|sqrt` act as a manual override that skips the A/B entirely. The full decision — `candidates_mae`, `chosen`, `selection_metric` (`"raw_scored_wf_mae"`), `skew`, `manual_override`, and `requested` — is recorded in `model_results.json` under `transform_selection`.

## Scored-Category Optimization

The submission contract can restrict scoring to a subset of the categories present in training — for example, an aggregate / hierarchical target where only certain level-codes appear in `sample_submission.csv`. The modeler detects this generically: for every candidate column shared between `sample_submission.csv` and the training frame (group columns probed first, then any remaining submission column, with target / id / time columns excluded), the distinct submission values are compared to the distinct training values. A column qualifies as **scored-restricting** only when the submission values form a **strict proper subset** of the training values. Nothing about the column name or category labels is hardcoded — if no candidate column passes the strict-subset check, the pipeline defaults to treating all training rows as scored (no restriction).

When a scored subset is detected, the pipeline **trains on every training row** (the unscored categories carry signal for hierarchical aggregates and shared seasonal structure) but **optimizes and reports MAE only on the scored subset**. The scored-only mask flows through:

- The Optuna objective inside each outer fold, so hyperparameter selection targets the metric the leaderboard actually uses.
- The recursive-vs-static-imputation comparison, so the chosen lag-handling method wins on the scored slice rather than on the global average.
- The honest OOF MAE and per-fold MAE reports that propagate into `model_results.json` and the report.

`model_results.json` carries both views: `oof_mae` and `per_fold_maes` report the scored-only metric (the decision metric), while `oof_mae_all_categories` and `per_fold_maes_all_categories` are sibling diagnostics over every training row. `scored_categories`, `scored_category_column`, and `per_scored_category_oof_mae` are also written so downstream tools can break down performance by category.

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

## Recursive (Iterated) Forecasting

For panel forecasting problems, the modeler selects between two methods for handling unknown future lags at prediction time, and logs the choice in `model_results.json` under `lag_forecasting`.

**Static cycle-aware imputation** freezes unknown future lags at a fixed seasonal estimate derived from training history (the same cycle-position mean used during feature engineering). Every horizon step sees the same pre-computed lag value regardless of what earlier steps predicted.

**Recursive (iterated) forecasting** predicts the horizon one step at a time, feeding each step's prediction forward as the next step's lag input. After each step, all target-derived features (lags, rolling means, recent values, slope features) are recomputed using the updated prediction history. Covariate features, shift-aware features, seasonality encodings, and static baseline features are taken as-is from the original feature frame and do not recurse.

Recursive forecasting reuses the already-trained model objects — no additional training passes occur, only additional prediction passes. The same ensemble blend (families and weights) used in the static path is applied at each recursive step. A runaway-ceiling guard clips each step's prediction to a sane multiple of the training maximum to prevent feedback-loop amplification.

**Method selection is empirical.** Both methods are scored on a walk-forward holdout (the final `n_holdout_steps` training periods, withheld from feature fitting), including per-step MAE arrays. Recursive forecasting is kept only when its holdout MAE ≤ the static imputation holdout MAE; otherwise the pipeline falls back to imputation. The comparison MAEs, per-step arrays, and selected method are logged under `lag_forecasting` in `model_results.json` and reflected in Section 7 of `report.pdf`.

The benefit of recursive forecasting appears in the `lag_forecasting` holdout comparison and in true out-of-sample error. It does not appear in standard OOF or strict-CV metrics, because those are computed on training periods where lags are already known — both methods produce identical results there. The gap between the two methods only materialises when predicting genuinely unknown future values.

## Adversarial Validation

`feature_engineering.py` trains a binary CatBoost classifier (5-fold StratifiedKFold, 200 iterations) to distinguish training rows (label 1) from validation rows (label 0). The classifier's OOF AUC measures multivariate covariate shift that per-column KS tests may miss.

**Activation conditions**: at least 500 combined rows, at least 100 training rows, at least 100 validation rows, and at least 3 numeric covariate columns. Datasets that do not meet all four conditions skip adversarial validation silently.

**AUC < 0.55**: no meaningful shift detected; uniform sample weights are used.  
**AUC ≥ 0.55**: each training row receives weight `w = clip(1 − P(is_train), 0.1, 10.0)`, then normalized so the mean weight equals 1.0. Rows that "look like" validation rows receive higher weight, nudging the model toward the actual prediction distribution.

The weights are stored as an `adversarial_weights` column in `data/features_train.parquet`. The modeler reads this column and passes it as `sample_weight` to CatBoost (Optuna probe, WF probe, all 5 full-data retrain seeds) and to the Ridge diagnostic baseline. The top five shift-revealing features (by classifier importance), the AUC, and whether weights were applied are logged in `features.json` under `adversarial_validation` and propagated to `model_results.json` under `adaptive_choice.adversarial_validation`.

## Codebook-Based Date Features

For datasets where the time column uses opaque identifiers (e.g. base-64 hashes like `"uTjgI1Sv"`) instead of readable dates, `schema_analyst` looks for a JSON codebook file in `data/` mapping those IDs to calendar dates. The following filenames are checked in order: `period_id_codebook.json`, `period_codebook.json`, `period_id.json`, `dates.json`, `time_codebook.json`.

When a valid codebook is found, `profile.json` gains a `time_codebook` field that includes the filename, number of entries, detected mapping direction (`id_to_date` or `date_to_id`), and up to five sample pairs. `feature_engineering.py` then adds three calendar features: `month_of_year` (1–12), `quarter_of_year` (1–4), and `is_quarter_start` (1 if month ∈ {1, 4, 7, 10}). Rows with unmapped time values are imputed with the training-set median month. If more than 10% of training rows are unmapped, date features are skipped entirely and a warning is logged. When no codebook is present, this block is a silent no-op that leaves all other features unchanged.

## Robustness Safeguards

The pipeline includes three defensive rails that activate only under specific adverse conditions. On normally-structured datasets all three are complete no-ops. Their purpose is to convert potential catastrophic failures — an out-of-memory kill, a silently inverted train/val split, corrupted seasonal features — into safe, logged degradations with a working submission. They do not improve MAE on datasets that already run cleanly.

**Rail 1 — Memory / feature-budget gate (`feature_engineering.py`).**
Before generating the five expansive covariate feature families (extended rolling windows, covariate ratios, group-level covariate aggregates, slope features, entropy features), the pipeline estimates the projected extra column count and total cell count (`n_rows × projected_extra_cols`), and optionally reads available RAM via `psutil` (gracefully skipped when `psutil` is absent). If the estimate exceeds either named threshold (`FEATURE_BUDGET_CELL_THRESHOLD = 2,000,000,000` cells or `FEATURE_BUDGET_COL_THRESHOLD = 1,000` extra columns) or if available RAM is below 2.0 GB, those five families are skipped entirely and only the base target features (lags, rolling stats, group baselines, seasonality, encodings, and base covariate derivatives) are computed. On normal-sized datasets both estimates are well below the thresholds and the gate never fires. The decision — `downgraded` flag, estimates, skipped families, and skip reason — is logged in `features.json` under `feature_budget`.

**Rail 2 — Robust file and codebook detection (`profile_data.py`).**
Data file detection is case-insensitive. The highest-priority check is content: any CSV file containing the target column with fewer than 50% null values is routed to train regardless of its filename, and one with an entirely null target is treated as a held-out submission file. After the content check, filename substring keywords are used as a fallback: files whose names contain "train" or "fit" are assigned to train; files containing "val", "test", "eval", or "feature" are assigned to val. A second-pass content swap then re-examines files that reached val via filename heuristic — if any such file actually contains the target column with non-null values, it is corrected to train. This prevents a filename mismatch from silently swapping the splits. Codebook detection applies an analogous two-tier strategy: exact filename matches (case-insensitive) against the conventional list are tried first; any JSON file in `data/` whose stem contains "codebook", "period", "dates", or "time" qualifies as a fallback candidate. Every file-role assignment and its triggering reason are logged in `profile.json` under `file_role_log`.

**Rail 3 — Time-granularity ambiguity fallback (`feature_engineering.py`).**
Granularity (hourly / daily / weekly / monthly) is detected from the median inter-observation step within groups. For datetime time columns, the pipeline also measures cadence regularity: the fraction of inter-observation gaps equal to the modal gap (`< 0.60` → irregular) and the coefficient of variation of gap sizes (`> 0.50` → irregular). A cadence is considered ambiguous only when it is **both** irregular **and** within 12 hours of a classification boundary (2 h, 48 h, or 216 h). When ambiguous, cycle-specific sin/cos harmonics and cycle-modulo time features are replaced with a single granularity-agnostic relative-time position feature (`{time_col}_rel_pos` ∈ [0, 1]) that encodes no period assumption. A clean cadence near a boundary, or an irregular cadence far from any boundary, classifies normally. Integer time columns have no meaningful hour-unit boundary check and always classify as weekly by convention. The detected granularity, regularity metrics, and whether the fallback fired are logged in `features.json` under `time_granularity`.

## Constraints

- CPU only (no GPU)
- No network access or external data during analysis
- 2-hour wall-clock budget; 1M token budget (input + output, all agents combined)
- Single-predictor pipeline (CatBoost) with Ridge as a diagnostic-only linear baseline

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
- Stacking is not used. Target encoding **is** used and made leak-safe by per-fold fitting: a `TargetEncoderCV` (smoothed empirical-Bayes encoder with an inner KFold) is rebuilt inside every outer fold's adaptive Pipeline and `fit` only on that fold's training rows. The Ridge diagnostic consumes the encoded columns directly; CatBoost consumes the raw categoricals as `cat_features` and handles them via ordered boosting, so it sees no target-derived numerics. Unseen categories at transform time fall back to the global training mean.
- For panel forecasting, the modeler selects empirically between static cycle-aware imputation and recursive (iterated) forecasting for unknown future lags (see [Recursive (Iterated) Forecasting](#recursive-iterated-forecasting)). Even with recursive forecasting, compounding prediction error accumulates over long horizons; the holdout comparison measures — but cannot fully eliminate — this degradation.
- The holdout-selected lag method improves alignment between internal metrics and real test MAE, but a residual gap remains: OOF and strict-CV metrics are computed on training periods where lags are already known, so neither method's advantage is visible there. The gap manifests only on genuinely unknown future lags.
- On datasets with severe distribution shift, internal CV estimates may still underestimate true generalization error even after shift-aware features are applied, because the KS-weighted ensemble adjustments are bounded and cannot fully correct for extreme covariate drift.
- Expanded covariate features add substantial feature count, often doubling or tripling the feature set on datasets with many numeric covariates. Runtime increases proportionally during feature engineering and training. The benefit varies by dataset: datasets where existing features already capture the predictive signal show minimal change, while datasets with rich covariate information show meaningful improvement on strict CV.

- The pipeline uses a single predictor family (CatBoost). Cross-family ensembling was deliberately removed; variance reduction relies solely on the 5-seed CatBoost ensemble and the recursive-vs-imputation method selection.

- True multiclass classification (unordered categories like species or diagnosis codes) uses the CatBoost classifier path with Logloss. Probability aggregation across families is not implemented because the pipeline only has one predictor family. Ordinal regression targets (integer counts/scores with natural order) use the regression path correctly and are not affected by this limitation.

The `report.pdf` produced during analysis contains detailed methodology and results for the specific run.
