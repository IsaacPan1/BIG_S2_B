# Autonomous Data Analysis Pipeline (Award B)

An autonomous tabular forecasting / regression / classification pipeline built on
Claude Code. Given an unknown dataset in `data/`, it produces `submission.csv` and
`report.pdf` within a 2-hour budget with no human intervention after the trigger
prompt.

---

## 1. Overview

The pipeline is driven by a Claude Code session (the orchestrator) that sequences
six sub-agents plus one direct subprocess. Each stage writes a marker file
(`reports/{stage}_was_here.txt`) - the authoritative completion signal. All
inter-stage communication is through files; no sub-agent calls another directly.

The modeler (Step 3) is the exception to sub-agent dispatch: it runs as a **direct
blocking subprocess** (`python tools/run_modeler.py`), not through the Task tool.
This is intentional - see [Section 5](#5-pipeline-stages) for why.

---

## 2. Quick Start

1. Set up the environment - see [Section 3](#3-environment-setup).
2. Place the dataset in `data/` - see [Section 4](#4-required-data-structure).
3. Launch Claude Code from the repo root:
   ```bash
   claude --dangerously-skip-permissions
   ```
4. Prompt: `Do the data analysis.`
5. Wait. A full run is roughly 60-80 minutes end to end; the modeler is the long
   pole at ~46-58 min. The 2-hour wall-clock budget is a hard ceiling, not the
   expected runtime.
6. Collect `submission.csv` and `report.pdf` at the repo root.

> **Memory note.** The modeler is memory-bound. On machines with limited free RAM
> it will swap and slow dramatically - runs are several times slower under memory
> pressure. Run with several GB of free RAM for best performance.

---

## 3. Environment Setup

Python 3.11 required (specifically 3.11.x - the pipeline invokes the `python3.11`
binary; a 3.12-only system has no `python3.11` in PATH). The venv **must be created
from a Python 3.11 executable** so that `python3.11` exists in the venv's `bin/` -
all pipeline tools are invoked via `python3.11` explicitly (not bare `python`).

> **Required before running.** The pipeline's preflight gate calls
> `python3.11 tools/preflight_check.py`. If `python3.11` is absent or resolves
> to a version below 3.11, the pipeline hard-stops before any stage runs.
> The fix is always: create your venv from Python 3.11 (step below).

```bash
python3.11 -m venv /path/to/envs/award_b
source /path/to/envs/award_b/bin/activate   # Linux/macOS

python3.11 --version   # confirm 3.11.x

pip install \
  "pandas>=2.0" "numpy>=1.24" "scikit-learn>=1.3" "catboost>=1.2" \
  "optuna>=3.4" "matplotlib>=3.7" "reportlab>=4.0" "pydantic>=2.0" \
  pyarrow
```

Or, if `pyproject.toml` is in sync: `pip install -e .`

Verify the install:
```bash
python3.11 -c "import pandas, numpy, sklearn, catboost, optuna, matplotlib, reportlab, pydantic, pyarrow; print('all imports OK')"
```

**Shared clusters.** If pip complains about disk quota, point caches and the venv
at a work/scratch volume (`pip install --cache-dir /work/<you>/.pip-cache ...`).
**Launch Claude Code from inside the activated venv** so every stage uses the
same interpreter. All pipeline tools invoke `python3.11` explicitly - activation
alone is sufficient as long as the venv was created from Python 3.11.

---

## 4. Required Data Structure

### 4.1 Files

| File | Required | Role |
|------|----------|------|
| `DATA_DESCRIPTION.md` | **Required** | Problem statement, target column name, evaluation metric |
| `sample_submission.csv` | **Required** | Output format: exact column names and row IDs |
| Training CSV(s) | **Required** | Rows with the target column present and non-null |
| Validation / test CSV(s) | **Required** | Rows with the target absent or all-null |
| JSON time codebook | Optional | Maps opaque time IDs to real dates |
| Image sidecar files | Optional | `.png` files matched by group/time naming convention |

The pipeline **never reads** `data/_truth/` (reserved for local scoring only),
external URLs, or pretrained model weights.

### 4.2 Layout conventions

The profiler auto-detects one of three layouts, tried in order:

| Convention | Typical files |
|------------|---------------|
| **Split** | `covariates_train.csv` + `target_train.csv` + `covariates_val.csv` |
| **Combined** | `train.csv` (target embedded) + `val_features.csv` |
| **Kaggle subfolder** | `data/train/*.csv` + `data/test/covariates.csv` |

Detection is content-first: a CSV whose target column is present with <50% nulls
is classified as train; an all-null target column is treated as the submission
template. Filename keywords (train / val / test / sample / submission) serve as
fallback only.

### 4.3 Target column detection priority

1. Explicit `predict X` sentence in `DATA_DESCRIPTION.md`.
2. Column shared between training data and the submission template.
3. Heuristic scan of column names and dtype.

---

## 5. Pipeline Stages

The orchestrator runs Steps 0-5 in order. **Absence of a marker file while the
process is still alive is NOT a failure** - it means the stage is in progress.

| Step | Stage | Invocation | Marker |
|------|-------|------------|--------|
| 0 | Initialize `pipeline_run.json` | Orchestrator inline (Python) | - |
| 1 | `schema_analyst` | Task sub-agent | `reports/schema_analyst_was_here.txt` |
| 2 | `feature_engineer` | Task sub-agent | `reports/feature_engineer_was_here.txt` |
| 3 | Modeler - `tools/run_modeler.py` | **Direct blocking subprocess** | `reports/modeler_was_here.txt` |
| 3.5 | `validator` | Task sub-agent | `reports/validator_was_here.txt` |
| 3.6 | `critic` | Task sub-agent | `reports/critic_was_here.txt` |
| 4 | `submission_writer` | Task sub-agent | `reports/submission_writer_was_here.txt` |
| 5 | `report_writer` | Task sub-agent | `reports/report_writer_was_here.txt` |

**Why Step 3 is a direct subprocess.** Prior sub-agent dispatch caused the verify
gate to fire against half-written artifacts (agent returned before the script
finished) and left orphaned training processes running outside the orchestrator's
process tree. The direct subprocess makes the orchestrator the synchronous parent:
control returns only after the child has fully exited.

**Step 3.6 retune.** The critic may request at most one retune cycle (cap = 1,
enforced via `pipeline_run.json["retune_cap"]`). If triggered and both the cycle
cap and a 25-minute budget guard pass, the modeler, validator, and critic re-run
before proceeding to Step 4. **Step 0** initializes a fresh `pipeline_run.json`
(session start time, budget, modeler run ID, critic cycle count), overwriting any
leftover record.

---

## 6. Problem Types Supported

The profiler infers one of five subtypes from unique-value count, dtype, value
range, and `DATA_DESCRIPTION.md` keywords:

| Subtype | Detection criteria | CatBoost path | Metric |
|---------|--------------------|---------------|--------|
| `continuous_regression` | Float target or high unique count | CatBoostRegressor | MAE |
| `ordinal_regression` | Integer target, few consecutive ordered values; keywords: score / rating / severity / count | CatBoostRegressor; predictions rounded to nearest valid integer | MAE |
| `panel_forecasting` | Time axis + repeated group observations | CatBoostRegressor; walk-forward CV | MAE |
| `binary_classification` | Exactly 2 unique target values | CatBoostClassifier, Logloss | OOF MAE (= error rate; F1 used internally for threshold tuning) |
| `multiclass_classification` | 3-50 unordered unique values | CatBoostClassifier, MultiClass | OOF MAE (= error rate) |

When ambiguous between ordinal and multiclass (integer target, 5-15 values), the
pipeline defaults to ordinal regression to preserve MAE-evaluable behavior.

**CatBoost is the sole predictor in every submission.** A Ridge baseline runs as
a diagnostic only (logged in `model_results.json`) and never appears in
`predictions.csv` or `submission.csv`.

---

## 6.5 Cross-Validation Scheme

`scheme_analysis.py` is the sole authority for the CV plan: it classifies the
problem, selects a scheme, enumerates leakage risks, and writes a **frozen**
`reports/cv_plan.json`. The plan is marked immutable (`frozen: true`) and no
downstream stage may modify it - `load_cv_plan` asserts the freeze on every read.

**Scheme selection by structure** (not by problem-type label):

| Structure | Scheme |
|---|---|
| Time column + group columns | `TimeSeriesExpanding` (panel forecasting) |
| Time column only | `TimeSeriesExpanding` |
| Group columns, no time | `GroupKFold` |
| <=2 unique target values | `StratifiedKFold` |
| <=50 unique integer values | `StratifiedKFold` |
| Otherwise | `KFold` |

**Recent-vs-full drift gate (expanding vs sliding).** The default for any
time-series problem is an expanding window (all history). The pipeline switches
to a sliding window *only* when restricting training to recent periods
measurably reduces the train-validation gap.

The diagnostic deliberately avoids the naive approach (triggering sliding on high
KS or adversarial-classifier AUC). On a forecasting holdout, validation is a
future window, so those signals saturate - they read "large shift" by
construction whether or not recency would help, and would pick sliding almost
always, discarding history for no gain. Instead, the gate computes a
**standardized mean shift** (an effect size that does not saturate) per numeric
covariate, comparing full-train-vs-val against recent-train-vs-val, and measures
whether the recent window actually narrows the distance.

Sliding requires all three: `frac_improved >= 0.60` (share of features whose gap narrows with recency), `rel >= 0.25` (relative gap improvement), `n_features_scanned >= 12` (minimum evidence breadth).

Any failure - including a non-runnable diagnostic, too few periods, or no time
axis - falls back to expanding. Sliding is never the default; it must be
affirmatively justified. (Seasonality and time-index features are excluded from
the scan, since they separate train from val by construction rather than by drift.)

**Leakage controls.** The plan enumerates and mitigates leakage risks: future
timestamps (validator enforces `max(train_time)+gap = min(valid_time)`), group
overlap (disjoint groups for `GroupKFold`), target-derived features (recomputed
per-fold from training indices only), and ID-like columns (dropped). The
validator runs an independent strict re-audit to confirm the modeler's CV was
not optimistic.

---

## 6.6 Feature Families

`feature_engineering.py` organizes features into named families registered via
`_reg()`. Family membership is recorded in `reports/features.json` and consumed by
`family_ablation.py` (in the critic stage) for leave-one-family-out diagnostics.

**Panel path** (activated when `time_col` is present):

| Family | Contents |
|--------|----------|
| `group_encodings` | Label encoding of each group column |
| `seasonality` | Fourier sin/cos at 1 and 2 cycles of the detected period; sub-year harmonics when granularity supports them |
| `relative_time` | Relative temporal position within the full time range (`{time_col}_rel_pos`) |
| `group_baselines` | Per-group mean/std of the target at hour-of-day or day-of-week resolution (hourly/daily only) |
| `time_derived` | Time-column cycle index, quarter, month, linear trend |
| `date_features` | Calendar features (month_of_year, quarter_of_year, is_quarter_start) when a JSON codebook resolves opaque IDs to real dates |
| `lags` | AR lags of the target: core [1,2,3,4] always; long [8,12,26] and extended [52] when history allows (lag_k requires k+4 periods per group) |
| `rolling_mean` | Rolling mean of the target: core windows [4,8]; long [13,26] when history allows |
| `rolling_std` | Rolling std of the target, same windows as `rolling_mean` |
| `cov_lags` | One-period lag of each numeric covariate |
| `cov_deltas` | Period-over-period first difference of each numeric covariate |
| `cov_rolls` | Four-period rolling mean of each numeric covariate |
| `cov_rolls_ext` | Extended rolling windows (8, 13, 26-period) per covariate -- budget-gated |
| `cov_ratios` | Pair ratios within covariate prefix groups -- budget-gated |
| `cov_entropy` | Shannon entropy across covariate prefix groups -- budget-gated |
| `interactions` | Domain interactions (e.g. temperature^2) |
| `horizon` | Step-ahead distance from the last known observation |
| `covariates` | Raw numeric covariate pass-through |
| `image_features` | 21-23 spatial statistics per matched image sidecar (when present) |

**Cross-sectional path** (no `time_col`): registers `group_encodings`,
`interactions` (polynomial terms, domain composites, log1p transforms),
`time_derived` (age-derived flags when an "age" column exists), `covariates`,
`horizon` (constant 0), and `image_features` when sidecars are detected.

**Budget gate.** Before the expansive families (`cov_rolls_ext`, `cov_ratios`,
`cov_group_stats`, `slope_features`, `cov_entropy`), three checks run: extra
columns > 1,000; cells > 2e9; free RAM < 2.0 GB. Any trigger skips all five;
reason logged under `features.json -> feature_budget`.

**Image features.** When `.png` sidecars are present, `_extract_img_features()`
computes 21-23 spatial statistics: intensity moments, per-quadrant mean/std,
center-vs-edge ratio, brightness distribution, RGB color variance
(grayscale: 21; RGB: 23). No CNN or semantic embeddings.

---

## 6.7 Modeler Internals

### Scored-category filter

`run_modeler.py` Step 2.5 reads `data/sample_submission.csv` and detects "scored
filters": columns whose distinct submission values are a **strict proper subset**
of their training values. The decision metric (OOF MAE, Optuna objective, transform
A/B) is restricted to those scored rows; training still uses all rows. Results
logged in `model_results.json` under `scored_filters` and
`per_scored_category_oof_mae`.

### Transform auto-selection

The target transform is chosen by a data-driven A/B over three candidates
(`_TRANSFORM_CANDIDATES = ("none", "sqrt", "log1p")`). One CatBoost probe per
candidate runs on the walk-forward 80/20 split; predictions are inverse-mapped to
raw space before scoring. The winner has the lowest **scored WF MAE** (stored in
`model_results.json` as `transform_selection.chosen`). Target skewness is logged
but does not drive the choice -- optimizing de-skewing is not the same as
optimizing the scored metric. Classification always uses `"none"` and skips the
A/B. Manual override: `--transform={none,sqrt,log1p}`.

### 5-seed ensemble

The production model retrains with seeds `[42, 7, 123, 2024, 999]`
(`FINAL_RETRAIN_SEEDS`). Outer-fold CV uses a 3-seed subset `(42, 7, 123)`
(`OUTER_FOLD_FINAL_SEEDS`). Aggregation is `np.mean` by default; the critic's
retune path can request `np.median` (stored as `oof_per_seed.agg`). For
classification, the ensemble averages probability arrays before argmax.

### Adversarial weighting

`feature_engineering.py` trains a 5-fold CatBoost classifier (train=1, val=0) on
numeric covariates. OOF AUC < 0.55 -- no weights. AUC >= 0.55: weights =
`clip(1 - P(train | row), 0.1, 10.0)` normalized to mean 1.0, stored as
`adversarial_weights` in `data/features_train.parquet`. Consumed at three points:
(1) transform A/B probe fits; (2) every CatBoost fit in nested CV and final
retrain; (3) WF-split residual weighting in level correction (Section 6.9).

---

## 6.8 Multi-Ruler MAE Design

The pipeline uses five distinct MAE estimates that serve different purposes.
Using a single estimate for all decisions would conflate optimization pressure
with evaluation quality.

| Estimate | Stored in | What it drives |
|----------|-----------|----------------|
| `oof_mae` (scored OOF) | `model_results.json` | Optuna objective; critic/validator benchmark; the pipeline's primary reported metric |
| `oof_mae_all_categories` | `model_results.json` | Diagnostic -- shows gap between scored subset and full population |
| `walk_forward_mae_scored` (`probe_mae_80_20_scored`) | `model_results.json` | Transform A/B winner selection; never reported as the pipeline's final score |
| `lag_forecasting.recursive_holdout_mae` | `model_results.json` | Recursive vs imputation method selection (Section 6.9) |
| `strict_cv_mae` | `validator_review.json` | Independent re-run by the validator (no modeler artifacts); audits OOF for optimism; gap > 25% triggers CRITICAL |

---

## 6.9 Forecast Method and Level Correction

### Recursive vs imputation method selection

For `panel_forecasting` datasets, Step 12 of `run_modeler.py` compares two
strategies on the walk-forward holdout:

- **Imputation**: last known target used as constant lag seed for all steps.
- **Recursive**: each prediction fed forward as the lag for the next step.

Both run in full. Recursive is chosen only when valid **and** lower scored
holdout MAE. Per-step MAE stored in `lag_forecasting` under
`per_step_mae_recursive` / `per_step_mae_imputation`; method in
`lag_forecasting.method_used`.

### Level correction (post-hoc bias adjustment)

After the final ensemble is assembled, a bias estimate is computed from the
walk-forward **probe** model's residuals (not the 5-seed production model, to
avoid in-sample collapse). Residuals are weighted by `adversarial_weights` from
the WF split to focus on validation-like rows. Per-group estimates are used when
a group has >= 30 WF holdout rows (`MIN_GROUP_HOLDOUT = 30`), shrunk toward the
global with lambda = 0.5 (`BIAS_CORRECTION_LAMBDA`); smaller groups fall back to
the global. The correction is applied only when weighted holdout MAE improves by
> 1.5% of baseline (`BIAS_CORRECTION_REL_MARGIN = 0.015`). Outcome stored in
`model_results.json` under `level_correction`. Skipped for classification.

---

## 6.10 Validation and Critic Quality Control

Steps 3.5 and 3.6 form an independent quality-control layer. The modeler writes
its artifacts and exits; the validator and critic then read those artifacts and
issue verdicts the modeler never sees. Neither stage can block submission --
their verdicts surface in the report and may trigger at most one modeler retune.

### Validator -- independent CV re-audit (validate.py)

The validator re-runs CatBoost using the modeler's best hyperparameters (from
`model_results.json`) but builds its own folds from scratch, independent of the
modeler's fold state.

**Strict CV scheme** (validate.py:181-200):

| Dataset structure | Scheme |
|---|---|
| `time_col` present | 4-fold purged+embargoed walk-forward; 2 periods purged at each boundary (`N_STRICT_FOLDS=4`, `EMBARGO_PERIODS=2`) |
| `group_cols`, no time | GroupKFold on first group_col; folds auto-reduced to min(4, n_unique_groups) |
| Neither | 80/20 sequential split |

**Three named checks** (validate.py:330-373):

| Check | Trigger | Verdict |
|---|---|---|
| `cv_integrity` | `(strict_mae / reported_mae - 1) > CV_GAP_CRITICAL_PCT` (0.25) | CRITICAL |
| `cv_integrity` | same gap `> CV_GAP_WARNING_PCT` (0.10) | WARNING |
| `cv_integrity` | classification problem | always PASS (regression-calibrated) |
| `importance_concentration` | max feature share `> IMPORTANCE_WARNING_SHARE` (0.50) | WARNING |
| `leakage_two_signal` | any two-signal suspect (see below) | CRITICAL |

**Two-signal leakage detection** (validate.py:487-511):

Leakage is flagged only when BOTH a statistical AND a structural signal fire on
the same feature. Neither alone is sufficient (validate.py:655-660).

- **Statistical signal**: importance share `>= LOO_CANDIDATE_SHARE` (0.30) AND
  `|loo_delta| / strict_mae < LOO_STAT_SIGNAL_THRESHOLD` (0.05). Near-zero LOO
  delta means removing the feature does not degrade strict-CV MAE -- it earns
  its importance through a channel other than predictive generalization.
- **Structural signal**: feature name matches any of six `STRUCTURAL_PATTERNS`
  regexes (validate.py:53-60): `_lead_`, `_future`, `_t+N`, `_lag0`/`lag0`,
  `_leak_`; or the target column name appears in the feature name without a
  lag/window suffix.

`suspect = (stat_signal AND struct_signal)` (validate.py:499). `leakage_two_signal`
fires CRITICAL if any feature is suspect.

**Gap attribution** (gap_attribution.py:46-140):

After strict CV, `gap_attribution.py` classifies the OOF->strict gap and appends
a `gap_attribution` block to `validator_review.json`.

| Classification | Condition |
|---|---|
| `CV_SCHEME` | Latest fold MAE within `LATEST_FOLD_MATCH_PCT` (0.15) of reported MAE -- expanding-window scheme pessimism, not overfit |
| `REAL_DIVERGENCE` | Latest fold MAE diverges > 15%; gap unexplained by scheme |
| `UNKNOWN` | Fewer than 2 folds or `reported_mae = 0` |

Monotonicity check: `MONOTONE_MIN_FRACTION` (0.60) of consecutive fold pairs
(sorted by ascending training size) must show more-data->lower-MAE for
high-confidence `CV_SCHEME`. `CV_SCHEME` also assigned when only `latest_match`
holds, at lower confidence (gap_attribution.py:124-131).

The critic consumes this: `CV_SCHEME` downgrades the validator's CRITICAL to
WARNING and WARNING to PASS (run_critic.py:80-87).

### Critic -- quality checks + retune decision (run_critic.py)

The critic runs 5-6 checks (`prediction_collapse` fires for classification only),
then accepts or requests a retune.

**Six named checks** (run_critic.py:64-198):

| Check | Trigger | Severity |
|---|---|---|
| `validator_concordance` | Validator verdict pass-through. `CV_SCHEME` downgrades CRITICAL->WARNING, WARNING->PASS. `REAL_DIVERGENCE`/`UNKNOWN`: no downgrade. | WARNING or CRITICAL |
| `prediction_distribution` | Regression only. Mean bias `> PRED_MEAN_BIAS_CRITICAL` (0.30) of `|train_mean|` -> CRITICAL; `pred_std < PRED_STD_CRITICAL` (0.40) `* train_std` -> CRITICAL; `pred_std < PRED_STD_WARNING` (0.65) `* train_std` -> WARNING. Skipped for classification. | WARNING or CRITICAL |
| `prediction_collapse` | Classification only. All predictions degenerate to one label (`len(unique)==1`). | WARNING |
| `mae_plausibility` | `wf_mae < WF_MAE_SUSPICIOUS_LOW` (0.03) `* train_std` AND validator != PASS -> WARNING; `wf_mae > WF_MAE_POOR_RATIO` (0.80) `* train_std` -> WARNING. | WARNING |
| `feature_concentration` | Top feature's share of top-10 importance `> TOP_FEATURE_SHARE_WARNING` (0.50) -> WARNING. Requires >= 10 features. | WARNING |
| `prediction_sanity` | NaN predictions or negatives in non-negative target -> CRITICAL. `pred_max > PRED_MAX_HIGH_RATIO` (3.0) `* train_max` or `< PRED_MAX_LOW_RATIO` (0.40) `* train_max` -> WARNING. | WARNING or CRITICAL |

**Acceptance rule** (run_critic.py:232-248): WARNINGs are accepted.
`CV_GAP_CRITIC_ACCEPT = 0.15` (line 12) documents the design: the critic accepts
the 10-25% WARNING band but not a CRITICAL gap, unless `CV_SCHEME` attribution
suppresses it. Any CRITICAL triggers `retune_requested` -- first cycle only.

**Family ablation trigger** (run_critic.py:200-230): First cycle only. Runs
`tools/family_ablation.py` (subprocess, timeout=300s). If `net_harmful_families`
is non-empty AND no CRITICALs are present, `retune_reason = "ablation"`. The
CRITICAL path has priority for the single retune slot (line 240). Ablation
failure is non-fatal.

**Retune cap** (run_critic.py:24, 248): Second-cycle detection via marker
`reports/critic_retune_attempted.txt`. On the second cycle all checks run but the
outcome is force-accepted regardless. Cap = 1.

---

## 7. Imbalanced Classification

When minority class fraction falls below 10% (`IMBALANCE_THRESHOLD = 0.10`),
four stages activate:

1. **Detection.** `profile_data.py` sets `is_imbalanced = True` in `profile.json`.
2. **Class weights.** `auto_class_weights = "Balanced"` on every CatBoost fit
   (training folds, Optuna inner folds, production retrain).
3. **Probability averaging.** The seed ensemble averages class probability arrays
   before argmax rather than majority-voting hard labels.
4. **Threshold tuning (binary only).** Sweeps thresholds on OOF class-1
   probabilities; applies the F1-optimal threshold to validation predictions only
   when OOF F1 gain >= 0.02 absolute. Multiclass uses argmax directly.

---

## 8. Hard Constraints

| Constraint | Value |
|------------|-------|
| Wall-clock budget | 2 hours |
| Token budget | 1,000,000 input + output combined, all agents |
| GPU | Not available - CPU only |
| External data | None - no downloads, no web search, no API calls, no pretrained weights |
| Retune cap | 1 retune cycle maximum per pipeline run |
| Submission | Must always be written, even on failure |

---

## 9. Output Files

| File | Location | Description |
|------|----------|-------------|
| `submission.csv` | repo root | Graded deliverable: exactly two columns - `row_id` and the target column. Row count matches `sample_submission.csv`. |
| `submission_with_cov.csv` | repo root | Full diagnostic copy: all columns from `sample_submission.csv` plus the target. Same predictions as `submission.csv`, not scored. |
| `report.pdf` | repo root | Methodology report with feature importance, prediction diagnostics, limitations |
| `reports/profile.json` | `reports/` | Problem type, column roles, imbalance flags, KS shift results |
| `reports/schema_analysis.md` | `reports/` | Human-readable schema summary and CV plan narrative |
| `reports/features.json` | `reports/` | Feature families built, adversarial validation AUC, budget gate log |
| `reports/model_results.json` | `reports/` | OOF MAE, feature importance, transform selection, threshold tuning record |
| `reports/predictions.csv` | `reports/` | Raw validation predictions (before submission formatting) |
| `reports/oof_predictions.csv` | `reports/` | Out-of-fold predictions from training folds |
| `reports/validator_review.json` | `reports/` | Strict-CV MAE, fold MAEs, gap attribution |
| `reports/critic_review.json` | `reports/` | 5-check quality review result and retune decision |
| `reports/submission_summary.json` | `reports/` | Round-trip audit result for `submission.csv` |
| `reports/*_completion.json` | `reports/` | Per-stage completion records: status, exit code, artifact paths, modeler run ID |

`pipeline_config.json` at the repo root is regenerated each run and is not tracked
by git.

---

## 10. Robustness Safeguards

### Per-stage failure handling

| Stage fails | Severity | Orchestrator action |
|-------------|----------|---------------------|
| `schema_analyst` | **Fatal** | Write baseline submission (group-mean if target identifiable, else zeros matching sample shape) + minimal one-page `report.pdf`. Stop. |
| `feature_engineer` | Non-fatal | Fall back to raw covariates; continue to modeler |
| Modeler | Non-fatal | Compute group-mean predictions from training target; write to `reports/predictions.csv`; continue to `submission_writer` |
| `report_writer` | Non-fatal | Write minimal one-page `report.pdf` directly via reportlab; if reportlab unavailable, write `report.txt` and log the degradation |

The validator and critic are diagnostic-only - their failure or an adverse verdict
never blocks submission.

### Code-level defensive rails

**Memory / feature-budget gate.** Before building the five expansive covariate
families, `feature_engineering.py` checks projected column count, cell count, and
available RAM (thresholds: > 1,000 columns; > 2e9 cells; < 2.0 GB free). Any
trigger skips all five families and logs the reason under
`features.json -> feature_budget`.

**Additional edge-case protections.** Two-phase ID detection (name-match then
dtype-gated near-unique scan); dtype-gate blocking ID-like integers from the
categorical encoder; content-first file routing (non-null target = train;
all-null = template; second-pass swap for mislabeled splits); submission
round-trip audit verifying every `row_id` maps to a real prediction.

---

## 11. Recent Changes

See `git log --oneline` for full history. The `custom-cv` branch adds the CV
scheme analysis and drift gate (Section 6.5), the 16-family feature engineering
framework (Section 6.6), shift-aware level correction (Section 6.9), and full
classification routing with 4-stage imbalance handling (Section 7).

---

## 12. Limitations

**Distribution shift.** OOF and strict-CV MAE are within-training-distribution
estimates. Under severe train-val shift (high adversarial validation AUC), true
test error can exceed these by an amount no within-training CV can quantify.
Shift-aware weighting and level correction partially compensate but are bounded
heuristics.

**Single model family.** No cross-family ensembling; variance reduction comes
from the 5-seed CatBoost ensemble, target-transform selection, and
recursive-vs-imputation method selection.

**Long-horizon compounding.** Recursive forecasting reduces but cannot eliminate
error accumulation over many steps. OOF and strict-CV do not reveal it because
lags are known in training folds.

**File layout detection.** Only three layout conventions are auto-detected.
Unusual file structures fall back to heuristics that may misclassify train vs
validation.

**Image features.** When image sidecars are present, 21-23 spatial statistics
are computed via PIL/numpy (see Section 6.6). No CNN or GPU required; semantic
content is not captured.
