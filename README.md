# Autonomous Data Analysis Pipeline

An autonomous data-analysis pipeline built on Claude Code. Given an unknown tabular
dataset placed in `data/`, it produces `submission.csv` and `report.pdf` within a
2-hour budget, with no human intervention. It handles forecasting, regression, and
classification problems, detecting the problem type and adapting its feature
engineering, cross-validation, and modeling accordingly.

---

## 1. Quick Start

1. **Set up the environment** (one time) — see [Section 2](#2-environment-setup).
2. Place your dataset in `data/` along with `DATA_DESCRIPTION.md` and
   `sample_submission.csv`.
3. Launch Claude Code from the repo root:
   ```
   claude --dangerously-skip-permissions
   ```
4. Prompt: `Do the data analysis.`
5. Wait for completion. A full run is roughly **35–65 minutes** end to end: the
   modeler (5-fold nested CV + Optuna + 5-seed ensemble) is the long pole at
   ~25–45 min on a machine with adequate RAM, and the remaining stages
   (schema, features, validation, critic, report) add ~10–20 min.
6. Collect `submission.csv` and `report.pdf` at the repo root.

> **Resource note.** The modeler is memory-bound. On a machine with little free RAM
> it will swap and slow dramatically (a run that takes ~7 min with headroom can take
> over an hour while swapping). Run with several GB of free RAM for best results; the
> pipeline degrades gracefully under memory pressure (see
> [Robustness Safeguards](#robustness-safeguards)) but runs much faster with headroom.

---

## 2. Environment Setup

The pipeline needs Python 3.11+ and a handful of scientific packages. The most
reliable setup is a dedicated virtual environment (avoids polluting a base
environment and sidesteps conda/quota issues on shared clusters).

### 2.1 Create and activate a venv

```bash
# create the environment (use any path you like)
python3 -m venv /path/to/envs/award_b
source /path/to/envs/award_b/bin/activate      # Linux/macOS
#  .\path\to\envs\award_b\Scripts\Activate.ps1  # Windows PowerShell

python --version    # confirm 3.11+
```

### 2.2 Install dependencies

If the repo's `pyproject.toml` dependency list is correct, the simplest install is:

```bash
pip install -e .
```

Otherwise install the packages directly (this is the known-good set):

```bash
pip install \
  "pandas>=2.0" "numpy>=1.24" "scikit-learn>=1.3" "catboost>=1.2" \
  "optuna>=3.4" "matplotlib>=3.7" "reportlab>=4.0" "pydantic>=2.0" \
  pyarrow
```

> `pyarrow` is required for reading/writing the `.parquet` feature files even though
> it is not always listed as a direct dependency — install it explicitly.

### 2.3 Verify

```bash
python -c "import pandas, numpy, sklearn, catboost, optuna, matplotlib, reportlab, pydantic, pyarrow; print('all imports OK')"
```

### 2.4 Notes for shared clusters

- If `pip`/conda complain about **disk quota**, point caches and the environment at
  a work/scratch volume rather than your home directory
  (e.g. `pip install --cache-dir /work/<you>/.pip-cache ...`, and create the venv
  under `/work/...`). Home directories on HPC systems are frequently too small.
- Launch Claude Code from **inside** the activated venv so every stage uses the same
  interpreter. If sub-stages can't find packages, ensure they invoke the venv's
  Python explicitly rather than a system `python3`.

---

## 3. Inputs and Outputs

### Inputs — place in `data/` before running

- `DATA_DESCRIPTION.md` — describes the problem, target variable, and evaluation
  metric. (Detection is case-insensitive.)
- Training and validation data files (see layout conventions below).
- `sample_submission.csv` — defines the expected output format (columns, row count).
- *(Optional)* Image sidecar files — see [Known Limitations](#known-limitations).
- *(Optional)* A JSON codebook mapping opaque time-IDs to dates — see
  [Codebook-Based Date Features](#codebook-based-date-features).

The pipeline auto-detects one of three file-layout conventions:

| Convention | Files |
|------------|-------|
| **Split**  | `covariates_train.csv` + `target_train.csv` + `covariates_val.csv` |
| **Combined** | `train.csv` (target embedded) + `val_features.csv` |
| **Kaggle subfolder** | `data/train/*.csv` + `data/test/covariates.csv` |

### Outputs — written to the repo root

- `submission.csv` — predictions matching `sample_submission.csv`.
- `report.pdf` — multi-section methodology report with feature-importance charts and
  a limitations section.

---

## 4. Architecture

Seven sub-agents run in sequence, each in its own context window. A marker file
(`reports/{agent}_was_here.txt`) confirms each sub-agent actually ran and was not
inlined by the orchestrator.

| Sub-agent | Role |
|-----------|------|
| `schema_analyst` | Discovers dataset structure, problem type, distribution shifts, and optional time codebook (`tools/profile_data.py`); authors the frozen CV plan (`tools/scheme_analysis.py`). Writes `profile.json`, `schema_analysis.md`, `cv_plan.json`. |
| `feature_engineer` | Generates features adapted to schema, time granularity, distribution shift, and adversarial validation (`tools/feature_engineering.py`). |
| `modeler` | Trains the model (`tools/run_modeler.py`). CatBoost is the sole predictor; Ridge runs as a linear diagnostic only. Nested CV from the frozen plan, Optuna tuning inside each fold, 5-seed final ensemble, empirical recursive-vs-static lag handling. |
| `validator` | Independent strict-CV audit via purged walk-forward (`tools/validate.py`). Diagnostic only — never blocks. |
| `critic` | 5-check quality review with at most one retune request (`tools/run_critic.py`). |
| `submission_writer` | Validates format and writes `submission.csv` (`tools/build_submission.py`). |
| `report_writer` | Assembles `report.pdf` (`tools/generate_report.py`). |

**Design principle.** All substantive logic lives in deterministic Python tools; the
sub-agents orchestrate and make the dataset-dependent *decisions* (problem-type
inference, feature-family selection, accept/retune judgment). On a dataset that
resembles the tools' assumptions these decisions look obvious; their value is in
generalizing to datasets the tools were not hand-tuned for.

---

## 5. How It Works

### 5.1 Problem-Type and Subtype Detection

The pipeline first classifies the broad problem type, then a finer subtype that
selects the modeling path.

**Subtypes:**

- **continuous_regression** — float target or many unique values.
- **ordinal_regression** — integer target, few consecutive ordered values
  (counts, severity scores, 1–5 ratings). Evaluated by MAE, *not* classification.
- **panel_forecasting** — time-indexed regression with group structure.
- **binary_classification** — exactly 2 unique target values.
- **multiclass_classification** — 3–50 unordered target values.

Subtype is inferred from unique-value count, dtype, value range/consecutiveness, and
`DATA_DESCRIPTION.md` keywords (days/score/count/rating/severity suggest ordinal).
When ambiguous between ordinal and multiclass (integer target, 5–15 values), it
defaults to ordinal regression for MAE-evaluable behavior.

### 5.2 Model Selection and Tuning

**CatBoost is the sole predictor. Ridge is a diagnostic-only linear baseline** whose
predictions never reach the submission. Cross-family ensembling was deliberately
removed in favor of a leaner, leak-proof pipeline; variance reduction comes from the
5-seed CatBoost ensemble and the recursive-vs-imputation method selection.

The CatBoost class used is dataset-driven: continuous_regression, ordinal_regression,
and panel_forecasting subtypes use CatBoostRegressor with MAE; binary_classification
uses CatBoostClassifier with Logloss; multiclass_classification uses CatBoostClassifier
with MultiClass loss. The validator and submission writer follow the same routing —
metric thresholds suppress for classification (gap thresholds are MAE-calibrated), and
the submission validator checks predicted class labels are valid members of the
training label set.

| Family | Role | In submission? |
|--------|------|----------------|
| CatBoost | Sole predictor | Yes |
| Ridge | Linear diagnostic baseline | No |

*(LightGBM/XGBoost appear in the CV plan's `interface_only_backends` slot but are
stubs that raise `NotImplementedError` — never trained.)*

**Ensemble path by subtype:**

| Subtype | Loss |
|---------|------|
| continuous_regression / ordinal_regression / panel_forecasting | MAE |
| binary / multiclass classification | LogLoss |

Ordinal regression uses the *regression* path (distance-aware MAE) rather than
classification, then rounds predictions to the nearest valid integer class — the
rounding offset is optimized on OOF predictions and logged under `ordinal_rounding`.

**Hyperparameter tuning.** Optuna runs inside each outer fold (up to `OPTUNA_N_TRIALS`,
8 in full mode), bounded by a per-fold deadline from a global tuning budget (default
25 min). Folds whose budget is exhausted fit CatBoost with default hyperparameters and
still contribute their MAE — no fold is silently dropped. `--debug` drops to 1 trial,
a 60-second budget, and 1 seed; **debug OOF is not a valid score.**

**Ridge diagnostic.** Fit on the walk-forward holdout with alpha selected from
`{0.01, 0.1, 1, 10, 100}`. Produces a linear-baseline OOF MAE and top-10 coefficients
as a linear-importance signal. Never in `predictions.csv`.

### 5.3 Target Transform Selection

`--transform {auto, none, log1p, sqrt}` (default `auto`). Under `auto`, a data-driven
A/B fits a CatBoost probe under each of `{none, sqrt, log1p}` on the same walk-forward
split, **inverse-transforms predictions back to raw scale before scoring**, and picks
the argmin scored MAE. The choice is structural and measured — skewness is logged as
context but does not drive the decision. Explicit `--transform=none|log1p|sqrt`
overrides the A/B. Full record in `model_results.json → transform_selection`.

### 5.4 Scored-Category Optimization

When `sample_submission.csv`'s categories form a strict proper subset of the training
categories (e.g. a hierarchical target where only some level-codes are scored), the
pipeline **trains on all rows** (unscored categories carry shared signal) but
**optimizes and reports MAE only on the scored subset**. The scored mask flows through
the Optuna objective, the recursive-vs-static comparison, and the reported OOF.
Detection is generic (strict-subset check on shared columns, no hardcoded names);
if nothing qualifies, all rows are treated as scored. `model_results.json` carries
both scored (`oof_mae`) and all-category (`oof_mae_all_categories`) views.

### 5.5 Recursive (Iterated) Forecasting

For panel forecasting, the modeler chooses empirically between two ways to handle
unknown future lags:

- **Static cycle-aware imputation** — freezes unknown lags at a seasonal estimate.
- **Recursive forecasting** — predicts one step at a time, feeding each prediction
  forward; target-derived features (lags, rolling means, slopes) are recomputed each
  step. Covariate, shift-aware, seasonality, and static-baseline features do not
  recurse. A ceiling guard clips runaway feedback.

Both are scored on a walk-forward holdout; recursive is kept only if its holdout MAE
≤ static's. It reuses trained models (extra prediction passes, no extra training).
**Note:** this benefit does *not* appear in OOF/strict-CV (those are on training
periods with known lags) — only on genuinely unknown future values. The ordinal time
column is derived from the dataset's time_col (`f"{time_col}_ord"`), so recursion
engages on any panel dataset regardless of the time column's name — not only when
that column happens to be called period_id. Logged under `lag_forecasting`.

### 5.6 Post-Hoc Level Correction (Shift-Aware, Gated)

After final val predictions are produced, an optional level correction estimates and
applies a shift-aware bias adjustment. On the walk-forward holdout (where labels are
available because it's a training slice), the modeler computes per-row residuals from
the out-of-sample probe model and weights them by the adversarial-validation density
ratio (P(val-like)) to estimate the global bias the model carries under the train→val
covariate shift. A per-group bias estimate is computed for groups with at least
MIN_GROUP_HOLDOUT rows and shrunk toward the global estimate (hierarchical pooling,
weight LAMBDA).

The correction is gated: it is applied to final predictions ONLY if it reduces the
weighted holdout MAE by more than BIAS_CORRECTION_REL_MARGIN × wf_wmae_before. On
datasets with minimal shift or no in-train-visible bias, the gate does not fire and
the correction is a no-op. The correction is regression-only (skipped for
classification subtypes) and uses the OOS probe model — never the in-sample
production fit. Logged under model_results.json → level_correction with global bias,
per-group bias, before/after holdout MAE, and the applied flag.

Important limit: the correction can only catch bias that is visible on the in-train
walk-forward holdout. Pure out-of-distribution shift that does not surface in-train is
not honestly correctable from training data alone.

---

## 6. Cross-Validation

### 6.1 Frozen CV Contract

CV is authored exactly once per run. `schema_analyst` invokes `scheme_analysis.py` —
the **only** module permitted to write `cv_plan.json` — which selects a scheme from
the inferred problem type (time series → `TimeSeriesExpanding`; tabular IID →
`GroupKFold`/`StratifiedKFold`/`KFold`) and marks it `frozen: true`. No downstream
agent may modify it; the critic can request **at most one** replan via `CV_INVALID`,
which deletes the plan and re-invokes `schema_analyst` once.

`cv_engine.CVEngine` materializes the plan into deterministic folds and enforces
leakage invariants (no index overlap; `max(train_time) + gap ≤ min(valid_time)` for
time series; disjoint groups for `GroupKFold`). The modeler's outer folds come
directly from this plan.

**End-anchored expanding window.** For `TimeSeriesExpanding`, the last fold's
validation block ends at the final time index *T*, mirroring the real train → test
boundary. Training is always `[0:train_end]`, so it expands as the fold index advances.

**Rank-resolved time axis.** When the time column is an opaque codebook-resolved ID,
`attach_period_rank` builds a dense integer rank used for all CV math; the raw hash
remains the join/submission key.

**Drift gate (expanding vs sliding).** Defaults to expanding; switches to sliding
**only** when a recent-vs-full drift diagnostic shows the shift is recency-reducible —
all three gates must hold: `frac_improved ≥ 0.60`, `rel ≥ 0.25`,
`n_features_scanned ≥ 12`. Every fallback path returns expanding. Uses validation
*covariates* only — the validation target is never read.

### 6.2 Cross-Validation Decision Record

This section documents the **selection criteria** — the rules the pipeline applies to
*any* dataset to choose a problem type and CV scheme. It is not a record of one run's
outcome: the same logic runs on every dataset, and a different input selects a
different branch. For a given run, the branch actually taken and the evidence behind it
are re-derived from `cv_plan.json`, `features.json`, and `validator_review.json` and
surfaced in `report.pdf`. Nothing is hardcoded to a particular dataset.

**Decision 1 — Problem Type.** Inferred from the presence of a time axis and group
structure, plus target characteristics. The pipeline selects the branch whose
preconditions the data satisfies:

| Branch | Selected when | Why, and what the alternatives would cost |
|--------|---------------|-------------------------------------------|
| `plain_regression` / classification (IID) | No ordered time column is detected | Rows are exchangeable, so random/stratified k-fold is valid. Imposing a time-series scheme here would waste folds on a non-existent ordering; treating it as panel would invent group structure that isn't there. |
| `univariate_time_series` | Time column present, **no** usable group structure | A single ordered series; expanding/sliding CV applies. Pooling across non-existent groups is impossible; IID k-fold would leak future into past. |
| `panel_forecasting` | Time column present **and** repeated observations across multiple groups | Cross-group signal (shared seasonality, shared covariate effects) is poolable while temporal order is preserved. Treating it as IID regression would leak future→past; treating each group as an isolated series would discard the cross-group signal and lower per-group signal-to-noise. |

The decision is evidence-driven: a regular time cadence is confirmed from the median
inter-observation step and its regularity (`features.json → time_granularity`); group
structure is confirmed from repeated (group, time) keys (`cv_plan.json →
group_columns`). When no time axis is found, the pipeline falls to the IID branch and
chooses a non-temporal CV scheme — it does **not** force temporal machinery onto
non-temporal data.

**Decision 2 — CV Scheme.** Follows directly from the problem type, then a data-driven
refinement for the time-series case:

| Problem type | CV scheme chosen | Rationale |
|--------------|------------------|-----------|
| IID regression / classification, **with** group columns | `GroupKFold` | Keeps each group entirely within one fold so the model is tested on unseen groups, not memorized ones. |
| IID classification, discrete target, no groups | `StratifiedKFold` | Preserves class balance across folds for a stable estimate. |
| IID regression, no groups | `KFold` | Standard random partition; valid because rows are exchangeable. |
| Time series (univariate or panel) | `TimeSeriesExpanding` (default) or `TimeSeriesSliding` | Respects the arrow of time; the last fold is anchored at the final period *T*, mirroring the real train→submission boundary. |

For the time-series branch, **expanding vs sliding** is itself a data-driven decision,
not a fixed choice. The pipeline defaults to expanding (use all history) and switches to
sliding (fixed recent window) **only** when a recent-vs-full drift diagnostic shows the
train→val shift is *recency-reducible* — i.e. restricting training to recent periods
measurably narrows the covariate gap. All three gates must hold simultaneously:
`frac_improved ≥ 0.60`, `rel ≥ 0.25`, `n_features_scanned ≥ 12`; every fallback path
returns expanding. This means older data is discarded only when there is breadth of
evidence that it has become unrepresentative — never by default.

**Why these schemes and not the simpler defaults**, in general terms:

| Rejected default | Why it is avoided (when a better-fitting scheme applies) |
|------------------|----------------------------------------------------------|
| Random k-fold on ordered data | Trains on future periods to predict past ones — leaks the answer and produces an optimistic estimate that does not transfer to a forecast. |
| Single train/test holdout | High-variance estimate dependent on one split point; gives no fold-ensemble variance reduction and no evidence of how error scales with training size. |
| Group-blind k-fold when groups exist | Lets the model memorize group identity, inflating the score relative to performance on unseen groups. |

**Leakage controls applied to every scheme.** `cv_engine.CVEngine` enforces no
train/valid index overlap; for time-series schemes it enforces
`max(train_time) + gap ≤ min(valid_time)`; for `GroupKFold` it enforces disjoint
groups. The validator's independent re-audit additionally applies a purge/embargo
around fold boundaries, dropping boundary-adjacent observations so lag/rolling features
whose lookback spans the boundary cannot leak across it.

**Nested tuning** is applied regardless of scheme: hyperparameter search runs inside
each outer fold on inner-training rows only, so the outer-fold metric estimates the
performance of the configuration the pipeline *would have selected without seeing the
held-out data* — not a hindsight best.

**Decision 3 — Why the OOF Is Honest, and Its Limit**

Honest within the training distribution because, whatever scheme was selected, each row
is predicted by a model that never saw it during training or hyperparameter selection:
the fold construction guarantees train/validation separation (temporal ordering for
time-series schemes, disjoint groups for `GroupKFold`, no index overlap for all), and
nested tuning keeps held-out rows out of the search. For time-series schemes, the
validator's independent purged re-audit additionally classifies any gap between the
modeler's CV and the strict re-audit: a `CV_SCHEME` classification means the gap is
structural pessimism (smaller early-fold training windows), not model overfit, and a
high `monotone_score` means error improves in the expected direction as training data
grows.

The limit is **distribution shift**, and it applies to any scheme. Adversarial
validation trains a classifier to separate training from validation rows on covariates
alone; a high AUC means the model will be evaluated on data that looks systematically
different from what it trained on. **The OOF metric is a within-training-distribution
estimate** — when adversarial AUC is high, true test error exceeds OOF by an amount no
within-training CV can quantify, because the correction would require the (unavailable)
test distribution. Adversarial sample weighting partially compensates but is a bounded
heuristic, not an exact correction. The OOF metric is therefore reported as a
within-distribution estimate, **not** as a prediction of leaderboard position.

**Decision 4 — What This Framework Does Not Claim**

| Claim | Why withheld |
|-------|-------------|
| The internal metric ≈ leaderboard score | The leaderboard is out-of-sample; the internal metric is within-training-distribution. When distribution shift is present (high adversarial AUC), it drives a gap no within-training CV can quantify. |
| The model predicts within-cell / within-group fine-grained variation | The model predicts the expected value at each unit (e.g. group × period); residual within-unit deviation that is noise from the model's perspective is not presented as predictable signal. |
| Every category contributes equally to the score | When the metric is an unweighted average over rows but category magnitudes differ, high-magnitude categories dominate the absolute error. The pipeline reports per-category breakdowns rather than implying uniform contribution. |
| More tuning guarantees better generalization | Tuning minimizes a CV-estimated objective; if that estimate carries residual optimism (e.g. from shift), further tuning toward it can increase overfit rather than improve test performance. |
| The submission covers the full training distribution | When the submission template restricts to a subset of training categories, only that scored subset reflects in the leaderboard metric; performance on unscored categories is not directly evaluated. |

---

## 7. Robustness Safeguards

Three defensive rails that are complete no-ops on normal datasets; they convert
catastrophic failures into safe, logged degradations. They do not improve MAE on
clean runs.

- **Rail 1 — Memory / feature-budget gate.** Before building the five expansive
  covariate families, estimates column/cell counts and (if `psutil` present) free RAM.
  If over `2e9` cells, `1000` extra columns, or under 2 GB free, those families are
  skipped and only base features are computed. Logged under `features.json →
  feature_budget`.
- **Rail 2 — Robust file/codebook detection.** Content-first file routing (a CSV
  containing the target with <50% nulls is train; an all-null target is the
  submission file), with filename keywords as fallback and a second-pass content swap
  to fix mislabeled splits. Case-insensitive. Logged under `profile.json →
  file_role_log`. Sequential row-index columns (e.g. an 'id' covariate with 100%
  unique values) are excluded from shift detection and adversarial weighting via a
  uniqueness-ratio filter (ID_UNIQUENESS_THRESHOLD = 0.95 in the cross-sectional
  path), preventing a meaningless KS=1.0 alarm and wasted shift-aware features on
  row identifiers.
- **Rail 3 — Time-granularity ambiguity fallback.** Cadence is flagged ambiguous only
  when both irregular *and* within 12 h of a classification boundary; then cycle-
  specific harmonics are replaced with a single granularity-agnostic relative-time
  feature. Logged under `features.json → time_granularity`.

---

## 8. Constraints

- CPU only (no GPU); no network or external data during analysis.
- 2-hour wall-clock budget; 1M-token budget across all agents.
- Single predictor (CatBoost) with Ridge as a diagnostic-only baseline.

---

## 9. Known Limitations

- **Distribution shift.** Walk-forward and purged-CV MAE can underestimate true
  out-of-sample error under severe shift, even with shift-aware features and bounded
  adversarial weighting.
- **Long-horizon compounding.** Recursive forecasting reduces but cannot eliminate
  error accumulation over long horizons; OOF/strict-CV don't reveal it (lags known
  there).
- **Single family.** No cross-family ensembling; variance reduction is the 5-seed
  CatBoost ensemble plus method selection only.
- **Multiclass.** True unordered multiclass uses the CatBoost-classifier path with
  LogLoss; cross-family probability aggregation is not implemented (one family).
  Ordinal targets correctly use the regression path and are unaffected.
- **File layouts.** Only the three documented conventions are auto-detected; unusual
  layouts fall back to heuristics that could misclassify train/val.
- **Expanded covariate features** can double/triple the feature count and runtime;
  benefit varies — minimal where base features already capture the signal, meaningful
  where covariates are rich.
- **Image features** (when image sidecars present): 23 hand-crafted spatial statistics
  per image via PIL/numpy (intensity stats, 2×2 quadrant mean/std, center-vs-edge
  contrast, brightness distribution, color variance). Panel data matched by filename
  pattern `{group}_{time}.png`; cross-sectional by direct row mapping. Activation:
  ≥10 images, PIL importable. All features passed to the model regardless of
  correlation (tree importance decides usage); max abs correlation logged as a
  diagnostic. These are *not* deep-learning embeddings — semantic content (object
  recognition) would need CNN embeddings, GPU, and weight downloads, which are out of
  scope. All failures (missing PIL, no images, load errors) are caught and the
  pipeline continues with standard features.

---

## 10. Reference: Observability Artifacts

These are **derived, observability-only** — no agent reads them for decisions and they
have no effect on the analysis or submission.

### Knowledge Graph (`reports/knowledge_graph.json`)

Append-only run audit trail via `tools/kg.py` (`kg_set_stage`, `kg_append_event`,
`kg_record_rejected`). Records stage transitions, per-stage events (inferred problem
type, resource counts, Optuna outcomes), and rejected hypotheses. Purpose: transparency
and post-hoc debugging, not control.

### Per-Seed OOF Diagnostic (`reports/oof/oof_per_seed.csv`)

Persists the per-seed prediction arrays the outer-fold scoring step already computes
(slot `k` ↔ seed `OUTER_FOLD_FINAL_SEEDS[k]`; uncovered slots `NaN`). Nothing is
added, reordered, reseeded, or refit. One row per covered OOF observation; identifier
columns resolved generically from `profile.json` (`row_index`, group columns, time
column, `fold_id`, `y_true`, one `pred_seed_{k}` per seed).

**Integrity check.** After writing, the modeler rebuilds OOF predictions row-wise using
the same aggregation callable the run used (`np.mean`, or `np.median` under the
critic-triggered `median_seed_aggregation` retune path) over the `pred_seed_*` columns
(NaN-aware). The result must reproduce the in-memory `oof_preds` to within `1e-9` or the
run aborts with an `AssertionError`; a companion assertion checks the row count. Under
`--debug` only one seed is present, so this view is degenerate (debug OOF is not valid).
Pointer block in `model_results.json → oof_per_seed`.

---

## 11. Reference: Feature Engineering Details

### Distribution-Shift-Aware Features

Each numeric covariate is KS-tested for train-vs-val shift. Covariates with KS > 0.15
get five derived features: z-score (training mean/std), rolling 4-period z-score,
percentile rank in the training distribution, group-level deviation from training mean,
and a covariate × normalized-time interaction. All from training rows only. No shift →
no-op.

### Expanded Covariate Families

For panel forecasting with numeric covariates, `time_col`, and sufficient history, six
extra families (all training-row-only, mapped onto the full frame):

- **Extended rolling windows** — rolling mean (8, 13) and rolling std (4, 8, 13) per
  covariate (skipped if > half of `min_periods_per_group`).
- **Covariate ratios** — for covariates sharing a name prefix; safe division, outlier-
  capped, max 10.
- **Group-level aggregates** — per-group historical mean, recent-4 mean, and drift.
- **Slope features** — target-on-time linear slope over recent 6 and 12 obs per group
  (skipped if `min_periods_per_group < 12`).
- **Centered covariates** — each covariate minus its overall training mean.
- **Entropy features** — Shannon entropy within covariate prefix groups (≥2 sharing a
  prefix).

### Time Granularity Detection

Inferred from median inter-observation step within groups. Datetime thresholds:
< 2 h hourly, < 48 h daily, < 216 h weekly, else monthly; integer time columns →
weekly by convention. Granularity drives extra seasonality features:

| Granularity | Extra features |
|-------------|----------------|
| Hourly | hour-of-day & day-of-week harmonics; per-(group × cycle-position) target mean/std |
| Daily | day-of-week, month-of-year, week-of-year sin/cos |
| Monthly | month-of-year sin/cos; calendar quarter |
| Weekly | base annual harmonics only |

For panel data, val lags that look into the validation period are imputed with a
cycle-aware seasonal mean (matching hour/day-of-week/week/month) rather than the last
training value, preserving seasonal pattern far ahead of training. Falls back to last-
known value when the seasonal position can't be determined.

### Adversarial Validation

A binary CatBoost classifier (5-fold stratified, 200 iterations) distinguishes train
(1) from val (0) rows; its OOF AUC measures multivariate shift KS may miss. Activation:
≥500 combined rows, ≥100 train, ≥100 val, ≥3 numeric covariates. AUC < 0.55 → uniform
weights; AUC ≥ 0.55 → each train row weighted `clip(1 − P(is_train), 0.1, 10)`,
mean-normalized. Weights stored in `data/features_train.parquet` and passed as
`sample_weight` to all CatBoost fits and the Ridge diagnostic. Top-5 shift-revealing
features, AUC, and weight flag logged in `features.json → adversarial_validation`.

### Codebook-Based Date Features

For opaque time-ID columns (e.g. base-64 hashes), `schema_analyst` looks for a JSON
codebook in `data/` (checked in order: `period_id_codebook.json`, `period_codebook.json`,
`period_id.json`, `dates.json`, `time_codebook.json`). When found, `profile.json` gains
a `time_codebook` field and `feature_engineering.py` adds `month_of_year`,
`quarter_of_year`, `is_quarter_start`. Unmapped rows get the median month; if > 10%
unmapped, date features are skipped with a warning. No codebook → silent no-op.

### Leak-Safe Target Encoding

Stacking is not used. Target encoding **is**, made leak-safe per-fold: a
`TargetEncoderCV` (smoothed empirical-Bayes encoder with inner KFold) is rebuilt inside
every outer fold and fit only on that fold's training rows. Ridge consumes the encoded
columns; CatBoost consumes raw categoricals as `cat_features` (ordered boosting, no
target-derived numerics). Unseen categories fall back to the global training mean.

---

*The `report.pdf` produced during each run contains the detailed methodology and
results specific to that run.*