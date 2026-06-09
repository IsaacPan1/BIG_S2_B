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
periods with known lags) — only on genuinely unknown future values. Logged under
`lag_forecasting`.

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

This records the CV design as an explicit decision record — options, choice, evidence,
and the cost of the rejected alternative — re-derived from `cv_plan.json`,
`features.json`, and `validator_review.json` each run. Nothing is invented.

**Decision 1 — Problem Type** (`plain_regression` / `univariate_time_series` /
`panel_forecasting`)

| Candidate | Outcome | Reason |
|-----------|---------|--------|
| `plain_regression` | REJECTED | Treats rows as IID; CV would train on future to predict past (temporal leakage → optimistic, fails at submission). |
| `univariate_time_series` | REJECTED | One model per group discards cross-group shared seasonality/covariate effects; lower per-group signal. |
| `panel_forecasting` | **SELECTED** | Time column (regular cadence) + group columns detected; covariates keyed on (group, time); cross-group signal available, ordering respected. |

**Decision 2 — CV Scheme** (random k-fold / single holdout / expanding / sliding)

Choice: **`TimeSeriesExpanding`** (frozen in `cv_plan.json`). Random k-fold is invalid
for ordered data; expanding window respects the arrow of time with the last fold
anchored at *T*. The expanding-vs-sliding gate is data-driven (the three gates above);
sliding is taken only on demonstrated recency-reducible shift.

| Rejected scheme | Cost |
|-----------------|------|
| Random k-fold | Optimistic, meaningless estimate for ordered targets. |
| Single holdout | High variance; no scaling evidence; no fold-ensemble variance reduction. |
| Sliding (no gate) | Discards still-valid older data without demonstrated benefit. |

Embargo/purge: the validator's strict re-audit uses a 2-period embargo, dropping
boundary-adjacent observations to prevent lag-feature leakage across the fold edge.

Nested tuning: Optuna runs inside each outer fold (`inner_folds = 3`) on inner-train
rows only, so outer-fold MAE is an honest estimate of the configuration the pipeline
*would have selected without seeing held-out data* — not a hindsight best.

**Decision 3 — Why the OOF Is Honest, and Its Limit**

Honest within the training distribution because each row is predicted by a model that
never saw it (expanding split + nested tuning), and the validator's independent purged
re-audit classifies any modeler-vs-strict gap: `CV_SCHEME` = structural pessimism from
smaller early-fold windows (not overfit); `monotone_score = 1.0` = MAE improves as the
training window grows.

The limit is **distribution shift.** Adversarial validation trains a classifier to
separate train from val rows on covariates; a high AUC means the model will be
evaluated on systematically different data. **OOF MAE is a within-training-distribution
estimate** — when adversarial AUC is high, true test error exceeds OOF by an amount no
within-training CV can quantify (the correction needs the unavailable test
distribution). Adversarial sample weighting partially compensates but is a bounded
heuristic. **OOF MAE is not presented as a leaderboard prediction.**

**Decision 4 — What This Framework Does Not Claim**

| Claim | Why withheld |
|-------|-------------|
| OOF MAE ≈ leaderboard | Leaderboard is out-of-sample; shift drives a gap CV cannot quantify. |
| Model predicts within-cell variation | It predicts the cell's expected value; within-cell deviation is noise-floor variance. |
| All categories contribute equally | MAE with differing category means is size-weighted; large categories dominate. |
| More Optuna trials guarantee better generalization | Tuning toward an optimistic CV objective can increase overfit. |
| Submission covers the full training distribution | Only the submission-covered category subset is scored. |

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
  file_role_log`.
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