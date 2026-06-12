# Autonomous Data Analysis Pipeline (Award B)

An autonomous tabular forecasting / regression / classification pipeline built on
Claude Code. Given an unknown dataset in `data/`, it produces `submission.csv` and
`report.pdf` within a 2-hour budget with no human intervention after the trigger
prompt.

---

## 1. Overview

The pipeline is driven by a Claude Code session (the orchestrator) that sequences
six sub-agents plus one direct subprocess. Each stage writes a marker file
(`reports/{stage}_was_here.txt`) — the authoritative completion signal. All
inter-stage communication is through files; no sub-agent calls another directly.

The modeler (Step 3) is the exception to sub-agent dispatch: it runs as a **direct
blocking subprocess** (`python tools/run_modeler.py`), not through the Task tool.
This is intentional — see [Section 5](#5-pipeline-stages) for why.

---

## 2. Quick Start

1. Set up the environment — see [Section 3](#3-environment-setup).
2. Place the dataset in `data/` — see [Section 4](#4-required-data-structure).
3. Launch Claude Code from the repo root:
   ```bash
   claude --dangerously-skip-permissions
   ```
4. Prompt: `Do the data analysis.`
5. Wait. A full run is roughly 60–80 minutes end to end; the modeler is the long
   pole at ~46–58 min. The 2-hour wall-clock budget is a hard ceiling, not the
   expected runtime.
6. Collect `submission.csv` and `report.pdf` at the repo root.

> **Memory note.** The modeler is memory-bound. On machines with limited free RAM
> it will swap and slow dramatically — runs are several times slower under memory
> pressure. Run with several GB of free RAM for best performance.

---

## 3. Environment Setup

Python 3.11 required (specifically 3.11.x — the pipeline invokes the `python3.11`
binary; a 3.12-only system has no `python3.11` in PATH). The venv **must be created
from a Python 3.11 executable** so that `python3.11` exists in the venv's `bin/` —
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
same interpreter. All pipeline tools invoke `python3.11` explicitly — activation
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

If target detection fails inside `profile_data.py`, the profiler defaults the
problem subtype to `continuous_regression` and continues — it does **not** abort
or write a fallback submission. The fatal-failure branch (baseline submission +
minimal report + stop) fires only if `schema_analyst` cannot produce a valid
`profile.json` at all; that path is handled by the orchestrator per the CLAUDE.md
"schema_analyst fails" section.

---

## 5. Pipeline Stages

The orchestrator runs Steps 0–5 in order. **Absence of a marker file while the
process is still alive is NOT a failure** — it means the stage is in progress.

| Step | Stage | Invocation | Marker |
|------|-------|------------|--------|
| 0 | Initialize `pipeline_run.json` | Orchestrator inline (Python) | — |
| 1 | `schema_analyst` | Task sub-agent | `reports/schema_analyst_was_here.txt` |
| 2 | `feature_engineer` | Task sub-agent | `reports/feature_engineer_was_here.txt` |
| 3 | Modeler — `tools/run_modeler.py` | **Direct blocking subprocess** | `reports/modeler_was_here.txt` |
| 3.5 | `validator` | Task sub-agent | `reports/validator_was_here.txt` |
| 3.6 | `critic` | Task sub-agent | `reports/critic_was_here.txt` |
| 4 | `submission_writer` | Task sub-agent | `reports/submission_writer_was_here.txt` |
| 5 | `report_writer` | Task sub-agent | `reports/report_writer_was_here.txt` |

**Why Step 3 is a direct subprocess, not a Task sub-agent.** Prior sub-agent
dispatches for the modeler failed in two characteristic ways: (a) the sub-agent
returned before `run_modeler.py` finished, so the verify gate fired against
half-written artifacts; (b) the sub-agent backgrounded the training script and
exited, leaving an orphaned Python worker consuming CPU outside the orchestrator's
process tree for up to an hour. Both failures share the same root cause — the
orchestrator was not the direct parent and had no synchronous wait on the child's
exit. The direct-subprocess contract fixes this: the orchestrator's Bash tool call
is the direct parent and blocks until the child exits, so when control returns to
the verify gate, all modeler work is fully done.

**Step 3.6 retune.** The critic may request at most one retune cycle (cap = 1,
enforced via `pipeline_run.json["retune_cap"]`). If triggered and both the cycle
cap and a 25-minute budget guard pass, the orchestrator re-runs the modeler (same
direct-subprocess contract), validator, and critic before continuing to Step 4.

**Step 0** initializes a fresh `pipeline_run.json` at the start of every run,
overwriting any leftover record from a prior run. This file tracks session start
time, budget, modeler run ID, and critic cycle count — it is the orchestrator's
authoritative run-state record.

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
| `multiclass_classification` | 3–50 unordered unique values | CatBoostClassifier, MultiClass | OOF MAE (= error rate) |

When ambiguous between ordinal and multiclass (integer target, 5–15 values), the
pipeline defaults to ordinal regression to preserve MAE-evaluable behavior.

**CatBoost is the sole predictor in every submission.** A Ridge baseline runs as
a diagnostic only (logged in `model_results.json`) and never appears in
`predictions.csv` or `submission.csv`.

---

## 6.5 Cross-Validation Scheme

`scheme_analysis.py` is the sole authority for the CV plan: it classifies the
problem, selects a scheme, enumerates leakage risks, and writes a **frozen**
`reports/cv_plan.json`. The plan is marked immutable (`frozen: true`) and no
downstream stage may modify it � `load_cv_plan` asserts the freeze on every read.

**Scheme selection by structure** (not by problem-type label):

| Structure | Scheme |
|---|---|
| Time column + group columns | `TimeSeriesExpanding` (panel forecasting) |
| Time column only | `TimeSeriesExpanding` |
| Group columns, no time | `GroupKFold` |
| =2 unique target values | `StratifiedKFold` |
| =50 unique integer values | `StratifiedKFold` |
| Otherwise | `KFold` |

**Recent-vs-full drift gate (expanding vs sliding).** The default for any
time-series problem is an expanding window (all history). The pipeline switches
to a sliding window *only* when restricting training to recent periods
measurably reduces the train?validation gap.

The diagnostic deliberately avoids the naive approach (triggering sliding on high
KS or adversarial-classifier AUC). On a forecasting holdout, validation is a
future window, so those signals saturate � they read "large shift" by
construction whether or not recency would help, and would pick sliding almost
always, discarding history for no gain. Instead, the gate computes a
**standardized mean shift** (an effect size that does not saturate) per numeric
covariate, comparing full-train-vs-val against recent-train-vs-val, and measures
whether the recent window actually narrows the distance.

Sliding is selected **only when all three conditions hold**:
- `frac_improved = 0.60` � share of features whose gap shrinks with recency (primary gate)
- `rel = 0.25` � relative mean improvement (secondary gate)
- `n_features_scanned = 12` � evidence-breadth floor

Any failure � including a non-runnable diagnostic, too few periods, or no time
axis � falls back to expanding. Sliding is never the default; it must be
affirmatively justified. (Seasonality and time-index features are excluded from
the scan, since they separate train from val by construction rather than by drift.)

**Leakage controls.** The plan enumerates and mitigates leakage risks: future
timestamps (validator enforces `max(train_time)+gap = min(valid_time)`), group
overlap (disjoint groups for `GroupKFold`), target-derived features (recomputed
per-fold from training indices only), and ID-like columns (dropped). The
validator runs an independent strict re-audit to confirm the modeler's CV was
not optimistic.

---

## 7. Imbalanced Classification

When the minority class fraction falls below 10% (`IMBALANCE_THRESHOLD = 0.10` in
`tools/profile_data.py`), the pipeline activates a four-stage handling sequence.
Stages 1–3 (detection, class weights, and probability averaging) apply to both binary
and multiclass classification. Stage 4 (threshold tuning) is binary-only — multiclass
uses argmax on the averaged probability arrays directly and receives no threshold sweep.

**Stage 1 — Detection** (`1acf5e2`). `profile_data.py` computes `min_class_fraction`
across all target classes and sets `is_imbalanced = True` in `profile.json["target_chars"]`.
No model changes at this stage — detection only.

**Stage 2 — Class weights** (`09daf31`). The modeler reads `is_imbalanced` from
`profile.json`. When true and the subtype is classification, `auto_class_weights = "Balanced"`
is set on every CatBoost fit: training folds, Optuna inner folds, walk-forward probe,
and the seed ensemble production pass.

**Stage 3 — Probability averaging** (`d453055`). For classification subtypes, the
seed ensemble averages class probability arrays (`predict_proba`) across seeds before
taking argmax, rather than majority-voting hard labels. This preserves probability
calibration and reduces per-seed variance before the argmax step.

**Stage 4 — Threshold tuning** (`1e7f9af`). For binary classification, the modeler
sweeps thresholds on OOF class-1 probabilities and selects the F1-optimal threshold.
The tuned threshold is applied to validation predictions **only** if the OOF F1 gain
meets a minimum floor (`THRESHOLD_TUNE_MIN_F1_GAIN = 0.02` absolute F1 improvement
over the default threshold of 0.5). If the gain is below this floor, the default
threshold is kept. Threshold tuning is binary-only; multiclass uses argmax directly.

**Safety check** (`253a692`). The critic includes a `prediction_collapse` check: if
all validation predictions degenerate to a single class label, it emits a WARNING
(non-blocking — submission proceeds). The critic's `prediction_distribution` check
(`bdc92a7`) is suppressed for classification subtypes because its gap threshold is
calibrated for regression MAE distributions and is not meaningful for class-label
or probability outputs.

---

## 8. Hard Constraints

| Constraint | Value | Source |
|------------|-------|--------|
| Wall-clock budget | 2 hours | CLAUDE.md hard constraints table |
| Token budget | 1,000,000 input + output combined, all agents | CLAUDE.md hard constraints table |
| GPU | Not available — CPU only | CLAUDE.md hard constraints table |
| External data | None — no downloads, no web search, no API calls, no pretrained weights | CLAUDE.md network policy |
| Retune cap | 1 retune cycle maximum per pipeline run | CLAUDE.md; `pipeline_run.json["retune_cap"] = 1` |
| Submission | Must always be written, even on failure | CLAUDE.md hard constraints table |

**Per-stage time ceilings** (from CLAUDE.md time budget table):

| Stage | Ceiling | Notes |
|-------|---------|-------|
| `schema_analyst` | 5 min | Non-negotiable |
| `feature_engineer` | 15 min | |
| Modeler | 90 min | Subprocess hard kill ceiling; measured typical run ~46–58 min |
| `validator` | 10 min | Diagnostic; never blocks |
| `critic` | 5 min | Advisory; never blocks |
| `submission_writer` | 10 min | |
| `report_writer` | 20 min | |
| Buffer | 15 min | For retries and fallback logic |

---

## 9. Output Files

| File | Location | Description |
|------|----------|-------------|
| `submission.csv` | repo root | Graded deliverable: exactly two columns — `row_id` and the target column. Row count matches `sample_submission.csv`. |
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

Failure behavior per CLAUDE.md:

| Stage fails | Severity | Orchestrator action |
|-------------|----------|---------------------|
| `schema_analyst` | **Fatal** | Write baseline submission (group-mean if target identifiable, else zeros matching sample shape) + minimal one-page `report.pdf`. Stop. |
| `feature_engineer` | Non-fatal | Fall back to raw covariates; continue to modeler |
| Modeler | Non-fatal | Compute group-mean predictions from training target; write to `reports/predictions.csv`; continue to `submission_writer` |
| `report_writer` | Non-fatal | Write minimal one-page `report.pdf` directly via reportlab; if reportlab unavailable, write `report.txt` and log the degradation |

The validator and critic are diagnostic-only — their failure or an adverse verdict
never blocks submission.

> **Note.** The `schema_analyst` fatal path fires only when the agent cannot produce
> a valid `profile.json` at all. If `profile_data.py` encounters a target-detection
> failure internally, it defaults to `continuous_regression` and continues writing a
> complete `profile.json` — the fatal orchestrator path does not fire (see §4.3).

### Code-level defensive rails

**Memory / feature-budget gate.** Before building large covariate families,
`feature_engineering.py` estimates column/cell counts and free RAM (via `psutil`).
If over 2×10⁹ cells, 1,000 extra columns, or under 2 GB free, those families are
skipped and only base features are computed. Logged under `features.json → feature_budget`.

**Two-phase ID column detection** (`e4bf090`). Exact name-match for common ID column
names runs before the dtype-gated near-unique scan. Prevents false-positive ID
classification on meaningful integer columns that happen to be high-cardinality.

**dtype-gate for CatBoost encoding** (`ce8ee6e`). ID-like integer columns are
blocked before reaching CatBoost's categorical encoder, preventing encoding errors
on columns that look like row identifiers.

**Content-first file routing.** A CSV with the target column and <50% nulls is
classified as train; all-null target → submission template. Filename keywords are
fallback only. A second-pass content swap corrects mislabeled splits.

**Forecasting-section suppression** (`e5e3b27`). The report omits the forecasting
diagnostics section when no time axis is detected, preventing misleading output on
non-temporal datasets.

**Submission round-trip audit.** `build_submission.py` verifies every row in
`sample_submission.csv` maps to a real prediction via the composite business key
before writing. NaN predictions fall back to training mean with a logged warning
(`a49dee4` handles the edge case where the composite key is empty).

---

## 11. Recent Changes

Commits from the `custom-cv` branch head (newest first). Hashes from
`git log --oneline origin/custom-cv..HEAD`.

**Imbalanced classification — 4-stage implementation**
`1acf5e2` `09daf31` `d453055` `1e7f9af`

End-to-end handling for class-imbalanced datasets. Detection at the 10% minority-
class-fraction threshold; `auto_class_weights = "Balanced"` applied to all CatBoost
fits; seed ensemble averages probabilities before argmax; OOF-optimized threshold
applied to binary predictions only when F1 gain ≥ 0.02.

`253a692` — **Critic: single-class collapse guard.** Adds a `prediction_collapse`
WARNING check (non-blocking) when all classification predictions degenerate to one
class label.

`bdc92a7` — **Critic: suppress regression-calibrated distribution check for
classification.** The `prediction_distribution` gap threshold is calibrated for MAE
distributions; suppressed for classification subtypes where it is not meaningful.

`38ba55d` — **Full classification routing.** `CatBoostClassifier` with Logloss /
MultiClass loss selection, F1/accuracy metric reporting, and label-set validation in
the submission writer.

`e4bf090` — **Profiler: two-phase id_col detection.** Exact name-match before
dtype-gated near-unique scan; prevents false-positive ID classification.

`bee6414` — **Profiler: reconcile `problem_type` with subtype classifier.** Ensures
the top-level `problem_type` field is consistent with the subtype classifier for
numeric binary targets (a `{0,1}` integer target is classified as
`binary_classification`, not `ordinal_regression`).

`ce8ee6e` — **dtype-gate: guard ID-like integer columns before CatBoost encoding.**
Prevents encoding errors on high-cardinality integer columns that reach the
categorical pipeline.

`a49dee4` — **Submission: handle empty composite key in round-trip audit.** When no
group or time columns form the composite key, falls back to row-index matching with
an explicit log instead of silently skipping the audit.

`8190bde` — **Report: derive CV-decision narrative from profile structure.** The CV
narrative in `report.pdf` is generated at report time from `profile.json` and
`cv_plan.json` rather than from hardcoded panel text that became stale across runs.

`e5e3b27` — **Report: suppress forecasting section when no time axis present.**
Prevents misleading forecasting diagnostics in reports for non-temporal datasets.

`3cae09b` — **Shift-aware post-hoc level correction.** Optional bias adjustment
weighted by adversarial validation density ratio; applied to regression predictions
only if it reduces weighted holdout MAE beyond a relative margin gate.

`7281b7e` — **Revert IS density ratio sharpening.** Reverts `f33ff71` (tempered IS
density ratio for adversarial weighting) — introduced instability on low-shift
datasets with marginal benefit.

---

## 12. Limitations

**Distribution shift.** OOF and strict-CV MAE are within-training-distribution
estimates. Under severe train→val shift (high adversarial validation AUC), true test
error can exceed these by an amount no within-training CV can quantify. Shift-aware
weighting and level correction partially compensate but are bounded heuristics.

**Single model family.** No cross-family ensembling; variance reduction is the
5-seed CatBoost ensemble plus target-transform selection and recursive-vs-static
method selection.

**Long-horizon compounding.** Recursive forecasting reduces but cannot eliminate
error accumulation over many steps. OOF and strict-CV do not reveal it because lags
are known in training folds.

**File layout detection.** Only three layout conventions are auto-detected. Unusual
file structures fall back to heuristics that may misclassify train vs validation.

**Imbalanced multiclass threshold tuning.** Stage 4 threshold tuning is binary
classification only. Multiclass uses argmax directly on averaged probabilities.

**Image features.** When image sidecars are present, 23 hand-crafted spatial
statistics are computed via PIL/numpy — not deep-learning embeddings. Semantic
content (object recognition) requires CNN weights and a GPU, which are out of scope.

**Retune cap.** At most one critic-triggered retune cycle per pipeline run. A
dataset requiring multiple retuning passes will not benefit from them.

**Ordinal rounding.** Ordinal regression predictions are rounded to the nearest
valid integer using an OOF-optimized offset. Works well for dense integer ranges;
may degrade on sparse or irregular ordinal sets.
