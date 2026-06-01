# Award B — Autonomous Data Analysis Pipeline

This repository is an autonomous data analysis pipeline for a blind tabular
forecasting competition. When an evaluator drops an unknown dataset into `data/`
and prompts **"Do the data analysis"** (or any equivalent phrasing), this pipeline
must run without any human intervention, produce a valid `submission.csv` at the
repo root, and produce a `report.pdf` summarising the methodology and results —
all within **2 hours wall-clock** and **1,000,000 tokens** (input + output combined).

---

## Trigger phrases

Start the workflow immediately when the user says any of:
- "Do the data analysis"
- "Analyze the data"
- "Run the pipeline"
- "Start the analysis"
- any close paraphrase of the above

Do not ask clarifying questions. Do not wait. Begin at Step 1 of the workflow.

---

## Hard constraints

| Constraint | Value |
|------------|-------|
| Wall-clock budget | 2 hours from first tool call |
| Token budget | 1,000,000 (input + output, all agents combined) |
| Model | claude-sonnet-4-6 only |
| External data | None — no downloads, no web search |
| Human intervention | None after the initial prompt |
| GPU | Not available — CPU-only training |
| Submission | **Must always be written**, even on failure |

---

## Workflow

Execute the following sub-agents in order using the **Task tool**.
Each sub-agent is defined in `.claude/agents/` — use its registered name exactly.
After each agent completes, read its primary output file and verify it is
non-empty and sensible before continuing.

### Step 1 — schema_analyst (ALWAYS first)

```
Use the schema_analyst sub-agent on the data in data/.
```

- Reads: `data/DATA_DESCRIPTION.md`, all CSVs in `data/`
- Runs: `python tools/profile_data.py --data-dir data/ --output reports/profile.json`
- Writes: `reports/profile.json`, `reports/schema_analysis.md`
- Budget: **5 minutes**

Verify after completion:
- `reports/schema_analyst_was_here.txt` exists — if missing, the sub-agent was not
  invoked (work was inlined instead); STOP and report the failure immediately
- `reports/profile.json` exists and `problem_type` is set
- `reports/schema_analysis.md` exists and contains a "Problem Classification" section
- `target_col` is non-null in the JSON

If verification fails: write a minimal `submission.csv` (group-mean baseline) and a
one-page `report.pdf` explaining the failure, then stop.

### Step 2 — feature_engineer

After schema_analyst completes, invoke the feature_engineer sub-agent using the Agent tool.
Wait for it to write `reports/features.json` before proceeding.

```
Use the feature_engineer sub-agent to engineer features for the dataset.
```

- Reads: `reports/schema_analysis.md`, `reports/profile.json`, `data/`
- Writes: `reports/features.json`, `data/features_train.parquet`, `data/features_val.parquet`
- Budget: **15 minutes**

Verify after completion:
- `reports/features.json` exists and lists at least one feature group
- `reports/feature_engineer_was_here.txt` exists (proves the sub-agent ran, not inline work)
- `data/features_train.parquet` exists and has more columns than the raw data

If this step fails or times out: fall back to raw covariates only (no lag features).
Log the fallback in stdout and continue to the modeler with whatever feature files exist.

### Step 3 — modeler

Invoke the modeler sub-agent via the Agent tool. Wait for it to write
`reports/model_results.json`, `reports/predictions.csv`, AND
`reports/modeler_was_here.txt`. Do not proceed until the marker file exists.
Do NOT perform modeling inline — always delegate to the modeler sub-agent.

```
Use the modeler sub-agent to train models and generate predictions.
```

- Reads: `reports/schema_analysis.md`, `reports/features.json`,
         `data/features_train.parquet`, `data/features_val.parquet`
- Writes: `reports/model_results.json`, `reports/predictions.csv`,
          `reports/modeler_was_here.txt`
- Budget: **60 minutes**

Verify after completion:
- `reports/modeler_was_here.txt` exists (proves the sub-agent ran, not inline work)
- `reports/predictions.csv` exists
- Row count matches the validation set (check against `n_val_rows` in `reports/profile.json`)
- Prediction column name matches `target_col` from the profile
- No NaN predictions

If this step fails: generate a group-mean baseline prediction for `reports/predictions.csv`
using Python directly (do not invoke another agent), log the fallback, and continue.

### Step 3.5 — validator

Invoke the validator sub-agent via the Agent tool after modeler completes.
Wait for it to write `reports/validator_review.json` AND
`reports/validator_was_here.txt`. Do not proceed until the marker file exists.
Do NOT perform validation inline — always delegate to the validator sub-agent.
The validator is **diagnostic only** — it does NOT modify predictions and does NOT
block submission regardless of verdict.

```
Use the validator sub-agent to audit the modeler's CV integrity.
```

- Reads: `reports/profile.json`, `reports/model_results.json`, `reports/features.json`,
         `data/features_train.parquet`, `reports/oof_predictions.csv` (if present),
         `reports/schema_analysis.md` (optional)
- Writes: `reports/validator_review.json`, `reports/validator_was_here.txt`
- Budget: **10 minutes**

Verify after completion:
- `reports/validator_was_here.txt` exists (proves sub-agent ran, not inline work)
- `reports/validator_review.json` exists and has all required keys:
  `verdict`, `reported_cv_mae`, `strict_cv_mae`, `honest_cv_mae`,
  `cv_gap_abs`, `cv_gap_pct`, `strict_cv_scheme`, `feature_suspicion`,
  `checks`, `notes`
- `verdict` ∈ {PASS, WARNING, CRITICAL}

If this step fails: write a minimal `reports/validator_review.json` with
`verdict="WARNING"` and `notes` explaining the failure. Always continue to
`submission_writer` — a missing validator review never blocks submission.

### Step 4 — submission_writer

Invoke the submission_writer sub-agent via the Agent tool. Wait for it to write
`submission.csv` AND `reports/submission_writer_was_here.txt`. Do not proceed until
the marker file exists. Do NOT write submission.csv inline — always delegate to the
submission_writer sub-agent.

```
Use the submission_writer sub-agent to validate and write submission.csv.
```

- Reads: `reports/predictions.csv`, `data/DATA_DESCRIPTION.md`, `data/sample_submission.csv`
- Writes: `submission.csv` at the repo root, `reports/submission_summary.json`,
          `reports/submission_writer_was_here.txt`
- Budget: **10 minutes**

Verify after completion:
- `reports/submission_writer_was_here.txt` exists (proves the sub-agent ran, not inline work)
- `submission.csv` exists at the repo root
- Its schema matches `data/sample_submission.csv` (same columns, same row count)

If this step fails: copy `reports/predictions.csv` to `submission.csv` with a warning
logged to stdout.

### Step 5 — report_writer

Invoke the report_writer sub-agent via the Agent tool. Wait for it to write
`report.pdf` AND `reports/report_writer_was_here.txt`. Do not proceed until
the marker file exists. Do NOT generate report.pdf inline — always delegate to
the report_writer sub-agent. Verify the marker file exists; if missing, the
sub-agent was not invoked.

```
Use the report_writer sub-agent to produce report.pdf.
```

- Reads: all files in `reports/`, `data/DATA_DESCRIPTION.md`
- Writes: `report.pdf` at the repo root, `reports/report_writer_was_here.txt`,
          optionally `reports/feature_importance.png`, `reports/prediction_histogram.png`
- Budget: **20 minutes**

Verify after completion:
- `reports/report_writer_was_here.txt` exists (proves the sub-agent ran, not inline work)
- `report.pdf` exists at the repo root and is non-zero bytes

If this step fails: write a minimal one-page `report.pdf` using Python + reportlab
directly (do not invoke another agent) with: dataset name, problem type, model used,
final metric from `reports/model_results.json`.

---

## Time budget allocation

| Phase | Budget | Notes |
|-------|--------|-------|
| schema_analyst | 5 min | Non-negotiable — never skip or shorten |
| feature_engineer | 15 min | Reduce feature complexity if falling behind |
| modeler | 60 min | Cap Optuna at 30 trials if time is short |
| validator | 10 min | Diagnostic only; never blocks submission |
| submission_writer | 10 min | Should be fast; escalate if it stalls |
| report_writer | 20 min | Reduce to text-only PDF if time is short |
| Buffer | 20 min | Reserved for retries and fallback logic |
| **Total** | **140 min** | Hard limit: 120 min; 20 min slack in buffer |

**Self-pacing rule**: After each phase, estimate remaining wall-clock time.
If ≥ 75 % of the token budget or time budget is consumed and fewer than
two phases are complete, switch all remaining phases to their simplest
fallback variants immediately.

---

## Tools available

| Tool | Usage |
|------|-------|
| `tools/profile_data.py` | Profiles a data directory; outputs `profile.json` with problem type, group/time/target columns, per-column stats, KS shift flags, and image detection. Run with `--data-dir data/ --output reports/profile.json`. |

Additional tools will be documented here as they are built.

---

## File contracts between agents

| Agent | Reads | Writes |
|-------|-------|--------|
| `schema_analyst` | `data/`, `data/DATA_DESCRIPTION.md` | `reports/profile.json`, `reports/schema_analysis.md`, `reports/schema_analyst_was_here.txt` |
| `feature_engineer` | `reports/schema_analysis.md`, `reports/profile.json`, `data/` | `reports/features.json`, `data/features_train.parquet`, `data/features_val.parquet` |
| `modeler` | `reports/schema_analysis.md`, `reports/features.json`, `data/features_train.parquet`, `data/features_val.parquet` | `reports/model_results.json` (includes `feature_importance_all`, `oof_mae`), `reports/predictions.csv` (columns: `row_id`, identifier cols, `predicted_target`), `reports/oof_predictions.csv` (columns: identifier cols, `fold`, `predicted_target`), `reports/modeler_was_here.txt` |
| `validator` | `reports/profile.json`, `reports/model_results.json`, `reports/features.json`, `data/features_train.parquet`, `reports/oof_predictions.csv` (opt), `reports/schema_analysis.md` (opt) | `reports/validator_review.json`, `reports/validator_was_here.txt` |
| `submission_writer` | `reports/predictions.csv`, `data/DATA_DESCRIPTION.md`, `data/sample_submission.csv` | `submission.csv` (repo root), `reports/submission_summary.json`, `reports/submission_writer_was_here.txt` — renames `predicted_target` → actual target column name from `DATA_DESCRIPTION.md` |
| `report_writer` | `reports/` (all files), `data/DATA_DESCRIPTION.md` | `report.pdf` (repo root), `reports/report_writer_was_here.txt`, optional `.png` charts |

All inter-agent communication happens through files. Agents do not call each other
directly. The orchestrator (this CLAUDE.md) is responsible for sequencing and
passing the right context to each sub-agent invocation.

---

## Failure handling

### schema_analyst fails
This is a fatal failure. The target column and problem type are unknown.

1. Log the error clearly in stdout.
2. Read `data/DATA_DESCRIPTION.md` directly and attempt to identify the target column
   by scanning for a column named in a "predict X" sentence.
3. If identifiable: write `submission.csv` with the group-mean baseline for that column.
4. If not identifiable: write `submission.csv` with a single column of zeros matching
   `data/sample_submission.csv` shape.
5. Write a minimal `report.pdf` (one page) explaining the failure.
6. Stop.

### feature_engineer fails
Non-fatal. Fall back to raw covariates.

1. Log the fallback in stdout.
2. Copy raw `data/*.csv` train/val files as the feature source.
3. Proceed to `modeler` with a note that only raw features are available.

### modeler fails
Non-fatal. Use the group-mean baseline.

1. Log the failure in stdout.
2. Compute group-mean predictions: group the training target by the group columns
   identified in `reports/profile.json`, take the mean, broadcast to the val set.
3. Write the result to `reports/predictions.csv`.
4. Proceed to `submission_writer`.

### report_writer fails
Non-fatal. Write a minimal report directly.

1. Use Python + reportlab to write a one-page `report.pdf` containing:
   - Dataset: value of `data_dir` from `reports/profile.json`
   - Problem type and target column
   - Model used (from `reports/model_results.json`, if it exists)
   - Final metric (from `reports/model_results.json`, if it exists)
   - Note that full report generation failed and why.

---

## What NOT to do

- **Do not improvise sub-agents** not registered in `.claude/agents/`.
  If a sub-agent file does not exist, implement its logic as inline Python/bash
  and note the gap — do not invent a new `.md` agent on the fly.
- **Do not write `submission.csv` directly** without going through `submission_writer`.
  The submission_writer validates column names, row counts, and value ranges.
  The only exception is the fatal-failure fallback described above.
- **Do not exceed budget.** At 75 % token consumption, immediately switch all
  remaining work to its simplest fallback. Log the switch clearly.
- **Do not skip `schema_analyst`**, even if the data "looks obvious" or column names
  are familiar from prior runs. The schema_analyst produces `reports/profile.json`
  which all downstream agents depend on.
- **Do not download external data**, call external APIs, or use any data source
  other than the files already in `data/`.
- **Do not read or use `data/_truth/`** if it exists — that directory contains
  hidden ground-truth labels for local scoring only and must not be used in
  training or validation.

---


## Network and external resources policy

This agent performs analysis ONLY on the data provided in data/. 
The agent must NOT:
- Use web_search, web_fetch, or any network access during analysis
- Load pretrained models from external sources at runtime
- Download datasets, even if the rules technically permit network 
  access
- Use any information about the problem domain beyond what is 
  stated in data/DATA_DESCRIPTION.md

All signal must be derived from computation on the provided data. 
Sophisticated analysis (seasonality detection, distribution shift 
testing, statistical feature engineering) is encouraged. External 
information lookup is prohibited.

This is a deliberate engineering choice for robustness and 
transparency, beyond the minimum competition requirements.

---


## Sub-agents registered in .claude/agents/

| File | Name | Status | When invoked |
|------|------|--------|-------------|
| `schema_analyst.md` | `schema_analyst` | **exists** | Step 1 — always, immediately |
| `feature_engineer.md` | `feature_engineer` | **exists** | Step 2 — after schema_analyst succeeds |
| `modeler.md` | `modeler` | **exists** | Step 3 — after feature_engineer succeeds or falls back |
| `validator.md` | `validator` | **exists** | Step 3.5 — after modeler produces predictions; diagnostic only |
| `submission_writer.md` | `submission_writer` | **exists** | Step 4 — after validator completes (or fails gracefully) |
| `report_writer.md` | `report_writer` | **exists** | Step 5 — after submission.csv is written |
