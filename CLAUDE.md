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
- "Do the data analysis" (most common)
- "Run the data analysis" / "Run the pipeline"
- "Analyze the data"
- "Perform the data analysis"
- "Execute the pipeline" / "Execute the analysis"
- "Start the analysis" / "Begin the analysis"
- Any natural language expression of intent to run the autonomous data analysis pipeline

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

### Sub-agent completion contract (read before invoking any stage)

**MARKER IS AUTHORITATIVE — ABSENCE ≠ FAILURE.**
A sub-agent is COMPLETE when and only when its `*_was_here.txt` marker file exists.
If the marker is absent, the sub-agent is either STILL RUNNING or genuinely failed —
these are two different states and must NOT be conflated. Absence of the marker is NOT
proof of failure. Do NOT re-invoke a sub-agent solely because its outputs are not yet
present.

**NEVER DOUBLE-INVOKE.**
Before invoking any sub-agent, check whether its marker already exists OR whether a
process for it is already running. NEVER launch a second instance of a sub-agent while
a prior instance may still be running — concurrent writes to the same `reports/` files
corrupt outputs. If a stage appears stalled, wait; do not relaunch.

**HOW TO DISTINGUISH 'STILL RUNNING' FROM 'FAILED'.**
The only valid failure signal is: the process has EXITED (with any exit code) AND the
marker file is still absent after exit. Re-invocation or fallback logic is permitted ONLY
in that genuine-failure state — never on a not-yet-present check during an active run.

**Expected wall-clock durations per stage** (the marker will be absent throughout this
window during normal operation — this is expected, not a failure):

| Stage | Typical duration | Outer budget |
|-------|-----------------|--------------|
| feature_engineer | 2–5 min | 15 min |
| modeler | 8–15 min (Optuna tuning across families) | 60 min |
| validator | 3–8 min | 10 min |
| critic | 2–5 min | 5 min |

The modeler is the longest-running stage. It is normal for `reports/modeler_was_here.txt`
to be absent for 10+ minutes after invocation while Optuna trials are running. Do not
treat this as a failure. Poll for the marker periodically; continue waiting as long as
the launched process is alive.

### Stage handoff contracts

Every inter-stage handoff is an **artifact contract**, never a verbal one. A downstream
sub-agent's precondition is the set of files produced by the upstream stage plus a
small `reports/<stage>_completion.json` record (schema below). The orchestrator's
"verify after completion" block is the gate that decides whether the next stage runs.

**Completion-record schema** — written by every sub-agent whose contract requires one
(currently: modeler; the same schema applies to any stage that adopts the pattern):

```json
{
  "stage": "modeler",
  "status": "ok",                       // "ok" | "failed"
  "dispatch_time": "2026-01-01T00:00:00",// ISO8601 UTC, captured before the script runs
  "exit_code": 0,
  "artifacts": {
    "model_results":   "reports/model_results.json",
    "predictions":     "reports/predictions.csv",
    "oof_predictions": "reports/oof_predictions.csv",
    "marker":          "reports/modeler_was_here.txt"
  },
  "notes": ""                           // on failure: last ~50 lines of stdout/stderr
}
```

**Stale-marker guard.** A marker file existing is not enough — it may be left over
from a prior pipeline run. The orchestrator's gate must verify `marker.mtime >
dispatch_time`. Sub-agents that write a completion record must capture
`dispatch_time` BEFORE launching the backing script.

#### Pipeline run state — `reports/pipeline_run.json`

The above completion records are **write-once** — each sub-agent writes its own
record at the end of its work. `reports/pipeline_run.json` is the one
**mutated-mid-run** state file: the orchestrator's run-state record. It
turns the freshness checks, budget guard, and cycle bound from heuristic
fallbacks (documented per-stage) into real enforcement.

**Schema** (no other fields; this file is strictly the cross-stage state the
orchestrator needs):

```json
{
  "session_start_iso":      "2026-06-08T21:01:12.540453+00:00",  // tz-aware UTC; convenience copy
  "session_start_epoch":    1780952472.540453,                   // POSIX epoch float, UTC — primary reference
  "total_budget_seconds":   7200,                                // 2 h, per the Hard constraints table
  "current_modeler_run_id": null,                                // null at start; set after each modeler verify pass
  "critic_cycle":           0,                                   // 0 at start; incremented before each retune dispatch
  "retune_cap":             1                                    // matches critic.md contract; orchestrator-enforced
}
```

**Ownership.** `pipeline_run.json` is **orchestrator-owned**. Sub-agents
**READ** it; only the orchestrator **WRITES** it. There are no concurrent
writers. Last-write-wins applies — but in practice, every write is a
deliberate orchestrator action between agent dispatches.

**Write semantics.** The file mutates at four well-defined points:

1. **CREATE / OVERWRITE at pipeline start (Step 0).** Initialize a fresh
   record with `session_start_iso` / `session_start_epoch` set from
   `datetime.now(timezone.utc)`, `total_budget_seconds = 7200`,
   `current_modeler_run_id = null`, `critic_cycle = 0`, `retune_cap = 1`.
   **If a `pipeline_run.json` already exists from a prior run, OVERWRITE it
   — never reuse.** Reusing a leftover record is the stale-state hole: it
   would silently validate prior-run completion records as fresh. Overwrite
   unconditionally.
2. **UPDATE `current_modeler_run_id` after Step 3 modeler verify passes
   (initial pass).** Copy
   `modeler_completion.json["modeler_run_id"]` into
   `pipeline_run.json["current_modeler_run_id"]`. Do this only after the
   Step 3 artifact gate has passed.
3. **UPDATE `critic_cycle` in the Step 3.6 retune branch — BEFORE
   dispatching the retune modeler pass.** Increment `critic_cycle` and
   persist BEFORE re-invoking the modeler. The cap check and budget guard
   run against the post-increment value. You cannot afford to start a
   retune cycle you can't finish in budget — check before dispatch, never
   after.
4. **UPDATE `current_modeler_run_id` after Step 3 modeler verify passes
   (retune pass).** Same as (2). The field always tracks the **latest**
   modeler pass; freshness checks downstream point at the most recent run.

`session_start_*`, `total_budget_seconds`, and `retune_cap` are written
once at Step 0 and never modified.

**Enforcement this record powers** (each replacing a previously
heuristic-only check; documented for clarity, not yet wired through all
agent contracts — see "Remaining contract work" at the bottom):

| Concern | Old fallback | Strict enforcement via pipeline_run.json |
|---|---|---|
| Freshness | submission_writer.md / report_writer.md / critic.md 3 h mtime window | every downstream completion record's `modeler_run_id` must equal `pipeline_run.current_modeler_run_id` |
| Budget guard | critic.md "< 25 min" heuristic guess | `remaining = session_start_epoch + total_budget_seconds − now` — a real number |
| Cycle cap | Step 3.6 narrative "Maximum one retune cycle per pipeline run" | `critic_cycle < retune_cap`, checked at orchestrator before dispatching a retune |

**Producer → artifact → consumer table**

| Producer | Artifact | Consumer | Precondition for consumer |
|---|---|---|---|
| modeler | `reports/model_results.json` | validator, critic, report_writer | exists, non-empty, parses as JSON |
| modeler | `reports/predictions.csv` | submission_writer, critic, report_writer | exists, non-empty, row count == `n_val_rows`, `predicted_target` has no NaN |
| modeler | `reports/oof_predictions.csv` | validator, critic | exists, non-empty |
| modeler | `reports/modeler_was_here.txt` | orchestrator | exists, `mtime > dispatch_time` |
| modeler | `reports/modeler_completion.json` | validator, orchestrator | exists, `status == "ok"` |
| orchestrator | `reports/pipeline_run.json` | critic, submission_writer, report_writer | exists (created at Step 0); critic + writers compare upstream `modeler_run_id` against `current_modeler_run_id`; critic also reads `session_start_epoch` for the budget guard and `critic_cycle` for the cycle counter |

The validator's precondition is the full set of modeler artifacts above
(`status == "ok"` plus the four named files). Verbal-only handoffs are NOT
permitted; if a downstream stage cannot find its required artifacts, it must
FAIL its own gate and not invent fallbacks based on the subagent's narrative
report.

### Step 0 — initialize `reports/pipeline_run.json` (ALWAYS first concrete action)

Before invoking ANY sub-agent — even schema_analyst — the orchestrator
creates a fresh `reports/pipeline_run.json` per the schema in "Stage handoff
contracts → Pipeline run state".

```python
import json, datetime, pathlib
pathlib.Path("reports").mkdir(parents=True, exist_ok=True)
now = datetime.datetime.now(datetime.timezone.utc)
record = {
    "session_start_iso":      now.isoformat(),
    "session_start_epoch":    now.timestamp(),
    "total_budget_seconds":   7200,        # 2 h, per Hard constraints
    "current_modeler_run_id": None,
    "critic_cycle":           0,
    "retune_cap":             1,
}
with open("reports/pipeline_run.json", "w") as f:
    json.dump(record, f, indent=2)
```

If `reports/pipeline_run.json` already exists from a prior pipeline run:
**OVERWRITE it unconditionally.** Reusing a leftover record is the
stale-state hole — it would silently validate prior-run completion records
as fresh. There is no recovery / resume path off a prior pipeline_run.json;
each pipeline run starts clean.

Step 0 has no marker file and no sub-agent — it's a one-liner the
orchestrator does itself.

### Step 1 — schema_analyst (ALWAYS first sub-agent)

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

After schema_analyst completes, invoke the feature_engineer sub-agent using the Task tool.
Wait for it to write `reports/feature_engineer_was_here.txt` before proceeding. This stage
typically takes 2–5 minutes; the marker may be absent for several minutes during normal
operation. See the Sub-agent completion contract above — do NOT re-invoke if the marker
has not yet appeared.

```
Use the feature_engineer sub-agent to engineer features for the dataset.
```

- Reads: `reports/schema_analysis.md`, `reports/profile.json`, `data/`
- Writes: `reports/features.json`, `data/features_train.parquet`, `data/features_val.parquet`
- Budget: **15 minutes**

Verify after completion:
- `reports/feature_engineer_was_here.txt` exists (proves the sub-agent ran, not inline work)
- `reports/features.json` exists and lists at least one feature group
- `data/features_train.parquet` exists and has more columns than the raw data

If the process has exited AND `reports/feature_engineer_was_here.txt` is still absent
(genuine failure): fall back to raw covariates only (no lag features).
Log the fallback in stdout and continue to the modeler with whatever feature files exist.

### Step 3 — modeler

Invoke the modeler sub-agent via the Task tool. Wait for it to write
`reports/model_results.json`, `reports/predictions.csv`, AND
`reports/modeler_was_here.txt`. Do not proceed until the marker file exists.
Do NOT perform modeling inline — always delegate to the modeler sub-agent.

This is the longest-running stage: Optuna hyperparameter tuning typically takes
8–15 minutes. The marker `reports/modeler_was_here.txt` will be absent throughout
this time — that is expected, not a failure. Poll for the marker periodically;
continue waiting as long as the launched process is alive. See the Sub-agent
completion contract above.

```
Use the modeler sub-agent to train models and generate predictions.
```

- Reads: `reports/schema_analysis.md`, `reports/features.json`,
         `data/features_train.parquet`, `data/features_val.parquet`
- Writes: `reports/model_results.json`, `reports/predictions.csv`,
          `reports/oof_predictions.csv`, `reports/modeler_was_here.txt`,
          `reports/modeler_completion.json`
- Budget: **60 minutes**

**Capture `dispatch_time` BEFORE invoking the sub-agent**, as a timezone-aware
UTC value. Two equivalent options:

- Python: `datetime.datetime.now(datetime.timezone.utc)` — store as both the
  ISO8601 string (with `+00:00` offset) and the POSIX epoch float
  (`.timestamp()`); record the string in the completion file, use the epoch
  float for the mtime comparison.
- Shell: `date -u +%s` for the epoch float plus `date -u --iso-8601=seconds`
  for the string.

NEVER reparse a naive ISO string with `datetime.fromisoformat(...).timestamp()`
later — that interprets the string as local time and breaks the comparison on
any machine outside UTC. Always carry the epoch float forward, or always carry
a tz-aware ISO string that round-trips.

Verify after completion — this is an **INDEPENDENT artifact gate**; do NOT trust
the sub-agent's verbal "done" report. Re-run every check from this side:

1. The sub-agent process has exited. If the marker is absent but the process
   is still alive, continue waiting — this is NOT a failure state.
2. `reports/modeler_completion.json` exists, parses as JSON, and has
   `status == "ok"` and `exit_code == 0`.
3. `reports/modeler_was_here.txt` exists AND its mtime is strictly newer than
   `dispatch_time` (compare in epoch-float seconds; rejects leftover markers
   from prior runs).
4. `reports/model_results.json` exists, size > 0, parses as JSON.
5. `reports/predictions.csv` exists, size > 0, the prediction column matches
   `target_col` (or is `predicted_target` for submission_writer to rename),
   no NaN predictions, **and `len(predictions.csv) == len(features_val.parquet)`
   exactly**. The validation-feature parquet is the authoritative row-count
   reference because feature_engineer may expand the raw validation rows
   (e.g. cross-joining missing category levels) — `profile.json.n_val_rows`
   is the RAW row count and will NOT match. Do not use
   `model_results.json.n_val_rows` either: that is producer self-reporting
   and gives circular agreement.
6. `reports/oof_predictions.csv` exists, size > 0.

**After all six checks pass, propagate `modeler_run_id` into the run-state
record:** read `modeler_completion.json["modeler_run_id"]` and write it into
`reports/pipeline_run.json["current_modeler_run_id"]`. This is the orchestrator's
job; it applies to BOTH the initial Step 3 dispatch and any Step 3.6 retune
re-dispatch — the field always tracks the latest modeler pass. After this
update, downstream stages can use the strict `modeler_run_id`-match freshness
check instead of the 3 h mtime heuristic their agent files currently document
(once those agents are updated to consume it — see "Remaining contract work"
at the bottom).

```python
import json
mc = json.load(open("reports/modeler_completion.json"))
pr = json.load(open("reports/pipeline_run.json"))
pr["current_modeler_run_id"] = mc["modeler_run_id"]
json.dump(pr, open("reports/pipeline_run.json", "w"), indent=2)
```

Only on full pass (all six checks AND the `pipeline_run.json` update): dispatch
the validator (Step 3.5).

On any failure (process exited with nonzero code, completion record missing
or `status == "failed"`, stale marker, missing/empty/invalid artifact):
generate a group-mean baseline prediction for `reports/predictions.csv` using
Python directly (do not invoke another agent), log the fallback, and continue
to `submission_writer`. Do NOT dispatch the validator on partial state.

### Step 3.5 — validator

Invoke the validator sub-agent via the Task tool after modeler completes.
Wait for it to write `reports/validator_review.json` AND
`reports/validator_was_here.txt`. Do not proceed until the marker file exists.
Do NOT perform validation inline — always delegate to the validator sub-agent.
The validator is **diagnostic only** — it does NOT modify predictions and does NOT
block submission regardless of verdict. This stage typically takes 3–8 minutes;
see the Sub-agent completion contract above — do NOT re-invoke if the marker
has not yet appeared.

```
Use the validator sub-agent to audit the modeler's CV integrity.
```

- Reads: `reports/profile.json`, `reports/model_results.json`, `reports/features.json`,
         `data/features_train.parquet`, `reports/oof_predictions.csv` (if present),
         `reports/schema_analysis.md` (optional)
- Writes: `reports/validator_review.json`, `reports/validator_was_here.txt`
- Budget: **10 minutes**

Verify after completion:
- Before verifying, confirm the sub-agent process has exited. If the marker is absent
  but the process is still alive, continue waiting — this is NOT a failure state.
- `reports/validator_was_here.txt` exists (proves sub-agent ran, not inline work)
- `reports/validator_review.json` exists and has all required keys:
  `verdict`, `reported_cv_mae`, `strict_cv_mae`, `honest_cv_mae`,
  `cv_gap_abs`, `cv_gap_pct`, `strict_cv_scheme`, `feature_suspicion`,
  `checks`, `notes`
- `verdict` ∈ {PASS, WARNING, CRITICAL}

If the process has exited AND `reports/validator_was_here.txt` is still absent (genuine
failure): write a minimal `reports/validator_review.json` with `verdict="WARNING"` and
`notes` explaining the failure. Always continue to `submission_writer` — a missing
validator review never blocks submission.

### Step 3.6 — Critic Review

Invoke the critic sub-agent via the Task tool. Wait for
reports/critic_was_here.txt and reports/critic_review.json. This stage typically
takes 2–5 minutes; see the Sub-agent completion contract above — do NOT re-invoke
if the marker has not yet appeared.

```
Use the critic sub-agent to review the validator output and modeler predictions.
```

- Reads: `reports/validator_review.json`, `reports/model_results.json`,
         `reports/predictions.csv`, `reports/features.json`,
         `reports/profile.json`, `data/features_train.parquet`
- Writes: `reports/critic_review.json`, `reports/critic_was_here.txt`,
          optionally `reports/critic_retune_requested.json` and
          `reports/critic_retune_attempted.txt`
- Budget: **5 minutes**

NOTE: the retune cycle below is an INTENTIONAL, gated re-invocation of the modeler —
distinct from the accidental double-invocation that the Sub-agent completion contract
prohibits. The trigger is now the critic's structured completion status, with the
cycle cap and budget guard enforced by `reports/pipeline_run.json` at the
orchestrator — not by file-existence heuristics.

**Retune-vs-proceed decision (orchestrator-side):**

1. Read `reports/critic_completion.json`. If its `status != "retune_requested"`
   (i.e. it is `"ok"`, `"failed"`, or `"blocked"`), proceed to Step 4. No
   retune.
2. If `status == "retune_requested"`:
   - Read `reports/pipeline_run.json`. Compute
     `remaining_budget = session_start_epoch + total_budget_seconds − now()`.
   - **Cycle cap check.** If `critic_cycle >= retune_cap` (cap = 1, matching
     `critic.md`): force-proceed to Step 4. Log the cap-hit; do NOT dispatch
     the retune. The critic should already have force-accepted at this cycle
     by its own rules, so this branch is a defense-in-depth.
   - **Budget guard.** If `remaining_budget < 25 * 60` seconds (a typical
     retune cycle: modeler + validator + critic + slack): force-proceed to
     Step 4. Log the budget-exhaustion reason; do NOT dispatch the retune.
   - **Both checks pass — dispatch the retune.** In this order, atomically:
     1. Increment `pipeline_run.json["critic_cycle"]` (write to disk BEFORE
        re-dispatching — a partial run that crashes mid-retune must not
        leave `critic_cycle` unincremented and re-attempt the loop on a
        future invocation).
     2. Re-invoke modeler (it reads `reports/critic_retune_requested.json`
        and applies the suggested change). After the modeler verify gate
        passes, the orchestrator updates
        `pipeline_run.json["current_modeler_run_id"]` per Step 3's final
        bullet — same as the initial pass.
     3. Re-invoke validator (audits the new modeler output).
     4. Re-invoke critic. At this cycle the critic's `cycle_cap_will_block`
        (per `critic.md` Step 3) fires; the critic force-accepts; the
        retune branch terminates.
3. After at most one cap-bounded retune cycle, proceed to Step 4. The
   pipeline always reaches submission_writer regardless of which branch
   fires.

The cap value lives in `pipeline_run.json["retune_cap"]` (default `1`,
matching `critic.md`). Raising the cap requires the coordinated changes
spelled out in `critic.md` § "Future option: raising the cap to 2 retunes" —
do NOT raise it here unilaterally.

Do NOT generate critic review inline; always delegate to the critic
sub-agent.

Verify after completion:
- Before verifying, confirm the sub-agent process has exited. If the marker is absent
  but the process is still alive, continue waiting — this is NOT a failure state.
- `reports/critic_was_here.txt` exists (proves the sub-agent ran, not inline work)
- `reports/critic_review.json` exists with a `status` field

If the process has exited AND `reports/critic_was_here.txt` is still absent (genuine
failure): write a minimal `reports/critic_review.json` with `status="accepted"` and a
warning note. Always continue to `submission_writer` — a missing critic review never
blocks submission.

### Step 4 — submission_writer

Invoke the submission_writer sub-agent via the Task tool. Wait for it to write
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

Invoke the report_writer sub-agent via the Task tool. Wait for it to write
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
| critic | 5 min | Advisory + retune; never blocks submission |
| submission_writer | 10 min | Should be fast; escalate if it stalls |
| report_writer | 20 min | Reduce to text-only PDF if time is short |
| Buffer | 15 min | Reserved for retries and fallback logic |
| **Total** | **140 min** | Nominal total assuming no retune cycle; phases must compress if falling behind the 120-minute wall-clock limit |

If critic triggers a retune, modeler + validator + critic re-run, adding ~75 minutes. Subsequent phases must reduce to their fallback variants if this happens.

**Self-pacing rule**: After each phase, estimate remaining wall-clock time.
If ≥ 75 % of the token budget or time budget is consumed and fewer than
two phases are complete, switch all remaining phases to their simplest
fallback variants immediately.

---

## Tools available

| Tool | Invoked by | Usage |
|------|------------|-------|
| `tools/profile_data.py` | schema_analyst | Profiles a data directory; outputs `reports/profile.json` with problem type, group/time/target columns, per-column stats, KS shift flags, and image detection. Run with `--data-dir data/ --output reports/profile.json`. |
| `tools/feature_engineering.py` | feature_engineer | Dataset-agnostic panel feature engineering. Reads `reports/profile.json` for schema; adapts lag/rolling depths to available training history. Writes `data/features_train.parquet`, `data/features_val.parquet`, `reports/features.json`, `reports/feature_engineer_was_here.txt`. |
| `tools/run_modeler.py` | modeler | CatBoost training (sole predictor) with walk-forward 80/20 holdout, Optuna hyperparameter tuning, and boundary reflection. Trains a Ridge diagnostic baseline for OOF MAE comparison and top linear coefficients (not in submission). Writes `reports/model_results.json`, `reports/predictions.csv`, `reports/oof_predictions.csv`, `reports/modeler_was_here.txt`. |
| `tools/validate.py` | validator | CV integrity and leakage audit. Computes purged walk-forward MAE and compares to reported CV MAE. Writes `reports/validator_review.json` (including `fold_maes`, `fold_train_sizes`); calls `tools/gap_attribution.py` internally. Run with `--repo-root PATH`. |
| `tools/gap_attribution.py` | validator (auto) | Dataset-agnostic OOF→strict CV gap attribution. Reads `fold_maes`/`fold_train_sizes` from `validator_review.json`; classifies gap as CV_SCHEME (scheme pessimism) or REAL_DIVERGENCE; appends `gap_attribution` block to `validator_review.json`. No model training. |
| `tools/family_ablation.py` | critic (auto) | Leave-one-family-out strict-CV ablation using `feature_families` from `features.json`. Budget-gated (skips if time insufficient). Flags families as NET-HARMFUL if `strict_mae_without + margin < strict_mae_full` where `margin = max(3*seed_std, 0.005)`. Writes `reports/family_ablation.json`. **DIAGNOSTIC ONLY** — results are recorded for analysis but the critic must NOT automatically drop families or trigger a retune based on `net_harmful_families` unless the `auto_drop_harmful_families` flag is explicitly enabled (default: off). |
| `tools/run_critic.py` | critic | 5-check quality review covering CV gap (with gap-attribution downgrade), prediction bias/variance, feature concentration, walk-forward plausibility, and prediction sanity. Invokes `family_ablation.py`. Writes `reports/critic_review.json`, `reports/critic_was_here.txt`, and optionally `reports/critic_retune_requested.json`. |
| `tools/build_submission.py` | submission_writer | Validates predictions against `data/sample_submission.csv` and writes `submission.csv` at the repo root. Renames `predicted_target` to the actual target column name. Run with `--repo-root PATH`. |
| `tools/generate_report.py` | report_writer | Assembles `report.pdf` from `reports/` artefacts using reportlab. Includes methodology, feature importance, prediction diagnostics, and limitations sections. |

---

## File contracts between agents

| Agent | Reads | Writes |
|-------|-------|--------|
| `orchestrator` (this CLAUDE.md, not a sub-agent) | — | `reports/pipeline_run.json` — created at Step 0, mutated at Step 3 verify (post-pass: `current_modeler_run_id` updated) and Step 3.6 retune dispatch (pre-dispatch: `critic_cycle` incremented). Read by critic, submission_writer, report_writer. See "Stage handoff contracts → Pipeline run state". |
| `schema_analyst` | `data/`, `data/DATA_DESCRIPTION.md` | `reports/profile.json`, `reports/schema_analysis.md`, `reports/schema_analyst_was_here.txt` |
| `feature_engineer` | `reports/schema_analysis.md`, `reports/profile.json`, `data/` | `reports/features.json`, `data/features_train.parquet`, `data/features_val.parquet` |
| `modeler` | `reports/schema_analysis.md`, `reports/features.json`, `data/features_train.parquet`, `data/features_val.parquet` | `reports/model_results.json` (includes `feature_importance_all`, `oof_mae`), `reports/predictions.csv` (columns: `row_id`, identifier cols, `predicted_target`), `reports/oof_predictions.csv` (columns: identifier cols, `fold`, `predicted_target`), `reports/modeler_was_here.txt`, `reports/modeler_completion.json` (status / dispatch_time / exit_code / artifact paths — see Stage handoff contracts) |
| `validator` | `reports/modeler_completion.json` (precondition: `status == "ok"`), `reports/profile.json`, `reports/model_results.json`, `reports/features.json`, `data/features_train.parquet`, `reports/oof_predictions.csv` (opt), `reports/schema_analysis.md` (opt) | `reports/validator_review.json` (includes `fold_maes`, `fold_train_sizes`, `gap_attribution` block), `reports/validator_was_here.txt` |
| `critic` | `reports/validator_review.json` (reads `gap_attribution`), `reports/model_results.json`, `reports/predictions.csv`, `reports/features.json`, `reports/profile.json`, `data/features_train.parquet` | `reports/critic_review.json` (includes `gap_attribution_used`, `family_ablation`), `reports/critic_was_here.txt`, `reports/family_ablation.json` (via ablation tool), optionally `reports/critic_retune_requested.json` (may include `net_harmful_families`) and `reports/critic_retune_attempted.txt` |
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

- **Do not re-invoke a sub-agent because its marker is absent** — absence of a marker
  while a process is still running means the stage is in progress, not failed. See the
  Sub-agent completion contract. The only permitted re-invocations are: (a) genuine
  failure (process exited AND marker still absent), and (b) the critic-triggered retune
  cycle (gated by `reports/critic_retune_requested.json`, never by a missing marker).
- **Do not create new sub-agent definitions** (`.md` files in `.claude/agents/`) for
  tasks not already registered. If a registered sub-agent fails or is missing, implement
  its fallback logic inline using Python or bash — this is allowed and expected. What is
  forbidden is inventing and registering a novel agent on the fly to handle a gap.
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
| `critic.md` | `critic` | **exists** | Step 3.6 — after validator completes; advisory + retune; never blocks submission |
| `submission_writer.md` | `submission_writer` | **exists** | Step 4 — after critic completes (or fails gracefully) |
| `report_writer.md` | `report_writer` | **exists** | Step 5 — after submission.csv is written |

---

## Remaining contract work

This pass wired `reports/pipeline_run.json` and the Step 3.6 retune branch.
Two follow-up passes are required before every freshness check switches
from heuristic-fallback to strict-enforcement and before the Step 3.5 / 3.6
/ 4 / 5 verify blocks match the per-stage completion-record contracts.

### Next pass — `modeler_run_id` propagation through four agent files

The strict freshness check (`upstream.modeler_run_id ==
pipeline_run.current_modeler_run_id`) cannot fire until every upstream
completion record carries `modeler_run_id`. Each agent file below needs a
small targeted edit:

| File | Edit |
|---|---|
| `.claude/agents/modeler.md` | Add `modeler_run_id` to the modeler_completion.json schema as a REQUIRED field; specify generation as a `<UTC_compact>_<hex8>` nonce fresh per dispatch, written into modeler_completion.json by the agent before returning OK. Today the field is only implied (critic.md assumes it); modeler.md does not yet mandate writing it. |
| `.claude/agents/validator.md` | One-line: copy `modeler_run_id` from `modeler_completion.json` into `validator_completion.json`. |
| `.claude/agents/submission_writer.md` | (a) Copy `modeler_run_id` through into `submission_writer_completion.json`. (b) Switch the strict freshness check from `session_start_epoch >= dispatch_time` to `modeler_completion.json["modeler_run_id"] == pipeline_run.json["current_modeler_run_id"]`. The 3 h mtime heuristic stays as defensive fallback for the (now-impossible-in-normal-runs) case where `pipeline_run.json` is absent. |
| `.claude/agents/report_writer.md` | Same as submission_writer, applied to every upstream completion record (modeler / validator / submission_writer). |

### Next pass — Step 3.5 / 3.6 / 4 / 5 verify-block reconciliation

The orchestrator-side verify blocks below were written before the
completion-record contract existed. Each one currently gates on `*_was_here.txt`
+ analytic-fields-in-review-JSON only; none of them gate on the
corresponding `*_completion.json`. Update each to mirror the per-stage
completion contract:

| Step | What to add to the verify block |
|---|---|
| 3.5 (validator) | Gate on `reports/validator_completion.json` exists, parses, `status == "ok"`, `exit_code == 0`. Update "Writes" list to include `validator_completion.json`. |
| 3.6 (critic) | Gate on `reports/critic_completion.json` exists, parses, `status ∈ {"ok", "failed", "blocked", "retune_requested"}`. The retune branch already documented above triggers on `status == "retune_requested"`; `"failed"` and `"blocked"` fall through to Step 4 per the existing no-block-on-critic-failure rule; only `"ok"` is a clean accept. Update "Writes" list to include `critic_completion.json`. |
| 4 (submission_writer) | Gate on `reports/submission_writer_completion.json` exists, parses, `status == "ok"`. The scored-deliverable post-exit gate already lives in submission_writer.md and does not need to be duplicated at the orchestrator level. Update "Writes" list to include `submission_writer_completion.json`. |
| 5 (report_writer) | Gate on `reports/report_writer_completion.json` exists, parses, `status == "ok"`. Update "Writes" list. |

### Future (gated on tool + orchestrator both changing) — raise the retune cap to 2

Tracked under `.claude/agents/critic.md` § "Future option: raising the cap
to 2 retunes". Requires `tools/run_critic.py` sentinel → persisted-counter
upgrade AND a corresponding orchestrator change (raise
`pipeline_run.json["retune_cap"]` from 1 to 2 and adjust the Step 3.6
narrative). A doc-only bump in either place desynchronises the contract
from what is actually enforced — never do that.
