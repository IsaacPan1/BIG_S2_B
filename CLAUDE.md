# Award B — Autonomous Data Analysis Pipeline

Blind tabular forecasting pipeline. On trigger phrase, run without human intervention to produce `submission.csv` and `report.pdf` within **2 hours wall-clock** and **1,000,000 tokens** (input + output combined).

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
| External data | None — no downloads, no web search, no pretrained model fetches at runtime |
| Human intervention | None after the initial prompt |
| GPU | Not available — CPU-only training |
| Submission | **Must always be written**, even on failure |
| Network access | Prohibited — no `web_search`, `web_fetch`, external APIs, or data beyond `data/` |

---

## Workflow

Execute the following sub-agents in order using the **Task tool**.
Each sub-agent is defined in `.claude/agents/` — use its registered name exactly.
After each agent completes, read its primary output file and verify it is
non-empty and sensible before continuing.

### Sub-agent completion contract (read before invoking any stage)

**MARKER IS AUTHORITATIVE — ABSENCE ≠ FAILURE.** A sub-agent is COMPLETE when its `*_was_here.txt` marker file exists. Absence means STILL RUNNING or genuinely failed — two different states, must NOT be conflated. Do NOT re-invoke solely because outputs are not yet present.

**NEVER DOUBLE-INVOKE.** Before invoking any sub-agent, check whether its marker already exists OR a process for it is already running. NEVER launch a second instance while a prior one may still be running — concurrent writes to `reports/` corrupt outputs. If a stage appears stalled, wait; do not relaunch.

**HOW TO DISTINGUISH 'STILL RUNNING' FROM 'FAILED'.** The only valid failure signal: process has EXITED (any exit code) AND marker is still absent after exit. Re-invocation or fallback logic is permitted ONLY in that genuine-failure state — never on a not-yet-present check during an active run.

| Stage | Typical duration | Outer budget |
|-------|-----------------|--------------|
| feature_engineer | 2–5 min | 15 min |
| modeler | 8–15 min (Optuna tuning across families) | 90 min |
| validator | 3–8 min | 10 min |
| critic | 2–5 min | 5 min |

### Stage handoff contracts

Every handoff is an **artifact contract** — downstream preconditions are upstream stage files plus a completion-record; verbal handoffs are NOT permitted. The orchestrator's verify block is the gate.

**Completion-record schema** (written by modeler; same schema for stages adopting the pattern):

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

**Stale-marker guard:** A marker may be left from a prior run. Gates must verify `marker.mtime > dispatch_time`. Capture `dispatch_time` BEFORE launching the backing script.

#### Pipeline run state — `reports/pipeline_run.json`

Orchestrator-owned mutable state record. Sub-agents READ; only the orchestrator WRITES.

**Schema:**

```json
{
  "session_start_iso":      "2026-06-08T21:01:12.540453+00:00",
  "session_start_epoch":    1780952472.540453,
  "total_budget_seconds":   7200,
  "current_modeler_run_id": null,
  "critic_cycle":           0,
  "retune_cap":             1
}
```

**Write points** (`session_start_*`, `total_budget_seconds`, `retune_cap` are write-once at Step 0):

| When | Field updated | Value |
|------|--------------|-------|
| Step 0 — CREATE (overwrite unconditionally if prior file exists) | all fields | fresh session values |
| Step 3 verify passes — initial pass AND retune pass | `current_modeler_run_id` | `modeler_completion.json["modeler_run_id"]` |
| Step 3.6 retune branch BEFORE dispatch | `critic_cycle` | incremented (+1, written to disk before re-dispatch) |

**Enforcement:** `remaining_budget = session_start_epoch + total_budget_seconds − now()` (budget guard in Step 3.6); `critic_cycle < retune_cap` (cycle cap); every downstream `modeler_run_id` must equal `current_modeler_run_id` (freshness — checked by validator, critic, submission_writer, report_writer).

### Step 0 — preflight check + initialize `reports/pipeline_run.json` (ALWAYS first concrete action)

#### Step 0a — interpreter preflight (HARD STOP on failure)

Before any other action, verify that `python3.11` — the same invocation
all pipeline tools use — resolves to Python 3.11 (any 3.11.x patch). Run this via the Bash tool:

```bash
python3.11 tools/preflight_check.py
```

**If exit code is non-zero: HARD STOP immediately.** Do not create
`pipeline_run.json`. Do not invoke any sub-agent. Do not attempt a
degraded run or fallback submission. Print the script's error message to
the user and stop — the tools cannot import cleanly under the detected
interpreter, and any submission produced would be invalid. The fix is to
ensure `python3.11` is in PATH per README Section 3 and relaunch.

**Why `python3.11` and not bare `python`.** Claude Code's Bash tool does not
inherit the launch-environment PATH, so bare `python` resolves to the system
interpreter (potentially an older version) even when Claude Code is launched
from inside an activated 3.11 venv. All pipeline tools are invoked via
`python3.11` explicitly, so the preflight checks the same versioned binary they
use. If `python3.11` is absent ("command not found"), Bash exits with code 127
before the script runs — the HARD STOP fires regardless.

#### Step 0b — initialize `reports/pipeline_run.json`

After the preflight passes, create a fresh `reports/pipeline_run.json`
per the schema in "Stage handoff contracts → Pipeline run state".

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

Step 0 has no marker file and no sub-agent — it's two orchestrator-inline
actions (preflight Bash call + pipeline_run.json write).

### Step 1 — schema_analyst (ALWAYS first sub-agent)

```
Use the schema_analyst sub-agent on the data in data/.
```

- Reads: `data/DATA_DESCRIPTION.md`, all CSVs in `data/`
- Runs: `python3.11 tools/profile_data.py --data-dir data/ --output reports/profile.json`
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

**Execute `tools/run_modeler.py` synchronously, in the foreground, from the
orchestrator's own Bash tool call. Do NOT invoke the `modeler` sub-agent via
the Task tool. Do NOT use `run_in_background`. Do NOT use trailing `&`,
`nohup`, `Start-Process -NoWait`, PowerShell jobs, or any other detach /
backgrounding mechanism.** (Task-tool dispatch previously caused orphaned processes and half-written artifacts.)

**Two structural invariants this enforces — neither can orphan:**
1. The orchestrator's tool call is the direct parent of `run_modeler.py`.
   No `Agent` / `Task` indirection sits between them.
2. The orchestrator's tool call does not return until the child exits.
   No background flag, no detachment, no polling loop racing the child.

- Reads: `reports/schema_analysis.md`, `reports/features.json`,
         `data/features_train.parquet`, `data/features_val.parquet`
- Writes (by `tools/run_modeler.py`):
         `reports/model_results.json`, `reports/predictions.csv`,
         `reports/oof_predictions.csv`, `reports/modeler_was_here.txt`
- Writes (by the orchestrator, after the script exits):
         `reports/modeler_completion.json`
- Budget: **90 minutes**

**Invocation contract.** Mint `dispatch_time` AND `modeler_run_id` BEFORE
the script runs (the orchestrator now owns both — `run_modeler.py` itself
mints neither), then issue a SINGLE blocking subprocess call:

```python
import datetime, json, os, secrets, subprocess, sys

now            = datetime.datetime.now(datetime.timezone.utc)
dispatch_iso   = now.isoformat()                                  # tz-aware, "+00:00"
dispatch_epoch = now.timestamp()                                  # POSIX float, UTC
modeler_run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{secrets.token_hex(4)}"

# BLOCKING. Returns only when the child has fully exited.
# No run_in_background, no shell &, no nohup, no detach.
proc = subprocess.run(
    [sys.executable, "tools/run_modeler.py"],
    cwd=os.getcwd(),
    check=False,
    capture_output=True,
    text=True,
    timeout=90 * 60,        # 90 min hard ceiling — single source of truth for Step 3 & 3.6.
                            # Basis: measured full-mode pass ~46 min clean / ~58 min under
                            # memory pressure; 90 min clears the worst case while leaving
                            # ~30 min for validator/critic/submission/report inside the 2 h
                            # total budget. NOT 120 min — that would starve downstream stages.
)
exit_code = proc.returncode
tail = "\n".join(
    (proc.stdout or "").splitlines()[-25:]
    + (proc.stderr or "").splitlines()[-25:]
)
```

If invoked through the Bash tool rather than `subprocess.run`, the
equivalent command is a plain `python3.11 tools/run_modeler.py` line WITHOUT
`run_in_background=true` set on the tool call. The Bash tool's default
foreground semantics already block until exit and propagate the exit
code; the only failure mode is enabling backgrounding, so do not.

NEVER reparse a naive ISO string with `datetime.fromisoformat(...).
timestamp()` later — that interprets the string as local time and breaks
the mtime comparison on any machine outside UTC. Always carry
`dispatch_epoch` forward as a float, or always carry a tz-aware ISO
string that round-trips.

**After the script exits — synthesize the completion record FIRST, then
gate.** `tools/run_modeler.py` writes `modeler_was_here.txt` and the
three analytic artifacts, but it does NOT write `modeler_completion.json`
and it does NOT mint `modeler_run_id`. That ownership now sits with the
orchestrator. Write the record unconditionally — downstream gates need a
parsable record on both the success and failure paths:

```python
status = "ok" if exit_code == 0 else "failed"
completion = {
    "stage": "modeler",
    "status": status,
    "dispatch_time": dispatch_iso,
    "exit_code": exit_code,
    "modeler_run_id": modeler_run_id,
    "artifacts": {
        "model_results":   "reports/model_results.json",
        "predictions":     "reports/predictions.csv",
        "oof_predictions": "reports/oof_predictions.csv",
        "marker":          "reports/modeler_was_here.txt",
    },
    "notes": "" if status == "ok" else tail,
}
with open("reports/modeler_completion.json", "w") as f:
    json.dump(completion, f, indent=2)
```

Verify after completion — this is an **INDEPENDENT artifact gate**; do
NOT trust the recorded status alone. Re-run every check from this side:

1. `subprocess.run` returned (the child process has exited). This is now
   structurally guaranteed by the invocation contract above — there is
   no "still running" branch to handle. If the call raised
   `TimeoutExpired`, treat it as `exit_code = -1`, `status = "failed"`,
   and `notes = "timeout at 5400s"` when synthesizing the completion
   record, then fall through to the failure branch below.
2. `reports/modeler_completion.json` exists, parses as JSON, and has
   `status == "ok"` and `exit_code == 0`.
3. `reports/modeler_was_here.txt` exists AND its mtime is strictly newer
   than `dispatch_epoch` (compare in epoch-float seconds; rejects
   leftover markers from prior runs).
4. `reports/model_results.json` exists, size > 0, parses as JSON.
5. `reports/predictions.csv` exists, size > 0, the prediction column
   matches `target_col` (or is `predicted_target` for submission_writer
   to rename), no NaN predictions, **and `len(predictions.csv) ==
   len(features_val.parquet)` exactly**. The validation-feature parquet
   is the authoritative row-count reference because feature_engineer may
   expand the raw validation rows (e.g. cross-joining missing category
   levels) — `profile.json.n_val_rows` is the RAW row count and will
   NOT match. Do not use `model_results.json.n_val_rows` either: that is
   producer self-reporting and gives circular agreement.
6. `reports/oof_predictions.csv` exists, size > 0.

**After all six checks pass, propagate `modeler_run_id` into the
run-state record.** The value is already in
`modeler_completion.json["modeler_run_id"]` because the orchestrator
wrote it there one step earlier; copy it into
`reports/pipeline_run.json["current_modeler_run_id"]`. This applies to
BOTH the initial Step 3 dispatch and any Step 3.6 retune re-dispatch —
the field always tracks the latest modeler pass. After this update, the
downstream validator / critic / submission_writer / report_writer
freshness checks all fire against `current_modeler_run_id` in their own
Step 1 precondition gates AND at the orchestrator-side Step 3.x verify
gates — every downstream agent contract consumes the field.

```python
import json
mc = json.load(open("reports/modeler_completion.json"))
pr = json.load(open("reports/pipeline_run.json"))
pr["current_modeler_run_id"] = mc["modeler_run_id"]
json.dump(pr, open("reports/pipeline_run.json", "w"), indent=2)
```

Only on full pass (all six checks AND the `pipeline_run.json` update):
dispatch the validator (Step 3.5).

On any failure (non-zero exit code or timeout, missing/empty/invalid
artifact, stale marker): the orchestrator-written completion record
already captures `status == "failed"` and the stdout/stderr tail —
nothing to synthesize. Generate a group-mean baseline prediction for
`reports/predictions.csv` using Python directly (do not invoke another
agent), log the fallback, and continue to `submission_writer`. Do NOT
dispatch the validator on partial state.

**Step 3.6 retune re-dispatch uses this same procedure.** When the
Step 3.6 retune branch fires, the orchestrator re-runs `tools/run_modeler.py`
through the identical blocking-subprocess contract above — including the
same 90-min `timeout=90 * 60` ceiling (the Step 3 literal is the single
source of truth; do not redeclare it here) — minting a fresh
`dispatch_time` and a fresh `modeler_run_id`, synthesizing a fresh
`modeler_completion.json`, re-running the six-check gate, and updating
`current_modeler_run_id` on pass. The Task-subagent path is not used on
the retune either.

### Step 3.5 — validator

Invoke the validator sub-agent via the Task tool after modeler completes.
Do NOT perform validation inline — always delegate to the validator sub-agent.
The validator is **diagnostic only** — it does NOT modify predictions and does NOT
block submission regardless of verdict. This stage typically takes 3–8 minutes;
see the Sub-agent completion contract above — do NOT re-invoke if the marker
has not yet appeared.

**Capture `dispatch_time` as UTC ISO + epoch-float pair BEFORE invoking — same convention as Step 3.**

```
Use the validator sub-agent to audit the modeler's CV integrity.
```

- Reads: `reports/modeler_completion.json` (precondition: `status == "ok"`),
         `reports/pipeline_run.json` (freshness),
         `reports/profile.json`, `reports/model_results.json`, `reports/features.json`,
         `data/features_train.parquet`, `reports/oof_predictions.csv` (if present),
         `reports/schema_analysis.md` (optional)
- Writes: `reports/validator_review.json`, `reports/validator_was_here.txt`,
          `reports/validator_completion.json`
- Budget: **10 minutes**

Verify after completion — this is an **INDEPENDENT artifact gate**; do NOT
trust the sub-agent's verbal "done" report. Re-run every check from this side:

1. The sub-agent process has exited. If the marker is absent but the process
   is still alive, continue waiting — this is NOT a failure state.
2. `reports/validator_completion.json` exists, parses as JSON, has
   `status ∈ {"ok", "failed", "blocked"}` and an integer `exit_code` field.
3. `reports/validator_was_here.txt` exists AND its mtime is strictly newer
   than `dispatch_time` (epoch-float comparison; rejects leftover markers
   from prior runs).
4. `reports/validator_review.json` exists, size > 0, parses as JSON, and has
   a top-level `verdict ∈ {"PASS", "WARNING", "CRITICAL"}` (HARD). Other
   analytic keys (`reported_cv_mae`, `strict_cv_mae`, `fold_maes`,
   `gap_attribution`, etc.) are SOFT — log if missing, do NOT fail the gate
   on them.
5. **Freshness:** `validator_completion.json["modeler_run_id"] ==
   pipeline_run.json["current_modeler_run_id"]`. Mismatch → gate fails
   (the validator ran against a pre-retune modeler pass).

Status branch (on full gate pass):
- `status == "ok"` → dispatch Step 3.6 (critic).
- `status == "failed"` / `"blocked"` → record the situation; continue to
  Step 3.6. The validator is diagnostic and never blocks submission; a
  failed/blocked validator is not a pipeline-blocking outcome — the report
  surfaces it.

**Distinguish two failure modes that must NOT be conflated:**
- **Gate fail** (record missing/stale/unparseable, marker absent or stale,
  freshness mismatch) is an orchestrator-detected breakage. Synthesize a
  minimal `reports/validator_completion.json` with `status="failed"` AND a
  minimal `reports/validator_review.json` with `verdict="WARNING"` +
  `notes` naming the orchestrator-detected breakage. Continue to Step 3.6.
- **Verdict fail** (the validator ran cleanly and returned
  `verdict ∈ {"WARNING", "CRITICAL"}`) is NOT a gate fail — it is the
  validator doing its job. Pass through unchanged; the verdict surfaces in
  the report; nothing else changes.

### Step 3.6 — Critic Review

Invoke the critic sub-agent via the Task tool. This stage typically takes
2–5 minutes; see the Sub-agent completion contract above — do NOT re-invoke
if the marker has not yet appeared. Do NOT generate critic review inline;
always delegate to the critic sub-agent.

**Capture `dispatch_time` as UTC ISO + epoch-float pair BEFORE invoking — same convention as Step 3.**

```
Use the critic sub-agent to review the validator output and modeler predictions.
```

- Reads: `reports/modeler_completion.json` (precondition: `status == "ok"`),
         `reports/pipeline_run.json` (freshness + cycle counter),
         `reports/validator_review.json`, `reports/model_results.json`,
         `reports/predictions.csv`, `reports/features.json`,
         `reports/profile.json`, `data/features_train.parquet`
- Writes: `reports/critic_review.json`, `reports/critic_was_here.txt`,
          `reports/critic_completion.json` (extended status set incl.
          `"retune_requested"`; carries `modeler_run_id`, `cycle`,
          `retune_reason`), optionally
          `reports/critic_retune_requested.json` and
          `reports/critic_retune_attempted.txt`
- Budget: **5 minutes**

Verify after completion — this is an **INDEPENDENT artifact gate**; do NOT
trust the sub-agent's verbal "done" report. Re-run every check from this
side. The gate runs FIRST; the retune-vs-proceed decision below consumes
its outcome.

1. The sub-agent process has exited. If the marker is absent but the process
   is still alive, continue waiting — this is NOT a failure state.
2. `reports/critic_completion.json` exists, parses as JSON, has
   `status ∈ {"ok", "failed", "blocked", "retune_requested"}` (extended set
   per `critic.md`) and an integer `exit_code` field.
3. `reports/critic_was_here.txt` exists AND its mtime is strictly newer than
   `dispatch_time` (epoch-float comparison).
4. `reports/critic_review.json` exists, size > 0, parses as JSON, and has a
   top-level `status` field (the script's vocabulary,
   `{"accepted", "retune_requested"}` — distinct from the completion
   record's extended set). Other analytic keys (`checks`,
   `gap_attribution_used`, `family_ablation`, `cycle`, `decision_rationale`)
   are SOFT — log if missing, do NOT fail the gate.
5. **Freshness:** `critic_completion.json["modeler_run_id"] ==
   pipeline_run.json["current_modeler_run_id"]`. Mismatch → gate fails (the
   critic ran against a pre-retune modeler pass).
6. If `critic_completion.json["status"] == "retune_requested"`:
   `reports/critic_retune_requested.json` also exists and parses (the
   conditional retune-signal artifact named in the critic's outputs).
   Absence → gate fails.

**Gate-fail action** (record missing/stale/unparseable, marker absent or
stale, freshness mismatch, retune-signal absent when status claims it):
synthesize a minimal `reports/critic_completion.json` with `status="failed"`
AND a minimal `reports/critic_review.json` with `status="accepted"` +
`notes` describing the orchestrator-detected breakage. Continue to Step 4
per "never blocks submission". **NEVER trigger the retune loop on a
gate-fail.**

**Status branch** (on full gate pass) — the gate runs FIRST; the branch
consumes its outcome:

**`status == "ok"`** → proceed to Step 4.

**`status == "retune_requested"`** — INTENTIONAL, gated re-invocation of
the modeler, distinct from the accidental double-invocation that the
Sub-agent completion contract prohibits. The trigger is the critic's
structured completion status; the cycle cap and budget guard are enforced
by `reports/pipeline_run.json` at the orchestrator, not by file-existence
heuristics:

   1. Read `reports/pipeline_run.json`. Compute
      `remaining_budget = session_start_epoch + total_budget_seconds − now()`.
   2. **Cycle cap check.** If `critic_cycle >= retune_cap` (cap = 1, matching
      `critic.md`): force-proceed to Step 4. Log the cap-hit; do NOT dispatch
      the retune. The critic should already have force-accepted at this cycle
      by its own rules, so this branch is a defense-in-depth.
   3. **Budget guard.** If `remaining_budget < 25 * 60` seconds (a typical
      retune cycle: modeler + validator + critic + slack): force-proceed to
      Step 4. Log the budget-exhaustion reason; do NOT dispatch the retune.
   4. **Both checks pass — dispatch the retune.** In this order, atomically:
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

The cap value lives in `pipeline_run.json["retune_cap"]` (default `1`,
matching `critic.md`). Raising the cap requires the coordinated changes
spelled out in `critic.md` § "Future option: raising the cap to 2 retunes" —
do NOT raise it here unilaterally.

**`status == "failed"` / `"blocked"`** → continue to Step 4 per "never
blocks submission". Do NOT enter the retune loop. The critic ran but
either could not produce a clean verdict (`"failed"`) or could not start
because its precondition failed (`"blocked"`). The agent already wrote an
honest `critic_completion.json` — no synthesis needed. If `critic_review.
json` is absent, synthesize a minimal one with `status="accepted"` so the
report_writer can call it out without crashing on a missing file.

After at most one cap-bounded retune cycle, proceed to Step 4. The pipeline
always reaches submission_writer regardless of which branch fires.

### Step 4 — submission_writer

Invoke the submission_writer sub-agent via the Task tool. Do NOT write
`submission.csv` inline — always delegate to the submission_writer sub-agent.

**Capture `dispatch_time` as UTC ISO + epoch-float pair BEFORE invoking — same convention as Step 3.**

```
Use the submission_writer sub-agent to validate and write submission.csv.
```

- Reads: `reports/modeler_completion.json` (precondition: `status == "ok"`),
         `reports/pipeline_run.json` (freshness),
         `reports/predictions.csv`, `data/DATA_DESCRIPTION.md`,
         `data/sample_submission.csv`
- Writes: `submission.csv` at the repo root, `reports/submission_summary.json`,
          `reports/submission_writer_was_here.txt`,
          `reports/submission_writer_completion.json`
- Budget: **10 minutes**

Verify after completion — this is an **INDEPENDENT artifact gate**.
`submission.csv` is the scored deliverable; the gate MUST validate the
file itself, not just the completion record. Re-run every check from this
side:

1. The sub-agent process has exited. If the marker is absent but the process
   is still alive, continue waiting.
2. `reports/submission_writer_completion.json` exists, parses as JSON, has
   `status ∈ {"ok", "failed", "blocked"}` and an integer `exit_code` field.
3. `reports/submission_writer_was_here.txt` exists AND its mtime is strictly
   newer than `dispatch_time` (epoch-float comparison).
4. `reports/submission_summary.json` exists, parses as JSON, and contains
   `round_trip_audit.status == "PASS"` (HARD — the audit is the script's
   own integrity check). Other analytic keys (`prediction_stats`,
   `join_key_used`, `target_column`, etc.) are SOFT — log if missing, do
   NOT fail the gate.
5. **Scored-deliverable check on `submission.csv` itself — do NOT trust the
   completion record alone:**
   - File at the repo root exists, parses as CSV.
   - Exactly two columns: `["row_id", <target_col>]` where `<target_col>`
     is `profile.json["target_col"]`. Any other column set is a gate fail.
   - Row count equals `len(data/sample_submission.csv)`.
   - The target column has no NaN values.
6. **Freshness:** `submission_writer_completion.json["modeler_run_id"] ==
   pipeline_run.json["current_modeler_run_id"]`. Mismatch → gate fails.

Only on full pass: dispatch Step 5.

**Gate-fail action — build a fallback `submission.csv` with the CORRECT
schema.** The graded file must be exactly `[row_id, <target_col>]` — two
columns, nothing else. The fallback always produces a schema-correct file;
values may be wrong, but the submission is well-formed:

1. Build a two-column DataFrame `[row_id, <target_col>]` using the
   `row_id` values from `data/sample_submission.csv`.
2. Fill the target column. Preferred source: a composite-key join of
   `reports/predictions.csv` onto `sample_submission.csv` (composite
   business key from `profile.json`'s `group_cols + time_col` — same join
   key `tools/build_submission.py` uses; do NOT join on `row_id`). If the
   join cannot be performed (predictions absent, key columns missing, no
   matches), fill the target column with the training-target mean from
   whichever file the profile points to as the train target source.
3. Log the failure reason and the chosen fill source. Continue to Step 5.

### Step 5 — report_writer

Invoke the report_writer sub-agent via the Task tool. Do NOT generate
`report.pdf` inline — always delegate to the report_writer sub-agent.

**Capture `dispatch_time` as UTC ISO + epoch-float pair BEFORE invoking — same convention as Step 3.**

```
Use the report_writer sub-agent to produce report.pdf.
```

- Reads: `reports/modeler_completion.json`,
         `reports/validator_completion.json`,
         `reports/submission_writer_completion.json` (union of upstream
         completion records — all must satisfy
         `status == "ok"` per `report_writer.md` § Step 1),
         `reports/pipeline_run.json` (freshness),
         `reports/` (all analytic files), `data/DATA_DESCRIPTION.md`
- Writes: `report.pdf` at the repo root, `reports/report_writer_was_here.txt`,
          `reports/report_writer_completion.json`,
          optionally `reports/feature_importance.png`,
          `reports/prediction_histogram.png`
- Budget: **20 minutes**

Verify after completion — this is an **INDEPENDENT artifact gate**. Re-run
every check from this side:

1. The sub-agent process has exited. If the marker is absent but the process
   is still alive, continue waiting.
2. `reports/report_writer_completion.json` exists, parses as JSON, has
   `status ∈ {"ok", "failed", "blocked"}` and an integer `exit_code` field.
3. `reports/report_writer_was_here.txt` exists AND its mtime is strictly
   newer than `dispatch_time` (epoch-float comparison).
4. `report.pdf` at the repo root exists and size > 0. **`report.txt`-only is
   treated as a gate FAIL** — `report.txt` is the script's degraded fallback
   when `reportlab` is unavailable; the contract requires PDF, matching
   `report_writer.md`'s own post-exit gate.
5. **Freshness:** `report_writer_completion.json["modeler_run_id"] ==
   pipeline_run.json["current_modeler_run_id"]`. Mismatch → gate fails.

Only on full pass: pipeline complete.

**Gate-fail action.** Write a minimal one-page `report.pdf` using Python +
reportlab directly (do not invoke another agent) with: dataset name,
problem type, model used, final metric from `reports/model_results.json`.
If `reportlab` is unavailable in the orchestrator's environment, write a
minimal `report.txt` and log that PDF generation is impossible — `report.txt`
is not contract-compliant but is the only fallback when `reportlab` is
genuinely missing system-wide.

---

## Time budget allocation

| Phase | Budget | Notes |
|-------|--------|-------|
| schema_analyst | 5 min | Non-negotiable — never skip or shorten |
| feature_engineer | 15 min | Reduce feature complexity if falling behind |
| modeler | 90 min | Subprocess hard ceiling. Basis: measured ~46 min clean / ~58 min under memory pressure; 90 min clears the worst case. Cap Optuna at 30 trials if time is short |
| validator | 10 min | Diagnostic only; never blocks submission |
| critic | 5 min | Advisory + retune; never blocks submission |
| submission_writer | 10 min | Should be fast; escalate if it stalls |
| report_writer | 20 min | Reduce to text-only PDF if time is short |
| Buffer | 15 min | Reserved for retries and fallback logic |
| **Total** | **170 min** | Typical modeler runs ~46–58 min, not 90. Phases compress if behind 120-min wall-clock limit; 90-min modeler ceiling is a kill-switch, not a target. If critic triggers a retune, modeler + validator + critic re-run (~75 min); subsequent phases must use fallback variants. |

**Self-pacing rule**: After each phase, estimate remaining wall-clock time.
If ≥ 75 % of the token budget or time budget is consumed and fewer than
two phases are complete, switch all remaining phases to their simplest
fallback variants immediately.

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

## Sub-agents registered in .claude/agents/

| File | Name | When invoked |
|------|------|-------------|
| `schema_analyst.md` | `schema_analyst` | Step 1 — always, immediately |
| `feature_engineer.md` | `feature_engineer` | Step 2 — after schema_analyst succeeds |
| `modeler.md` | `modeler` | Step 3 — NOT via Task tool; run `tools/run_modeler.py` directly (blocking) |
| `validator.md` | `validator` | Step 3.5 — after modeler; diagnostic only |
| `critic.md` | `critic` | Step 3.6 — after validator; advisory + retune; never blocks submission |
| `submission_writer.md` | `submission_writer` | Step 4 — after critic completes (or fails gracefully) |
| `report_writer.md` | `report_writer` | Step 5 — after submission.csv is written |
