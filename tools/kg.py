"""
tools/kg.py — Project Knowledge Graph helper (observability only).

The KG is a derived, append-only JSON file: reports/knowledge_graph.json.
Per-agent files (profile.json, features.json, model_results.json, etc.)
remain the SOLE source of truth.  The KG is a cross-stage audit trail.

What the KG does NOT do
-----------------------
* Act as source of truth — always read per-agent JSON files for decisions.
* Gate, block, or alter control flow — a missing or corrupt KG must never
  stop an agent; every call is wrapped in try/except.
* Store large data payloads — only summaries, pointers, and top-N slices.
* Replace marker files (*_was_here.txt) as proof of agent execution.
* Grow unbounded — hypothesis_log and resource_usage are capped at 10.

JSON schema (reports/knowledge_graph.json)
------------------------------------------
{
  "schema_version": "1.0",
  "project_status": {
    "current_stage": "init|schema_analyst|feature_engineer|modeler|
                      validator|critic|submission_writer|complete|failed",
    "last_updated":    "<ISO-8601 UTC>",
    "pipeline_start":  "<ISO-8601 UTC>",
    "retune_cycles_used": 0,   // shared mutable counter; the only mutable field
    "retune_budget":      1
  },
  "inferred_problem": {
    "problem_type":    null,
    "problem_subtype": null,
    "target_col":      null,
    "n_train":         null,
    "n_val":           null,
    "group_cols":      [],
    "time_col":        null,
    "value_constraints": {
      "non_negative": null,   // bool — true if train target min >= 0
      "min_val":      null,   // float
      "max_val":      null,   // float
      "is_integer":   null    // bool — true for ordinal/count subtypes
    }
  },
  "source_of_truth": {        // static path registry — never write data here
    "profile_json":            "reports/profile.json",
    "schema_analysis_md":      "reports/schema_analysis.md",
    "features_json":           "reports/features.json",
    "features_train_parquet":  "data/features_train.parquet",
    "features_val_parquet":    "data/features_val.parquet",
    "model_results_json":      "reports/model_results.json",
    "predictions_csv":         "reports/predictions.csv",
    "oof_predictions_csv":     "reports/oof_predictions.csv",
    "validator_review_json":   "reports/validator_review.json",
    "critic_review_json":      "reports/critic_review.json",
    "submission_csv":          "submission.csv",
    "report_pdf":              "report.pdf"
  },
  "hypothesis_log": [         // append-only; capped at 10 most-recent entries
    {
      "ts":         "<ISO-8601 UTC>",
      "stage":      "<agent name>",
      "hypothesis": "<brief claim being tested>",
      "outcome":    "confirmed|rejected|pending",
      "detail":     "<one-line summary of evidence>"
    }
  ],
  "optuna_reflection": {      // last modeler Optuna run (merged, not appended)
    "n_trials":      null,
    "best_params":   {},
    "best_oof_mae":  null,
    "ensemble_path": null,
    "n_seeds":       null
  },
  "rejected_hypotheses": [    // infrequent; uncapped
    {
      "ts":         "<ISO-8601 UTC>",
      "stage":      "<agent name>",
      "hypothesis": "<claim>",
      "reason":     "<why rejected>"
    }
  ],
  "resource_usage": [         // append-only; capped at 10 most-recent entries
    {
      "ts":              "<ISO-8601 UTC>",
      "stage":           "<agent name>",
      "n_features":      null,
      "n_families":      null,
      "elapsed_seconds": null
    }
  ],
  "feature_importance_snapshot": {  // top-10 slice from last modeler run
    "top_10":         [],   // [{"feature": str, "importance": float}, ...]
    "total_features": null,
    "n_families":     null
  }
}

Integration (one try/except block per agent, ≤5 code lines each)
-----------------------------------------------------------------
schema_analyst    → kg_init() + kg_set_stage() + kg_append_event("inferred_problem")
feature_engineer  → kg_append_event("resource")  + kg_set_stage()
modeler           → kg_append_event("optuna") + kg_append_event("feature_importance") + kg_set_stage()
validator         → kg_append_event("hypothesis") + kg_set_stage()
critic            → kg_append_event("hypothesis") + kg_set_stage(retune_cycles_used=N)
submission_writer → kg_set_stage()
report_writer     → kg_set_stage("complete")
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KG_PATH = Path("reports/knowledge_graph.json")
_MAX_LOG = 10          # max entries kept in hypothesis_log and resource_usage
_SCHEMA_VERSION = "1.0"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scaffold() -> dict:
    now = _now()
    return {
        "schema_version": _SCHEMA_VERSION,
        "project_status": {
            "current_stage": "init",
            "last_updated": now,
            "pipeline_start": now,
            "retune_cycles_used": 0,
            "retune_budget": 1,
        },
        "inferred_problem": {
            "problem_type": None,
            "problem_subtype": None,
            "target_col": None,
            "n_train": None,
            "n_val": None,
            "group_cols": [],
            "time_col": None,
            "value_constraints": {
                "non_negative": None,
                "min_val": None,
                "max_val": None,
                "is_integer": None,
            },
        },
        "source_of_truth": {
            "profile_json": "reports/profile.json",
            "schema_analysis_md": "reports/schema_analysis.md",
            "features_json": "reports/features.json",
            "features_train_parquet": "data/features_train.parquet",
            "features_val_parquet": "data/features_val.parquet",
            "model_results_json": "reports/model_results.json",
            "predictions_csv": "reports/predictions.csv",
            "oof_predictions_csv": "reports/oof_predictions.csv",
            "validator_review_json": "reports/validator_review.json",
            "critic_review_json": "reports/critic_review.json",
            "submission_csv": "submission.csv",
            "report_pdf": "report.pdf",
        },
        "hypothesis_log": [],
        "optuna_reflection": {
            "n_trials": None,
            "best_params": {},
            "best_oof_mae": None,
            "ensemble_path": None,
            "n_seeds": None,
        },
        "rejected_hypotheses": [],
        "resource_usage": [],
        "feature_importance_snapshot": {
            "top_10": [],
            "total_features": None,
            "n_families": None,
        },
    }


def _load_raw() -> dict:
    """Load KG from disk; return fresh scaffold on any failure (corrupt file, missing)."""
    try:
        if KG_PATH.exists():
            with open(KG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and data.get("schema_version") == _SCHEMA_VERSION:
                return data
    except Exception:
        pass
    return _scaffold()


def _atomic_write(kg: dict) -> None:
    """Write KG via tempfile + os.replace to prevent torn reads on crash."""
    KG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(KG_PATH.parent), suffix=".kg.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(kg, fh, indent=2, default=str)
        os.replace(tmp, str(KG_PATH))
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


# ── Public API ────────────────────────────────────────────────────────────────

def kg_init(pipeline_start: str | None = None) -> None:
    """
    Create (or reset) the KG scaffold.  Call once at the start of the pipeline
    (schema_analyst).  Preserves pipeline_start if provided.
    """
    try:
        kg = _scaffold()
        if pipeline_start:
            kg["project_status"]["pipeline_start"] = pipeline_start
        _atomic_write(kg)
    except Exception as exc:
        print(f"[KG] kg_init failed (non-fatal): {exc}", file=sys.stderr)


def kg_load() -> dict:
    """
    Return the current KG dict.  Returns an empty scaffold on any failure.
    Agents may call this for diagnostics but must NEVER branch on its content.
    """
    try:
        return _load_raw()
    except Exception as exc:
        print(f"[KG] kg_load failed (non-fatal): {exc}", file=sys.stderr)
        return _scaffold()


def kg_set_stage(stage: str, extra: dict[str, Any] | None = None) -> None:
    """
    Update project_status.current_stage and optionally merge extra fields
    (e.g. retune_cycles_used) into project_status.
    """
    try:
        kg = _load_raw()
        kg["project_status"]["current_stage"] = stage
        kg["project_status"]["last_updated"] = _now()
        if extra:
            kg["project_status"].update(extra)
        _atomic_write(kg)
    except Exception as exc:
        print(f"[KG] kg_set_stage failed (non-fatal): {exc}", file=sys.stderr)


def kg_append_event(
    stage: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """
    Append or merge an event into the KG.

    event_type          action
    ────────────────    ──────────────────────────────────────────────────────
    "hypothesis"      → appended to hypothesis_log; list capped at _MAX_LOG
    "resource"        → appended to resource_usage;  list capped at _MAX_LOG
    "inferred_problem"→ merged (update) into inferred_problem
    "optuna"          → merged (update) into optuna_reflection
    "feature_importance" → merged (update) into feature_importance_snapshot
    """
    try:
        kg = _load_raw()
        ts = _now()

        if event_type == "hypothesis":
            kg["hypothesis_log"].append({"ts": ts, "stage": stage, **payload})
            kg["hypothesis_log"] = kg["hypothesis_log"][-_MAX_LOG:]

        elif event_type == "resource":
            kg["resource_usage"].append({"ts": ts, "stage": stage, **payload})
            kg["resource_usage"] = kg["resource_usage"][-_MAX_LOG:]

        elif event_type == "inferred_problem":
            kg["inferred_problem"].update(payload)

        elif event_type == "optuna":
            kg["optuna_reflection"].update(payload)

        elif event_type == "feature_importance":
            kg["feature_importance_snapshot"].update(payload)

        else:
            print(f"[KG] unknown event_type={event_type!r} — skipped", file=sys.stderr)
            return

        kg["project_status"]["last_updated"] = ts
        _atomic_write(kg)
    except Exception as exc:
        print(f"[KG] kg_append_event failed (non-fatal): {exc}", file=sys.stderr)


def kg_record_rejected(stage: str, hypothesis: str, reason: str) -> None:
    """
    Append one entry to rejected_hypotheses.
    Use when a model family or feature hypothesis is explicitly discarded
    (e.g. Ridge excluded because OOF MAE > 1.5× best tree).
    """
    try:
        kg = _load_raw()
        kg["rejected_hypotheses"].append({
            "ts": _now(),
            "stage": stage,
            "hypothesis": hypothesis,
            "reason": reason,
        })
        _atomic_write(kg)
    except Exception as exc:
        print(f"[KG] kg_record_rejected failed (non-fatal): {exc}", file=sys.stderr)
