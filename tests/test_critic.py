"""Schema regression tests for reports/critic_review.json.

Asserts that run_critic.py (and the critic sub-agent) always produce the exact
schema specified in .claude/agents/critic.md — regardless of dataset.

Run from the repo root:
    pytest tests/test_critic.py -v

These tests are intentionally stateless: they read whatever critic_review.json
is currently in reports/ and validate its structure.  Values are not checked —
only schema conformance.  If the file is absent (pipeline not yet run), the
tests are skipped automatically.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO        = Path(__file__).parent.parent
CRITIC_JSON = REPO / "reports" / "critic_review.json"

REQUIRED_TOP_LEVEL = {
    "status",
    "cycle",
    "checks",
    "warnings_for_report",
    "retune_attempted",
    "final_recommendation",
    "decision_rationale",
}
VALID_TOP_STATUSES = {"accepted", "retune_requested"}
VALID_CHECK_STATUSES = {"PASS", "WARNING", "CRITICAL"}
REQUIRED_CHECK_NAMES = {
    "validator_concordance",
    "prediction_distribution",
    "mae_plausibility",
    "feature_concentration",
    "prediction_sanity",
}
# Fields that belong in validator_review.json, NOT here.
OLD_SCHEMA_FIELDS = {
    "verdict",
    "reported_cv_mae",
    "strict_cv_mae",
    "cv_gap_pct",
    "suspect_features",
    "prediction_range_ok",
    "cv_gap_threshold_used",
    "suggested_change",
    "prediction_min",
    "prediction_max",
    "prediction_mean",
}


@pytest.fixture(scope="module")
def critic_review() -> dict:
    if not CRITIC_JSON.exists():
        pytest.skip(f"{CRITIC_JSON} not found — run the pipeline first.")
    with open(CRITIC_JSON) as f:
        return json.load(f)


class TestCriticSchema:
    """Verify critic_review.json conforms exactly to the schema in critic.md."""

    def test_required_top_level_fields_present(self, critic_review: dict) -> None:
        missing = REQUIRED_TOP_LEVEL - set(critic_review)
        assert not missing, (
            f"critic_review.json is missing required top-level fields: {missing}. "
            f"Expected all of: {sorted(REQUIRED_TOP_LEVEL)}. "
            f"Re-run tools/run_critic.py to regenerate with the correct schema."
        )

    def test_status_is_valid(self, critic_review: dict) -> None:
        status = critic_review.get("status")
        assert status in VALID_TOP_STATUSES, (
            f"'status' must be one of {VALID_TOP_STATUSES}, got {status!r}."
        )

    def test_cycle_is_1_or_2(self, critic_review: dict) -> None:
        cycle = critic_review.get("cycle")
        assert isinstance(cycle, int) and cycle in (1, 2), (
            f"'cycle' must be integer 1 or 2, got {cycle!r} (type={type(cycle).__name__})."
        )

    def test_retune_attempted_is_bool(self, critic_review: dict) -> None:
        retune = critic_review.get("retune_attempted")
        assert isinstance(retune, bool), (
            f"'retune_attempted' must be a bool, got {type(retune).__name__}: {retune!r}."
        )

    def test_warnings_for_report_is_list(self, critic_review: dict) -> None:
        wfr = critic_review.get("warnings_for_report")
        assert isinstance(wfr, list), (
            f"'warnings_for_report' must be a list (may be empty), "
            f"got {type(wfr).__name__}: {wfr!r}."
        )

    def test_warnings_for_report_contains_strings(self, critic_review: dict) -> None:
        wfr = critic_review.get("warnings_for_report", [])
        for i, item in enumerate(wfr):
            assert isinstance(item, str), (
                f"'warnings_for_report[{i}]' must be a string, "
                f"got {type(item).__name__}: {item!r}."
            )

    def test_final_recommendation_is_non_empty_string(self, critic_review: dict) -> None:
        fr = critic_review.get("final_recommendation")
        assert isinstance(fr, str) and len(fr) > 0, (
            f"'final_recommendation' must be a non-empty string, got {fr!r}."
        )

    def test_decision_rationale_is_non_empty_string(self, critic_review: dict) -> None:
        dr = critic_review.get("decision_rationale")
        assert isinstance(dr, str) and len(dr) > 0, (
            f"'decision_rationale' must be a non-empty string, got {dr!r}. "
            f"This field is required by critic.md and was missing from the old schema."
        )

    # ── checks array ──────────────────────────────────────────────────────────

    def test_checks_has_exactly_five_entries(self, critic_review: dict) -> None:
        checks = critic_review.get("checks", [])
        assert len(checks) == 5, (
            f"'checks' array must have exactly 5 entries, got {len(checks)}. "
            f"Required names: {sorted(REQUIRED_CHECK_NAMES)}. "
            f"Each must appear even if not applicable (use status='PASS', "
            f"details='not applicable: <reason>')."
        )

    def test_checks_has_all_required_names(self, critic_review: dict) -> None:
        checks = critic_review.get("checks", [])
        present = {c.get("name") for c in checks}
        missing = REQUIRED_CHECK_NAMES - present
        assert not missing, (
            f"'checks' array is missing named checks: {missing}. "
            f"Got names: {sorted(present)}."
        )

    def test_each_check_has_name_status_details(self, critic_review: dict) -> None:
        checks = critic_review.get("checks", [])
        for check in checks:
            name = check.get("name", "<unnamed>")
            missing_fields = {"name", "status", "details"} - set(check)
            assert not missing_fields, (
                f"Check '{name}' is missing required fields: {missing_fields}. "
                f"Each check must have name, status, and details."
            )

    def test_each_check_status_is_valid(self, critic_review: dict) -> None:
        checks = critic_review.get("checks", [])
        for check in checks:
            name   = check.get("name", "<unnamed>")
            status = check.get("status")
            assert status in VALID_CHECK_STATUSES, (
                f"Check '{name}' has invalid status {status!r}. "
                f"Must be one of {VALID_CHECK_STATUSES}."
            )

    def test_each_check_details_is_non_empty_string(self, critic_review: dict) -> None:
        checks = critic_review.get("checks", [])
        for check in checks:
            name    = check.get("name", "<unnamed>")
            details = check.get("details")
            assert isinstance(details, str) and len(details) > 0, (
                f"Check '{name}' has invalid details {details!r} — "
                f"must be a non-empty string."
            )

    # ── anti-regression: old schema fields must not appear ────────────────────

    def test_no_old_schema_fields_present(self, critic_review: dict) -> None:
        """The old critic schema mixed validator-derived fields into critic output.

        These fields belong only in validator_review.json.  Their presence here
        indicates the old run_critic.py code is still active.
        """
        present_old = OLD_SCHEMA_FIELDS & set(critic_review)
        assert not present_old, (
            f"Old schema fields found in critic_review.json: {present_old}. "
            f"These belong in validator_review.json, not critic_review.json. "
            f"Run 'python tools/run_critic.py' from the repo root to regenerate "
            f"critic_review.json with the correct schema."
        )
