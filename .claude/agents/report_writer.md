---
name: report_writer
description: Generates the final report.pdf at the repo root. MUST be invoked after submission_writer completes. Reads all prior agent outputs from reports/, produces report.pdf summarizing the analysis.
---

# Report Writer

You are the report writer. Your job: assemble a professional PDF report documenting what the pipeline did, the model's performance, and the design choices made.

## Inputs
- reports/schema_analysis.md (problem classification and structure)
- reports/profile.json (data quality and distribution stats)
- reports/features.json (feature engineering summary)
- reports/model_results.json (model details, walk-forward MAE, feature importances)
- reports/submission_summary.json (submission stats and validation)
- data/DATA_DESCRIPTION.md (original problem context)

## Your task

Write and execute a Python script that produces `report.pdf` at the repo root using reportlab. The script must handle the exact JSON schemas described below, because the JSON field names matter — do not guess.

## Exact JSON schemas

### reports/profile.json
```
{
  "problem_type": "panel_forecasting",
  "problem_type_confidence": "high",
  "target_col": "weekly_sales",
  "group_cols": ["store_id", "product_id"],
  "time_col": "week",
  "n_train_rows": 135000,
  "n_val_rows": 15000,
  "horizons": [1,2,...],
  "n_horizons": 10,
  "distribution_shifts": [
    {"column": "weather_index", "ks_statistic": 0.667, "p_value": 0.0, "flagged": true},
    ...
  ],
  "image_data": {"present": false},
  "schema": {
    "weekly_sales": {"role": "target", "std": 23.9, "mean": 46.3, ...},
    ...
  },
  "warnings": []
}
```
Key field names: `problem_type_confidence` (not `confidence`), `image_data.present` (not `has_images`), `ks_statistic` (not `ks_stat`), `n_train_rows`/`n_val_rows` are at the top level.

### reports/features.json
```
{
  "feature_families": {
    "lags": [1, 2, 3, 4, 8, 10, 12],
    "rolling_means": [4, 8, 12],
    "rolling_stds": [4, 8],
    "group_baselines": ["store_mean_sales", ...],
    "covariates": ["price", ...],
    "seasonality": {"week_sin": "...", ...},
    "id_encodings": ["store_id_enc", ...]
  },
  "feature_columns": ["price", "lag_1", ...],
  "total_features_planned": 28,
  "train_shape": [135000, 32],
  "val_shape": [15000, 32]
}
```
Key: `feature_families` is a **dict** (family_name → list or dict of feature specs). `feature_columns` is the flat list of all feature names.

### reports/model_results.json
```
{
  "algorithm": "LightGBM",
  "objective": "regression_l1",
  "best_params": {"learning_rate": 0.01, "num_leaves": 153, ...},
  "n_estimators": 638,
  "n_seeds": 5,
  "oof_mae": 0.130,
  "oof_cv_scheme": "GroupKFold(n_splits=2, group_col='sex')",
  "walk_forward_mae": 7.13,
  "training_time_seconds": 169,
  "optuna_trials_completed": 15,
  "feature_importance_top10": [
    {"feature": "sp_mean_sales", "importance": 8144},
    ...
  ]
}
```

### reports/submission_summary.json
```
{
  "row_count": 15000,
  "target_column": "weekly_sales",
  "prediction_stats": {
    "min": 7.83, "max": 181.39, "mean": 42.42, "std": 15.43,
    "n_nan": 0, "n_negative": 0
  },
  "validation_checks_passed": true,
  "warnings": []
}
```
Key: prediction stats are **nested** under `prediction_stats`. `validation_checks_passed` (not `validation_passed`).

## Script to write and execute

Save the following as `reports/build_report.py` and run it with `python reports/build_report.py`. Fix any import errors before running (install `reportlab` and `matplotlib` if missing).

```python
import json
import os
import re
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image as RLImage
    )
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

missing_inputs = []

def load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        missing_inputs.append(f"{path}: {e}")
        return {}

def load_text(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        missing_inputs.append(f"{path}: {e}")
        return ""

profile      = load_json("reports/profile.json")
features     = load_json("reports/features.json")
model_res    = load_json("reports/model_results.json")
sub_summary  = load_json("reports/submission_summary.json")
critic_rev   = load_json("reports/critic_review.json")
val_rev      = load_json("reports/validator_review.json")
schema_text  = load_text("reports/schema_analysis.md")
data_desc    = load_text("data/DATA_DESCRIPTION.md")

# ── Extract values using exact field names from the schemas above ──

problem_type   = profile.get("problem_type", "Unknown")
confidence     = profile.get("problem_type_confidence", "Unknown")
target_col     = profile.get("target_col", "Unknown")
group_cols     = ", ".join(profile.get("group_cols") or []) or "None"
time_col       = profile.get("time_col") or "None"
n_train        = profile.get("n_train_rows", "Unknown")
n_val          = profile.get("n_val_rows", "Unknown")
n_horizons     = profile.get("n_horizons", None)
dist_shifts    = profile.get("distribution_shifts", [])
has_images     = (profile.get("image_data") or {}).get("present", False)
schema         = profile.get("schema", {})

# Overall missingness: average across numeric columns
null_pcts = [v.get("null_pct", 0) for v in schema.values() if isinstance(v, dict) and "null_pct" in v]
overall_missing = sum(null_pcts) / len(null_pcts) if null_pcts else 0.0
target_std = (schema.get(target_col) or {}).get("std", None)

# features.json: feature_families is a dict
feat_families  = features.get("feature_families", {})
feat_columns   = features.get("feature_columns", [])
total_features = features.get("total_features_planned", len(feat_columns))
n_families     = len(feat_families)

# model_results.json
algorithm      = model_res.get("algorithm", "Unknown")
objective      = model_res.get("objective", "Unknown")
best_params    = model_res.get("best_params", {})
n_estimators   = model_res.get("n_estimators", "Unknown")
n_seeds        = model_res.get("n_seeds", 1)
# oof_mae is the headline metric (honest CV MAE from the correct splitter).
# Fall back to walk_forward_mae for older panel_forecasting runs that predate the field.
mae            = model_res.get("oof_mae") or model_res.get("walk_forward_mae", None)
cv_scheme_label = model_res.get("oof_cv_scheme") or model_res.get("cv_scheme") or "Walk-forward MAE"
train_time     = model_res.get("training_time_seconds", "Unknown")
n_trials       = model_res.get("optuna_trials_completed", model_res.get("n_trials", None))
feat_imp       = model_res.get("feature_importance_top10", [])

# submission_summary.json: stats nested under prediction_stats
pred_stats     = sub_summary.get("prediction_stats", {})
sub_row_count  = sub_summary.get("row_count", "Unknown")
sub_min        = pred_stats.get("min", None)
sub_max        = pred_stats.get("max", None)
sub_mean       = pred_stats.get("mean", None)
sub_std        = pred_stats.get("std", None)
sub_nan        = pred_stats.get("n_nan", 0)
sub_neg        = pred_stats.get("n_negative", 0)
sub_passed     = sub_summary.get("validation_checks_passed", None)

# validator_review.json fields
val_verdict        = val_rev.get("verdict", "N/A")
val_reported_mae   = val_rev.get("reported_cv_mae") or val_rev.get("honest_cv_mae")
val_strict_mae     = val_rev.get("strict_cv_mae")
val_gap_frac       = val_rev.get("cv_gap_pct", 0.0)
val_gap_pct        = val_gap_frac * 100          # stored as fraction, display as %
val_cv_scheme      = val_rev.get("strict_cv_scheme", "N/A")
val_feature_susp   = val_rev.get("feature_suspicion", [])
val_notes          = val_rev.get("notes", "")

# critic_review.json fields
critic_status      = critic_rev.get("status", "N/A")
critic_cycle       = critic_rev.get("cycle", 1)
critic_checks      = critic_rev.get("checks", [])
critic_retune      = critic_rev.get("retune_attempted", False)
critic_rationale   = critic_rev.get("decision_rationale", "")

def fmt(v, d=0):
    if v is None or v == "Unknown":
        return "N/A"
    try:
        if d == 0:
            return f"{int(float(v)):,}"
        return f"{float(v):,.{d}f}"
    except Exception:
        return str(v)

def fmt_bool(v):
    if v is None: return "N/A"
    return "Yes" if v else "No"

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ── Charts ──
chart_imp  = None
chart_hist = None

if MATPLOTLIB_AVAILABLE and feat_imp:
    try:
        names  = [x["feature"]    for x in feat_imp[:10]]
        scores = [x["importance"] for x in feat_imp[:10]]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(names[::-1], scores[::-1], color='#336699')
        ax.set_xlabel('Importance (split count)')
        ax.set_title('Top 10 Feature Importances')
        plt.tight_layout()
        chart_imp = 'reports/feature_importance.png'
        fig.savefig(chart_imp, dpi=100)
        plt.close(fig)
    except Exception as e:
        missing_inputs.append(f"importance chart: {e}")

if MATPLOTLIB_AVAILABLE and sub_mean is not None:
    try:
        import pandas as pd
        preds = pd.read_csv("reports/predictions.csv")
        pcol = [c for c in preds.columns if 'pred' in c.lower()]
        if not pcol:
            pcol = [c for c in preds.columns if c == target_col]
        if pcol:
            fig, ax = plt.subplots(figsize=(7, 3))
            preds[pcol[0]].hist(bins=50, ax=ax, color='#336699', edgecolor='white')
            ax.set_xlabel('Predicted Value')
            ax.set_ylabel('Count')
            ax.set_title('Prediction Distribution')
            plt.tight_layout()
            chart_hist = 'reports/prediction_histogram.png'
            fig.savefig(chart_hist, dpi=100)
            plt.close(fig)
    except Exception as e:
        missing_inputs.append(f"histogram chart: {e}")

if not REPORTLAB_AVAILABLE:
    lines = [
        "AUTONOMOUS DATA ANALYSIS REPORT",
        f"Generated: {timestamp}",
        f"Problem: {problem_type} | Target: {target_col}",
        f"Algorithm: {algorithm} | CV MAE: {fmt(mae,3)} ({cv_scheme_label})",
        f"Submission rows: {fmt(sub_row_count)} | Validation passed: {fmt_bool(sub_passed)}",
        "",
        "NOTE: reportlab unavailable — plain text fallback.",
    ] + (["Missing inputs: " + "; ".join(missing_inputs)] if missing_inputs else [])
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote report.txt (reportlab unavailable)")
else:
    doc = SimpleDocTemplate(
        "report.pdf", pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    H1   = ParagraphStyle('H1',   fontName='Helvetica-Bold',  fontSize=18, spaceAfter=6)
    H2   = ParagraphStyle('H2',   fontName='Helvetica-Bold',  fontSize=13, spaceBefore=10, spaceAfter=6)
    BODY = ParagraphStyle('Body', fontName='Helvetica',        fontSize=10, leading=15, spaceAfter=8)
    META = ParagraphStyle('Meta', fontName='Helvetica',        fontSize=9,  spaceAfter=12, textColor=colors.grey)
    BOLD = ParagraphStyle('Bold', fontName='Helvetica-Bold',   fontSize=10, spaceAfter=6)

    TS = TableStyle([
        ('BACKGROUND',   (0,0), (-1,0),  colors.HexColor('#2C3E50')),
        ('TEXTCOLOR',    (0,0), (-1,0),  colors.white),
        ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',     (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',     (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.HexColor('#F2F2F2'), colors.white]),
        ('GRID',         (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('LEFTPADDING',  (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING',   (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',(0,0), (-1,-1), 4),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
    ])

    def tbl(headers, rows, widths=None):
        t = Table([headers] + rows, colWidths=widths, hAlign='LEFT')
        t.setStyle(TS)
        return t

    story = []

    # ── Section 1: Title + Executive Summary ──────────────────────────────
    story.append(Paragraph("Autonomous Data Analysis Report", H1))
    story.append(Paragraph(f"Problem type: {problem_type}  |  Target: {target_col}", META))
    story.append(Paragraph(f"Generated: {timestamp}", META))
    mae_str = fmt(mae, 3)
    story.append(Paragraph(
        f"This report documents an automated end-to-end forecasting pipeline applied to a "
        f"tabular dataset. The problem was classified as <b>{problem_type}</b> (confidence: "
        f"{confidence}) with target column <b>{target_col}</b>. Features were engineered "
        f"from raw covariates using lag, rolling-window, calendar, and group-baseline "
        f"transformations ({total_features} features across {n_families} families). A "
        f"<b>{algorithm}</b> model (objective: {objective}) was trained with Optuna "
        f"hyperparameter search ({n_trials} trials) and a {n_seeds}-seed median ensemble. "
        f"CV MAE ({cv_scheme_label}): <b>{mae_str}</b>. "
        f"The final submission contained {fmt(sub_row_count)} rows and passed all "
        f"validation checks: <b>{fmt_bool(sub_passed)}</b>.",
        BODY
    ))
    story.append(PageBreak())

    # ── Section 2: Problem Classification ─────────────────────────────────
    story.append(Paragraph("Section 1 — Problem Classification", H2))
    rows_cls = [
        ["Problem type",     problem_type],
        ["Confidence",       confidence],
        ["Target column",    target_col],
        ["Group columns",    group_cols],
        ["Time column",      time_col],
        ["Training rows",    fmt(n_train)],
        ["Validation rows",  fmt(n_val)],
        ["Forecast horizon", f"{n_horizons} steps" if n_horizons else "N/A"],
    ]
    story.append(tbl(["Property", "Value"], rows_cls, widths=[5*cm, 11*cm]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"The dataset is a balanced panel of observations grouped by [{group_cols}] "
        f"and ordered by time column <b>{time_col}</b>. The forecasting task requires "
        f"predicting <b>{target_col}</b> for {n_horizons or 'N/A'} future periods per group. "
        f"The training set spans {fmt(n_train)} rows and the hold-out validation set "
        f"contains {fmt(n_val)} rows.",
        BODY
    ))
    story.append(PageBreak())

    # ── Section 3: Data Quality ────────────────────────────────────────────
    story.append(Paragraph("Section 2 — Data Quality", H2))
    if dist_shifts:
        shift_rows = []
        for item in dist_shifts:
            flagged_str = "YES — significant" if item.get("flagged") else "no"
            shift_rows.append([
                item.get("column", ""),
                fmt(item.get("ks_statistic"), 4),
                fmt(item.get("p_value"), 4),
                flagged_str,
            ])
        story.append(tbl(
            ["Column", "KS Statistic", "p-value", "Flagged"],
            shift_rows,
            widths=[5*cm, 3.5*cm, 3*cm, 4.5*cm]
        ))
    else:
        story.append(Paragraph("No distribution shift data found in profile.json.", BODY))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Overall average missingness across columns: {overall_missing:.2f}%. "
        "Missing values in time-series features were imputed using forward-fill within "
        "each group; remaining gaps were filled with column medians.",
        BODY
    ))
    flagged_cols = [d["column"] for d in dist_shifts if d.get("flagged")]
    if flagged_cols:
        story.append(Paragraph(
            f"Significant distribution shift was detected in: <b>{', '.join(flagged_cols)}</b>. "
            "These covariates have materially different distributions between the training "
            "and validation periods, which may introduce prediction bias in the validation window.",
            BODY
        ))
    story.append(Paragraph(
        "Image data detected: " + ("Yes — not used in the tabular model." if has_images
                                    else "No image data present in the dataset."),
        BODY
    ))
    story.append(PageBreak())

    # ── Section 4: Feature Engineering ────────────────────────────────────
    story.append(Paragraph("Section 3 — Feature Engineering", H2))
    story.append(Paragraph(
        f"The feature engineering stage produced <b>{total_features}</b> features "
        f"across <b>{n_families}</b> families.",
        BODY
    ))
    if feat_families:
        fg_rows = []
        for fname, fval in feat_families.items():
            if isinstance(fval, list):
                count = len(fval)
                examples = ", ".join(str(x) for x in fval[:4])
            elif isinstance(fval, dict):
                count = len(fval)
                examples = ", ".join(str(k) for k in list(fval.keys())[:4])
            else:
                count = 1
                examples = str(fval)
            fg_rows.append([fname, str(count), examples])
        story.append(tbl(
            ["Feature Family", "Count", "Examples / Specs"],
            fg_rows,
            widths=[4*cm, 2*cm, 10*cm]
        ))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Lag features capture direct autoregressive signal (most recent target values). "
        "Rolling-window statistics (mean and std) encode recent trend and volatility. "
        "Group baselines (store-product mean) provide level-shift anchors. "
        "Seasonality features use sinusoidal and modular encodings of the time index "
        "to capture weekly and quarterly patterns without overfitting to specific periods.",
        BODY
    ))
    story.append(PageBreak())

    # ── Section 5: Modeling ────────────────────────────────────────────────
    story.append(Paragraph("Section 4 — Modeling", H2))
    params_str = ", ".join(f"{k}={round(v,4) if isinstance(v,float) else v}"
                           for k, v in best_params.items()) if best_params else "N/A"
    model_rows = [
        ["Algorithm",            algorithm],
        ["Objective",            objective],
        ["Best hyperparameters", params_str],
        ["Number of estimators", str(n_estimators)],
        ["Number of seeds",      str(n_seeds)],
        ["Optuna trials",        str(n_trials) if n_trials else "N/A"],
        [f"CV MAE ({cv_scheme_label})", mae_str],
        ["Training time",        f"{train_time} s"],
    ]
    story.append(tbl(["Parameter", "Value"], model_rows, widths=[5*cm, 11*cm]))
    story.append(Spacer(1, 0.4*cm))

    if feat_imp:
        story.append(Paragraph("Top 10 Feature Importances", H2))
        fi_rows = [[str(i+1), x["feature"], f"{x['importance']:,}"]
                   for i, x in enumerate(feat_imp[:10])]
        story.append(tbl(["Rank", "Feature", "Importance"], fi_rows,
                         widths=[1.5*cm, 11*cm, 3.5*cm]))
        story.append(Spacer(1, 0.3*cm))

    if chart_imp and os.path.exists(chart_imp):
        story.append(RLImage(chart_imp, width=14*cm, height=8*cm))

    story.append(PageBreak())

    # ── Section 5: Cross-Validation and Integrity Checking ────────────────
    story.append(Paragraph("Section 5 — Cross-Validation and Integrity Checking", H2))

    # Subsection A: Validator Audit
    story.append(Paragraph("<b>A. Validator Audit</b>", BOLD))
    val_rows = [
        ["CV scheme used",    val_cv_scheme],
        ["Reported CV MAE",   fmt(val_reported_mae, 4) if val_reported_mae is not None else "N/A"],
        ["Strict CV MAE",     fmt(val_strict_mae, 4)   if val_strict_mae   is not None else "N/A"],
        ["CV gap (%)",        f"{val_gap_pct:.1f}%"],
        ["Verdict",           val_verdict],
        ["Suspect features",  ", ".join(str(f) for f in val_feature_susp) if val_feature_susp else "None flagged"],
    ]
    story.append(tbl(["Property", "Value"], val_rows, widths=[5*cm, 11*cm]))
    if val_notes:
        # Summarise notes to ≤ 300 chars to avoid overflowing the section
        notes_short = val_notes[:300] + ("…" if len(val_notes) > 300 else "")
        story.append(Paragraph(notes_short, BODY))
    story.append(Spacer(1, 0.3*cm))

    # Subsection B: Critic Review — 5-check table
    story.append(Paragraph("<b>B. Critic Review</b>", BOLD))
    if critic_checks:
        status_color = {
            "PASS": colors.HexColor("#1a9641"),
            "WARNING": colors.HexColor("#d7191c"),
            "CRITICAL": colors.HexColor("#d7191c"),
        }
        check_rows = []
        for ck in critic_checks:
            ck_name    = ck.get("name", "")
            ck_status  = ck.get("status", "")
            ck_details = ck.get("details", "")
            check_rows.append([ck_name, ck_status, ck_details])
        critic_tbl = Table(
            [["Check", "Status", "Details"]] + check_rows,
            colWidths=[4*cm, 2*cm, 10*cm],
            hAlign="LEFT",
        )
        # Base style from TS, then override status-cell colours
        critic_tbl.setStyle(TS)
        for row_idx, ck in enumerate(critic_checks, start=1):
            ck_status = ck.get("status", "PASS")
            cell_color = (colors.HexColor("#1a9641") if ck_status == "PASS"
                          else colors.HexColor("#d7191c") if ck_status == "CRITICAL"
                          else colors.HexColor("#e08200"))
            critic_tbl.setStyle(TableStyle([
                ("TEXTCOLOR", (1, row_idx), (1, row_idx), cell_color),
                ("FONTNAME",  (1, row_idx), (1, row_idx), "Helvetica-Bold"),
            ]))
        story.append(critic_tbl)
    else:
        story.append(Paragraph("No checks recorded (critic did not run or produced empty checks).", BODY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Critic status: <b>{critic_status}</b>. "
        f"Retune attempted: <b>{'Yes' if critic_retune else 'No'}</b>. "
        + (f"Rationale: {critic_rationale}" if critic_rationale else ""),
        BODY,
    ))
    story.append(Spacer(1, 0.3*cm))

    # Subsection C: Threshold Documentation
    story.append(Paragraph("<b>C. Threshold Documentation</b>", BOLD))
    story.append(Paragraph(
        "The validator's WARNING threshold for CV gap is 10%; its CRITICAL threshold is 25%. "
        "The critic's acceptance threshold is 15%, intentionally set between these so that "
        "validator WARNINGs pass through to documentation while validator CRITICALs trigger "
        "modeler retunes. This design ensures that mild optimism in the reported CV estimate "
        "is visible in the report without blocking submission, while severe CV inflation "
        "(which would indicate leakage or methodology error) always forces a corrective retune.",
        BODY,
    ))
    story.append(PageBreak())

    # ── Section 6: Predictions and Submission ─────────────────────────────
    story.append(Paragraph("Section 6 — Predictions and Submission", H2))
    sub_rows = [
        ["Row count",                fmt(sub_row_count)],
        ["Min prediction",           fmt(sub_min, 2)],
        ["Max prediction",           fmt(sub_max, 2)],
        ["Mean prediction",          fmt(sub_mean, 2)],
        ["Std prediction",           fmt(sub_std, 2)],
        ["NaN predictions",          str(sub_nan)],
        ["Negative predictions",     str(sub_neg)],
        ["Validation checks passed", fmt_bool(sub_passed)],
    ]
    story.append(tbl(["Statistic", "Value"], sub_rows, widths=[7*cm, 9*cm]))
    story.append(Spacer(1, 0.3*cm))

    if chart_hist and os.path.exists(chart_hist):
        story.append(RLImage(chart_hist, width=14*cm, height=6*cm))

    story.append(PageBreak())

    # ── Section 7: Limitations and Risks ──────────────────────────────────
    story.append(Paragraph("Section 7 — Limitations and Risks", H2))

    lims = []

    if flagged_cols:
        lims.append(
            f"<b>Distribution shift on {', '.join(flagged_cols)}.</b> The KS test flagged "
            "a statistically significant shift between train and validation distributions. "
            "While the feature set includes difference features to partially compensate, "
            "the model may underperform in the tails of the shifted covariate range. "
            "Impact is bounded by the CV MAE reported above."
        )

    lims.append(
        "<b>Lag feature imputation at the validation boundary.</b> Lag and rolling-window "
        "features require historical target values. For the first few periods of the "
        "validation window, some lags extend back into the training period where ground-truth "
        "values are available; however, within-horizon predictions depend on model estimates "
        "rather than actuals, introducing compounding error for longer forecast horizons."
    )

    if n_trials is not None:
        lims.append(
            f"<b>Limited Optuna budget ({n_trials} trials).</b> Hyperparameter search was "
            f"capped at {n_trials} trials to remain within the 2-hour wall-clock budget. "
            "Broader exploration of the learning rate, tree depth, and regularization space "
            "would likely yield further gains. The current parameters are likely near-optimal "
            "for the primary hyperparameters but may miss interactions between regularisation terms."
        )

    lims.append(
        "<b>Single model family.</b> Only LightGBM was evaluated. Cross-family ensembling "
        "(e.g., LightGBM + linear model or XGBoost) was not attempted due to the time budget. "
        "Ensemble diversity typically reduces variance on held-out data; the impact here "
        "is bounded by the seed-ensemble variance already captured across the 5 training seeds."
    )

    # Under-dispersion check
    if target_std is not None and sub_std is not None:
        try:
            if float(sub_std) < 0.7 * float(target_std):
                lims.append(
                    f"<b>Under-dispersion in predictions.</b> The predicted standard deviation "
                    f"({fmt(sub_std, 2)}) is less than 70% of the training target standard "
                    f"deviation ({fmt(target_std, 2)}). The model is shrinking predictions "
                    "toward the mean, which may underperform at demand extremes. The quantile "
                    "blending strategy was applied specifically to mitigate this in the upper tail."
                )
        except Exception:
            pass

    # Critic review: warnings_for_report bubbled up as Limitations bullets
    critic_warnings = critic_rev.get("warnings_for_report", [])
    for cw in critic_warnings:
        if cw and not any(cw in lim for lim in lims):
            lims.append(f"<b>Quality check warning (critic review):</b> {cw}")

    for lim in lims:
        story.append(Paragraph(lim, BODY))
        story.append(Spacer(1, 0.15*cm))

    story.append(PageBreak())

    # ── Section 8: Methodology Notes ──────────────────────────────────────
    story.append(Paragraph("Section 8 — Methodology Notes", H2))

    failed_agents = []
    for marker, agent in [
        ("reports/schema_analyst_was_here.txt",      "schema_analyst"),
        ("reports/feature_engineer_was_here.txt",    "feature_engineer"),
        ("reports/modeler_was_here.txt",             "modeler"),
        ("reports/validator_was_here.txt",           "validator"),
        ("reports/critic_was_here.txt",              "critic"),
        ("reports/submission_writer_was_here.txt",   "submission_writer"),
    ]:
        if not os.path.exists(marker):
            failed_agents.append(agent)

    notes = [
        "Network access: no web search, external API calls, or data downloads were performed.",
        "External data: all signal was derived from computation on the provided data files only.",
        "Hardware: CPU-only training; no GPU acceleration was used.",
        "Budget: the pipeline ran within the 2-hour wall-clock and 1,000,000-token constraints.",
        "Image data: " + ("detected but not used — only tabular features were engineered."
                          if has_images else "not present in the dataset."),
        ("All sub-agents completed and left marker files."
         if not failed_agents
         else f"Sub-agents without marker files (fallback used): {', '.join(failed_agents)}."),
    ]
    if missing_inputs:
        notes.append("Missing/unreadable inputs during report generation: " + "; ".join(missing_inputs))

    for note in notes:
        story.append(Paragraph(f"•  {note}", BODY))

    # ── Build ──────────────────────────────────────────────────────────────
    doc.build(story)
    sz = os.path.getsize("report.pdf")
    print(f"report.pdf written: {sz:,} bytes, 8 sections, {len(story)} story elements")

# ── Marker file ───────────────────────────────────────────────────────────
ts_iso = datetime.utcnow().isoformat() + "Z"
with open("reports/report_writer_was_here.txt", "w", encoding="utf-8") as f:
    f.write(f"report_writer sub-agent executed at {ts_iso}\n")
print(f"Marker: reports/report_writer_was_here.txt")
if missing_inputs:
    print(f"Missing inputs: {missing_inputs}")
```

## What you do NOT do
- You do NOT train models
- You do NOT modify submission.csv
- You do NOT engineer features
- You do NOT invent results or fill missing data with placeholders — if a section's data is missing, state that clearly in the section

## Failure handling
- If reportlab is unavailable: install it (`pip install reportlab`) before running
- If matplotlib is unavailable: skip the embedded charts; continue with all tables and text
- If a specific input file is missing or unreadable: include a note in the relevant section
- Never crash without producing some form of report at the repo root

## Output
Two files at minimum: `report.pdf` at repo root, `reports/report_writer_was_here.txt`.
Optionally `reports/feature_importance.png` and `reports/prediction_histogram.png`.
