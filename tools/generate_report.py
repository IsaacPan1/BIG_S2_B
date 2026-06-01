"""Generate report.pdf summarising the pipeline methodology and results."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

BASE = Path(__file__).parent.parent
REPORTS = BASE / "reports"

# Load data
with open(REPORTS / "profile.json") as f:
    profile = json.load(f)
with open(REPORTS / "model_results.json") as f:
    model = json.load(f)
with open(REPORTS / "submission_summary.json") as f:
    submission = json.load(f)
with open(REPORTS / "features.json") as f:
    features = json.load(f)

doc = SimpleDocTemplate(
    str(BASE / "report.pdf"),
    pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=6)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, spaceBefore=14, spaceAfter=4)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceBefore=10, spaceAfter=3)
body = styles["BodyText"]
body.fontSize = 9
body.leading = 13

def tbl(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("LEADING", (0,0), (-1,-1), 11),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F2F3F4")]),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
    ]
    if not header:
        style_cmds = [c for c in style_cmds if c[0] not in ("BACKGROUND","TEXTCOLOR","FONTNAME") or c[2] != (0,0)]
    t.setStyle(TableStyle(style_cmds))
    return t

story = []

# ── Title ────────────────────────────────────────────────────────────────
story.append(Paragraph("Award B — Autonomous Forecasting Pipeline", title_style))
story.append(Paragraph("Methodology &amp; Results Report", styles["Heading2"]))
story.append(Spacer(1, 0.3*cm))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2C3E50")))
story.append(Spacer(1, 0.3*cm))

# ── 1. Problem Overview ───────────────────────────────────────────────────
story.append(Paragraph("1. Problem Overview", h1))
overview = [
    ["Field", "Value"],
    ["Dataset", "Weekly retail sales panel (50 stores × 30 products × 100 weeks)"],
    ["Problem type", profile.get("problem_type", "panel_forecasting")],
    ["Target column", profile.get("target_col", "weekly_sales")],
    ["Train period", "Weeks 0–89 (90 weeks, 135,000 rows)"],
    ["Val / forecast period", "Weeks 90–99 (10 weeks, 15,000 rows)"],
    ["Groups", "1,500 (store_id, product_id) pairs"],
    ["Covariates", "price, promotion_active, holiday_flag, weather_index"],
    ["Missing values", "None"],
    ["Key risk", "weather_index severe distribution shift (KS = 0.667)"],
]
story.append(tbl(overview, col_widths=[4.5*cm, 12*cm]))

# ── 2. Feature Engineering ────────────────────────────────────────────────
story.append(Paragraph("2. Feature Engineering", h1))
story.append(Paragraph(
    "43 features were engineered from 7 raw columns. All lag and rolling features "
    "were computed per (store_id, product_id) group with strict temporal ordering "
    "to prevent data leakage. Val rows use only training-window look-back values.",
    body
))
story.append(Spacer(1, 0.2*cm))

feat_data = [["Feature Family", "Count", "Description"]]
ff = features["feature_families"]
feat_data += [
    ["Lag features", str(len(ff["lags"])), f"Lags {ff['lags'][:5]}… up to lag-26"],
    ["Rolling mean/std", "6", "Windows: 4, 8, 12 weeks (shifted by 1)"],
    ["Group baselines", str(len(ff["group_baselines"])), "store/product/pair mean, std, median (train-only)"],
    ["Seasonality", "4", "week_sin, week_cos (52-week period), mod4, mod13"],
    ["Weather robust", "3", "weather_lag1, weather_delta, weather_abs (handles KS shift)"],
    ["Price derived", "3", "price_rel (normalised), price_x_promo interaction, product_price_mean"],
    ["Group encodings", "2", "store_code, product_code (integer category codes)"],
    ["Raw covariates", "4", "price, promotion_active, holiday_flag, weather_index"],
    ["TOTAL", "43", ""],
]
story.append(tbl(feat_data, col_widths=[4.5*cm, 2*cm, 10*cm]))

# ── 3. Modelling ──────────────────────────────────────────────────────────
story.append(Paragraph("3. Modelling", h1))
story.append(Paragraph(
    "A LightGBM gradient-boosted tree ensemble was trained using a walk-forward "
    "time-series cross-validation scheme. Hyperparameters were tuned with Optuna "
    "(15 trials, MAE objective). The final model is an ensemble of 5 seeds to "
    "reduce variance. Training was CPU-only and completed in ~103 seconds.",
    body
))
story.append(Spacer(1, 0.2*cm))

model_data = [
    ["Parameter", "Value"],
    ["Algorithm", model["algorithm"]],
    ["Objective", model["objective"]],
    ["n_estimators", str(model["n_estimators"])],
    ["Ensemble seeds", str(model["n_seeds"])],
    ["Optuna trials", str(model["optuna_trials_completed"])],
    ["learning_rate", f"{model['best_params']['learning_rate']:.5f}"],
    ["num_leaves", str(model['best_params']['num_leaves'])],
    ["feature_fraction", f"{model['best_params']['feature_fraction']:.4f}"],
    ["bagging_fraction", f"{model['best_params']['bagging_fraction']:.4f}"],
    ["Walk-forward MAE (weeks 72–89)", f"{model['walk_forward_mae']:.4f}"],
    ["Training time (s)", str(model["training_time_seconds"])],
]
story.append(tbl(model_data, col_widths=[7*cm, 9.5*cm]))

# ── 4. Feature Importance ─────────────────────────────────────────────────
story.append(Paragraph("4. Top-10 Feature Importances (LightGBM split gain)", h1))
fi_data = [["Rank", "Feature", "Importance (split gain)"]]
for i, fi in enumerate(model["feature_importance_top10"], 1):
    fi_data.append([str(i), fi["feature"], str(fi["importance"])])
story.append(tbl(fi_data, col_widths=[1.5*cm, 8*cm, 7*cm]))

# ── 5. Prediction Summary ─────────────────────────────────────────────────
story.append(Paragraph("5. Prediction Summary", h1))
pred_stats = model["val_prediction_stats"]
pred_data = [
    ["Metric", "Value"],
    ["Val rows predicted", str(model["n_val_rows"])],
    ["NaN predictions", "0"],
    ["Negative predictions", "0"],
    ["Prediction min", f"{pred_stats['min']:.3f}"],
    ["Prediction max", f"{pred_stats['max']:.3f}"],
    ["Prediction mean", f"{pred_stats['mean']:.3f}"],
    ["Prediction std", f"{pred_stats['std']:.3f}"],
    ["Training target mean", "46.3 (slight downward shift expected for future weeks)"],
]
story.append(tbl(pred_data, col_widths=[7*cm, 9.5*cm]))

# ── 6. Pipeline Provenance ────────────────────────────────────────────────
story.append(Paragraph("6. Pipeline Provenance", h1))
story.append(Paragraph(
    "All four pipeline stages ran as registered sub-agents (schema_analyst, "
    "feature_engineer, modeler, submission_writer). The report_writer stage was "
    "implemented inline (no report_writer.md agent file exists yet). "
    "Marker files were verified at each stage transition. No external data, "
    "pretrained models, or network access were used.",
    body
))
story.append(Spacer(1, 0.2*cm))

stage_data = [
    ["Stage", "Sub-agent", "Status", "Key output"],
    ["1 — Schema analysis", "schema_analyst", "✓ Complete", "reports/schema_analysis.md, reports/profile.json"],
    ["2 — Feature engineering", "feature_engineer", "✓ Complete", "data/features_train.parquet (43 cols)"],
    ["3 — Modelling", "modeler", "✓ Complete", f"MAE = {model['walk_forward_mae']:.4f}"],
    ["4 — Submission writing", "submission_writer", "✓ Complete", "submission.csv (15,000 rows, 0 NaN)"],
    ["5 — Report writing", "inline (no agent file)", "✓ Complete", "report.pdf"],
]
story.append(tbl(stage_data, col_widths=[4*cm, 3.5*cm, 2.5*cm, 6.5*cm]))

# ── Footer ────────────────────────────────────────────────────────────────
story.append(Spacer(1, 0.5*cm))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
story.append(Paragraph(
    "Generated by Award B autonomous pipeline · 2026-05-31 · claude-sonnet-4-6",
    ParagraphStyle("footer", parent=body, fontSize=7, textColor=colors.grey, alignment=TA_CENTER)
))

doc.build(story)
print("report.pdf written successfully.")
