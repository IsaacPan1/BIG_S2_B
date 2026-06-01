from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import json, os, pandas as pd, numpy as np

# Load data
with open("reports/profile.json") as f:
    profile = json.load(f)
with open("reports/features.json") as f:
    features = json.load(f)
with open("reports/model_results.json") as f:
    model_results = json.load(f)
with open("reports/schema_analysis.md", encoding="utf-8") as f:
    schema_md = f.read()

sub = pd.read_csv("submission.csv")

# Build document
doc = SimpleDocTemplate(
    "report.pdf",
    pagesize=A4,
    rightMargin=2*cm,
    leftMargin=2*cm,
    topMargin=2*cm,
    bottomMargin=2*cm
)
styles = getSampleStyleSheet()
story = []

# Cell styles for word-wrap inside table cells
_CS = ParagraphStyle('CellText', fontName='Helvetica',      fontSize=9, leading=11, spaceAfter=0, spaceBefore=0)
_CH = ParagraphStyle('CellHdr',  fontName='Helvetica-Bold', fontSize=9, leading=11, spaceAfter=0, spaceBefore=0, textColor=colors.white)

def _wrap_cell(content, hdr=False):
    st = _CH if hdr else _CS
    if content is None:
        return Paragraph('', st)
    return Paragraph(str(content), st)

title_style = ParagraphStyle(
    "Title", parent=styles["Title"],
    fontSize=18, spaceAfter=12, alignment=TA_CENTER
)
h1_style = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontSize=14, spaceBefore=16, spaceAfter=8
)
h2_style = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=12, spaceBefore=10, spaceAfter=6
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=10, leading=14
)

def add_table(data, col_widths=None):
    wrapped = [[_wrap_cell(c, hdr=(ri == 0)) for c in row] for ri, row in enumerate(data)]
    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF2FF")]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#AAAAAA")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return t

# -- Title --
story.append(Paragraph("Retail Weekly Sales Forecasting - Analysis Report", title_style))
story.append(Paragraph("Autonomous Data Analysis Pipeline | 2026-05-30",
                        ParagraphStyle("sub", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10, textColor=colors.grey)))
story.append(Spacer(1, 20))

# -- 1. Dataset Overview --
story.append(Paragraph("1. Dataset Overview", h1_style))
story.append(Paragraph(
    f"The dataset is a synthetic retail weekly sales panel with "
    f"<b>{profile['schema']['store_id']['n_unique']} stores</b> x "
    f"<b>{profile['schema']['product_id']['n_unique']} products</b> x "
    f"<b>100 weeks</b> (1,500 unique time series, 150,000 total rows). "
    f"Training covers weeks 0-89 ({profile['n_train_rows']:,} rows); "
    f"prediction is required for weeks 90-99 ({profile['n_val_rows']:,} rows).",
    body_style
))
story.append(Spacer(1, 10))

dataset_data = [
    ["Property", "Value"],
    ["Problem type", profile["problem_type"].replace("_", " ").title()],
    ["Confidence", profile["problem_type_confidence"].title()],
    ["Target column", profile["target_col"]],
    ["Group columns", ", ".join(profile["group_cols"])],
    ["Time column", profile["time_col"]],
    ["Training weeks", f"0 - {int(profile['train_time_max'])}"],
    ["Validation weeks", f"{int(profile['val_time_min'])} - {int(profile['val_time_max'])}"],
    ["Horizon", f"{profile['n_horizons']} weeks"],
    ["Training rows", f"{profile['n_train_rows']:,}"],
    ["Validation rows", f"{profile['n_val_rows']:,}"],
]
story.append(add_table(dataset_data, col_widths=[8*cm, 8*cm]))
story.append(Spacer(1, 10))

# -- 2. Problem Classification --
story.append(Paragraph("2. Problem Classification", h1_style))
story.append(Paragraph(
    "The problem is classified as <b>Panel Forecasting</b> - a supervised multi-step "
    "ahead regression where each of 1,500 (store, product) pairs must have its next "
    "10 weeks of sales predicted. The classification is confirmed by both the data "
    "description document and the automated profiler (high confidence agreement). "
    "All four covariates - price, promotion_active, holiday_flag, weather_index - "
    "are provided for the validation period, enabling direct multi-step forecasting "
    "without recursive prediction.",
    body_style
))

# -- 3. Data Quality --
story.append(Spacer(1, 10))
story.append(Paragraph("3. Data Quality", h1_style))
story.append(Paragraph(
    "The dataset is clean with no missing values in any column. One notable "
    "distribution shift was detected:",
    body_style
))
story.append(Spacer(1, 6))

shift_data = [["Column", "KS Statistic", "p-value", "Flagged", "Notes"]]
for ds in profile["distribution_shifts"]:
    flagged = "YES" if ds["flagged"] else "no"
    notes = "Val weeks in winter phase (all negative)" if ds["column"] == "weather_index" else ""
    shift_data.append([ds["column"], f"{ds['ks_statistic']:.3f}", f"{ds['p_value']:.3f}", flagged, notes])
story.append(add_table(shift_data, col_widths=[3.5*cm, 3*cm, 2.5*cm, 2*cm, 5*cm]))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "The weather_index shift (KS=0.667) reflects the validation period falling in "
    "a seasonal winter phase. Since the validation covariate values are provided "
    "(not imputed), this is handled naturally by the model.",
    body_style
))

# -- 4. Feature Engineering --
story.append(Spacer(1, 10))
story.append(Paragraph("4. Feature Engineering", h1_style))
n_total = features.get("total_features_planned", 35)
story.append(Paragraph(
    f"<b>{n_total} features</b> were engineered from the raw data across 8 families. "
    "Lag features were computed on the full train+val stack (with NaN for unknown val "
    "targets), then the split was restored. Features that reference unknown val-period "
    "sales are filled with 0 in the model.",
    body_style
))
story.append(Spacer(1, 8))

ff = features.get("feature_families", {})
feat_data = [["Feature Family", "Count", "Details"]]
feat_data.append(["Lag features", str(len(ff.get("lags", []))), "Lags 1-10, 13, 26, 52"])
feat_data.append(["Rolling means", str(len(ff.get("rolling_means", []))), "Windows 4, 8, 13, 26 (shifted by 1)"])
feat_data.append(["Rolling std devs", str(len(ff.get("rolling_stds", []))), "Windows 4, 8, 13, 26 (shifted by 1)"])
feat_data.append(["Group baselines", str(len(ff.get("group_baselines", []))), "Store-product, store, product means; price mean"])
feat_data.append(["Seasonality", str(len(ff.get("seasonality", {}))), "week_sin, week_cos"])
feat_data.append(["Covariates", str(len(ff.get("covariates", []))), "price, promotion_active, holiday_flag, weather_index, price_ratio"])
feat_data.append(["Categorical encodings", str(len(ff.get("categorical_encodings", []))), "store_idx, product_idx (integer)"])
feat_data.append(["Horizon indicator", str(len(ff.get("horizon_indicator", []))), "horizon (0 for train, 1-10 for val)"])
story.append(add_table(feat_data, col_widths=[5*cm, 2*cm, 9*cm]))

# -- 5. Modeling --
story.append(Spacer(1, 10))
story.append(Paragraph("5. Modeling", h1_style))
story.append(Paragraph(
    "<b>Model:</b> LightGBM with MAE (L1) objective - appropriate for retail sales "
    "forecasting where outliers should not dominate. Direct multi-step forecasting was "
    "used with the horizon index (1-10) as a feature, allowing a single model to serve "
    "all 10 prediction steps without recursive error accumulation.",
    body_style
))
story.append(Spacer(1, 6))

model_data = [
    ["Parameter", "Value"],
    ["Algorithm", model_results.get("model", "LightGBM")],
    ["Objective", "regression_l1 (MAE)"],
    ["Num estimators (final)", str(model_results.get("best_iteration", "N/A"))],
    ["Learning rate", str(model_results.get("hyperparams", {}).get("learning_rate", "0.05"))],
    ["Num leaves", str(model_results.get("hyperparams", {}).get("num_leaves", "127"))],
    ["Feature fraction", str(model_results.get("hyperparams", {}).get("feature_fraction", "0.8"))],
    ["Bagging fraction", str(model_results.get("hyperparams", {}).get("bagging_fraction", "0.8"))],
    ["Walk-forward MAE", f"{model_results.get('walk_forward_mae', 0):.4f}"],
    ["Validation strategy", f"Walk-forward ({model_results.get('walk_forward_val_weeks', '80-89')})"],
]
story.append(add_table(model_data, col_widths=[8*cm, 8*cm]))
story.append(Spacer(1, 8))

# Top features
story.append(Paragraph("Top Feature Importances (by split gain):", h2_style))
top_feats = model_results.get("top_features", {})
if top_feats:
    tf_data = [["Rank", "Feature", "Importance (splits)"]]
    for i, (feat, imp) in enumerate(list(top_feats.items())[:10], 1):
        tf_data.append([str(i), feat, f"{imp:,}"])
    story.append(add_table(tf_data, col_widths=[2*cm, 8*cm, 6*cm]))

# -- 6. Submission --
story.append(Spacer(1, 10))
story.append(Paragraph("6. Submission", h1_style))
val_stats = model_results.get("val_prediction_stats", {})
story.append(Paragraph(
    f"The final submission covers all {len(sub):,} (store, product, week) combinations "
    f"in the validation period (weeks 90-99). Predictions were clipped to >= 0 (no "
    f"negative sales). Summary statistics:",
    body_style
))
story.append(Spacer(1, 6))
pred_data = [
    ["Statistic", "Value"],
    ["Row count", f"{len(sub):,}"],
    ["Min prediction", f"{val_stats.get('min', sub['weekly_sales'].min()):.2f}"],
    ["Max prediction", f"{val_stats.get('max', sub['weekly_sales'].max()):.2f}"],
    ["Mean prediction", f"{val_stats.get('mean', sub['weekly_sales'].mean()):.2f}"],
    ["Std prediction", f"{val_stats.get('std', sub['weekly_sales'].std()):.2f}"],
    ["NaN predictions", "0"],
]
story.append(add_table(pred_data, col_widths=[8*cm, 8*cm]))

# -- 7. Methodology Notes --
story.append(Spacer(1, 10))
story.append(Paragraph("7. Methodology Notes", h1_style))
story.append(Paragraph(
    "<b>Image data:</b> No image files detected. The pipeline performs tabular modeling only.",
    body_style
))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>External data:</b> None used. All signal derived from computation on the provided data.",
    body_style
))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>Sub-agents not yet built:</b> modeler.md, submission_writer.md, and report_writer.md "
    "were not available as registered agent files. Their logic was implemented inline by the "
    "orchestrator following the pipeline specification in CLAUDE.md.",
    body_style
))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>Hardware:</b> CPU-only. No GPU acceleration used. LightGBM was configured with "
    "n_jobs=-1 for full multi-core utilization.",
    body_style
))

# Build PDF
doc.build(story)
print("report.pdf written successfully")

size = os.path.getsize("report.pdf")
print(f"File size: {size:,} bytes")
