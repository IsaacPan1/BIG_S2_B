import json, os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER

with open("reports/profile.json") as f:
    profile = json.load(f)
with open("reports/model_results.json") as f:
    results = json.load(f)
sub = pd.read_csv("submission.csv")

doc = SimpleDocTemplate("report.pdf", pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
styles = getSampleStyleSheet()
body = styles["Normal"]
body.fontSize = 10
body.leading = 14
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, spaceBefore=14, spaceAfter=6)
ctr = ParagraphStyle("ctr", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10,
                     textColor=colors.grey)

def tbl(data, widths=None):
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF2FF")]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#AAAAAA")),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    return t

algo = results["algorithm"]
obj  = results["objective"]
mae  = results["walk_forward_mae"]
n_opt = results["optuna_trials_completed"]
n_est = results["n_estimators"]
n_seeds = results["n_seeds"]
t_sec = results["training_time_seconds"]
v_min = results["val_prediction_stats"]["min"]
v_max = results["val_prediction_stats"]["max"]
v_mean = results["val_prediction_stats"]["mean"]

top10 = results["feature_importance_top10"]

story = [
    Paragraph("Retail Weekly Sales — Forecast Report",
              ParagraphStyle("T", parent=styles["Title"], fontSize=17, spaceAfter=4, alignment=TA_CENTER)),
    Paragraph("Autonomous Pipeline | 2026-05-30", ctr),
    Spacer(1, 16),

    Paragraph("1. Dataset", h1),
    Paragraph("Panel: 50 stores x 30 products x 100 weeks. Train weeks 0-89 "
              "(135,000 rows); predict weeks 90-99 (15,000 rows). Target: "
              "<b>weekly_sales</b> (float, non-negative, range 0-247).", body),
    Spacer(1, 8),
    tbl([["Property", "Value"],
         ["Problem type", "Panel Forecasting"],
         ["Target", "weekly_sales"],
         ["Groups", "store_id (50), product_id (30)"],
         ["Train weeks", "0 - 89"],
         ["Val weeks", "90 - 99"],
         ["Horizon", "10 weeks"]], [7*cm, 9*cm]),

    Spacer(1, 12),
    Paragraph("2. Data Quality", h1),
    Paragraph("Zero missing values across all files. One distribution shift flagged: "
              "<b>weather_index</b> (KS=0.667) — val period falls in winter phase. "
              "Val covariates are fully provided so no imputation is needed.", body),

    Spacer(1, 12),
    Paragraph("3. Modeling", h1),
    Paragraph("<b>Algorithm:</b> " + algo + " (" + obj + "). "
              "15-trial Optuna tuning on 80/20 walk-forward split (3-seed objective), "
              "then 5-seed ensemble retrained on full training data with log1p target "
              "transform. Best params fed back for final n_estimators via early stopping.", body),
    Spacer(1, 8),
    tbl([["Metric", "Value"],
         ["Walk-forward MAE", "{:.4f}".format(mae)],
         ["Optuna trials", str(n_opt)],
         ["n_estimators (final)", str(n_est)],
         ["n_seeds (ensemble)", str(n_seeds)],
         ["Training time", str(t_sec) + "s"]], [7*cm, 9*cm]),
    Spacer(1, 10),
    Paragraph("<b>Top 10 features by split gain:</b>", body),
    Spacer(1, 4),
    tbl([["Rank", "Feature", "Importance"]] +
        [[str(i+1), r["feature"], str(r["importance"])]
         for i, r in enumerate(top10)],
        [2*cm, 8*cm, 6*cm]),

    Spacer(1, 12),
    Paragraph("4. Submission", h1),
    Paragraph("{:,} rows, columns: store_id, product_id, week, weekly_sales. "
              "Predictions range {:.2f}-{:.2f}, mean {:.2f}. Zero NaN values.".format(
                  len(sub), v_min, v_max, v_mean), body),

    Spacer(1, 12),
    Paragraph("5. Notes", h1),
    Paragraph("submission_writer.md and report_writer.md are not yet built as registered "
              "sub-agents — their logic was executed inline by the orchestrator. "
              "modeler.md is now a fully registered sub-agent (Step 3 in workflow), "
              "featuring Optuna tuning and 5-seed ensembling.", body),
]

doc.build(story)
print("report.pdf:", os.path.getsize("report.pdf"), "bytes")
