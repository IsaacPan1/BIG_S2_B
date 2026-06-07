"""report_writer.py — CV-aware report generator.

Reads CV_PLAN, fold artefacts, validator + critic verdicts, and writes
report.pdf (or report.txt if reportlab is unavailable) at the repo root.
"""
from __future__ import annotations

import json
import datetime
from pathlib import Path

import pandas as pd


def _load(p: Path, parquet: bool = False):
    if not p.exists():
        return None
    return pd.read_parquet(p) if parquet else json.load(open(p))


def run(reports_dir: str | Path = "reports", out_path: str | Path = "report.pdf") -> str:
    reports_dir = Path(reports_dir)
    out_path = Path(out_path)

    plan = _load(reports_dir / "cv_plan.json") or {}
    folds = _load(reports_dir / "cv_folds.json") or {}
    fm = _load(reports_dir / "fold_metrics.json") or {}
    mr = _load(reports_dir / "model_results.json") or {}
    vr = _load(reports_dir / "validator_review.json") or {}
    cr = _load(reports_dir / "critic_review.json") or {}

    verdict = cr.get("verdict", "N/A")
    cv = plan.get("cv", {})

    def f(v, prec=4):
        if v is None: return "N/A"
        try:
            return f"{float(v):.{prec}f}"
        except Exception:
            return str(v)

    # Try reportlab first
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
        )
        HAVE_RL = True
    except ImportError:
        HAVE_RL = False

    if HAVE_RL:
        out_path = out_path.with_suffix(".pdf")
        doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        H1 = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=18, spaceAfter=8)
        H2 = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=13,
                            spaceBefore=10, spaceAfter=6)
        BODY = ParagraphStyle("Body", fontName="Helvetica", fontSize=10,
                              leading=14, spaceAfter=6)
        TS = TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#F4F4F4"), colors.white]),
        ])
        def tbl(headers, rows, widths=None):
            t = Table([headers] + rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
            t.setStyle(TS)
            return t

        story = []
        story.append(Paragraph("CV-as-Contract Pipeline Report", H1))
        story.append(Paragraph(
            f"Problem: <b>{plan.get('problem_type', 'N/A')}</b> | "
            f"CV: <b>{cv.get('cv_type', 'N/A')}</b> | "
            f"OOF MAE: <b>{f(fm.get('oof_mae'))}</b> | "
            f"Critic: <b>{verdict}</b>", BODY))

        story.append(Paragraph("Section 2 — CV_PLAN (Contract)", H2))
        plan_rows = [
            ["plan_id", str(plan.get("plan_id"))],
            ["problem_type", str(plan.get("problem_type"))],
            ["problem_subtype", str(plan.get("problem_subtype"))],
            ["cv_type", str(cv.get("cv_type"))],
            ["n_splits", str(cv.get("n_splits"))],
            ["time_column", str(plan.get("time_column"))],
            ["group_columns", ", ".join(plan.get("group_columns") or [])],
            ["horizon", str(plan.get("horizon"))],
            ["gap", str(cv.get("gap"))],
            ["frozen", str(plan.get("frozen"))],
        ]
        story.append(tbl(["Field", "Value"], plan_rows, widths=[5*cm, 11*cm]))

        risks = plan.get("leakage_risks", [])
        if risks:
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph("Leakage Risk Register", H2))
            risk_rows = [[r.get("id",""), r.get("kind",""),
                          str(r.get("column","")), r.get("mitigation","")] for r in risks]
            story.append(tbl(["ID", "Kind", "Column", "Mitigation"], risk_rows,
                             widths=[2.5*cm, 4*cm, 3.5*cm, 6*cm]))
        story.append(PageBreak())

        story.append(Paragraph("Section 4 — Model Performance per Fold", H2))
        story.append(Paragraph(
            "Active backends: CatBoost + Ridge. Blend = equal-weight mean. "
            "LightGBM/XGBoost interface-only.", BODY))
        pf_rows = []
        for r in mr.get("per_fold", []):
            pf_rows.append([
                str(r["fold_id"]),
                f(r.get("catboost_mae")),
                f(r.get("ridge_mae")),
                f(r["mae"]),
            ])
        if pf_rows:
            story.append(tbl(["Fold", "CatBoost MAE", "Ridge MAE", "Blend MAE"], pf_rows,
                             widths=[1.5*cm, 4*cm, 4*cm, 4*cm]))

        story.append(Paragraph("Section 5 — OOF Aggregate and Stability", H2))
        oof_rows = [
            ["OOF MAE", f(fm.get("oof_mae"))],
            ["fold MAE mean", f(fm.get("fold_mae_mean"))],
            ["fold MAE std", f(fm.get("fold_mae_std"))],
            ["fold MAE min", f(fm.get("fold_mae_min"))],
            ["fold MAE max", f(fm.get("fold_mae_max"))],
        ]
        story.append(tbl(["Statistic", "Value"], oof_rows, widths=[5*cm, 11*cm]))

        story.append(Paragraph("Section 6 — Critic Verdict", H2))
        story.append(Paragraph(f"<b>{verdict}</b> — {cr.get('recommendation','')}", BODY))
        check_rows = [[c["name"], c["status"], c.get("details","")[:200]]
                      for c in cr.get("checks", [])]
        if check_rows:
            story.append(tbl(["Check", "Status", "Details"], check_rows,
                             widths=[4*cm, 3*cm, 9*cm]))

        doc.build(story)
    else:
        # Plain text fallback
        out_path = out_path.with_suffix(".txt")
        lines = [
            "CV-as-Contract Pipeline Report",
            f"Problem: {plan.get('problem_type')}  CV: {cv.get('cv_type')}  "
            f"OOF MAE: {f(fm.get('oof_mae'))}  Critic verdict: {verdict}",
            "",
            "CV_PLAN:",
            f"  plan_id={plan.get('plan_id')}",
            f"  cv_type={cv.get('cv_type')}  n_splits={cv.get('n_splits')}",
            f"  time={plan.get('time_column')}  groups={plan.get('group_columns')}",
            f"  horizon={plan.get('horizon')}  gap={cv.get('gap')}",
            "",
            f"OOF MAE = {f(fm.get('oof_mae'))}",
            f"fold MAE mean/std = {f(fm.get('fold_mae_mean'))} / {f(fm.get('fold_mae_std'))}",
            "",
            "Critic verdict:",
            f"  {verdict} — {cr.get('recommendation','')}",
        ]
        for c in cr.get("checks", []):
            lines.append(f"  - {c['name']}: {c['status']} ({c.get('details','')[:120]})")
        out_path.write_text("\n".join(lines), encoding="utf-8")

    with open(reports_dir / "report_writer_was_here.txt", "w") as fmark:
        fmark.write(f"report_writer executed at {datetime.datetime.utcnow().isoformat()}Z\n")

    return str(out_path)
