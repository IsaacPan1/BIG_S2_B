"""Generate report.pdf — user-friendly pipeline results report."""
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image as RLImage, HRFlowable,
    )
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
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

BASE    = Path(__file__).resolve().parent.parent
REPORTS = BASE / "reports"
missing_inputs = []

def load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        missing_inputs.append(f"{Path(path).name}: {e}")
        return {}

def load_text(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        missing_inputs.append(f"{Path(path).name}: {e}")
        return ""

profile     = load_json(REPORTS / "profile.json")
features    = load_json(REPORTS / "features.json")
model_res   = load_json(REPORTS / "model_results.json")
sub_summary = load_json(REPORTS / "submission_summary.json")
critic_rev  = load_json(REPORTS / "critic_review.json")
val_rev     = load_json(REPORTS / "validator_review.json")
load_text(REPORTS / "schema_analysis.md")
load_text(BASE / "data" / "DATA_DESCRIPTION.md")

# ── Field extraction — exact field names from the JSON schemas ────────────────
problem_type = profile.get("problem_type", "Unknown")
confidence   = profile.get("problem_type_confidence", "Unknown")
target_col   = profile.get("target_col", "Unknown")
group_cols   = ", ".join(profile.get("group_cols") or []) or "None"
time_col     = profile.get("time_col") or "None"
n_train      = profile.get("n_train_rows", "Unknown")
n_val        = profile.get("n_val_rows", "Unknown")
n_horizons   = profile.get("n_horizons", None)
dist_shifts  = profile.get("distribution_shifts", [])
has_images   = (profile.get("image_data") or {}).get("present", False)
schema       = profile.get("schema", {})
flagged_cols = [d["column"] for d in dist_shifts if d.get("flagged")]

null_pcts       = [v.get("null_pct", 0) for v in schema.values()
                   if isinstance(v, dict) and "null_pct" in v]
overall_missing = sum(null_pcts) / len(null_pcts) if null_pcts else 0.0
target_std      = (schema.get(target_col) or {}).get("std", None)

feat_families   = features.get("feature_families", {})
feat_columns    = features.get("feature_columns", [])
total_features  = features.get("total_features_planned", len(feat_columns))
n_feat_families = len(feat_families)
adv_val         = features.get("adversarial_validation", {})
image_feat_info = features.get("image_features", {})
smart_lag_info  = features.get("smart_lag_imputation", {})

algorithm    = model_res.get("algorithm", "Unknown")
objective    = model_res.get("objective", "Unknown")
best_params  = model_res.get("best_params", {})
n_estimators = model_res.get("n_estimators", "Unknown")
n_seeds      = model_res.get("n_seeds", 1)
mae          = model_res.get("oof_mae") or model_res.get("walk_forward_mae", None)
cv_scheme    = model_res.get("oof_cv_scheme") or model_res.get("cv_scheme") or "CV MAE"
train_time   = model_res.get("training_time_seconds", "Unknown")
n_trials     = model_res.get("optuna_trials_completed", model_res.get("n_trials", None))
feat_imp     = model_res.get("feature_importance_top10", [])
families     = model_res.get("families", {})
baseline_mae = model_res.get("baseline_mae", None)
grp_baseline = model_res.get("group_baseline_mae", None)

_ac             = model_res.get("adaptive_choice", {})
ens_branch      = _ac.get("branch", "N/A")
ens_reasoning   = _ac.get("reasoning", "N/A")
ens_weighting   = _ac.get("ensemble_weighting", "equal_median")
weight_reason   = _ac.get("weighting_reason", "")
adv_info_model  = _ac.get("adversarial_validation", {})
ensemble_blend  = _ac.get("ensemble_blend", None)
blend_weights   = _ac.get("blend_weights", {})
blend_mae_equal = _ac.get("blend_holdout_mae_equal", None)
blend_mae_inv   = _ac.get("blend_holdout_mae_inv", None)
optuna_refl     = model_res.get("optuna_reflection", {})
postproc        = model_res.get("postprocessing", {})
ensemble_oof_mae = model_res.get("ensemble_oof_mae", None)

pred_stats    = sub_summary.get("prediction_stats", {})
sub_row_count = sub_summary.get("row_count", "Unknown")
sub_min       = pred_stats.get("min", None)
sub_max       = pred_stats.get("max", None)
sub_mean      = pred_stats.get("mean", None)
sub_std       = pred_stats.get("std", None)
sub_nan       = pred_stats.get("n_nan", 0)
sub_neg       = pred_stats.get("n_negative", 0)
sub_passed    = sub_summary.get("validation_checks_passed", None)

val_verdict      = val_rev.get("verdict", "N/A")
val_rep_mae      = val_rev.get("reported_cv_mae") or val_rev.get("honest_cv_mae")
val_strict       = val_rev.get("strict_cv_mae")
val_gap_frac     = val_rev.get("cv_gap_pct", 0.0)
val_gap_pct      = val_gap_frac * 100
val_cv_scheme    = val_rev.get("strict_cv_scheme", "N/A")
val_feature_susp = val_rev.get("feature_suspicion", [])
val_notes        = val_rev.get("notes", "")

critic_status    = critic_rev.get("status", "N/A")
critic_cycle     = critic_rev.get("cycle", 1)
critic_checks    = critic_rev.get("checks", [])
critic_retune    = critic_rev.get("retune_attempted", False)
critic_rationale = critic_rev.get("decision_rationale", "")
critic_warnings  = critic_rev.get("warnings_for_report", [])

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

n_warnings  = sum(1 for ck in critic_checks if ck.get("status") == "WARNING")
n_criticals = sum(1 for ck in critic_checks if ck.get("status") == "CRITICAL")
if val_verdict == "CRITICAL":
    n_criticals += 1
elif val_verdict == "WARNING":
    n_warnings += 1

def fmt(v, d=0):
    if v is None or v == "Unknown":
        return "N/A"
    try:
        return f"{int(float(v)):,}" if d == 0 else f"{float(v):,.{d}f}"
    except Exception:
        return str(v)

def fmt_bool(v):
    return "N/A" if v is None else ("Yes" if v else "No")

def mae_interpretation():
    if mae is None:
        return "N/A — metric not available"
    base = f"OOF MAE: {fmt(mae, 4)}"
    if val_verdict == "CRITICAL" or n_criticals >= 2:
        return f"{base} (may be inflated — see Section 5)"
    if val_verdict == "WARNING" or n_warnings >= 2:
        return f"{base} (borderline — minor optimism possible)"
    return f"{base} (clean cross-validation)"

def recommendation():
    if not sub_passed:
        return "FIX ISSUES — submission validation failed", '#d7191c'
    if val_verdict == "CRITICAL":
        return "REVIEW — critical CV integrity issue detected", '#d7191c'
    if val_verdict == "WARNING" or (critic_status not in ("accepted", "N/A", "")):
        return "SUBMIT WITH CAUTION — review Section 5", '#e08200'
    return "SUBMIT", '#1a9641'

# ── Charts ────────────────────────────────────────────────────────────────────
chart_imp  = None
chart_hist = None

if MATPLOTLIB_AVAILABLE and feat_imp:
    try:
        names  = [x.get("feature", "")    for x in feat_imp[:10]]
        scores = [x.get("importance", 0) for x in feat_imp[:10]]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(names[::-1], scores[::-1], color='#336699')
        ax.set_xlabel('Importance (split count)')
        ax.set_title('Top 10 Feature Importances')
        plt.tight_layout()
        chart_imp = str(REPORTS / 'feature_importance.png')
        fig.savefig(chart_imp, dpi=100)
        plt.close(fig)
    except Exception as e:
        missing_inputs.append(f"importance chart: {e}")

if MATPLOTLIB_AVAILABLE and sub_mean is not None:
    try:
        import pandas as pd
        preds_df = pd.read_csv(REPORTS / "predictions.csv")
        pcol = next((c for c in preds_df.columns if 'pred' in c.lower()), None)
        if pcol is None:
            pcol = next((c for c in preds_df.columns if c == target_col), None)
        if pcol:
            fig, ax = plt.subplots(figsize=(7, 3))
            preds_df[pcol].hist(bins=50, ax=ax, color='#336699', edgecolor='white')
            ax.set_xlabel('Predicted Value')
            ax.set_ylabel('Count')
            ax.set_title('Prediction Distribution')
            plt.tight_layout()
            chart_hist = str(REPORTS / 'prediction_histogram.png')
            fig.savefig(chart_hist, dpi=100)
            plt.close(fig)
    except Exception as e:
        missing_inputs.append(f"histogram chart: {e}")

# ── Plain-text fallback ───────────────────────────────────────────────────────
if not REPORTLAB_AVAILABLE:
    rec_text, _ = recommendation()
    lines = [
        "AUTONOMOUS DATA ANALYSIS REPORT",
        f"Generated: {timestamp}",
        f"Problem: {problem_type} | Target: {target_col}",
        f"Algorithm: {algorithm} | {mae_interpretation()}",
        f"Submission rows: {fmt(sub_row_count)} | Validation passed: {fmt_bool(sub_passed)}",
        f"Recommendation: {rec_text}",
        "", "NOTE: reportlab unavailable — plain text fallback.",
    ] + (["Missing: " + "; ".join(missing_inputs)] if missing_inputs else [])
    with open(BASE / "report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    _pdf_written = os.path.getsize(str(BASE / "report.txt")) > 0
    print("Wrote report.txt (reportlab unavailable)" if _pdf_written
          else "ERROR: report.txt empty after write — OS write not confirmed")
else:
    # ── Style definitions ─────────────────────────────────────────────────────
    H1   = ParagraphStyle('H1',   fontName='Helvetica-Bold', fontSize=18, spaceAfter=4)
    H2   = ParagraphStyle('H2',   fontName='Helvetica-Bold', fontSize=13, spaceBefore=14, spaceAfter=6)
    H3   = ParagraphStyle('H3',   fontName='Helvetica-Bold', fontSize=11, spaceBefore=8,  spaceAfter=4)
    BODY = ParagraphStyle('Body', fontName='Helvetica',       fontSize=10, leading=15, spaceAfter=8)
    META = ParagraphStyle('Meta', fontName='Helvetica',       fontSize=9,  spaceAfter=4,
                          textColor=colors.HexColor('#666666'))
    BOLD = ParagraphStyle('Bold', fontName='Helvetica-Bold',  fontSize=10, spaceAfter=6)
    FOOT = ParagraphStyle('Foot', fontName='Helvetica',       fontSize=7,
                          textColor=colors.grey, alignment=TA_CENTER)
    _CS  = ParagraphStyle('CS',   fontName='Helvetica',       fontSize=9, leading=11)
    _CH  = ParagraphStyle('CH',   fontName='Helvetica-Bold',  fontSize=9, leading=11,
                          textColor=colors.white)

    def _p(text, st=None):
        return Paragraph(str(text) if text is not None else '', st or _CS)

    _BASE_TS_CMDS = [
        ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#2C3E50')),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
        ('FONTNAME',      (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#F2F2F2'), colors.white]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('LEFTPADDING',   (0,0), (-1,-1), 6),
        ('RIGHTPADDING',  (0,0), (-1,-1), 6),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]
    BASE_TS = TableStyle(_BASE_TS_CMDS)

    def tbl(headers, rows, widths=None):
        data = [[_p(h, _CH) for h in headers]]
        data += [[_p(c) for c in row] for row in rows]
        t = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1)
        t.setStyle(BASE_TS)
        return t

    def callout(title, bullets, title_bg='#2980B9', body_bg='#EBF5FB'):
        """Highlighted callout panel with a title bar and bullet lines."""
        title_st = ParagraphStyle('CT', fontName='Helvetica-Bold', fontSize=9,
                                   textColor=colors.white)
        body_st  = ParagraphStyle('CB', fontName='Helvetica', fontSize=9, leading=13)
        body_html = "<br/>".join(f"• {b}" for b in bullets)
        t = Table(
            [[_p(f"  {title}", title_st)], [_p(body_html, body_st)]],
            colWidths=[16*cm], hAlign='LEFT',
        )
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor(title_bg)),
            ('BACKGROUND',    (0,1), (-1,-1), colors.HexColor(body_bg)),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
            ('RIGHTPADDING',  (0,0), (-1,-1), 8),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LINEBELOW',     (0,-1),(-1,-1), 0.8, colors.HexColor(title_bg)),
        ]))
        return t

    def exec_summary():
        rec_text, rec_hex = recommendation()
        rec_st = ParagraphStyle('Rec', fontName='Helvetica-Bold', fontSize=9,
                                 textColor=colors.HexColor(rec_hex))
        hdr_st = ParagraphStyle('EHdr', fontName='Helvetica-Bold', fontSize=10,
                                 textColor=colors.white, alignment=TA_CENTER)
        rows_data = [
            [_p('Executive Summary', hdr_st), _p('', hdr_st)],
            [_p('Result', _CS),         _p(mae_interpretation(), _CS)],
            [_p('Submission', _CS),     _p(
                f"{'Valid' if sub_passed else 'INVALID'} — "
                f"{fmt(sub_row_count)} rows, {sub_nan} NaN, {sub_neg} negative", _CS)],
            [_p('Quality issues', _CS), _p(
                f"{n_warnings} warning(s), {n_criticals} critical(s)", _CS)],
            [_p('Training time', _CS),  _p(f"{train_time} s", _CS)],
            [_p('Recommendation', _CS), _p(rec_text, rec_st)],
        ]
        alternating = ['#D6EAF8', '#EBF5FB', '#D6EAF8', '#EBF5FB', '#D6EAF8']
        ts_cmds = [
            ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#1A5276')),
            ('SPAN',          (0,0), (-1,0)),
            ('ALIGN',         (0,0), (-1,0),  'CENTER'),
            ('FONTNAME',      (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE',      (0,0), (-1,-1), 9),
            ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#85C1E9')),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
            ('RIGHTPADDING',  (0,0), (-1,-1), 8),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
            ('LINEABOVE',     (0,0), (-1,0),  2.5, colors.HexColor('#1A5276')),
            ('LINEBELOW',     (0,-1),(-1,-1), 2.5, colors.HexColor('#1A5276')),
        ]
        for i, bg in enumerate(alternating):
            ts_cmds.append(('BACKGROUND', (0, i+1), (-1, i+1), colors.HexColor(bg)))
        t = Table(rows_data, colWidths=[4.5*cm, 11.5*cm], hAlign='LEFT')
        t.setStyle(TableStyle(ts_cmds))
        return t

    def pipeline_status_table():
        """Status board: one row per agent, status column colour-coded by outcome.

        Width-check: if sum(colWidths) > 80 % of the page width the Key Output
        column is compressed to fit; if it cannot reach ≥ 5 cm the font drops to
        8 pt; if it still cannot reach ≥ 3 cm a 2-column vertical layout is used.
        Key Output text is capped with textwrap.shorten (~7.5 chars / cm) so that
        a single unbroken token cannot force the column to overflow into neighbours.
        """
        _PAGE_W  = A4[0]               # full A4 page width in points (~595 pt / 21 cm)
        _THRESH  = 0.80 * _PAGE_W      # 80 % cap (~476 pt / 16.8 cm)

        # Preferred column widths (Step | Agent | Status | Key Output)
        _w_step   = 1.2 * cm
        _w_agent  = 4.2 * cm
        _w_status = 2.2 * cm
        _w_key    = 8.4 * cm
        _total    = _w_step + _w_agent + _w_status + _w_key

        _font_sz  = 9
        _use_vert = False

        if _total > _THRESH:
            _w_key_fit = _THRESH - _w_step - _w_agent - _w_status
            if _w_key_fit >= 5.0 * cm:
                _w_key = _w_key_fit          # compress key column, keep font size
            elif _w_key_fit >= 3.0 * cm:
                _w_key  = _w_key_fit
                _font_sz = 8                 # also reduce font to compensate
            else:
                _use_vert = True             # column too narrow: fall back to vertical

        # textwrap limit: ~7.5 chars per cm at 9 pt Helvetica (accounting for padding)
        _KEY_CHARS = max(50, int(_w_key / cm * 7.5))

        # Tuple: (step, name, marker_path, detail_text, artifact_path)
        # artifact_path is checked when the marker is absent — a stage whose
        # output artifact exists on disk did not "not run" (marker missing ≠ not run).
        AGENTS = [
            ("1",   "schema_analyst",   "reports/schema_analyst_was_here.txt",
             f"{problem_type} ({confidence} conf)",
             "reports/schema_analysis.md"),
            ("2",   "feature_engineer", "reports/feature_engineer_was_here.txt",
             f"{total_features} features, {n_feat_families} families",
             "reports/features.json"),
            ("3",   "modeler",          "reports/modeler_was_here.txt",
             f"OOF MAE: {fmt(mae, 4)} | {algorithm[:35]}",
             "reports/predictions.csv"),
            ("3.5", "validator",        "reports/validator_was_here.txt",
             f"Verdict: {val_verdict}",
             "reports/validator_review.json"),
            ("3.6", "critic",           "reports/critic_was_here.txt",
             f"Status: {critic_status} | retune: {'Yes' if critic_retune else 'No'}",
             "reports/critic_review.json"),
            ("4",   "submission_writer","reports/submission_writer_was_here.txt",
             f"{fmt(sub_row_count)} rows | valid: {fmt_bool(sub_passed)}",
             "submission.csv"),
            ("5",   "report_writer",    "reports/report_writer_was_here.txt",
             "report.pdf",
             "report.pdf"),
        ]

        def _status(step, name, marker, artifact=None):
            # Self-reference guard: generate_report.py IS the report_writer.
            # The marker is written only after the PDF is built (which embeds this
            # table), so the exists()-check can never see it.  The fact that this
            # code is executing means the stage is running → PASS.
            if name == "report_writer":
                return "PASS", '#D5F5E3', '#145a32'
            if not os.path.exists(BASE / marker):
                # A stage whose output artifact exists on disk did run —
                # a missing marker just means the marker write was skipped or
                # the stage was called without the canonical tool.
                if artifact and os.path.exists(BASE / artifact):
                    return "RAN (marker missing)", '#FEF9E7', '#e08200'
                return "NOT RUN", '#FDECEA', '#d7191c'
            if name == "validator":
                if val_verdict == "CRITICAL":
                    return "CRITICAL", '#FDECEA', '#d7191c'
                if val_verdict == "WARNING":
                    return "WARNING", '#FEF9E7', '#e08200'
            if name == "critic" and critic_status not in ("accepted", "N/A", ""):
                return "REVIEW",  '#FEF9E7', '#e08200'
            if name == "submission_writer" and not sub_passed:
                return "FAILED",  '#FDECEA', '#d7191c'
            return "PASS", '#D5F5E3', '#145a32'

        # Per-cell styles sized to the resolved font size
        _cell_st = ParagraphStyle('PST_cell', fontName='Helvetica',
                                   fontSize=_font_sz, leading=_font_sz + 2)
        _hdr_st  = ParagraphStyle('PST_hdr',  fontName='Helvetica-Bold',
                                   fontSize=_font_sz, leading=_font_sz + 2,
                                   textColor=colors.white)

        # Build row data once; shared by both layout branches
        entries = []
        for step, name, marker, detail, artifact in AGENTS:
            lbl, row_bg, txt_hex = _status(step, name, marker, artifact)
            detail_safe = textwrap.shorten(detail, width=_KEY_CHARS, placeholder='…')
            st_style = ParagraphStyle(f'PST_s_{name}', fontName='Helvetica-Bold',
                                       fontSize=_font_sz,
                                       textColor=colors.HexColor(txt_hex))
            entries.append((step, name, lbl, row_bg, detail_safe, st_style))

        # ── Vertical layout: 2 columns ─────────────────────────────────────────
        if _use_vert:
            _lbl_w = _THRESH * 0.40
            _val_w = _THRESH - _lbl_w
            vrows = [[_p('Agent / Stage', _hdr_st), _p('Status  ·  Key Output', _hdr_st)]]
            vbgs  = []
            for step, name, lbl, row_bg, detail_safe, st_style in entries:
                vrows.append([
                    Paragraph(f"<b>{step}.  {name}</b>", _cell_st),
                    Paragraph(f"<b>{lbl}</b>  ·  {detail_safe}", st_style),
                ])
                vbgs.append(row_bg)
            ts_v = list(_BASE_TS_CMDS)
            ts_v.append(('FONTSIZE', (0,0), (-1,-1), _font_sz))
            for i, bg in enumerate(vbgs):
                alt = '#F2F2F2' if i % 2 == 0 else '#FFFFFF'
                ts_v.append(('BACKGROUND', (0, i+1), (0, i+1), colors.HexColor(alt)))
                ts_v.append(('BACKGROUND', (1, i+1), (1, i+1), colors.HexColor(bg)))
            ts_v = [c for c in ts_v if c[0] != 'ROWBACKGROUNDS']
            t = Table(vrows, colWidths=[_lbl_w, _val_w], hAlign='LEFT', repeatRows=1)
            t.setStyle(TableStyle(ts_v))
            return t

        # ── Standard 4-column layout ───────────────────────────────────────────
        col_widths = [_w_step, _w_agent, _w_status, _w_key]
        hdr = [_p('Step', _hdr_st), _p('Agent', _hdr_st),
               _p('Status', _hdr_st), _p('Key Output', _hdr_st)]
        rows = []
        status_bgs = []
        for step, name, lbl, row_bg, detail_safe, st_style in entries:
            rows.append([
                _p(step,                    _cell_st),
                Paragraph(f"<b>{name}</b>", _cell_st),
                Paragraph(lbl,              st_style),
                _p(detail_safe,             _cell_st),
            ])
            status_bgs.append(row_bg)

        data = [hdr] + rows
        ts_cmds = list(_BASE_TS_CMDS)
        ts_cmds.append(('FONTSIZE', (0,0), (-1,-1), _font_sz))
        # Per-row: status colour only on col 2; alternate bg on cols 0-1 and 3
        for i, bg in enumerate(status_bgs):
            alt = '#F2F2F2' if i % 2 == 0 else '#FFFFFF'
            ts_cmds.append(('BACKGROUND', (0, i+1), (1, i+1), colors.HexColor(alt)))
            ts_cmds.append(('BACKGROUND', (2, i+1), (2, i+1), colors.HexColor(bg)))
            ts_cmds.append(('BACKGROUND', (3, i+1), (3, i+1), colors.HexColor(alt)))
        ts_cmds = [c for c in ts_cmds if c[0] != 'ROWBACKGROUNDS']
        t = Table(data, colWidths=col_widths, hAlign='LEFT', repeatRows=1)
        t.setStyle(TableStyle(ts_cmds))
        return t

    # ── Build story ───────────────────────────────────────────────────────────
    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story += [
        _p("Autonomous Data Analysis Report", H1),
        _p(f"Problem: {problem_type}  |  Target: {target_col}  |  {timestamp}", META),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2C3E50")),
        Spacer(1, 0.3*cm),
    ]

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(exec_summary())
    story.append(Spacer(1, 0.6*cm))  # min-height buffer prevents overlap with next table

    # ── Pipeline Status ───────────────────────────────────────────────────────
    story.append(_p("Pipeline Status", H2))
    story.append(Spacer(1, 0.25*cm))  # gap between heading and table top border
    story.append(pipeline_status_table())

    dec_lines = [f"Axis 1 — dataset size: Branch {ens_branch} — {ens_reasoning}"]
    if ens_weighting:
        dec_lines.append(f"Axis 2 — shift weighting: {ens_weighting}"
                         + (f" — {weight_reason}" if weight_reason else ""))
    if critic_retune:
        dec_lines.append(f"Critic retune: cycle {critic_cycle}"
                         + (f" — {critic_rationale}" if critic_rationale else ""))
    story.append(Spacer(1, 0.2*cm))
    story.append(callout("Adaptive Decisions Made", dec_lines,
                          title_bg='#5D6D7E', body_bg='#F2F3F4'))
    story.append(PageBreak())

    # ── Pipeline Configuration ────────────────────────────────────────────────
    story.append(_p("Pipeline Configuration", H2))
    failed_agents = [
        name for marker, name in [
            ("reports/schema_analyst_was_here.txt",    "schema_analyst"),
            ("reports/feature_engineer_was_here.txt",  "feature_engineer"),
            ("reports/modeler_was_here.txt",            "modeler"),
            ("reports/validator_was_here.txt",          "validator"),
            ("reports/critic_was_here.txt",             "critic"),
            ("reports/submission_writer_was_here.txt",  "submission_writer"),
        ]
        if not os.path.exists(BASE / marker)
    ]
    config_rows = [
        ["Hardware",        "CPU-only; no GPU acceleration"],
        ["Network access",  "None — all signal derived from provided data files only"],
        ["External data",   "None — no downloads, no pretrained models"],
        ["Agents completed", "All sub-agents ran" if not failed_agents
                             else f"Fallback used for: {', '.join(failed_agents)}"],
        ["Image handling",  "Image features extracted" if has_images
                            else "No image data detected"],
    ]
    if missing_inputs:
        config_rows.append(["Report warnings",
                            "; ".join(missing_inputs[:3])
                            + (f" (+{len(missing_inputs)-3} more)" if len(missing_inputs) > 3 else "")])
    story.append(tbl(["Setting", "Value"], config_rows, widths=[4.5*cm, 11.5*cm]))
    story.append(PageBreak())

    # ── Section 1: Problem Classification ────────────────────────────────────
    story.append(_p("Section 1 — Problem Classification", H2))
    story.append(tbl(["Property", "Value"], [
        ["Problem type",     problem_type],
        ["Confidence",       confidence],
        ["Target column",    target_col],
        ["Group columns",    group_cols],
        ["Time column",      time_col],
        ["Training rows",    fmt(n_train)],
        ["Validation rows",  fmt(n_val)],
        ["Forecast horizon", f"{n_horizons} steps" if n_horizons else "N/A"],
    ], widths=[5*cm, 11*cm]))
    story.append(Spacer(1, 0.2*cm))
    story.append(_p(
        f"Dataset classified as <b>{problem_type}</b> (<b>{confidence}</b> confidence). "
        f"Target column: <b>{target_col}</b>. Groups defined by [{group_cols}], "
        f"ordered by <b>{time_col}</b>. "
        f"Training: {fmt(n_train)} rows; hold-out validation: {fmt(n_val)} rows.",
        BODY,
    ))
    story.append(PageBreak())

    # ── Section 2: Data Quality ────────────────────────────────────────────────
    story.append(_p("Section 2 — Data Quality", H2))
    if dist_shifts:
        story.append(tbl(
            ["Column", "KS Statistic", "p-value", "Flagged"],
            [[d.get("column",""), fmt(d.get("ks_statistic"),4),
              fmt(d.get("p_value"),4),
              "YES — significant" if d.get("flagged") else "no"]
             for d in dist_shifts],
            widths=[5*cm, 3.5*cm, 3*cm, 4.5*cm],
        ))
    else:
        story.append(_p("No distribution shift data found in profile.json.", BODY))
    story.append(Spacer(1, 0.2*cm))
    story.append(_p(
        f"Overall average missingness: {overall_missing:.2f}%. "
        "Time-series values forward-filled within each group; remaining gaps "
        "filled with column medians.",
        BODY,
    ))
    if flagged_cols:
        max_ks = max((d.get("ks_statistic",0) for d in dist_shifts if d.get("flagged")), default=0)
        story.append(Spacer(1, 0.1*cm))
        story.append(callout(
            f"Distribution Shift Detected — {len(flagged_cols)} column(s) flagged",
            [f"Column(s): {', '.join(flagged_cols)}  |  Max KS: {max_ks:.4f}",
             "Shift-aware features added: z-score normalisation, rolling z-score, "
             "percentile rank, group deviation, covariate × time interaction",
             f"Ensemble weighting adjusted to: {ens_weighting}"],
            title_bg='#C0392B', body_bg='#FADBD8',
        ))
    else:
        story.append(_p("No significant distribution shift detected. Standard features used.", BODY))
    story.append(PageBreak())

    # ── Section 3: Feature Engineering ────────────────────────────────────────
    story.append(_p("Section 3 — Feature Engineering", H2))
    story.append(_p(
        f"<b>{total_features}</b> features engineered across "
        f"<b>{n_feat_families}</b> families.",
        BODY,
    ))
    if feat_families:
        fg_rows = []
        for fname, fval in feat_families.items():
            if isinstance(fval, list):
                count, examples = len(fval), ", ".join(str(x) for x in fval[:4])
            elif isinstance(fval, dict):
                count, examples = len(fval), ", ".join(str(k) for k in list(fval.keys())[:4])
            else:
                count, examples = 1, str(fval)
            fg_rows.append([fname, str(count), examples])
        story.append(tbl(["Feature Family", "Count", "Examples / Specs"],
                         fg_rows, widths=[4*cm, 2*cm, 10*cm]))
    story.append(Spacer(1, 0.25*cm))

    # Unique feature callouts
    if flagged_cols:
        story.append(callout(
            "Shift-Aware Features Active",
            [f"Triggered by: {', '.join(flagged_cols[:6])}",
             "Per flagged column: z-score normalisation, rolling 4-period z-score, "
             "percentile rank (training distribution), group-level deviation, "
             "covariate × normalised-time interaction",
             "All statistics computed from training rows only — no leakage"],
            title_bg='#2980B9', body_bg='#EBF5FB',
        ))
        story.append(Spacer(1, 0.15*cm))

    if has_images and image_feat_info:
        n_img    = image_feat_info.get("n_images", "N/A")
        match_rt = image_feat_info.get("match_rate_pct", "N/A")
        story.append(callout(
            "Image Embedding Features Active",
            [f"23 hand-crafted spatial features extracted per image  |  "
             f"Images: {n_img}  |  Match rate: {match_rt}%",
             "Features: intensity stats, 2×2 quadrant mean/std, centre/edge contrast, "
             "bright/dark pixel fractions, inter-channel variance (RGB)",
             "Tree-based importance determines which features are actually used"],
            title_bg='#1E8449', body_bg='#D5F5E3',
        ))
        story.append(Spacer(1, 0.15*cm))
    elif has_images:
        story.append(callout(
            "Image Data Detected",
            ["Image files found; spatial features attempted via PIL extraction"],
            title_bg='#1E8449', body_bg='#D5F5E3',
        ))
        story.append(Spacer(1, 0.15*cm))

    if smart_lag_info and smart_lag_info.get("cells_imputed", 0) > 0:
        story.append(callout(
            "Smart Lag Imputation Active",
            [f"{fmt(smart_lag_info['cells_imputed'])} validation cells filled via "
             f"{smart_lag_info.get('method','cycle-aware')} imputation",
             "Uses seasonal cycle position (hour / day-of-week / week / month) "
             "instead of last-known training value",
             "Preserves seasonal signal far ahead of the training window"],
            title_bg='#7D3C98', body_bg='#F5EEF8',
        ))
        story.append(Spacer(1, 0.15*cm))

    if adv_val and adv_val.get("auc") is not None:
        auc       = adv_val.get("auc")
        applied   = adv_val.get("weights_applied", False)
        top_feats = adv_val.get("top_shift_features", [])
        story.append(callout(
            "Adversarial Validation Active",
            [f"AUC: {fmt(auc, 3)}  —  "
             f"{'sample weights applied to all model families' if applied else 'AUC < 0.55; uniform weights used'}",
             f"Top shift-revealing features: "
             f"{', '.join(str(f) for f in top_feats[:5]) if top_feats else 'N/A'}"],
            title_bg='#D35400', body_bg='#FDEBD0',
        ))
    story.append(PageBreak())

    # ── Section 4: Modeling ────────────────────────────────────────────────────
    story.append(_p("Section 4 — Modeling", H2))

    # 4A — Ensemble Architecture
    story.append(_p("4A — Ensemble Architecture", H3))
    fam_selected = _ac.get("families_selected", [])
    fam_included = _ac.get("families_included_in_ensemble", [])
    fam_excluded = [f for f in fam_selected if f not in fam_included]
    # Build blend weights string — e.g. "lightgbm=0.333, xgboost=0.333, catboost=0.333"
    blend_w_str = (", ".join(f"{k}={v:.4f}" for k, v in blend_weights.items())
                   if blend_weights else "N/A")
    arch_rows = [
        ["Families selected",      ", ".join(fam_selected) if fam_selected else algorithm],
        ["Families in ensemble",   ", ".join(fam_included) if fam_included else algorithm],
        ["Families excluded",      ", ".join(fam_excluded) if fam_excluded else "None"],
        ["Ensemble blend method",  ensemble_blend if ensemble_blend else "equal_median"],
        ["Blend weights",          blend_w_str],
        ["Ensemble weighting",     ens_weighting],
        ["Weighting rationale",    weight_reason or "Equal-weight median (no severe shift)"],
        ["Branch (Axis 1)",        f"Branch {ens_branch} — {ens_reasoning}"],
        ["Ensemble size",          str(model_res.get("n_families_in_ensemble", "N/A"))],
        ["Ensemble OOF MAE",       fmt(ensemble_oof_mae, 4) if ensemble_oof_mae is not None else "N/A"],
    ]
    if blend_mae_equal is not None:
        arch_rows.append(["Blend holdout MAE (equal)", fmt(blend_mae_equal, 4)])
    if blend_mae_inv is not None:
        arch_rows.append(["Blend holdout MAE (inv-wt)", fmt(blend_mae_inv, 4)])
    story.append(tbl(["Decision", "Value"], arch_rows, widths=[5*cm, 11*cm]))
    story.append(Spacer(1, 0.25*cm))

    # 4B — Hyperparameter Selection
    story.append(_p("4B — Hyperparameter Selection", H3))
    hp_rows = [
        ["Objective",       objective],
        ["Optuna trials",   str(n_trials) if n_trials else "N/A"],
        ["n_estimators",    str(n_estimators)],
        ["Ensemble seeds",  str(n_seeds)],
    ]
    if best_params:
        for k, v in list(best_params.items())[:8]:
            hp_rows.append([k, f"{round(v, 5) if isinstance(v, float) else v}"])
    # Optuna reflection block
    if optuna_refl:
        pinned = ", ".join(optuna_refl.get("pinned_params", [])) or "None"
        recentered = "Yes" if optuna_refl.get("recentered") else "No"
        mae_before = optuna_refl.get("best_mae_before")
        mae_after  = optuna_refl.get("best_mae_after")
        hp_rows += [
            ["Optuna reflection — pinned params",    pinned],
            ["Optuna reflection — recentered",       recentered],
            ["Optuna reflection — MAE before",       fmt(mae_before, 4) if mae_before is not None else "N/A"],
            ["Optuna reflection — MAE after",        fmt(mae_after, 4)  if mae_after  is not None else "N/A"],
        ]
    # Postprocessing block
    if postproc:
        rounding = "Yes" if postproc.get("ordinal_rounding_applied") else "No"
        raw_wf   = postproc.get("ordinal_raw_wf_mae")
        rnd_wf   = postproc.get("ordinal_round_wf_mae")
        hp_rows.append(["Postprocessing — ordinal rounding applied", rounding])
        if raw_wf is not None:
            hp_rows.append(["Postprocessing — raw WF MAE",     fmt(raw_wf, 4)])
            hp_rows.append(["Postprocessing — rounded WF MAE", fmt(rnd_wf, 4) if rnd_wf is not None else "N/A"])
    story.append(tbl(["Parameter", "Best Value"], hp_rows, widths=[5*cm, 11*cm]))
    story.append(Spacer(1, 0.25*cm))

    # 4C — Family Performance Comparison
    story.append(_p("4C — Family Performance Comparison", H3))
    if families:
        story.append(tbl(
            ["Family", "OOF MAE", "Train Time", "In Ensemble", "Exclusion Reason"],
            [[fname,
              fmt(fd.get("oof_mae"), 4),
              f"{fd.get('training_time_seconds','N/A')} s",
              "Yes" if fd.get("included_in_ensemble") else "No",
              "" if fd.get("included_in_ensemble")
                 else fd.get("exclusion_reason","competence check failed")]
             for fname, fd in families.items()],
            widths=[3.5*cm, 2.5*cm, 2.5*cm, 2*cm, 5.5*cm],
        ))
    else:
        story.append(tbl(["Parameter", "Value"], [
            ["Algorithm",           algorithm],
            [f"CV MAE ({cv_scheme})", fmt(mae, 4)],
            ["Training time",       f"{train_time} s"],
        ], widths=[5*cm, 11*cm]))

    # Adversarial weights callout in modeling context
    _adv_auc = (adv_info_model or adv_val or {}).get("auc")
    if _adv_auc is not None:
        _applied = (adv_info_model or adv_val or {}).get("weights_applied", False)
        story.append(Spacer(1, 0.2*cm))
        story.append(callout(
            "Adversarial Sample Weights Applied to All Families",
            [f"Adversarial AUC: {fmt(_adv_auc, 3)}",
             "Weights passed to: LightGBM OOF folds + retraining, "
             "XGBoost Optuna objective + final fit, Ridge",
             "Rows resembling the validation distribution receive higher training weight"],
            title_bg='#D35400', body_bg='#FDEBD0',
        ))

    story.append(Spacer(1, 0.25*cm))

    # Feature importances
    if feat_imp:
        story.append(_p("Top-10 Feature Importances", H3))
        def _fmt_imp(v):
            try:
                fv = float(v)
                # Float importances (gain-normalised): show as percentage
                if 0 < fv < 1:
                    return f"{fv*100:.3f}%"
                # Integer split-count importances
                return f"{int(fv):,}"
            except Exception:
                return str(v)
        story.append(tbl(["Rank", "Feature", "Importance"],
                         [[str(i+1), x.get("feature",""), _fmt_imp(x.get('importance',0))]
                          for i, x in enumerate(feat_imp[:10])],
                         widths=[1.5*cm, 11*cm, 3.5*cm]))
    if chart_imp and os.path.exists(chart_imp):
        story.append(Spacer(1, 0.2*cm))
        story.append(RLImage(chart_imp, width=14*cm, height=8*cm))
    story.append(PageBreak())

    # ── Section 5: CV and Integrity ────────────────────────────────────────────
    story.append(_p("Section 5 — Cross-Validation and Integrity Checking", H2))

    story.append(_p("<b>A. Validator Audit</b>", BOLD))
    story.append(tbl(["Property", "Value"], [
        ["CV scheme",         val_cv_scheme],
        ["Reported CV MAE",   fmt(val_rep_mae, 4) if val_rep_mae is not None else "N/A"],
        ["Strict CV MAE",     fmt(val_strict, 4)  if val_strict  is not None else "N/A"],
        ["CV gap (%)",        f"{val_gap_pct:.1f}%"],
        ["Verdict",           val_verdict],
        ["Suspect features",  ", ".join(str(f) for f in val_feature_susp)
                              if val_feature_susp else "None flagged"],
    ], widths=[5*cm, 11*cm]))
    if val_notes:
        story.append(_p(val_notes[:300] + ("…" if len(val_notes) > 300 else ""), BODY))
    story.append(Spacer(1, 0.25*cm))

    story.append(_p("<b>B. Critic Review</b>", BOLD))
    if critic_checks:
        check_rows = []
        for ck in critic_checks:
            hex_col = ('#1a9641' if ck.get("status") == "PASS"
                       else '#d7191c' if ck.get("status") == "CRITICAL"
                       else '#e08200')
            st_style = ParagraphStyle(f'Ck{id(ck)}', fontName='Helvetica-Bold',
                                       fontSize=9, textColor=colors.HexColor(hex_col))
            check_rows.append([_p(ck.get("name","")),
                                Paragraph(ck.get("status",""), st_style),
                                _p(ck.get("details",""))])
        ct = Table(
            [[_p("Check",_CH), _p("Status",_CH), _p("Details",_CH)]] + check_rows,
            colWidths=[4*cm, 2*cm, 10*cm], hAlign="LEFT",
        )
        ct.setStyle(BASE_TS)
        story.append(ct)
    else:
        story.append(_p("No checks recorded.", BODY))
    story.append(Spacer(1, 0.15*cm))
    story.append(_p(
        f"Critic status: <b>{critic_status}</b>.  "
        f"Retune attempted: <b>{'Yes' if critic_retune else 'No'}</b>."
        + (f"  Rationale: {critic_rationale}" if critic_rationale else ""),
        BODY,
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(_p("<b>C. Threshold Documentation</b>", BOLD))
    story.append(_p(
        "Validator WARNING threshold: 10% CV gap.  CRITICAL threshold: 25% CV gap.  "
        "Critic acceptance threshold: 15% (between WARNING and CRITICAL so that mild "
        "optimism is documented without blocking submission, while severe leakage "
        "triggers a corrective modeler retune).",
        BODY,
    ))
    story.append(PageBreak())

    # ── Section 6: Predictions and Submission ─────────────────────────────────
    story.append(_p("Section 6 — Predictions and Submission", H2))
    story.append(tbl(["Statistic", "Value"], [
        ["Row count",               fmt(sub_row_count)],
        ["Min prediction",          fmt(sub_min, 2)],
        ["Max prediction",          fmt(sub_max, 2)],
        ["Mean prediction",         fmt(sub_mean, 2)],
        ["Std prediction",          fmt(sub_std, 2)],
        ["NaN predictions",         str(sub_nan)],
        ["Negative predictions",    str(sub_neg)],
        ["Validation checks passed", fmt_bool(sub_passed)],
    ], widths=[7*cm, 9*cm]))
    story.append(Spacer(1, 0.25*cm))

    # Baseline comparison — shown only if the modeler wrote baseline fields
    if baseline_mae is not None or grp_baseline is not None:
        story.append(_p("Baseline Comparison", H3))
        base_rows = [["Model OOF MAE", fmt(mae, 4), "—"]]
        for label, bval in [("Group-mean baseline", grp_baseline),
                             ("Global-mean baseline", baseline_mae)]:
            if bval is None:
                continue
            try:
                skill = (1 - float(mae) / float(bval)) * 100 if mae is not None else None
                skill_str = f"{skill:.1f}% better than baseline" if skill is not None else "N/A"
            except Exception:
                skill_str = "N/A"
            base_rows.append([label, fmt(bval, 4), skill_str])
        story.append(tbl(["Predictor", "MAE", "Skill vs Baseline"],
                         base_rows, widths=[5*cm, 3*cm, 8*cm]))
    else:
        story.append(_p(
            "Baseline comparison: not available — the modeler did not write "
            "baseline_mae / group_baseline_mae to model_results.json.",
            META,
        ))

    if chart_hist and os.path.exists(chart_hist):
        story.append(Spacer(1, 0.2*cm))
        story.append(RLImage(chart_hist, width=14*cm, height=6*cm))
    story.append(PageBreak())

    # ── Section 7: Limitations and Risks ──────────────────────────────────────
    story.append(_p("Section 7 — Limitations and Risks", H2))

    # Build severity-tagged list: (severity, title, body, mitigation)
    lims = []
    SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    SEV_HEX   = {"CRITICAL": '#d7191c', "HIGH": '#e08200',
                 "MEDIUM":   '#2874A6', "LOW":  '#145a32'}
    SEV_BG    = {"CRITICAL": '#FDECEA', "HIGH": '#FEF9E7',
                 "MEDIUM":   '#EBF5FB', "LOW":  '#D5F5E3'}

    if flagged_cols:
        max_ks = max((d.get("ks_statistic",0) for d in dist_shifts if d.get("flagged")), default=0)
        lims.append(("HIGH",
            f"Distribution shift on {', '.join(flagged_cols)}",
            f"KS up to {max_ks:.4f}. Shift-aware features and Ridge weighting partially compensate. "
            "Residual risk: underperformance in the tails of shifted covariate ranges.",
            "Shift-aware features added; ensemble reweighted toward Ridge."))

    if val_verdict == "CRITICAL":
        lims.append(("CRITICAL",
            "CV integrity issue (validator CRITICAL)",
            "Significant gap between reported and purged walk-forward MAE. "
            "The reported metric may be optimistic.",
            "Modeler retune triggered." if critic_retune else "Manual review recommended."))

    lims.append(("MEDIUM",
        "Lag feature imputation at the validation boundary",
        "Within-horizon predictions rely on model estimates rather than actuals, "
        "introducing compounding error for longer forecast horizons.",
        "Cycle-aware imputation used where applicable."))

    if n_trials is not None:
        lims.append(("LOW",
            f"Limited Optuna budget ({n_trials} trials)",
            "Hyperparameter search capped to stay within the 2-hour wall-clock budget. "
            "Broader search would likely improve primary hyperparameters.",
            "Current parameters are near-optimal for primary dimensions."))

    if target_std is not None and sub_std is not None:
        try:
            if float(sub_std) < 0.7 * float(target_std):
                lims.append(("MEDIUM",
                    "Under-dispersion in predictions",
                    f"Predicted std ({fmt(sub_std,2)}) < 70% of training target std "
                    f"({fmt(target_std,2)}). Model shrinks toward the mean.",
                    "May underperform at demand extremes."))
        except Exception:
            pass

    for cw in critic_warnings:
        if cw and not any(cw in lm[2] for lm in lims):
            lims.append(("MEDIUM", "Quality check warning (critic)", cw, ""))

    lims.sort(key=lambda x: SEV_ORDER.get(x[0], 99))

    if lims:
        first = lims[0]
        story.append(_p("Critical Risk" if first[0] in ("CRITICAL","HIGH") else "Primary Risk", H3))
        story.append(callout(
            f"[{first[0]}]  {first[1]}",
            ([first[2], f"Mitigation: {first[3]}"] if first[3] else [first[2]]),
            title_bg=SEV_HEX.get(first[0], '#2C3E50'),
            body_bg=SEV_BG.get(first[0], '#F2F3F4'),
        ))
        if len(lims) > 1:
            story.append(Spacer(1, 0.25*cm))
            story.append(_p("Secondary Risks", H3))
            for sev, title, body_text, mitigation in lims[1:]:
                story.append(callout(
                    f"[{sev}]  {title}",
                    ([body_text, f"Mitigation: {mitigation}"] if mitigation else [body_text]),
                    title_bg=SEV_HEX.get(sev, '#5D6D7E'),
                    body_bg=SEV_BG.get(sev, '#F2F3F4'),
                ))
                story.append(Spacer(1, 0.1*cm))
    else:
        story.append(_p("No significant limitations identified.", BODY))

    # ── Footer ────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.5*cm),
        HRFlowable(width="100%", thickness=0.5, color=colors.grey),
        _p(f"Generated by Award B autonomous pipeline  ·  {timestamp}  ·  claude-sonnet-4-6",
           FOOT),
    ]

    n_story = len(story)
    _pdf_path = BASE / "report.pdf"
    _pdf_written = False
    try:
        with open(str(_pdf_path), "wb") as _pdf_fh:
            _doc = SimpleDocTemplate(
                _pdf_fh, pagesize=A4,
                leftMargin=2*cm, rightMargin=2*cm,
                topMargin=2*cm, bottomMargin=2*cm,
            )
            _doc.build(story)
        # File stream is closed by the 'with' block before we reach here
        sz = os.path.getsize(str(_pdf_path))
        if sz > 0:
            print(f"report.pdf written: {sz:,} bytes, 7 sections, {n_story} story elements")
            _pdf_written = True
        else:
            print("ERROR: report.pdf had 0 bytes after stream close — write not confirmed")
    except Exception as _build_err:
        print(f"ERROR: report.pdf build failed: {_build_err}")

# ── Marker file — written only after OS-confirmed file stream close ────────────
if missing_inputs:
    print(f"Missing inputs: {missing_inputs}")
if _pdf_written:
    ts_iso = datetime.now(timezone.utc).isoformat()
    with open(REPORTS / "report_writer_was_here.txt", "w", encoding="utf-8") as fh:
        fh.write(f"report_writer sub-agent executed at {ts_iso}\n")
    print("Marker: reports/report_writer_was_here.txt")
else:
    print("ERROR: marker not written — report file write unconfirmed by OS")

# ── KG: deferred status update — only on confirmed PDF write ──────────────────
try:
    from kg import kg_set_stage
    kg_set_stage("complete" if _pdf_written else "failed")
except Exception as _kg_e:
    print(f"[KG] non-fatal: {_kg_e}")
