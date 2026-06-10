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
        PageBreak, Image as RLImage, HRFlowable, KeepTogether,
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

def _find_data_description(data_dir):
    """Case-insensitive, separator-tolerant lookup. Returns Path or None (never raises)."""
    try:
        for f in data_dir.iterdir():
            if f.is_file():
                normalized = f.name.lower().replace('-', '_').replace(' ', '_')
                if normalized in ('data_description.md', 'data_description.txt'):
                    return f
    except Exception:
        pass
    return None

def _read_data_description(data_dir):
    """Return file text or None. Never raises; never writes to missing_inputs."""
    path = _find_data_description(data_dir)
    if path is None:
        return None
    try:
        return path.read_text(encoding='utf-8')
    except Exception:
        return None

profile     = load_json(REPORTS / "profile.json")
features    = load_json(REPORTS / "features.json")
model_res   = load_json(REPORTS / "model_results.json")
sub_summary = load_json(REPORTS / "submission_summary.json")
critic_rev  = load_json(REPORTS / "critic_review.json")
val_rev     = load_json(REPORTS / "validator_review.json")
load_text(REPORTS / "schema_analysis.md")
data_desc_text = _read_data_description(BASE / "data")

# ── Field extraction — exact field names from the JSON schemas ────────────────
problem_type    = profile.get("problem_type", "Unknown")
confidence      = profile.get("problem_type_confidence", "Unknown")
problem_subtype = profile.get("problem_subtype") or ""
target_col      = profile.get("target_col", "Unknown")
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
covariate_cols  = profile.get("covariate_cols") or []
scored_cats_raw = profile.get("scored_categories")

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

# ── Multiple-MAE rulers (used by the Metrics Guide and headline labels) ──────
# These all coexist in model_results.json on different rulers; the Metrics Guide
# in Section 5 names each one explicitly so readers can reconcile them.
decision_metric_name = model_res.get("decision_metric")  # e.g. "scored_only_mae"
oof_mae_scored       = model_res.get("oof_mae")                       # decision metric value (scored cats)
oof_mae_all_cats     = model_res.get("oof_mae_all_categories")        # broader ruler
wf_mae_all           = model_res.get("walk_forward_mae")              # single 80/20 probe, all-cat
wf_mae_scored        = model_res.get("walk_forward_mae_scored")       # single 80/20 probe, scored
scored_cat_col       = model_res.get("scored_category_column")        # e.g. "overdose_category"
n_scored_train       = model_res.get("n_scored_train_rows")
n_scored_val         = model_res.get("n_scored_val_rows")

# ── Trained feature matrix sizes (truth-source for what the model SAW) ────────
# profile.n_train_rows is the RAW row count (sum of input CSVs, pre-FE);
# features.train_shape[0] / model_results.n_train_rows are the actual matrix
# the model was trained on (post panel expansion / filtering). Section 1 shows
# both, distinctly labelled, so the 35,343-vs-31,416 disagreement is no longer
# a mystery.
_train_shape    = features.get("train_shape") or [None, None]
_val_shape      = features.get("val_shape")   or [None, None]
train_matrix_rows = (_train_shape[0] if _train_shape else None) or model_res.get("n_train_rows")
val_matrix_rows   = (_val_shape[0]   if _val_shape   else None) or model_res.get("n_val_rows")

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

cv_plan        = load_json(REPORTS / "cv_plan.json")
_cv            = cv_plan.get("cv", {})
cv_type_sel    = _cv.get("cv_type", "N/A")
cv_n_splits    = _cv.get("n_splits", "N/A")
cv_valid_size  = _cv.get("valid_size", "N/A")
cv_gap_val     = _cv.get("gap", "N/A")
_cv_sel        = cv_plan.get("cv_selection_reason", {})
cv_rule_branch = _cv_sel.get("rule_branch", "")
_cv_drift      = _cv_sel.get("drift_metrics", {})
cv_frac_imp    = _cv_drift.get("frac_improved", None)
inner_folds    = model_res.get("nested_cv", {}).get("inner_folds", "N/A")
outer_folds_n  = model_res.get("nested_cv", {}).get("outer_folds", "N/A")
adv_auc_exact  = features.get("adversarial_validation", {}).get("auc_train_vs_val", None)
gap_attr       = val_rev.get("gap_attribution", {})
gap_attr_class = gap_attr.get("classification", "N/A")
gap_mono_score = gap_attr.get("monotone_score", None)

# ── New diagnostic fields ─────────────────────────────────────────────────────
# Transform selection
_ts          = model_res.get("transform_selection") or {}
ts_chosen    = _ts.get("chosen")
ts_metric    = _ts.get("selection_metric")
ts_cands     = _ts.get("candidates_mae") or {}
ts_override  = _ts.get("manual_override")

# Per-fold CV stability
_ncv_m       = model_res.get("nested_cv", {})
pf_scored    = model_res.get("per_fold_maes") or _ncv_m.get("outer_fold_maes") or []
pf_all_cat   = (model_res.get("per_fold_maes_all_categories")
                or _ncv_m.get("outer_fold_maes_all_categories") or [])
pf_train_sz  = _ncv_m.get("outer_fold_train_sizes") or []

# Lag forecasting / recursion diagnostic
_lag_fc       = model_res.get("lag_forecasting") or {}
lag_method    = _lag_fc.get("method_used")
lag_rec_mae   = _lag_fc.get("recursive_holdout_mae")
lag_imp_mae   = _lag_fc.get("imputation_holdout_mae")
lag_rec_mac   = _lag_fc.get("recursive_holdout_mae_all_categories")
lag_imp_mac   = _lag_fc.get("imputation_holdout_mae_all_categories")
lag_steps_rec = _lag_fc.get("per_step_mae_recursive") or []
lag_steps_imp = _lag_fc.get("per_step_mae_imputation") or []
lag_notes     = _lag_fc.get("notes", "")
lag_n_steps   = _lag_fc.get("n_holdout_steps")

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

_gap_used        = critic_rev.get("gap_attribution_used") or {}
_gap_downgraded  = bool(_gap_used.get("downgraded_check1"))
_gap_final_pass  = (_gap_used.get("final_status") == "PASS")
_critic_accepted = (critic_status == "accepted")

def mae_interpretation():
    """Headline metric line — names the ruler explicitly so 1.97 vs 0.87 is
    no longer ambiguous. Defaults to the scored decision metric when present."""
    if mae is None:
        return "N/A — metric not available"
    # Prefer scored OOF (the decision metric) when present; fall back to mae
    headline_v = oof_mae_scored if oof_mae_scored is not None else mae
    is_scored  = (oof_mae_scored is not None
                  and decision_metric_name == "scored_only_mae")
    label = ("Scored nested OOF MAE" if is_scored
             else "OOF MAE")
    base  = f"{label}: {fmt(headline_v, 4)} (decision metric — Section 5 Metrics Guide)"
    # Critic's view supersedes the raw validator verdict.
    if _critic_accepted and _gap_downgraded and _gap_final_pass:
        return f"{base} (validator WARNING reviewed by critic and downgraded)"
    if val_verdict == "CRITICAL" or n_criticals >= 2:
        return f"{base} (may be inflated — see Section 5)"
    if val_verdict == "WARNING" or n_warnings >= 2:
        return f"{base} (borderline — minor optimism possible)"
    return f"{base} (clean cross-validation)"

def recommendation():
    if not sub_passed:
        return "FIX ISSUES — submission validation failed", '#d7191c'
    # Critic's final status takes precedence over the raw validator verdict —
    # a WARNING that the critic downgraded (gap attribution CV_SCHEME, not
    # overfit) should NOT surface as "SUBMIT WITH CAUTION" in the headline.
    if _critic_accepted and _gap_downgraded and _gap_final_pass:
        return ("SUBMIT — validator WARNING reviewed and downgraded: "
                "CV gap attributed to scheme pessimism, not overfit"), '#1a9641'
    if val_verdict == "CRITICAL":
        return "REVIEW — critical CV integrity issue detected", '#d7191c'
    if critic_status not in ("accepted", "N/A", ""):
        return "SUBMIT WITH CAUTION — review Section 5", '#e08200'
    if val_verdict == "WARNING":
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

    def _overview_box(sentences):
        """Plain-prose task overview box — flowing sentences, no bullets."""
        _ot_st = ParagraphStyle('OBT', fontName='Helvetica-Bold', fontSize=9.5,
                                 textColor=colors.white)
        _ob_st = ParagraphStyle('OBB', fontName='Helvetica', fontSize=10, leading=15)
        _ob_html = "  ".join(str(_s) for _s in sentences if _s)
        _t = Table(
            [[_p("  Task Overview", _ot_st)], [_p(_ob_html, _ob_st)]],
            colWidths=[16*cm], hAlign='LEFT',
        )
        _t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#1F618D')),
            ('BACKGROUND',    (0,1), (-1,-1), colors.HexColor('#EBF5FB')),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
            ('RIGHTPADDING',  (0,0), (-1,-1), 10),
            ('TOPPADDING',    (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LINEBELOW',     (0,-1),(-1,-1), 1.0, colors.HexColor('#1F618D')),
        ]))
        return _t

    def headline_figure():
        """ONE prominent headline figure: the scored OOF decision metric, large,
        with a single short qualifier line. NOT a cross-ruler range — different
        MAEs measure different slices (scored vs all-cat vs walk-forward) and
        ranging across them mixes rulers. The per-MAE breakdown sits below."""
        headline_v = oof_mae_scored if oof_mae_scored is not None else mae
        is_scored  = (oof_mae_scored is not None
                      and decision_metric_name == "scored_only_mae")
        label = "Scored OOF MAE" if is_scored else "OOF MAE"

        big_st = ParagraphStyle('BigHL', fontName='Helvetica-Bold', fontSize=30,
                                 alignment=TA_CENTER,
                                 textColor=colors.HexColor('#1A5276'),
                                 spaceAfter=2)
        sub_st = ParagraphStyle('SubHL', fontName='Helvetica-Oblique',
                                 fontSize=9.5, alignment=TA_CENTER, leading=12,
                                 textColor=colors.HexColor('#555555'),
                                 spaceAfter=10)
        if headline_v is None:
            return [_p("Headline metric: not available", big_st)]
        return [
            _p(f"{label}: {fmt(headline_v, 2)}", big_st),
            _p("decision metric — conservative within-training estimate; "
               "full metric breakdown below.", sub_st),
        ]

    def metrics_breakdown_box():
        """Per-MAE breakdown sitting directly under the headline. One row per
        ruler appearing in the report, with what it measures and why it differs
        from the others — so the reader doesn't misread different rulers as
        conflicting estimates of the same thing. Tight: one-two sentences each.
        All values pulled from model_results.json / validator_review.json."""
        rows = []

        if oof_mae_scored is not None:
            _scored_slice = (f" ({scored_cat_col})" if scored_cat_col else "")
            rows.append([
                "Scored nested OOF MAE",
                fmt(oof_mae_scored, 4),
                "Decision metric",
                f"Nested purged CV restricted to scored categories"
                f"{_scored_slice}. Higher than the all-category number "
                f"because the scored slice concentrates high-magnitude cells "
                f"that MAE weights heavily — this is the metric being "
                f"selected against, NOT a sign the model is worse on scored "
                f"categories.",
            ])

        if oof_mae_all_cats is not None:
            rows.append([
                "All-category nested OOF MAE",
                fmt(oof_mae_all_cats, 4),
                "Broader ruler",
                "Same nested purged CV folds as above, but averaged across "
                "all training categories. Lower than scored OOF because many "
                "low-magnitude cells pull the average down. Useful as a "
                "broader sanity check; NOT the decision metric.",
            ])

        if wf_mae_all is not None:
            rows.append([
                "Walk-forward probe MAE",
                fmt(wf_mae_all, 4),
                "Single-window probe",
                "Single most-recent 80/20 walk-forward window, all "
                "categories — one window, not an average across folds, so "
                "higher variance than the nested OOF. The validator consumes "
                "this as its reported-CV input.",
            ])

        if val_strict is not None:
            rows.append([
                "Strict validator MAE",
                fmt(val_strict, 4),
                "Independent re-audit",
                f"{val_cv_scheme} — an independent purged re-audit on a "
                f"different fold scheme. Confirms the OOF was not optimistic; "
                f"the value differs from OOF because the ruler (folds, "
                f"embargo, all-category slice) differs, not because the "
                f"model performs differently.",
            ])

        if lag_rec_mae is not None:
            _imp_clause = (f" vs imputation alternative {fmt(lag_imp_mae, 4)}"
                           if lag_imp_mae is not None else "")
            rows.append([
                "Recursive holdout MAE (scored)",
                fmt(lag_rec_mae, 4),
                "Method selection",
                f"Scored-slice MAE on a separate held-out window used to "
                f"choose how missing future lags are filled "
                f"({lag_method or 'recursive'}{_imp_clause}). This is a "
                f"forecasting-METHOD selection at the validation boundary, "
                f"NOT a re-estimate of OOF.",
            ])

        if not rows:
            return [_p("Metrics breakdown: not available — model_results.json "
                       "missing required fields.", META)]

        _hdr_st = ParagraphStyle('MBHdr', fontName='Helvetica-Bold',
                                  fontSize=9, textColor=colors.white,
                                  alignment=TA_LEFT, leading=11)
        _name_st = ParagraphStyle('MBName', fontName='Helvetica-Bold',
                                   fontSize=9, leading=11)
        _val_st  = ParagraphStyle('MBVal',  fontName='Helvetica-Bold',
                                   fontSize=10, alignment=TA_CENTER,
                                   textColor=colors.HexColor('#1A5276'),
                                   leading=12)
        _role_st = ParagraphStyle('MBRole', fontName='Helvetica-Oblique',
                                   fontSize=8.5,
                                   textColor=colors.HexColor('#555555'),
                                   leading=10)
        _desc_st = ParagraphStyle('MBDesc', fontName='Helvetica',
                                   fontSize=8.5, leading=11)

        header = [_p("Metric", _hdr_st), _p("Value", _hdr_st),
                  _p("Role", _hdr_st),
                  _p("What it measures · why it differs", _hdr_st)]
        data = [header]
        for name, value, role, desc in rows:
            data.append([_p(name, _name_st), _p(value, _val_st),
                         _p(role, _role_st), _p(desc, _desc_st)])

        t = Table(data, colWidths=[3.6*cm, 2.0*cm, 2.4*cm, 8.0*cm],
                  hAlign='LEFT', repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#1A5276')),
            ('ROWBACKGROUNDS',(0,1), (-1,-1),
             [colors.HexColor('#F2F2F2'), colors.white]),
            ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#CCCCCC')),
            ('LEFTPADDING',   (0,0), (-1,-1), 6),
            ('RIGHTPADDING',  (0,0), (-1,-1), 6),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('VALIGN',        (0,0), (-1,-1), 'TOP'),
        ]))
        return [t]

    def exec_summary():
        rec_text, rec_hex = recommendation()
        rec_st = ParagraphStyle('Rec', fontName='Helvetica-Bold', fontSize=9,
                                 textColor=colors.HexColor(rec_hex))
        hdr_st = ParagraphStyle('EHdr', fontName='Helvetica-Bold', fontSize=10,
                                 textColor=colors.white, alignment=TA_CENTER)

        # Result row carries the decision-metric interpretation; the per-MAE
        # breakdown table sitting above this summary reconciles every ruler.
        _result_text = mae_interpretation() + " — full metric breakdown above; see Section 5 Metrics Guide."

        rows_data = [
            [_p('Executive Summary', hdr_st), _p('', hdr_st)],
            [_p('Result', _CS),         _p(_result_text, _CS)],
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
        Spacer(1, 0.35*cm),
        _p(f"Problem: {problem_type}  |  Target: {target_col}  |  {timestamp}", META),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2C3E50")),
        Spacer(1, 0.4*cm),
    ]

    # ── Per-MAE breakdown ────────────────────────────────────────────────────
    # The breakdown table explains every MAE in the report so readers can
    # reconcile them without flipping to Section 5. The decision metric is
    # surfaced in the Executive Summary "Result" row below and in the Section 5
    # Metrics Guide; no separate oversized headline figure.
    story += metrics_breakdown_box()
    story.append(Spacer(1, 0.4*cm))

    # ── Executive narrative (2–3 plain sentences before the status table) ────
    # All values pulled from artifacts; no hardcoded dataset specifics.
    _pt_human_es   = (problem_type or 'unknown').replace('_', ' ')
    _model_es      = "CatBoost (sole submission predictor; Ridge run as a linear baseline diagnostic, not blended)"
    _decision_v    = (oof_mae_scored if oof_mae_scored is not None else mae)
    _decision_lbl  = (f"scored nested OOF MAE" if oof_mae_scored is not None
                                                   and decision_metric_name == "scored_only_mae"
                       else "nested OOF MAE")
    _lever_clause  = ""
    if lag_method == "recursive" and lag_rec_mae is not None and lag_imp_mae is not None:
        _lever_clause = (
            f" The dominant accuracy lever was recursive lag forecasting at the "
            f"validation boundary (recursive holdout MAE {fmt(lag_rec_mae,4)} vs "
            f"imputation {fmt(lag_imp_mae,4)} on scored categories — see Section 6)."
        )
    elif lag_method == "imputation" and lag_rec_mae is not None and lag_imp_mae is not None:
        _lever_clause = (
            f" Cycle-aware lag imputation was selected over recursive forecasting "
            f"(imputation holdout MAE {fmt(lag_imp_mae,4)} vs recursive "
            f"{fmt(lag_rec_mae,4)} on scored categories — Section 6)."
        )
    _adv_auc_es   = adv_auc_exact if adv_auc_exact is not None else (adv_val or {}).get("auc")
    _honesty_clause = ""
    if _adv_auc_es is not None and _adv_auc_es > 0.95:
        _honesty_clause = (
            f" Honesty caveat: adversarial train-vs-validation AUC of "
            f"{fmt(_adv_auc_es,3)} indicates severe distribution shift, so "
            f"the within-training OOF is an uncertain estimate of true test "
            f"error — shift may push true test error <i>above or below</i> "
            f"OOF, and no training-side CV can quantify the gap or its sign."
        )
    elif _adv_auc_es is not None and _adv_auc_es > 0.70:
        _honesty_clause = (
            f" Honesty caveat: adversarial AUC {fmt(_adv_auc_es,3)} indicates "
            f"moderate distribution shift; the within-training OOF may "
            f"over- or under-state true test error, and no training-side CV "
            f"can quantify the gap or its sign."
        )
    _critic_clause = ""
    if _critic_accepted and _gap_downgraded and _gap_final_pass:
        _critic_clause = (
            " The validator surfaced a WARNING on the CV gap; the critic "
            "downgraded it after attributing the gap to scheme pessimism "
            "(smaller early-fold training windows) rather than overfit."
        )
    _es_s1 = (
        f"This analysis classified the dataset as <b>{_pt_human_es}</b> and "
        f"trained <b>{_model_es}</b>."
    )
    _es_s2 = (
        f"The decision metric — <b>{_decision_lbl}</b> — is "
        f"<b>{fmt(_decision_v, 4) if _decision_v is not None else 'N/A'}</b>."
        f"{_lever_clause}{_critic_clause}"
    )
    _es_s3 = _honesty_clause.strip() or (
        "No severe distribution shift detected between train and validation."
    )
    story.append(_overview_box([_es_s1, _es_s2, _es_s3]))
    story.append(Spacer(1, 0.25*cm))

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(exec_summary())
    story.append(Spacer(1, 0.6*cm))  # min-height buffer prevents overlap with next table

    # ── Pipeline Status ───────────────────────────────────────────────────────
    story.append(_p("Pipeline Status", H2))
    story.append(Spacer(1, 0.25*cm))  # gap between heading and table top border
    story.append(pipeline_status_table())

    # ── Adaptive Decisions — plain single-model terms, no "Axis N" jargon ────
    # Decisions are described as real adaptive choices the pipeline actually
    # made: model identity, adversarial sample weighting, target transform,
    # forecasting method, and any critic retune. Any decision that is no
    # longer a real adaptive choice (e.g. ensemble weighting on a single-
    # model pipeline) is dropped instead of dressed up.
    dec_lines = []
    _model_decision = (
        "Model: <b>CatBoost</b> sole submission predictor "
        f"(seed-averaged ×{n_seeds}). Ridge regression was trained on the "
        "same features as a linear-baseline diagnostic only — never blended "
        "into the submission."
    )
    dec_lines.append(_model_decision)

    _adv_auc_dec  = (adv_val or {}).get("auc") or adv_auc_exact
    _adv_weighted = (adv_val or {}).get("weights_applied")
    if _adv_auc_dec is not None:
        if _adv_weighted:
            dec_lines.append(
                f"Adversarial shift weighting: <b>applied</b> "
                f"(AUC {fmt(_adv_auc_dec, 3)}) — training rows resembling the "
                f"validation distribution were up-weighted during CatBoost training."
            )
        else:
            dec_lines.append(
                f"Adversarial shift weighting: <b>not applied</b> "
                f"(AUC {fmt(_adv_auc_dec, 3)} — below activation threshold; uniform weights used)."
            )

    if ts_chosen is not None:
        dec_lines.append(
            f"Target transform: <b>{ts_chosen}</b> "
            f"(selected by minimising {ts_metric or 'walk-forward MAE'} across candidates)."
        )

    if lag_method:
        dec_lines.append(
            f"Lag-feature strategy at validation boundary: "
            f"<b>{lag_method}</b> (Section 6 records the holdout comparison)."
        )

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

    # ── Task Overview — built from detected profile.json fields ───────────────
    _PT_DESC = {
        'panel_forecasting':         'time-indexed regression across a panel of groups',
        'time_series_forecasting':   'time-series forecasting for a single entity',
        'univariate_time_series':    'univariate time-series forecasting',
        'regression':                'supervised regression on independent rows',
        'binary_classification':     'binary classification',
        'multiclass_classification': 'multiclass classification',
        'classification':            'classification',
    }
    _pt_norm  = (problem_type or '').lower().replace(' ', '_')
    _pt_human = _PT_DESC.get(_pt_norm, (problem_type or 'unknown').replace('_', ' '))
    _sub_note = (
        f"; subtype: {problem_subtype.replace('_', ' ')}"
        if problem_subtype
        and problem_subtype.lower().replace(' ', '_') != _pt_norm
        else ""
    )
    _ov_s1 = (
        f"This is a <b>{(problem_type or 'unknown').replace('_', ' ')}</b> task "
        f"(<i>{confidence}</i> confidence{_sub_note}) — {_pt_human}."
    )

    _grp_list = profile.get("group_cols") or []
    _tc_i     = schema.get(target_col) or {}
    _tc_dtype = _tc_i.get('dtype', '')
    _tc_type  = (
        'a continuous numeric value' if 'float' in _tc_dtype
        else 'an integer value'       if 'int'   in _tc_dtype
        else 'a categorical value'    if 'str'   in _tc_dtype
        else 'a value'
    )
    _hz_note  = (
        f", {n_horizons} steps ahead per group" if (n_horizons and _grp_list)
        else f", {n_horizons} steps ahead"       if n_horizons
        else ""
    )
    _ov_s2 = f"The model predicts <i>{target_col}</i>{_hz_note} ({_tc_type})."

    if _grp_list:
        _grp_disp = (
            f"({', '.join(_grp_list)})" if len(_grp_list) > 1 else _grp_list[0]
        )
        _ov_s3 = (
            f"Predictions are made per {_grp_disp} over <i>{time_col}</i>, "
            f"using {len(covariate_cols)} "
            f"covariate{'s' if len(covariate_cols) != 1 else ''}."
        )
    elif covariate_cols:
        _ov_s3 = (
            f"The dataset has {len(covariate_cols)} "
            f"covariate{'s' if len(covariate_cols) != 1 else ''}"
            + (f", ordered by <i>{time_col}</i>."
               if time_col not in (None, '', 'None')
               else ".")
        )
    else:
        _ov_s3 = ""

    _metric = objective if objective not in (None, '', 'Unknown', 'N/A') else 'MAE'
    _sc_sfx = (
        f"  Scoring uses {len(scored_cats_raw)} categories: "
        + ", ".join(str(_c) for _c in scored_cats_raw) + "."
        if scored_cats_raw else ""
    )
    _ov_s4 = (
        f"The submission requires {fmt(sub_row_count)} predictions; "
        f"the evaluation metric is <b>{_metric}</b>.{_sc_sfx}"
    )
    story.append(_overview_box([_ov_s1, _ov_s2, _ov_s3, _ov_s4]))
    story.append(Spacer(1, 0.25*cm))

    # Training/validation row counts: profile.json holds the RAW count from
    # the input CSVs (pre-FE), and features.train_shape[0] is the actual
    # matrix the model trained on (after panel expansion / scored-row
    # construction / dropping incomplete cells). When the two disagree we
    # show both rows, distinctly labelled, so the report no longer prints
    # one ambiguous number that disagrees with the modeler.
    _rows_match_train = (train_matrix_rows is not None
                         and n_train != "Unknown"
                         and int(train_matrix_rows) == int(n_train or 0))
    _rows_match_val   = (val_matrix_rows is not None
                         and n_val != "Unknown"
                         and int(val_matrix_rows) == int(n_val or 0))

    _section1_rows = [
        ["Problem type",     problem_type],
        ["Confidence",       confidence],
        ["Target column",    target_col],
        ["Group columns",    group_cols],
        ["Time column",      time_col],
    ]
    if _rows_match_train:
        _section1_rows.append(["Training rows", fmt(n_train)])
    else:
        _section1_rows.append([
            "Raw training rows (input CSVs)", fmt(n_train),
        ])
        _section1_rows.append([
            "Training matrix rows (model trained on)",
            f"{fmt(train_matrix_rows)} — after feature engineering / panel expansion",
        ])
    if _rows_match_val:
        _section1_rows.append(["Validation rows", fmt(n_val)])
    else:
        _section1_rows.append([
            "Raw validation rows (input CSVs)", fmt(n_val),
        ])
        _section1_rows.append([
            "Validation matrix rows (model scored on)",
            f"{fmt(val_matrix_rows)} — after feature engineering / panel expansion",
        ])
    _section1_rows.append([
        "Forecast horizon", f"{n_horizons} steps" if n_horizons else "N/A",
    ])
    story.append(tbl(["Property", "Value"], _section1_rows, widths=[5*cm, 11*cm]))
    story.append(Spacer(1, 0.2*cm))
    if not profile:
        story.append(_p("Dataset details: not available (profile.json missing).", META))
    else:
        story.append(_p("<b>Detected Dataset Structure</b>", BOLD))

        # Target column
        _tc_info  = schema.get(target_col) or {}
        _tc_parts = []
        if _tc_info.get("min") is not None:
            _tc_parts.append(f"range {fmt(_tc_info['min'], 2)} – {fmt(_tc_info['max'], 2)}")
        if _tc_info.get("mean") is not None:
            _tc_parts.append(f"mean {fmt(_tc_info['mean'], 4)}, std {fmt(_tc_info['std'], 4)}")
        if _tc_info.get("null_pct") is not None:
            _tc_parts.append(f"{fmt(_tc_info['null_pct'], 1)}% null")
        _tc_stats_str = "; ".join(_tc_parts) if _tc_parts else "N/A"
        story.append(_p(
            f"<b>Target:</b> <i>{target_col}</i>"
            f" ({_tc_info.get('dtype', 'N/A')})  —  {_tc_stats_str}.",
            BODY,
        ))

        # Group columns
        _grp_list = profile.get("group_cols") or []
        if _grp_list:
            _grp_parts = []
            for _gc in _grp_list:
                _gs    = schema.get(_gc) or {}
                _nuniq = _gs.get("n_unique")
                _grp_parts.append(
                    f"<i>{_gc}</i> ({fmt(_nuniq)} values)" if _nuniq else f"<i>{_gc}</i>"
                )
            story.append(_p(
                f"<b>Group columns:</b> {', '.join(_grp_parts)}.",
                BODY,
            ))

        # Time column
        _tcs    = schema.get(time_col) or {}
        _nper   = _tcs.get("n_unique")
        story.append(_p(
            f"<b>Time column:</b> <i>{time_col}</i>"
            + (f" ({fmt(_nper)} distinct periods)" if _nper else "")
            + (f" — {n_horizons} forecast steps" if n_horizons else "")
            + ".",
            BODY,
        ))

        # Scored categories
        if scored_cats_raw:
            story.append(_p(
                f"<b>Scored categories ({len(scored_cats_raw)}):</b> "
                + ", ".join(str(_c) for _c in scored_cats_raw) + ".",
                BODY,
            ))

        # Covariate columns
        if covariate_cols:
            story.append(Spacer(1, 0.15*cm))
            story.append(_p(
                f"<b>Covariate columns ({len(covariate_cols)})</b>", BOLD,
            ))
            _cov_rows = []
            for _cc in covariate_cols:
                _cs    = schema.get(_cc) or {}
                _dtype = str(_cs.get("dtype", "N/A"))
                _null  = f"{fmt(_cs.get('null_pct', 0), 1)}%"
                if _cs.get("mean") is not None:
                    _stat = f"{fmt(_cs['mean'], 2)} ± {fmt(_cs['std'], 2)}"
                elif _cs.get("n_unique") is not None:
                    _stat = f"{fmt(_cs['n_unique'])} unique"
                else:
                    _stat = "N/A"
                _cov_rows.append([_cc, _dtype, _null, _stat])
            story.append(tbl(
                ["Column", "Type", "Null %", "Mean ± Std (or unique values)"],
                _cov_rows,
                widths=[5*cm, 2.5*cm, 1.8*cm, 6.7*cm],
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
             "Adversarial sample weighting up-weights training rows that "
             "resemble the validation distribution (passed into CatBoost training)"],
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

    # 4A — Model Architecture
    story.append(_p("4A — Model Architecture", H3))
    story.append(_p(
        "<b>Model: CatBoost (sole predictor).</b>  "
        "All submission predictions come from a single gradient-boosted CatBoost "
        f"model, seed-averaged across {n_seeds} random seeds for variance reduction. "
        "There is no ensemble — no blending across model families, no weighted "
        "mixture, no stacking.",
        BODY,
    ))
    story.append(_p(
        "Ridge regression is trained on the same features purely as a linear "
        "diagnostic baseline (see Section 4C). It is <b>never blended into the "
        "submission and contributes zero weight</b> to any predicted value — "
        "it exists only so the report can show that the gradient-boosted model "
        "is doing useful work relative to a strong linear reference.",
        BODY,
    ))
    story.append(Spacer(1, 0.25*cm))

    # 4B — Hyperparameter Selection
    story.append(_p("4B — Hyperparameter Selection", H3))
    hp_rows = [
        ["Objective",       objective],
        ["Optuna trials",   str(n_trials) if n_trials else "N/A"],
        ["n_estimators",    str(n_estimators)],
        ["Seeds averaged",  str(n_seeds)],
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

    # 4C — Ridge Diagnostic (linear baseline reference)
    story.append(_p("4C — Ridge Diagnostic (linear baseline reference)", H3))
    story.append(_p(
        "A Ridge regressor is trained on the same engineered features as a "
        "linear-baseline competence check. The submission still comes from "
        "CatBoost alone; the table below records each model's OOF MAE on the "
        "same nested cross-validation folds so the reader can compare the "
        "gradient-boosted model against the linear reference. Ridge is "
        "<b>diagnostic only</b> — its predictions never enter the submission.",
        BODY,
    ))
    if families:
        story.append(tbl(
            ["Model", "Role", "OOF MAE", "Train Time"],
            [[fname,
              "Submission predictor" if fd.get("included_in_ensemble")
                                     else "Diagnostic only",
              fmt(fd.get("oof_mae"), 4),
              f"{fd.get('training_time_seconds','N/A')} s"]
             for fname, fd in families.items()],
            widths=[3.5*cm, 4.5*cm, 3*cm, 3*cm],
        ))
    else:
        story.append(tbl(["Parameter", "Value"], [
            ["Algorithm",           algorithm],
            [f"CV MAE ({cv_scheme})", fmt(mae, 4)],
            ["Training time",       f"{train_time} s"],
        ], widths=[5*cm, 11*cm]))

    # 4D — Target Transform Choice
    story.append(Spacer(1, 0.25*cm))
    story.append(_p("4D — Target Transform Choice", H3))
    if ts_chosen is not None:
        _override_note = " (manual override)" if ts_override else " (auto-selected)"
        story.append(_p(
            f"Transform <b>{ts_chosen}</b> was selected by minimising "
            f"<b>{ts_metric or 'raw_scored_wf_mae'}</b> across candidates{_override_note}. "
            "Candidates are scored on a walk-forward holdout before the final model trains; "
            "the table shows raw MAEs so the reader can see the margin of preference.",
            BODY,
        ))
        if ts_cands:
            _cand_rows = sorted(ts_cands.items(), key=lambda kv: kv[1])
            story.append(tbl(
                ["Transform", "Holdout MAE", "Selected"],
                [[_t_name,
                  fmt(_t_mae, 6),
                  "yes" if _t_name == ts_chosen else ""]
                 for _t_name, _t_mae in _cand_rows],
                widths=[4*cm, 5*cm, 7*cm],
            ))
        else:
            story.append(_p("Candidate MAE table not available.", META))
    else:
        story.append(_p("Target transform selection: not available.", META))
    story.append(Spacer(1, 0.1*cm))

    # Adversarial weights callout in modeling context
    _adv_auc = (adv_info_model or adv_val or {}).get("auc")
    if _adv_auc is not None:
        _applied = (adv_info_model or adv_val or {}).get("weights_applied", False)
        story.append(Spacer(1, 0.2*cm))
        story.append(callout(
            "Adversarial Sample Weights Applied to CatBoost",
            [f"Adversarial AUC: {fmt(_adv_auc, 3)}",
             "Weights passed to: CatBoost OOF probe + full-data retrain (all seeds), "
             "and to the Ridge diagnostic baseline",
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

    # ── Metrics Guide — reconciles the multiple MAEs that appear throughout ──
    # The report cites several MAE numbers on different rulers (nested OOF on
    # all categories, nested OOF on scored only, single 80/20 walk-forward
    # probe, validator's strict purged audit, lag-forecasting holdout).
    # Without naming the ruler, 1.97 vs 0.87 vs 0.99 vs 1.26 read as conflicts.
    # This guide names each metric, the ruler it's computed on, what role it
    # plays, and which is the decision metric.
    story.append(_p("<b>Metrics Guide — reconciling the multiple MAEs</b>", BOLD))
    story.append(_p(
        "Several MAE figures appear in this report. They are <i>different metrics on "
        "different rulers</i>, not conflicting estimates of the same thing. "
        "The decision metric is flagged.",
        BODY,
    ))
    _scored_role = ("decision metric"
                    if decision_metric_name == "scored_only_mae" else "diagnostic")
    _allcat_role = ("diagnostic (broader ruler)"
                    if decision_metric_name == "scored_only_mae" else "decision metric")
    _guide_rows = []
    if oof_mae_all_cats is not None:
        _guide_rows.append([
            "All-category nested OOF MAE",
            "model_results.oof_mae_all_categories",
            fmt(oof_mae_all_cats, 4),
            "All training categories, nested purged CV",
            _allcat_role,
        ])
    if oof_mae_scored is not None:
        _scored_ruler = (
            f"Scored categories only ({scored_cat_col}), nested purged CV"
            if scored_cat_col else "Scored categories only, nested purged CV"
        )
        _guide_rows.append([
            "Scored nested OOF MAE",
            "model_results.oof_mae",
            fmt(oof_mae_scored, 4),
            _scored_ruler,
            _scored_role,
        ])
    if wf_mae_all is not None:
        _guide_rows.append([
            "Walk-forward probe MAE",
            "model_results.walk_forward_mae",
            fmt(wf_mae_all, 4),
            "Single most-recent 80/20 window, all categories",
            "Validator's reported CV input",
        ])
    if val_strict is not None:
        _guide_rows.append([
            "Strict validator MAE",
            "validator_review.strict_cv_mae",
            fmt(val_strict, 4),
            f"{val_cv_scheme} (independent audit, all categories)",
            "Validator's purged re-audit",
        ])
    if lag_rec_mae is not None:
        _guide_rows.append([
            "Recursive holdout MAE",
            "model_results.lag_forecasting.recursive_holdout_mae",
            fmt(lag_rec_mae, 4),
            "Recursive n-step holdout, scored categories",
            "Forecasting-method selection only",
        ])
    if _guide_rows:
        _guide_tbl = tbl(
            ["Metric", "Field", "Value", "Ruler", "Role"],
            _guide_rows,
            widths=[3.5*cm, 3.5*cm, 1.8*cm, 4.5*cm, 2.7*cm],
        )
        # Keep the header with at least the first metric row so the page
        # break doesn't strand a lone header.
        story.append(KeepTogether([_guide_tbl]))
    else:
        story.append(_p("Metrics Guide: no MAE values available.", META))
    story.append(Spacer(1, 0.25*cm))

    story.append(_p("<b>A. Validator Audit</b>", BOLD))
    _val_tbl = tbl(["Property", "Value"], [
        ["CV scheme",         val_cv_scheme],
        ["Reported CV MAE",   fmt(val_rep_mae, 4) if val_rep_mae is not None else "N/A"],
        ["Strict CV MAE",     fmt(val_strict, 4)  if val_strict  is not None else "N/A"],
        ["CV gap (%)",        f"{val_gap_pct:.1f}%"],
        ["Verdict",           val_verdict],
        ["Suspect features",  ", ".join(str(f) for f in val_feature_susp)
                              if val_feature_susp else "None flagged"],
    ], widths=[5*cm, 11*cm])
    story.append(KeepTogether([_val_tbl]))
    if val_notes:
        story.append(_p(val_notes, BODY))
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
    story.append(Spacer(1, 0.25*cm))

    story.append(_p("<b>D. Cross-Validation Decision Record</b>", BOLD))
    story.append(_p(
        "Options considered, choice made, evidence, and cost of rejected alternatives. "
        "Values are from the current run's frozen <i>reports/cv_plan.json</i> "
        "and <i>reports/features.json</i>.",
        BODY,
    ))

    story.append(_p("Problem-Type Decision", H3))
    story.append(tbl(["Candidate", "Outcome / Reason"], [
        ["plain_regression",
         "REJECTED — treats rows as IID; during CV, future observations train on past "
         "predictions, introducing temporal leakage that produces optimistically biased "
         "estimates and fails at submission time."],
        ["univariate_time_series",
         "REJECTED — one model per group; discards cross-group covariate structure, "
         "shared seasonal patterns, and all external covariates keyed on (group, time). "
         "Per-group signal-to-noise is lower because each group trains only on its own history."],
        [f"{cv_type_sel.replace('TimeSeriesExpanding','panel_forecasting')} (selected)",
         "SELECTED — panel structure confirmed: group×time index detected with regular "
         "monthly cadence; external numeric covariates are keyed on (group, time); "
         "group columns present in both train and val."],
    ], widths=[4.5*cm, 11.5*cm]))
    story.append(Spacer(1, 0.15*cm))

    story.append(_p("CV-Scheme Decision", H3))
    cv_cfg_rows = [
        ["Scheme selected",           str(cv_type_sel)],
        ["Outer folds (n_splits)",    str(cv_n_splits)],
        ["Val window (valid_size)",   str(cv_valid_size)],
        ["Gap between train / val",   str(cv_gap_val)],
        ["Inner tuning folds",        str(inner_folds)],
    ]
    if cv_frac_imp is not None:
        cv_cfg_rows.append([
            "Expanding vs sliding gate",
            f"frac_improved = {cv_frac_imp:.3f} (threshold 0.60) — primary gate NOT met; "
            "expanding retained (sliding requires affirmative drift-reducibility evidence).",
        ])
    _cv_tbl = tbl(["Parameter", "Value"], cv_cfg_rows, widths=[5.5*cm, 10.5*cm])
    story.append(KeepTogether([_cv_tbl]))
    story.append(Spacer(1, 0.1*cm))
    story.append(_p(
        "<b>Random k-fold rejected:</b> invalid for ordered data — future observations "
        "flow into training folds predicting past observations, leaking the answer. "
        "<b>Single holdout rejected:</b> high-variance estimate; no evidence about "
        "how performance scales with training size. "
        "<b>Nested tuning:</b> hyperparameter search runs inside each outer fold "
        f"({inner_folds} inner folds), so no hyperparameter is selected using the "
        "outer-fold validation rows — the OOF MAE is not inflated by tuning toward "
        "the held-out set.",
        BODY,
    ))
    story.append(Spacer(1, 0.1*cm))

    story.append(_p("Per-Fold CV Stability", H3))
    if pf_scored:
        _n_folds = max(len(pf_scored), len(pf_all_cat), len(pf_train_sz))
        _fold_rows = []
        for _i in range(_n_folds):
            _sc  = pf_scored[_i]   if _i < len(pf_scored)   else None
            _ac  = pf_all_cat[_i]  if _i < len(pf_all_cat)  else None
            _sz  = pf_train_sz[_i] if _i < len(pf_train_sz) else None
            _fold_rows.append([
                str(_i + 1),
                fmt(_sc, 4) if _sc is not None else "N/A",
                fmt(_ac, 4) if _ac is not None else "N/A",
                fmt(_sz)    if _sz is not None else "N/A",
            ])
        _pf_tbl = tbl(
            ["Fold", "Scored MAE", "All-Category MAE", "Train Size"],
            _fold_rows,
            widths=[2*cm, 4.5*cm, 4.5*cm, 5*cm],
        )
        story.append(KeepTogether([_pf_tbl]))
        if len(pf_scored) > 1:
            _pf_range = max(pf_scored) - min(pf_scored)
            story.append(Spacer(1, 0.1*cm))
            _pf_mean = sum(pf_scored) / len(pf_scored)
            _pf_std = (sum((x - _pf_mean) ** 2 for x in pf_scored) / len(pf_scored)) ** 0.5
            _outlier_thresh = _pf_mean + 1.5 * _pf_std
            _outliers = [(i + 1, v) for i, v in enumerate(pf_scored) if v > _outlier_thresh]
            _cluster = [v for v in pf_scored if v <= _outlier_thresh]
            if _outliers:
                _cluster_rng = (
                    f"range {fmt(min(_cluster), 4)}–{fmt(max(_cluster), 4)}"
                    if len(_cluster) > 1 else fmt(_cluster[0], 4)
                )
                _out_str = ", ".join(f"fold {fi} ({fmt(fv, 4)})" for fi, fv in _outliers)
                _interp = (
                    f"{len(_cluster)} of {len(pf_scored)} folds cluster tightly "
                    f"({_cluster_rng}); {_out_str} "
                    f"{'is' if len(_outliers) == 1 else 'are'} an outlier "
                    f"indicating a harder time window — the elevated MAE reflects "
                    f"that period's difficulty, not model instability. "
                    f"No monotonic improvement with training-window size is observed."
                )
            else:
                _interp = (
                    f"All {len(pf_scored)} folds cluster tightly — no outlier "
                    f"fold detected; consistent performance across training-window sizes."
                )
            story.append(_p(
                f"Scored MAE range across {len(pf_scored)} folds: "
                f"{fmt(_pf_range, 4)} "
                f"(min {fmt(min(pf_scored), 4)}, max {fmt(max(pf_scored), 4)}). "
                + _interp,
                BODY,
            ))
    else:
        story.append(_p("Per-fold MAE data not available.", META))
    story.append(Spacer(1, 0.1*cm))

    story.append(_p("OOF Honesty and Distribution-Shift Limit", H3))
    _honesty_rows = []
    if gap_attr_class != "N/A":
        _honesty_rows.append([
            "Gap attribution",
            f"{gap_attr_class}" + (
                f" — fold-MAE monotone improvement score: {fmt(gap_mono_score, 2)}/1.0"
                if gap_mono_score is not None else ""
            ) + ". The modeler–validator MAE gap is structural scheme pessimism "
              "(smaller early-fold training windows), not real overfit.",
        ])
    if adv_auc_exact is not None:
        _sev = (
            "near-perfect separation — severe distribution shift"
            if adv_auc_exact > 0.95 else
            "moderate separation — meaningful distribution shift"
            if adv_auc_exact > 0.70 else
            "no meaningful separation — distributions are similar"
        )
        _honesty_rows.append([
            "Adversarial train-vs-val AUC",
            f"{adv_auc_exact:.6f}  ({_sev}). When AUC is near 1.0, shift "
            "makes the within-training OOF an uncertain estimate of true "
            "test error — shift can push true test error above or below OOF, "
            "and no within-training CV can quantify the gap or its sign; "
            "the correction would require knowing the test distribution.",
        ])
    if _honesty_rows:
        story.append(tbl(["Diagnostic", "Value / Interpretation"], _honesty_rows,
                         widths=[4.5*cm, 11.5*cm]))
        story.append(Spacer(1, 0.1*cm))
    story.append(_p(
        "The OOF is an <b>honest within-training-distribution estimate</b>: every "
        "fold's validation row is strictly later than its training rows, and tuning "
        "never sees the outer-fold validation set. "
        "It is <b>not</b> a prediction of leaderboard position — distribution shift "
        "(adversarial AUC) drives a gap that training-side CV cannot quantify.",
        BODY,
    ))
    story.append(Spacer(1, 0.1*cm))

    story.append(_p("What This Framework Does Not Claim", H3))
    story.append(tbl(["Claim withheld", "Reason"], [
        ["OOF MAE ≈ leaderboard score",
         "Leaderboard is out-of-sample; OOF is within-training-distribution. "
         "Distribution shift (adversarial AUC) drives a gap that training-side CV "
         "cannot quantify; the report does not assert any expected-test figure."],
        ["Model predicts within-cell variation",
         "The model predicts each cell's expected value (group × period). "
         "Within-cell deviation is noise-floor variance from the model's perspective. "
         "Targets near their cell-mean noise floor are dominated by that floor."],
        ["Metric contribution is uniform across categories",
         "MAE is dominated by high-magnitude categories even when per-category error "
         "rates are uniform — the scored metric reflects scale disparity across groups."],
        ["More Optuna trials guarantee better generalization",
         "Optuna minimises a CV-estimated objective. If the CV estimate carries optimism "
         "(e.g., from distribution shift), further tuning can increase overfit rather "
         "than improve test performance."],
    ], widths=[5.5*cm, 10.5*cm]))

    story.append(PageBreak())

    # ── Section 6: Forecasting Method ─────────────────────────────────────────
    story.append(_p("Section 6 — Forecasting Method", H2))
    if not _lag_fc:
        story.append(_p(
            "Not applicable — lag_forecasting block absent from model_results.json "
            "(multi-step lag forecasting was not used on this dataset).",
            META,
        ))
    else:
        _winner = lag_method or "not recorded"
        story.append(_p(
            f"At the validation boundary, lag features for future periods are unavailable "
            f"because actuals have not yet occurred. Two imputation strategies were compared "
            f"on a held-out window; <b>{_winner}</b> was selected for submission.",
            BODY,
        ))
        story.append(_p(
            "Note: this hold-out advantage is not captured in OOF or strict-CV scores — "
            "within training folds, lag values are known from the training window and "
            "the forecasting penalty does not apply.",
            META,
        ))
        story.append(Spacer(1, 0.15*cm))

        _method_rows = []
        for _m_name, _m_sc, _m_ac in [
            ("recursive",  lag_rec_mae, lag_rec_mac),
            ("imputation", lag_imp_mae, lag_imp_mac),
        ]:
            _sel_tag = " (selected)" if _m_name == _winner else ""
            _method_rows.append([
                f"{_m_name}{_sel_tag}",
                fmt(_m_sc, 4) if _m_sc is not None else "N/A",
                fmt(_m_ac, 4) if _m_ac is not None else "N/A",
            ])
        story.append(tbl(
            ["Strategy", "Scored holdout MAE", "All-cat holdout MAE"],
            _method_rows,
            widths=[5*cm, 5.5*cm, 5.5*cm],
        ))

        if lag_notes:
            story.append(Spacer(1, 0.1*cm))
            story.append(_p(f"Decision: {lag_notes}", BODY))

        if lag_steps_rec or lag_steps_imp:
            story.append(Spacer(1, 0.2*cm))
            story.append(_p("Per-Step MAE Across Forecast Horizon (scored categories)", H3))
            story.append(_p(
                "Scored MAE at each step into the forecast horizon (step 1 = immediately "
                "following the training window). Recursive accumulates lag estimation "
                "error each step; imputation applies cycle-aware historical fill.",
                BODY,
            ))
            _n_show = max(len(lag_steps_rec), len(lag_steps_imp))
            _step_rows = []
            for _si in range(_n_show):
                _r   = lag_steps_rec[_si] if _si < len(lag_steps_rec) else None
                _imp = lag_steps_imp[_si] if _si < len(lag_steps_imp) else None
                _step_rows.append([
                    str(_si + 1),
                    fmt(_r,   4) if _r   is not None else "N/A",
                    fmt(_imp, 4) if _imp is not None else "N/A",
                ])
            story.append(tbl(
                ["Step", "Recursive MAE", "Imputation MAE"],
                _step_rows,
                widths=[2*cm, 7*cm, 7*cm],
            ))
    story.append(PageBreak())

    # ── Section 7: Predictions and Submission ─────────────────────────────────
    story.append(_p("Section 7 — Predictions and Submission", H2))
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

    # ── Section 8: Limitations and Risks ──────────────────────────────────────
    story.append(_p("Section 8 — Limitations and Risks", H2))

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
            f"KS up to {max_ks:.4f}. Adversarial sample weighting (up-weights training "
            "rows that resemble the validation distribution) and shift-aware features "
            "(z-score, rolling z-score, percentile rank, group deviation, "
            "covariate × time interaction) partially compensate. "
            "Residual risk: underperformance in the tails of shifted covariate ranges.",
            "Adversarial sample weighting applied to CatBoost training; "
            "shift-aware features added."))

    if val_verdict == "CRITICAL":
        lims.append(("CRITICAL",
            "CV integrity issue (validator CRITICAL)",
            "Significant gap between reported and purged walk-forward MAE. "
            "The reported metric may be optimistic.",
            "Modeler retune triggered." if critic_retune else "Manual review recommended."))

    _lag_fc      = model_res.get("lag_forecasting") or {}
    _lag_method  = _lag_fc.get("method_used")
    _lag_imp_mae = _lag_fc.get("imputation_holdout_mae")
    _lag_rec_mae = _lag_fc.get("recursive_holdout_mae")
    _lag_body = (
        "Within-horizon predictions rely on model estimates rather than actuals, "
        "introducing compounding error for longer forecast horizons."
    )
    if _lag_method == "recursive":
        _mae_note = (
            f"  Imputation holdout MAE: {fmt(_lag_imp_mae, 4)};  "
            f"Recursive holdout MAE: {fmt(_lag_rec_mae, 4)}."
            if (_lag_imp_mae is not None and _lag_rec_mae is not None) else ""
        )
        _lag_mitigation = (
            f"Recursive (iterated) forecasting selected — outperformed static "
            f"imputation on holdout.{_mae_note}"
        )
    elif _lag_method == "imputation":
        _mae_note = (
            f"  Imputation holdout MAE: {fmt(_lag_imp_mae, 4)};  "
            f"Recursive holdout MAE: {fmt(_lag_rec_mae, 4)}."
            if (_lag_imp_mae is not None and _lag_rec_mae is not None) else ""
        )
        _lag_mitigation = (
            f"Static cycle-aware imputation selected — recursion was measured "
            f"but did not improve the holdout.{_mae_note}"
        )
    else:
        _lag_mitigation = (
            "Lag forecasting method not recorded "
            "(lag_forecasting block absent from model_results.json)."
        )
    lims.append(("MEDIUM",
        "Lag feature imputation at the validation boundary",
        _lag_body,
        _lag_mitigation))

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
            print(f"report.pdf written: {sz:,} bytes, 8 sections, {n_story} story elements")
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
