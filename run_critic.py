import pandas as pd, numpy as np, json, os, datetime, sys

# ── Threshold constants ──────────────────────────────────────────────────────
CV_GAP_VALIDATOR_WARNING  = 0.10
CV_GAP_VALIDATOR_CRITICAL = 0.25
CV_GAP_CRITIC_ACCEPT      = 0.15

PRED_MEAN_BIAS_CRITICAL   = 0.30
PRED_STD_CRITICAL         = 0.40
PRED_STD_WARNING          = 0.65
WF_MAE_SUSPICIOUS_LOW     = 0.03
WF_MAE_POOR_RATIO         = 0.80
TOP_FEATURE_SHARE_WARNING = 0.50
PRED_MAX_HIGH_RATIO       = 3.0
PRED_MAX_LOW_RATIO        = 0.40

# Step 1: Check for second cycle
second_cycle = os.path.exists("reports/critic_retune_attempted.txt")

missing_inputs = []

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        missing_inputs.append(f"{path}: {e}")
        return {}

# Step 2: Load all inputs
validator_review = load_json("reports/validator_review.json")
model_results    = load_json("reports/model_results.json")
features_meta    = load_json("reports/features.json")
profile          = load_json("reports/profile.json")

try:
    predictions = pd.read_csv("reports/predictions.csv")
except Exception as e:
    missing_inputs.append(f"reports/predictions.csv: {e}")
    predictions = pd.DataFrame({"predicted_target": []})

try:
    train_df     = pd.read_parquet("data/features_train.parquet")
    target_col   = features_meta.get("target_col") or profile.get("target_col", "")
    train_target = train_df[target_col].dropna() if target_col in train_df.columns else pd.Series(dtype=float)
    train_mean   = float(train_target.mean()) if len(train_target) > 0 else 0.0
    train_std    = float(train_target.std())  if len(train_target) > 0 else 1.0
    train_max    = float(train_target.max())  if len(train_target) > 0 else 0.0
    train_nonneg = bool((train_target >= 0).all()) if len(train_target) > 0 else True
except Exception as e:
    missing_inputs.append(f"data/features_train.parquet: {e}")
    train_mean, train_std, train_max, train_nonneg = 0.0, 1.0, 0.0, True

# Step 3: Run quality checks
checks = []
warnings_for_report = []

# CHECK 1: Validator concordance
verdict         = validator_review.get("verdict", "UNKNOWN")
reported_cv_mae = float(validator_review.get("reported_cv_mae") or validator_review.get("honest_cv_mae") or 0.0)
strict_cv_mae   = float(validator_review.get("strict_cv_mae") or reported_cv_mae)
cv_gap_frac     = float(validator_review.get("cv_gap_pct", 0.0))
cv_gap_display  = cv_gap_frac * 100

c1_status  = "CRITICAL" if verdict == "CRITICAL" else ("WARNING" if verdict == "WARNING" else "PASS")
c1_details = (f"validator reported cv_mae={reported_cv_mae:.4f}, strict_cv_mae={strict_cv_mae:.4f}, "
              f"cv_gap={cv_gap_display:.1f}%; verdict was {verdict}")
checks.append({"name": "validator_concordance", "status": c1_status, "details": c1_details})

# CHECK 2: Prediction distribution match
pred_stats = model_results.get("val_prediction_stats", {})
pred_mean  = float(pred_stats.get("mean", predictions["predicted_target"].mean() if len(predictions) > 0 else 0.0))
pred_std   = float(pred_stats.get("std",  predictions["predicted_target"].std()  if len(predictions) > 0 else 0.0))
mean_bias  = abs(pred_mean - train_mean) if train_mean != 0.0 else 0.0
bias_pct   = (pred_mean - train_mean) / abs(train_mean) * 100 if train_mean != 0.0 else 0.0
std_ratio  = pred_std / train_std if train_std > 0.0 else 1.0
c2_status  = "PASS"
c2_details = (f"pred_mean={pred_mean:.3f} vs train_mean={train_mean:.3f} (delta {bias_pct:+.1f}%); "
              f"pred_std={pred_std:.3f} vs train_std={train_std:.3f} (ratio {std_ratio:.3f})")
if train_mean != 0.0 and mean_bias > PRED_MEAN_BIAS_CRITICAL * abs(train_mean):
    c2_status = "CRITICAL"
elif train_std > 0.0 and pred_std < PRED_STD_CRITICAL * train_std:
    c2_status = "CRITICAL"
elif train_std > 0.0 and pred_std < PRED_STD_WARNING * train_std:
    c2_status = "WARNING"
checks.append({"name": "prediction_distribution", "status": c2_status, "details": c2_details})

# CHECK 3: Walk-forward MAE plausibility
wf_mae     = float(model_results.get("walk_forward_mae") or model_results.get("oof_mae", 0.0) or 0.0)
ratio_pct  = (wf_mae / train_std * 100) if train_std > 0.0 else 0.0
c3_status  = "PASS"
c3_details = f"walk_forward_mae={wf_mae:.4f}; target_std={train_std:.4f}; ratio={ratio_pct:.1f}%"
if train_std > 0.0 and wf_mae < WF_MAE_SUSPICIOUS_LOW * train_std and verdict != "PASS":
    c3_status = "WARNING"
elif train_std > 0.0 and wf_mae > WF_MAE_POOR_RATIO * train_std:
    c3_status = "WARNING"
checks.append({"name": "mae_plausibility", "status": c3_status, "details": c3_details})

# CHECK 4: Feature importance concentration
top10     = model_results.get("feature_importance_top10", [])
c4_status = "PASS"
if not top10:
    c4_details = "not applicable: no feature importance data available"
elif len(top10) < 10:
    c4_details = f"not applicable: only {len(top10)} features available (need >= 10)"
else:
    top_imp = top10[0]["importance"]; total_imp = sum(x["importance"] for x in top10)
    if total_imp <= 0:
        c4_details = "not applicable: all importance values are zero"
    else:
        top_share  = top_imp / total_imp
        c4_details = f"top feature share = {top_share*100:.1f}% of top-10 importance (top: {top10[0]['feature']})"
        if top_share > TOP_FEATURE_SHARE_WARNING:
            c4_status = "WARNING"
checks.append({"name": "feature_concentration", "status": c4_status, "details": c4_details})

# CHECK 5: Prediction sanity
pred_col  = predictions["predicted_target"] if "predicted_target" in predictions.columns else pd.Series(dtype=float)
pred_min  = float(pred_col.min())  if len(pred_col) > 0 else 0.0
pred_max_val = float(pred_col.max())  if len(pred_col) > 0 else 0.0
nan_count = int(pred_col.isna().sum())
neg_count = int((pred_col < 0).sum()) if len(pred_col) > 0 else 0
c5_status  = "PASS"
c5_details = (f"pred_min={pred_min:.3f}, pred_max={pred_max_val:.3f}; "
              f"train_max={train_max:.3f}; n_nan={nan_count}; n_negative={neg_count}")
if nan_count > 0:
    c5_status = "CRITICAL"
elif neg_count > 0 and train_nonneg:
    c5_status = "CRITICAL"
elif train_max > 0.0 and pred_max_val > PRED_MAX_HIGH_RATIO * train_max:
    c5_status = "WARNING"
elif train_max > 0.0 and pred_max_val < PRED_MAX_LOW_RATIO * train_max:
    c5_status = "WARNING"
checks.append({"name": "prediction_sanity", "status": c5_status, "details": c5_details})

# Step 4: Decide on action
critical_checks = [c for c in checks if c["status"] == "CRITICAL"]
warning_checks  = [c for c in checks if c["status"] == "WARNING"]
for c in warning_checks:
    warnings_for_report.append(f"Check '{c['name']}' raised WARNING: {c['details']}")
if missing_inputs:
    warnings_for_report.append("Some critic inputs were missing: " + "; ".join(missing_inputs))

status = "accepted" if (second_cycle or not critical_checks) else "retune_requested"

warn_names = [c["name"] for c in checks if c["status"] == "WARNING"]
crit_names = [c["name"] for c in checks if c["status"] == "CRITICAL"]
check_summary = "; ".join(f"{c['name']}={c['status']}" for c in checks)
if second_cycle:
    decision_rationale = f"Second retune cycle: accepted regardless of remaining check results. Check summary: {check_summary}."
elif status == "accepted":
    if not warn_names and not crit_names:
        decision_rationale = f"All {len(checks)} checks PASS. Model accepted without concerns."
    else:
        decision_rationale = (f"Model accepted: {len(warn_names)} WARNING(s) ({', '.join(warn_names)}), 0 CRITICALs. "
                              f"WARNINGs documented in warnings_for_report. Check summary: {check_summary}.")
else:
    decision_rationale = (f"Retune requested: {len(crit_names)} CRITICAL check(s) ({', '.join(crit_names)}). "
                          f"Check summary: {check_summary}.")

print(f"Critic decision: {status} (second_cycle={second_cycle})")
for c in checks:
    print(f"  {c['name']}: {c['status']} -- {c['details']}")

# Step 5: If retune requested
issue = suggested = None
if status == "retune_requested" and not second_cycle:
    first_critical = critical_checks[0]["name"]
    if first_critical == "validator_concordance":
        issue = f"Validator returned CRITICAL verdict: {validator_review.get('notes', 'see validator_review.json')}"
        fs = validator_review.get("feature_suspicion", [])
        suggested = (f"remove suspect features ({', '.join(str(f) for f in fs[:3])}) and retrain"
                     if fs else "review validator_review.json notes and address CV methodology issue before retraining")
    elif first_critical == "prediction_distribution":
        if train_mean != 0.0 and abs(pred_mean - train_mean) > PRED_MEAN_BIAS_CRITICAL * abs(train_mean):
            issue = f"Systematic mean bias: pred_mean={pred_mean:.3f} vs train_mean={train_mean:.3f} (>30% offset)"
            suggested = "verify val feature imputation uses training medians per group, not zeros"
        else:
            issue = f"Severe under-dispersion: pred_std={pred_std:.3f} < {PRED_STD_CRITICAL:.2f}*train_std={train_std:.3f}"
            suggested = "use median seed aggregation; expand Optuna num_leaves upper bound from 127 to 255"
    elif first_critical == "prediction_sanity":
        if nan_count > 0:
            issue = f"NaN predictions: {nan_count} NaN in predicted_target"
            suggested = "apply fill_vals to all val columns; add np.nan_to_num fallback after ensemble_preds"
        else:
            issue = f"Negative predictions: {neg_count} negative values in non-negative target"
            suggested = "apply np.clip(predictions, 0, None) after seed aggregation, before saving predictions.csv"
    else:
        issue = f"Critical check failed: {first_critical} -- {critical_checks[0]['details']}"
        suggested = "retrain with default hyperparameters to rule out numerical instability"

    with open("reports/critic_retune_requested.json", "w") as f:
        json.dump({"issue": issue, "suggested_change": suggested,
                   "previous_metrics": {"walk_forward_mae": float(wf_mae), "pred_mean": float(pred_mean),
                                        "pred_std": float(pred_std), "validator_verdict": verdict}}, f, indent=2)
    with open("reports/critic_retune_attempted.txt", "w") as f:
        f.write(f"critic retune marker written at {datetime.datetime.now().isoformat()}\n")
    print("Written reports/critic_retune_requested.json and reports/critic_retune_attempted.txt")

# Step 6: Write critic_review.json
review = {
    "status":               status,
    "cycle":                2 if second_cycle else 1,
    "checks":               checks,
    "warnings_for_report":  warnings_for_report,
    "retune_attempted":     second_cycle or (status == "retune_requested"),
    "final_recommendation": ("proceed to submission_writer" if status == "accepted"
                             else "retune requested -- orchestrator should re-invoke modeler, then validator, then critic"),
    "decision_rationale":   decision_rationale,
}
if issue is not None:
    review["retune_issue"]            = issue
    review["retune_suggested_change"] = suggested

with open("reports/critic_review.json", "w") as f:
    json.dump(review, f, indent=2)
with open("reports/critic_was_here.txt", "w") as f:
    f.write(f"critic sub-agent executed at {datetime.datetime.now().isoformat()}\n")
print(f"Written reports/critic_review.json with status={status}")
print("Written reports/critic_was_here.txt")
