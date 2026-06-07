# Medical Imaging Dataset

## Overview
Patient-level outcome prediction: **500 patients**, each associated with one 64×64 grayscale medical scan.
Train on 400 patients (target visible); predict for 100 patients (target hidden).

## Files

| File | Rows | Description |
|------|------|-------------|
| `covariates_train.csv` | 400 | Tabular features + image link, patients 1–400 |
| `covariates_val.csv` | 100 | Tabular features + image link, patients 401–500 (no target) |
| `target_train.csv` | 400 | Target for patients 1–400 |
| `sample_submission.csv` | 100 | Expected output format (`hospitalization_days` is NaN — fill it in) |
| `images/` | 500 PNGs | One 64×64 grayscale image per patient |

## Image Data

**Each row has an associated image at `images/{image_filename}` containing a medical scan.**

The `image_filename` column in the covariate files links each patient row to its scan.
For example, a row with `image_filename = img_0042.png` corresponds to `images/img_0042.png`.

Images are 64×64 pixels, single-channel (grayscale), saved as PNG.
Average pixel intensity is weakly correlated with the target — a patient with more severe
outcomes tends to produce a brighter scan and a larger central feature.

## Column Schema

### covariates_train.csv / covariates_val.csv

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | Patient identifier (`pat_0001` … `pat_0500`) |
| `age` | int | Patient age in years (18–90) |
| `sex` | string | `M` or `F` |
| `prior_admissions` | int | Number of prior hospital admissions in the past 5 years (0–5) |
| `comorbidity_score` | float | Aggregate comorbidity index (0.0–10.0; higher = more comorbidities) |
| `image_filename` | string | Filename of the associated medical scan (e.g. `img_0001.png`) |

### target_train.csv

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | Patient identifier |
| `hospitalization_days` | int | Total days hospitalised in the follow-up window (0–14) |

### sample_submission.csv
Same schema as `target_train.csv` but for the 100 validation patients.
`hospitalization_days` is `NaN` — replace with your predictions.

## Data-Generating Process (for reference)
- **Linear predictor**: 0.04 × (age − 54) + 0.70 × comorbidity_score + 1.10 × prior_admissions + Gaussian noise.
- **Target**: latent score normalised to [0, 14], rounded to integer, clipped.
- **Image**: uniform noise in [40, 160] + brightness boost of 6 × target + circle of radius (4 + target) pixels at centre + 35-point intensity lift.
- No true imaging biomarker — the signal is weak and tabular features dominate.

## Task
Predict `hospitalization_days` (integer, 0–14) for each patient in `covariates_val.csv`.
