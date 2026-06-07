# Data Description — STAI-X Challenge 2026

## Folder structure

```
data/
├── train/
│   ├── dose_sys_train.csv       # training target
│   ├── covariates.csv           # training covariates
│   └── images/mat_density/      # MAT density PNG sidecar
│       └── {ST}_{PERIOD_ID}.png
├── val/
│   ├── covariates.csv           # validation covariates (target hidden)
│   └── images/mat_density/      # MAT density PNG sidecar (val periods)
│       └── {ST}_{PERIOD_ID}.png
├── sample_submission.csv        # submission template — fill this and write to ../submission.csv
└── Data_Description.md          # this file
```

The **training** package (`train/`) contains the labeled data you train on.
The **validation** package (`val/`) contains covariates only — the target is hidden.
Predict `rate_per_10000_ed_visits` for every row in `sample_submission.csv`.

---

## Files

| File | Location | Description |
| --- | --- | --- |
| `dose_sys_train.csv` | `train/` | Per-(period_id, jurisdiction) nonfatal overdose ED rates. Training target. |
| `covariates.csv` | `train/` and `val/` | Per-(period_id, jurisdiction) covariates panel. Same column schema in both. |
| `sample_submission.csv` | `data/` (root of this folder) | Submission template: 918 rows = 6 validation period_ids × 51 jurisdictions × 3 categories. Fill `rate_per_10000_ed_visits` and write the result to `../submission.csv` (the repo root). |
| `images/mat_density/{ST}_{PERIOD_ID}.png` | `train/images/` and `val/images/` | Per-(jurisdiction, period_id) 256×256 PNG visualizing within-state MAT clinic density. Optional multimodal feature. |

---

## Columns

### `train/dose_sys_train.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `period_id` | str | Opaque 8-character period identifier. Joins to `covariates.csv` and `mat_density` image filenames. |
| `jurisdiction` | str | USPS two-letter state code, or `DC` (51 jurisdictions total) |
| `overdose_category` | str | Drug class. Scoring categories: `all_drugs`, `all_opioids`, `all_stimulants`. Categories are nested and non-exclusive — never sum across them. |
| `rate_per_10000_ed_visits` | float | Suspected nonfatal overdose ED visits per 10,000 total ED visits. `NaN` when suppressed. |

### `train/covariates.csv` and `val/covariates.csv`

Long panel keyed on `(period_id, jurisdiction)`. Identical column schema in both files.

| Column | Type | Meaning |
| --- | --- | --- |
| `period_id` | str | Joins to `dose_sys_train.csv` and `sample_submission.csv` |
| `jurisdiction` | str | USPS two-letter state code or `DC` |
| `unemployment_rate` | float | State unemployment rate, percent |
| `labor_force` | int | Total state labor force, persons |
| `temp_avg_f` | float | State average temperature for the period, °F |
| `precip_in` | float | State total precipitation for the period, inches |
| `gtrends_overdose` | float | Google Trends search interest for "overdose" (0–100) |
| `gtrends_fentanyl` | float | Google Trends search interest for "fentanyl" (0–100) |
| `gtrends_naloxone` | float | Google Trends search interest for "naloxone" (0–100) |
| `gtrends_opioid` | float | Google Trends search interest for "opioid" (0–100) |
| `gtrends_methamphetamine` | float | Google Trends search interest for "methamphetamine" (0–100) |
| `state_doh_release` | str | Raw concatenated state Department of Health press-release text for that (state, period_id), drug-related only, capped at 500 whitespace-delimited tokens. Empty when no qualifying releases were issued. Calendar references have been stripped. |

Suppressed or missing numeric values appear as `NaN`; missing text is the empty string.

### `sample_submission.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `row_id` | int | Stable identifier. Your output must include every `row_id` from this file — no more, no fewer. |
| `period_id` | str | Opaque period identifier for the validation row |
| `jurisdiction` | str | USPS state code or `DC` |
| `overdose_category` | str | One of `all_drugs`, `all_opioids`, `all_stimulants` |
| `rate_per_10000_ed_visits` | float | Replace with your prediction. Template ships with `0.0` placeholders. |

### Image sidecar (`images/mat_density/`)

Filenames: `{JURISDICTION}_{PERIOD_ID}.png` (e.g., `WV_uTjgI1Sv.png`).
Each image is a 256×256 PNG of the state polygon with a viridis heatmap of
MAT prescriber density. Bright yellow = high density; dark purple = sparse.
Feed to a vision encoder for multimodal features, or ignore and use the CSV
columns alone.