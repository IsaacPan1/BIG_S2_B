#!/usr/bin/env python3
"""Generate synthetic practice datasets for Award B pipeline testing.

Run from the repo root:
    python tests/generate_practice_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Dataset 1: retail_sales
# ---------------------------------------------------------------------------

def make_retail_sales() -> None:
    print("Generating retail_sales ...")

    N_STORES, N_PRODUCTS, N_WEEKS = 50, 30, 100
    TRAIN_CUT = 90  # weeks 0-89 → train; 90-99 → val

    out = Path("practice_data/retail_sales")
    out.mkdir(parents=True, exist_ok=True)
    (out / "_truth").mkdir(exist_ok=True)

    # --- fixed entity parameters ---
    product_base_price = RNG.uniform(5, 50, N_PRODUCTS)          # shape (P,)
    store_scale = np.exp(RNG.normal(0, 0.4, N_STORES))           # shape (S,)
    # base_demand[s, p] = exp(2 + 0.5*log(price[p])) * scale[s]
    base_demand = (
        np.exp(2 + 0.5 * np.log(product_base_price))[None, :]   # (1, P)
        * store_scale[:, None]                                    # (S, 1)
    )  # → (S, P)

    # --- week-level signals (same across all store-product pairs) ---
    weeks = np.arange(N_WEEKS)
    holiday_flag = np.zeros(N_WEEKS, dtype=np.int8)
    holiday_flag[RNG.choice(N_WEEKS, 8, replace=False)] = 1
    weather_index = (
        np.sin(2 * np.pi * weeks / 52) + RNG.normal(0, 0.3, N_WEEKS)
    )  # (W,)

    # --- price: (P, W) — ±5 % weekly jitter around base ---
    prices = product_base_price[:, None] * RNG.uniform(0.95, 1.05, (N_PRODUCTS, N_WEEKS))

    # --- promotions: (S, P, W) — 20 % probability ---
    promotions = (RNG.random((N_STORES, N_PRODUCTS, N_WEEKS)) < 0.20).astype(np.int8)

    # --- AR(1) sales: (S, P, W) ---
    RHO = 0.60
    noise_std = base_demand * 0.20                                # (S, P)
    # Pre-draw all noise; scale per entity
    raw_noise = RNG.standard_normal((N_STORES, N_PRODUCTS, N_WEEKS))
    noise = raw_noise * noise_std[:, :, None]                     # (S, P, W)

    sales = np.zeros((N_STORES, N_PRODUCTS, N_WEEKS))
    residuals = np.zeros((N_STORES, N_PRODUCTS))                  # AR residual tracker

    for w in range(N_WEEKS):
        seasonal = base_demand * 0.15 * np.sin(2 * np.pi * w / 52)               # (S, P)
        price_ratio = prices[:, w] / product_base_price                           # (P,)
        price_effect = base_demand * (-0.30) * (price_ratio[None, :] - 1)         # (S, P)
        promo_effect = base_demand * 0.40 * promotions[:, :, w]                   # (S, P)
        holiday_effect = base_demand * 0.10 * float(holiday_flag[w])              # (S, P)

        deterministic = base_demand + seasonal + price_effect + promo_effect + holiday_effect

        residuals = RHO * residuals + noise[:, :, w]
        sales[:, :, w] = np.maximum(0.0, deterministic + residuals)

    # --- flatten to DataFrame ---
    s_idx = np.repeat(np.arange(N_STORES), N_PRODUCTS * N_WEEKS)
    p_idx = np.tile(np.repeat(np.arange(N_PRODUCTS), N_WEEKS), N_STORES)
    w_idx = np.tile(weeks, N_STORES * N_PRODUCTS)

    store_ids = [f"store_{i:03d}" for i in range(N_STORES)]
    product_ids = [f"prod_{i:03d}" for i in range(N_PRODUCTS)]

    df = pd.DataFrame({
        "store_id": pd.Categorical([store_ids[i] for i in s_idx]),
        "product_id": pd.Categorical([product_ids[i] for i in p_idx]),
        "week": w_idx.astype(np.int16),
        "price": np.round(prices[p_idx, w_idx], 2),
        "promotion_active": promotions[s_idx, p_idx, w_idx],
        "holiday_flag": holiday_flag[w_idx],
        "weather_index": np.round(weather_index[w_idx], 4),
        "weekly_sales": np.round(sales[s_idx, p_idx, w_idx], 2),
    })

    train = df[df["week"] < TRAIN_CUT]
    val = df[df["week"] >= TRAIN_CUT]

    cov_cols = ["store_id", "product_id", "week", "price",
                "promotion_active", "holiday_flag", "weather_index"]
    tgt_cols = ["store_id", "product_id", "week", "weekly_sales"]

    train[cov_cols].to_csv(out / "covariates_train.csv", index=False)
    val[cov_cols].to_csv(out / "covariates_val.csv", index=False)
    train[tgt_cols].to_csv(out / "target_train.csv", index=False)
    val[tgt_cols].to_csv(out / "_truth/val_truth.csv", index=False)

    sub = val[tgt_cols].copy()
    sub["weekly_sales"] = np.nan
    sub.to_csv(out / "sample_submission.csv", index=False)

    print(f"  covariates_train : {train[cov_cols].shape}")
    print(f"  covariates_val   : {val[cov_cols].shape}")
    print(f"  target_train     : {train[tgt_cols].shape}")
    print(f"  sample_submission: {sub.shape}")
    print(f"  _truth/val_truth : {val[tgt_cols].shape}")


# ---------------------------------------------------------------------------
# Dataset 2: energy_load
# ---------------------------------------------------------------------------

def make_energy_load() -> None:
    print("\nGenerating energy_load ...")

    N_REGIONS = 20
    N_DAYS = 30
    N_HOURS = 24
    TRAIN_CUT_DAY = 26          # days 0-25 → train; 26-29 → val
    N_TOTAL_HOURS = N_DAYS * N_HOURS   # 720

    out = Path("practice_data/energy_load")
    out.mkdir(parents=True, exist_ok=True)
    (out / "_truth").mkdir(exist_ok=True)

    start = pd.Timestamp("2024-01-01")

    # Timestamps for every hour across 30 days
    timestamps = pd.date_range(start, periods=N_TOTAL_HOURS, freq="h")
    hours_of_day = timestamps.hour.to_numpy()                    # (T,)
    days = timestamps.normalize()
    day_indices = ((days - start) / pd.Timedelta("1D")).astype(int).to_numpy()  # (T,)
    days_of_week = timestamps.dayofweek.to_numpy()               # (T,)

    # Holiday days (same for all regions)
    holiday_days_set = set(int(x) for x in RNG.choice(N_DAYS, 3, replace=False))
    is_holiday_day = np.array([int(d in holiday_days_set) for d in day_indices])  # (T,)

    # Hour-of-day pattern — double peak (morning + evening)
    HOUR_PATTERN = np.array([
        0.60, 0.56, 0.52, 0.50, 0.52, 0.58,   # 00-05: night valley
        0.68, 0.82, 1.00, 0.98, 0.95, 0.93,   # 06-11: morning peak
        0.90, 0.88, 0.87, 0.88, 0.90, 0.97,   # 12-17: midday plateau
        1.02, 1.00, 0.95, 0.88, 0.78, 0.68,   # 18-23: evening peak + decline
    ])

    # Temperature: daily base per day + regional offset + hourly variation
    day_temp_base = (
        15.0
        + 10.0 * np.sin(np.pi * np.arange(N_DAYS) / N_DAYS)
        + RNG.normal(0, 2, N_DAYS)
    )  # (D,)
    region_temp_offset = RNG.normal(0, 3, N_REGIONS)              # (R,)
    # hourly correction: +5 °C at noon, -5 °C at midnight
    hour_temp_correction = 5.0 * np.sin(np.pi * (hours_of_day - 6) / 12)  # (T,)

    # temperature[r, t] — broadcast later
    region_base_load = RNG.uniform(500, 2000, N_REGIONS)          # (R,) MW

    # --- generate AR(1) load per region ---
    rows = []
    RHO = 0.65

    for r in range(N_REGIONS):
        r_id = f"region_{r:02d}"
        base = region_base_load[r]
        residual = 0.0
        noise_std = base * 0.03

        for t in range(N_TOTAL_HOURS):
            h = hours_of_day[t]
            d = day_indices[t]
            dow = days_of_week[t]
            is_hol = is_holiday_day[t]
            is_wknd_hol = int(dow >= 5 or is_hol)

            temp = (
                day_temp_base[d]
                + region_temp_offset[r]
                + hour_temp_correction[t]
                + RNG.normal(0, 0.5)
            )

            # Deterministic load
            det_load = (
                base
                * HOUR_PATTERN[h]
                * (0.85 if is_wknd_hol else 1.0)
                * (1.0
                   + 0.008 * max(0.0, temp - 20.0)    # cooling demand
                   + 0.005 * max(0.0, 10.0 - temp))   # heating demand
            )

            residual = RHO * residual + RNG.normal(0, noise_std)
            load_mw = max(0.0, det_load + residual)

            rows.append({
                "region_id": r_id,
                "timestamp": str(timestamps[t]),
                "hour_of_day": int(h),
                "day_of_week": int(dow),
                "temperature": round(float(temp), 2),
                "is_holiday": int(is_hol),
                "load_mw": round(float(load_mw), 2),
            })

    df = pd.DataFrame(rows)

    train_day_strs = set(
        str((start + pd.Timedelta(days=d)).date())
        for d in range(TRAIN_CUT_DAY)
    )
    train_mask = df["timestamp"].str[:10].isin(train_day_strs)
    train = df[train_mask]
    val = df[~train_mask]

    feat_cols = ["region_id", "timestamp", "hour_of_day", "day_of_week",
                 "temperature", "is_holiday"]
    tgt_cols = ["region_id", "timestamp", "load_mw"]
    all_cols = feat_cols + ["load_mw"]

    train[all_cols].to_csv(out / "train.csv", index=False)
    val[feat_cols].to_csv(out / "val_features.csv", index=False)
    val[tgt_cols].to_csv(out / "_truth/val_truth.csv", index=False)

    sub = val[tgt_cols].copy()
    sub["load_mw"] = np.nan
    sub.to_csv(out / "sample_submission.csv", index=False)

    print(f"  train            : {train[all_cols].shape}")
    print(f"  val_features     : {val[feat_cols].shape}")
    print(f"  sample_submission: {sub.shape}")
    print(f"  _truth/val_truth : {val[tgt_cols].shape}")


# ---------------------------------------------------------------------------
# Dataset 3: medical_imaging
# ---------------------------------------------------------------------------

def _write_png_grayscale(path: Path, array: np.ndarray) -> None:
    """Write a 2D uint8 numpy array as an 8-bit grayscale PNG (stdlib only)."""
    import struct
    import zlib

    height, width = array.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    # IHDR: width, height, bit-depth=8, color-type=0 (grayscale), compress=0, filter=0, interlace=0
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    # Filter-method 0 (None) prepended to each row
    raw = b"".join(b"\x00" + row.tobytes() for row in array)
    idat = zlib.compress(raw, level=6)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )


def make_medical_imaging() -> None:
    print("\nGenerating medical_imaging ...")

    N_PATIENTS = 500
    TRAIN_CUT = 400
    IMG_SIZE = 64

    out = Path("practice_data/medical_imaging")
    img_dir = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    (out / "_truth").mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    # --- tabular covariates ---
    patient_ids = [f"pat_{i+1:04d}" for i in range(N_PATIENTS)]
    image_filenames = [f"img_{i+1:04d}.png" for i in range(N_PATIENTS)]

    ages = RNG.integers(18, 91, N_PATIENTS)
    sexes = RNG.choice(["M", "F"], N_PATIENTS)
    prior_admissions = RNG.integers(0, 6, N_PATIENTS)
    comorbidity_score = RNG.uniform(0.0, 10.0, N_PATIENTS).round(2)

    # --- target: hospitalization_days (0-14) ---
    # Correlated with age, comorbidity, prior admissions
    latent = (
        0.04 * (ages - 54)
        + 0.70 * comorbidity_score
        + 1.10 * prior_admissions
        + RNG.normal(0, 1.5, N_PATIENTS)
    )
    # Shift so minimum ≈ 0, then clip to [0, 14]
    latent = latent - latent.min()
    latent = latent / latent.max() * 14
    hosp_days = np.clip(np.round(latent + RNG.normal(0, 0.8, N_PATIENTS)), 0, 14).astype(int)

    # --- synthetic images ---
    # Base: random noise in [40, 160]; brightness offset ∝ target (0-84 extra)
    # Also stamp a circle whose radius ∝ target so an embedding can extract signal.
    CX, CY = IMG_SIZE // 2, IMG_SIZE // 2
    yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]

    print(f"  Writing {N_PATIENTS} PNG images ...", end="", flush=True)
    for i in range(N_PATIENTS):
        base = RNG.integers(40, 161, (IMG_SIZE, IMG_SIZE), dtype=np.int16)
        brightness_boost = int(hosp_days[i] * 6)          # 0 → 0, 14 → 84
        base += brightness_boost

        # Circle stamp: radius 4–18 pixels, proportional to target
        radius = 4 + hosp_days[i]                          # 4–18 px
        mask = (xx - CX) ** 2 + (yy - CY) ** 2 <= radius ** 2
        base[mask] += 35

        img = np.clip(base, 0, 255).astype(np.uint8)
        _write_png_grayscale(img_dir / image_filenames[i], img)

    print(" done.")

    # --- build DataFrame and save CSVs ---
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": ages,
        "sex": sexes,
        "prior_admissions": prior_admissions,
        "comorbidity_score": comorbidity_score,
        "image_filename": image_filenames,
        "hospitalization_days": hosp_days,
    })

    train = df.iloc[:TRAIN_CUT]
    val = df.iloc[TRAIN_CUT:]

    cov_cols = ["patient_id", "age", "sex", "prior_admissions",
                "comorbidity_score", "image_filename"]
    tgt_cols = ["patient_id", "hospitalization_days"]

    train[cov_cols].to_csv(out / "covariates_train.csv", index=False)
    val[cov_cols].to_csv(out / "covariates_val.csv", index=False)
    train[tgt_cols].to_csv(out / "target_train.csv", index=False)
    val[tgt_cols].to_csv(out / "_truth/val_truth.csv", index=False)

    sub = val[tgt_cols].copy()
    sub["hospitalization_days"] = np.nan
    sub.to_csv(out / "sample_submission.csv", index=False)

    print(f"  covariates_train : {train[cov_cols].shape}")
    print(f"  covariates_val   : {val[cov_cols].shape}")
    print(f"  target_train     : {train[tgt_cols].shape}")
    print(f"  sample_submission: {sub.shape}")
    print(f"  _truth/val_truth : {val[tgt_cols].shape}")
    print(f"  images/          : {N_PATIENTS} x {IMG_SIZE}x{IMG_SIZE} PNGs")


if __name__ == "__main__":
    make_retail_sales()
    make_energy_load()
    make_medical_imaging()
    print("\nAll practice data written.")
