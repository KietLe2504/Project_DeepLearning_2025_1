"""
infer.py - Inference for next-day crude oil price prediction (LSTM model).

Requirements:
- artifacts/best_lstm_crude_oil.keras
- artifacts/feat_scaler.pkl
- artifacts/target_scaler.pkl
- artifacts/meta.json
- compiled_dataset.csv

Usage:
    python infer.py
    python infer.py --days 7
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# Paths
# =========================
DATA_PATH = "compiled_dataset.csv"
ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "best_lstm_crude_oil.keras")
FEAT_SCALER_PATH = os.path.join(ART_DIR, "feat_scaler.pkl")
TGT_SCALER_PATH = os.path.join(ART_DIR, "target_scaler.pkl")
META_PATH = os.path.join(ART_DIR, "meta.json")


# =========================
# Feature engineering (MUST match training)
# =========================
def create_technical_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()

    df["price_lag_1"] = df[target_col].shift(1)
    df["price_lag_3"] = df[target_col].shift(3)
    df["price_lag_7"] = df[target_col].shift(7)
    df["price_lag_14"] = df[target_col].shift(14)

    df["ma_7"] = df[target_col].rolling(window=7).mean()
    df["ma_14"] = df[target_col].rolling(window=14).mean()
    df["ma_30"] = df[target_col].rolling(window=30).mean()

    df["volatility_7d"] = df[target_col].rolling(window=7).std()
    df["volatility_14d"] = df[target_col].rolling(window=14).std()

    df["price_change_1d"] = df[target_col].pct_change(1)
    df["price_change_7d"] = df[target_col].pct_change(7)
    df["price_change_14d"] = df[target_col].pct_change(14)

    df["momentum_7d"] = df[target_col] - df[target_col].shift(7)
    df["momentum_14d"] = df[target_col] - df[target_col].shift(14)

    df["bb_middle"] = df[target_col].rolling(window=20).mean()
    df["bb_std"] = df[target_col].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (2 * df["bb_std"])
    df["bb_lower"] = df["bb_middle"] - (2 * df["bb_std"])
    df["bb_position"] = (df[target_col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["price_range_7d"] = df[target_col].rolling(window=7).max() - df[target_col].rolling(window=7).min()

    delta = df[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df = df.dropna().reset_index(drop=True)
    return df


def check_artifacts():
    missing = []
    for p in [MODEL_PATH, FEAT_SCALER_PATH, TGT_SCALER_PATH, META_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        raise FileNotFoundError(
            "Missing required artifact files:\n" + "\n".join(missing) +
            "\n=> Make sure you ran training and artifacts/ exists."
        )


def load_artifacts():
    check_artifacts()

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = tf.keras.models.load_model(MODEL_PATH)
    feat_scaler = joblib.load(FEAT_SCALER_PATH)
    target_scaler = joblib.load(TGT_SCALER_PATH)

    return model, feat_scaler, target_scaler, meta


def prepare_last_window(df: pd.DataFrame, feature_cols: list, window_size: int, feat_scaler) -> np.ndarray:
    """
    Return X_last: shape (1, window_size, num_features)
    """
    if len(df) < window_size:
        raise ValueError(f"Not enough rows after feature engineering: need >= {window_size}, got {len(df)}")

    last_block = df[feature_cols].iloc[-window_size:]  # (window_size, num_features)
    last_scaled = feat_scaler.transform(last_block)    # scale same as training
    X_last = np.expand_dims(last_scaled, axis=0)       # (1, window_size, num_features)
    return X_last


def predict_next_day_price(model, target_scaler, X_last: np.ndarray) -> float:
    pred_scaled = model.predict(X_last, verbose=0)     # (1, 1) scaled
    pred_price = target_scaler.inverse_transform(pred_scaled)[0, 0]
    return float(pred_price)


def main(days: int):
    model, feat_scaler, target_scaler, meta = load_artifacts()

    target_col = meta["target_col"]
    window_size = int(meta["window_size"])
    feature_cols = meta["feature_cols"]

    # Load data
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # (Optional) same missing handling as training
    if "eur_to_usd_exchange_rate" in df.columns:
        df["eur_to_usd_exchange_rate"] = df["eur_to_usd_exchange_rate"].ffill().bfill()

    # Create features
    df_feat = create_technical_features(df, target_col)

    # Basic checks: training feature list must exist
    missing_cols = [c for c in feature_cols if c not in df_feat.columns]
    if missing_cols:
        raise ValueError(
            "Your current dataset is missing feature columns saved in meta.json:\n"
            + "\n".join(missing_cols)
            + "\n=> Ensure the same compiled_dataset.csv structure as training."
        )

    last_date = df_feat["Date"].iloc[-1]
    last_real_price = df_feat[target_col].iloc[-1]

    if days <= 1:
        X_last = prepare_last_window(df_feat, feature_cols, window_size, feat_scaler)
        pred_next = predict_next_day_price(model, target_scaler, X_last)

        print("======================================")
        print("📌 NEXT-DAY PREDICTION")
        print("======================================")
        print(f"Last date in data : {last_date.date()}")
        print(f"Last real price   : {last_real_price:.4f}")
        print(f"Predicted next day: {pred_next:.4f}")
        print("======================================")
        return

    # Rolling multi-day forecast (recursive)
    # NOTE: Since features depend on future target, we must simulate by appending predicted target
    # and recomputing indicators each step. This is a common baseline approach.
    df_roll = df.copy()
    preds = []

    for step in range(days):
        df_roll_feat = create_technical_features(df_roll, target_col)

        X_last = prepare_last_window(df_roll_feat, feature_cols, window_size, feat_scaler)
        pred_next = predict_next_day_price(model, target_scaler, X_last)

        last_dt = df_roll_feat["Date"].iloc[-1]
        next_dt = last_dt + pd.Timedelta(days=1)  # calendar day step (simple)
        preds.append((next_dt, pred_next))

        # Append predicted row: keep other columns NaN -> forward-fill where possible
        new_row = {c: np.nan for c in df_roll.columns}
        new_row["Date"] = next_dt
        new_row[target_col] = pred_next

        df_roll = pd.concat([df_roll, pd.DataFrame([new_row])], ignore_index=True)

        # Forward-fill other columns (so features can be computed)
        for c in df_roll.columns:
            if c not in ["Date", target_col]:
                df_roll[c] = df_roll[c].ffill().bfill()

    print("======================================")
    print(f"📌 ROLLING FORECAST: next {days} days")
    print("======================================")
    for dt, p in preds:
        print(f"{dt.date()}  ->  {p:.4f}")
    print("======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1, help="Number of days to forecast (default: 1)")
    args = parser.parse_args()

    main(days=args.days)
