"""
LSTM baseline model for crude oil price prediction.

- Dataset: compiled_dataset.csv
- Target: cushing_crude_oil_price
- Approach:
    + Feature engineering (lag, MA, volatility, ROC, momentum, Bollinger Bands, RSI,...)
    + Windowed time-series input (window_size days -> next-day prediction)
    + 3-layer LSTM + Dropout + Huber loss
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# =========================================================
# 1. Config
# =========================================================
FILE_PATH = "compiled_dataset.csv"
TARGET_COL = "cushing_crude_oil_price"
TRAIN_RATIO = 0.8
WINDOW_SIZE = 50  # sử dụng 50 ngày quá khứ để dự đoán 1 ngày tương lai


# =========================================================
# 2. Feature engineering
# =========================================================
def create_technical_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Tạo các features kỹ thuật (tech indicators) từ cột giá target.
    Trả về df mới đã drop NaN.
    """

    df = df.copy()

    # Lag features (giá quá khứ)
    df["price_lag_1"] = df[target_col].shift(1)
    df["price_lag_3"] = df[target_col].shift(3)
    df["price_lag_7"] = df[target_col].shift(7)
    df["price_lag_14"] = df[target_col].shift(14)

    # Moving Averages (trung bình trượt)
    df["ma_7"] = df[target_col].rolling(window=7).mean()
    df["ma_14"] = df[target_col].rolling(window=14).mean()
    df["ma_30"] = df[target_col].rolling(window=30).mean()

    # Volatility (độ biến động)
    df["volatility_7d"] = df[target_col].rolling(window=7).std()
    df["volatility_14d"] = df[target_col].rolling(window=14).std()

    # Rate of Change (tốc độ thay đổi giá)
    df["price_change_1d"] = df[target_col].pct_change(1)
    df["price_change_7d"] = df[target_col].pct_change(7)
    df["price_change_14d"] = df[target_col].pct_change(14)

    # Momentum indicators (chỉ báo động lượng)
    df["momentum_7d"] = df[target_col] - df[target_col].shift(7)
    df["momentum_14d"] = df[target_col] - df[target_col].shift(14)

    # Bollinger Bands (dải giá)
    df["bb_middle"] = df[target_col].rolling(window=20).mean()
    df["bb_std"] = df[target_col].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (2 * df["bb_std"])
    df["bb_lower"] = df["bb_middle"] - (2 * df["bb_std"])
    df["bb_position"] = (df[target_col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Price range (biên độ giá)
    df["price_range_7d"] = df[target_col].rolling(window=7).max() - df[target_col].rolling(window=7).min()

    # RSI-like indicator (chỉ số tương đối)
    delta = df[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Drop NaN sinh ra do rolling/shift
    df = df.dropna().reset_index(drop=True)

    return df


# =========================================================
# 3. Tạo sequences cho LSTM
# =========================================================
def create_sequences(features: np.ndarray, target: np.ndarray, window_size: int):
    """
    Biến chuỗi time-series (features, target) thành dạng sequences cho LSTM.
    Input:
        features: (num_samples, num_features)
        target: (num_samples, 1) hoặc (num_samples,)
    Output:
        X: (num_sequences, window_size, num_features)
        y: (num_sequences, 1)
    """
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])
        y.append(target[i])
    return np.array(X), np.array(y)


# =========================================================
# 4. LSTM Baseline Model Class
# =========================================================
class LSTMBaselineModel:
    """
    Wrapper class cho LSTM baseline model.
    Giữ nguyên kiến trúc: 3 LSTM layers + Dropout + Dense(1),
    nhưng đóng gói trong class để format nhìn chuyên nghiệp hơn.
    """

    def __init__(self, window_size: int, num_features: int, lr: float = 0.001):
        self.window_size = window_size
        self.num_features = num_features
        self.lr = lr
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """
        Xây dựng mô hình LSTM 3 tầng với Dropout.
        """
        model = Sequential([
            # LSTM layer 1 - return_sequences=True để stack thêm LSTM
            LSTM(128, return_sequences=True, input_shape=(self.window_size, self.num_features)),
            Dropout(0.2),

            # LSTM layer 2
            LSTM(64, return_sequences=True),
            Dropout(0.2),

            # LSTM layer 3 - không return sequences vì đây là layer cuối
            LSTM(32),
            Dropout(0.2),

            # Output layer
            Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=1.0),  # Robust loss function
            metrics=["mae", "mse"]
        )

        return model

    def summary(self):
        return self.model.summary()

    def fit(self, X_train, y_train, validation_split, epochs, batch_size, callbacks):
        return self.model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def save(self, path: str):
        self.model.save(path)

    def load_weights(self, path: str):
        self.model.load_weights(path)


# =========================================================
# 5. Plot helper functions
# =========================================================
def plot_results(y_test_inv, y_pred_inv, history):
    """
    Vẽ 3 biểu đồ:
    - Real vs Predicted
    - Prediction Error
    - Training history (loss & val_loss)
    Và một scatter plot Real vs Predicted.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Biểu đồ 1: So sánh giá thật vs dự đoán
    axes[0].plot(y_test_inv, label="Real price", linewidth=2, alpha=0.7)
    axes[0].plot(y_pred_inv, label="Predicted price", linewidth=2, alpha=0.7)
    axes[0].set_title("LSTM - Crude Oil Price Prediction (Improved Features)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Time (test set index)")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Biểu đồ 2: Prediction Error
    error = y_test_inv.flatten() - y_pred_inv.flatten()
    axes[1].plot(error, linewidth=1, alpha=0.6)
    axes[1].axhline(y=0, linestyle='--', linewidth=1)
    axes[1].fill_between(range(len(error)), error, 0, alpha=0.3)
    axes[1].set_title("Prediction Error (Real - Predicted)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Time (test set index)")
    axes[1].set_ylabel("Error")
    axes[1].grid(True, alpha=0.3)

    # Biểu đồ 3: Training history
    axes[2].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[2].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[2].set_title("Model Training History", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Scatter plot: Real vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_inv, y_pred_inv, alpha=0.5, s=20)
    min_val = min(y_test_inv.min(), y_pred_inv.min())
    max_val = max(y_test_inv.max(), y_pred_inv.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    plt.xlabel('Real Price', fontsize=12)
    plt.ylabel('Predicted Price', fontsize=12)
    plt.title('Real vs Predicted Price Scatter', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# 6. Main pipeline
# =========================================================
def main():
    # -------------------------------
    # Đọc và tiền xử lý dữ liệu
    # -------------------------------
    print("📥 Loading data...")

    df = pd.read_csv(FILE_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Xử lý missing value (nếu có)
    if "eur_to_usd_exchange_rate" in df.columns:
        df["eur_to_usd_exchange_rate"] = df["eur_to_usd_exchange_rate"].ffill().bfill()

    # -------------------------------
    # Feature engineering
    # -------------------------------
    print("🔧 Creating advanced features...")
    df = create_technical_features(df, TARGET_COL)

    # Chọn features (loại Date và target)
    feature_cols = [c for c in df.columns if c not in ["Date", TARGET_COL]]

    print(f"✅ Số dòng dữ liệu sau khi feature engineering & drop NaN: {len(df)}")
    print(f"✅ Số features: {len(feature_cols)}")
    print(f"📊 Danh sách features:\n{feature_cols}\n")

    # -------------------------------
    # Train / Test split
    # -------------------------------
    train_size = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    print(f"🔹 Train size: {train_size}")
    print(f"🔹 Test size: {len(test_df)}\n")

    # -------------------------------
    # Scale
    # -------------------------------
    feat_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features_scaled = feat_scaler.fit_transform(train_df[feature_cols])
    test_features_scaled = feat_scaler.transform(test_df[feature_cols])

    train_target_scaled = target_scaler.fit_transform(train_df[[TARGET_COL]])
    test_target_scaled = target_scaler.transform(test_df[[TARGET_COL]])

    # -------------------------------
    # Tạo sequences cho LSTM
    # -------------------------------
    X_train, y_train = create_sequences(train_features_scaled, train_target_scaled, WINDOW_SIZE)
    X_test, y_test = create_sequences(test_features_scaled, test_target_scaled, WINDOW_SIZE)

    print(f"🔹 X_train shape: {X_train.shape}")
    print(f"🔹 y_train shape: {y_train.shape}")
    print(f"🔹 X_test shape: {X_test.shape}")
    print(f"🔹 y_test shape: {y_test.shape}\n")

    # -------------------------------
    # Build model (dùng class)
    # -------------------------------
    print("🏗️ Building LSTM model...")

    num_features = X_train.shape[2]
    baseline_model = LSTMBaselineModel(WINDOW_SIZE, num_features, lr=0.001)
    baseline_model.summary()

    # -------------------------------
    # Callbacks
    # -------------------------------
    es = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )

    # -------------------------------
    # Training
    # -------------------------------
    print("\n🚀 Training model...")

    history = baseline_model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=120,
        batch_size=32,
        callbacks=[es, reduce_lr]
    )

    # -------------------------------
    # Prediction & Evaluation
    # -------------------------------
    print("\n📈 Making predictions...")

    y_pred_scaled = baseline_model.predict(X_test)

    # Inverse transform về giá thật
    y_test_inv = target_scaler.inverse_transform(y_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred_scaled)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 50)
    print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error):     {mae:.4f}")
    print(f"MAPE (Mean Absolute % Error):  {mape:.2f}%")
    print("=" * 50 + "\n")

    # -------------------------------
    # Visualization
    # -------------------------------
    plot_results(y_test_inv, y_pred_inv, history)

    print("✅ Done! Model training and evaluation completed.")


if __name__ == "__main__":
    main()
