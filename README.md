📈 LSTM Baseline Model for Crude Oil Price Prediction

Dự án xây dựng mô hình LSTM 3 tầng để dự đoán giá dầu thô Cushing (WTI) dựa trên dữ liệu thời gian và các chỉ báo kỹ thuật nâng cao.

🚀 1. Mô tả tổng quan

Mô hình được phát triển với mục tiêu dự đoán giá ngày tiếp theo dựa trên 50 ngày dữ liệu quá khứ.

🔧 Kỹ thuật sử dụng

Feature engineering nâng cao:

Lag features (1, 3, 7, 14 ngày)

MA7, MA14, MA30

Volatility (7d, 14d)

ROC, momentum (7d, 14d)

Bollinger Bands (upper, lower, position)

RSI 14 ngày

Price range

Chuẩn hóa: MinMaxScaler

Windowed input: WINDOW_SIZE = 50

Kiến trúc:

LSTM(128, return_seq)

LSTM(64, return_seq)

LSTM(32)

Dropout 0.2

Dense(1)

Loss: Huber loss

Callbacks:

EarlyStopping

ReduceLROnPlateau

📊 2. Kết quả đánh giá
RMSE: 0.1529
MAE : 0.1378
MAPE: 3.14%

## 📊 3. Kết quả trực quan (Visualization)

### 🔹 Real vs Predicted
![real_vs_pred](https://raw.githubusercontent.com/KietLe2504/Project_DeepLearning_2025_1/main/LongLSTM/images/real_vs_pred.png)

### 🔹 Error Plot
![error_plot](https://raw.githubusercontent.com/KietLe2504/Project_DeepLearning_2025_1/main/LongLSTM/images/error_plot.png)

### 🔹 Loss Curve
![loss_curve](https://raw.githubusercontent.com/KietLe2504/Project_DeepLearning_2025_1/main/LongLSTM/images/loss_curve.png)

### 🔹 Real vs Predicted Scatter
![scatter_plot](https://raw.githubusercontent.com/KietLe2504/Project_DeepLearning_2025_1/main/LongLSTM/images/scatter_plot.png)

🛠️ 4. Cấu trúc code chính
Pipeline chính gồm:

Load & xử lý dữ liệu

Feature engineering

Train/Test split

Scaling

Tạo sequences cho LSTM

Build & train mô hình

Dự đoán & đánh giá

Vẽ biểu đồ

