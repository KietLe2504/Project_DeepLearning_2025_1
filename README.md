**📈 LSTM Baseline Model for Crude Oil Price Prediction**

Dự án xây dựng mô hình LSTM ba tầng nhằm dự đoán giá dầu thô Cushing (WTI) dựa trên dữ liệu chuỗi thời gian kết hợp các chỉ báo kỹ thuật nâng cao.

**🚀 1. Tổng quan mô hình**

Mục tiêu của mô hình là dự đoán giá dầu của ngày tiếp theo dựa trên 50 phiên giao dịch gần nhất.
Pipeline gồm các bước chính:

Tiền xử lý & chuẩn hóa dữ liệu

Feature engineering nâng cao

Tạo chuỗi thời gian dạng window

Huấn luyện mô hình LSTM 3 tầng

Đánh giá & trực quan hóa kết quả

🔧 2. Các kỹ thuật và thành phần chính
🔹 Feature Engineering

Áp dụng loạt chỉ báo kỹ thuật nhằm mô tả đầy đủ biến động giá:

Lag features: 1, 3, 7, 14 ngày

Moving Averages: MA7, MA14, MA30

Volatility 7d & 14d

Momentum: 7d & 14d

Rate of Change (ROC)

Bollinger Bands (upper / lower / position)

RSI 14 ngày

Price range 7 ngày

🔹 Tiền xử lý & Chuẩn hóa

MinMaxScaler cho toàn bộ input features

Window input: WINDOW_SIZE = 50

🔹 Kiến trúc mô hình

LSTM(128, return_sequences=True)

LSTM(64, return_sequences=True)

LSTM(32)

Dropout 0.2 mỗi tầng

Dense(1) cho output

Loss function: Huber Loss

Optimizer: Adam

🔹 Callbacks

EarlyStopping (restore_best_weights)

ReduceLROnPlateau (giảm LR khi mô hình chững)

**📊 2. Kết quả đánh giá**
RMSE: 0.1529
MAE : 0.1378
MAPE: 3.14%

## 📊 3. Kết quả trực quan (Visualization)

### 🔹 Real vs Predicted
![real_vs_pred](https://raw.githubusercontent.com/KietLe2504/Project_DeepLearning_2025_1/LongLSTM/images/real_vs_pred.png)

### 🔹 Error Plot
![error_plot](https://raw.githubusercontent.com/KietLe2504/Project_DeepLearning_2025_1/LongLSTM/images/error_plot.png)

### 🔹 Loss Curve
![loss_curve](https://raw.githubusercontent.com/KietLe2504/Project_DeepLearning_2025_1/LongLSTM/images/loss_curve.png)

### 🔹 Real vs Predicted Scatter
![scatter_plot](https://raw.githubusercontent.com/KietLe2504/Project_DeepLearning_2025_1/LongLSTM/images/scatter_plot.png)

**🛠️ 4. Pineline**

Toàn bộ chương trình được tổ chức theo một pipeline xử lý dữ liệu và huấn luyện mô hình gồm 8 bước, tuần tự như sau:

1️⃣ Load & xử lý dữ liệu

Đọc file dữ liệu gốc (compiled_dataset.csv)

Chuyển đổi kiểu dữ liệu ngày tháng

Sắp xếp theo thời gian và xử lý các giá trị thiếu (nếu có)

2️⃣ Feature Engineering

Tạo thêm các đặc trưng kỹ thuật (technical indicators) để mô tả hành vi giá, bao gồm:

Lag features

Moving Averages

Volatility

Momentum, ROC

Bollinger Bands

RSI

Price range
→ Sau đó loại bỏ toàn bộ các dòng sinh ra NaN.

3️⃣ Train/Test Split

Chia dữ liệu theo tỷ lệ 80% train – 20% test

Đảm bảo thứ tự thời gian được giữ nguyên (không shuffle)

4️⃣ Scaling

Chuẩn hóa toàn bộ features bằng MinMaxScaler

Chuẩn hóa riêng cột target

Lưu lại scaler để đảo ngược (inverse transform) khi đánh giá

5️⃣ Tạo sequences cho LSTM

Chuyển dữ liệu chuỗi thời gian thành dạng 3D:
(num_samples, window_size, num_features)

Với WINDOW_SIZE = 50, mô hình dùng 50 ngày trước để dự đoán ngày tiếp theo

6️⃣ Build & train mô hình LSTM

Xây dựng mô hình 3 tầng LSTM + Dropout

Compile với Adam + Huber Loss

Huấn luyện cùng EarlyStopping & ReduceLROnPlateau để tránh overfitting

7️⃣ Dự đoán & đánh giá

Dự đoán trên tập test

Inverse transform để đưa giá về dạng thật

Tính các chỉ số: RMSE, MAE, MAPE

8️⃣ Vẽ biểu đồ

Trực quan hóa kết quả gồm:

Biểu đồ Real vs Predicted

Biểu đồ sai số (Prediction Error)

Training loss / val_loss

Scatter plot so sánh dự đoán và giá thật
