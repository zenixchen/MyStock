import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ==========================================
# 1. 準備數據
# ==========================================
symbol = "2330.TW"
print(f"🚀 正在下載 {symbol} 歷史數據...")
df = yf.download(symbol, period="5y", interval="1d") # 訓練資料要長一點(5年)

# 只取「收盤價」
data = df[['Close']].values

# ★ 重要：數據標準化 (Normalization)
# 神經網絡喜歡 0~1 之間的數字，股價 1000 太大了會算不動
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 製作「滑動視窗」數據
# 設定：用過去 60 天 (prediction_days) 來預測 第 61 天
prediction_days = 60

x_train = []
y_train = []

for i in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[i-prediction_days:i, 0]) # 拿前60天當題目
    y_train.append(scaled_data[i, 0])                   # 拿當天當答案

x_train, y_train = np.array(x_train), np.array(y_train)

# LSTM 需要三維資料格式: (樣本數, 時間步長, 特徵數)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# ==========================================
# 2. 建立 LSTM 模型 (AI 的大腦)
# ==========================================
print("🧠 正在建構 LSTM 模型...")
model = Sequential()

# 第一層 LSTM
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# 第二層 LSTM
model.add(LSTM(units=50, return_sequences=False))
# 輸出層 (Dense) - 預測 1 個數字 (股價)
model.add(Dense(units=25))
model.add(Dense(units=1))

# 編譯模型
model.compile(optimizer='adam', loss='mean_squared_error')

# ==========================================
# 3. 開始訓練 (Training)
# ==========================================
print("🏋️‍♂️ AI 開始訓練中 (這可能需要幾分鐘)...")
# epochs=25 代表全部資料讀 25 遍，batch_size=32 代表一次讀 32 筆
model.fit(x_train, y_train, epochs=25, batch_size=32)

# ==========================================
# 4. 測試模型 (預測未來)
# ==========================================
print("🔮 正在測試預測能力...")

# 抓最新的測試數據 (這裡簡單起見，我們直接拿訓練資料的最後一部分來驗證)
# 實際應用應該要切分 Training Set 和 Test Set
test_start = len(scaled_data) - 200 # 看最後 200 天
test_inputs = scaled_data[test_start - prediction_days:]

x_test = []
for i in range(prediction_days, len(test_inputs)):
    x_test.append(test_inputs[i-prediction_days:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 讓 AI 預測
predicted_prices = model.predict(x_test)
# ★ 把預測出來的 0~1 變回 真實股價
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = data[test_start:]

# ==========================================
# 5. 畫圖驗證
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(real_prices, color='black', label=f"Real {symbol} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted {symbol} Price")
plt.title(f"{symbol} Share Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# 預測明天
real_data = [scaled_data[len(scaled_data) + 1 - prediction_days:len(scaled_data)+1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"\n======== 最終預測 ========")
print(f"根據過去 {prediction_days} 天的走勢...")
print(f"AI 預測明天的 {symbol} 收盤價約為: {prediction[0][0]:.2f}")
