import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import re
import importlib.util
import json
import time
import os
import random
import requests
import xml.etree.ElementTree as ET

# ==========================================
# ★★★ 0. God Mode: 鎖定隨機種子 ★★★
# ==========================================
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except: pass

set_seeds(42)

# ==========================================
# ★★★ 1. 套件檢查 ★★★
# ==========================================
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

try:
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except: HAS_GEMINI = False

# ==========================================
# 2. 頁面設定
# ==========================================
st.set_page_config(
    page_title="2026 量化戰情室 (Ultimate v22.4)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        h1, h2, h3, h4, h5, h6, span, div, p { color: #d1d4dc !important; font-family: 'Roboto', sans-serif; }
        div[data-testid="stMetric"] { background-color: #1c202a; border: 1px solid #2a2e39; border-radius: 8px; padding: 10px; }
        div[data-testid="stMetricLabel"] > div { color: #787b86 !important; }
        div[data-testid="stMetricValue"] > div { color: #d1d4dc !important; }
        section[data-testid="stSidebar"] { background-color: #161920; border-right: 1px solid #2a2e39; }
        .stButton > button { background-color: #2962ff; color: white; border: none; border-radius: 4px; font-weight: 600; }
        .stButton > button:hover { background-color: #1e4bd1; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { background-color: #1c202a; border-radius: 4px 4px 0 0; color: #d1d4dc; }
        .stTabs [aria-selected="true"] { background-color: #2962ff; color: white; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# ★★★ 核心模組：AI 交易日記系統 ★★★
# ==========================================
LEDGER_FILE = os.path.join(os.getcwd(), "ai_prediction_history.csv")

def get_real_live_price(symbol):
    try:
        t = yf.Ticker(symbol)
        price = t.fast_info.get('last_price')
        if price is None or np.isnan(price):
            df = yf.download(symbol, period='1d', interval='1m', progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                return float(df['Close'].iloc[-1])
        return float(price) if price else None
    except: return None

def save_prediction(symbol, direction, confidence, entry_price, target_days=5):
    try:
        today = datetime.now().date()
        target_date = today + timedelta(days=target_days)
        new_record = {
            "Date": today, "Symbol": symbol, "Direction": direction,
            "Confidence": round(float(confidence), 4), "Entry_Price": round(float(entry_price), 2),
            "Target_Date": target_date, "Status": "Pending", "Exit_Price": 0.0, "Return": 0.0
        }
        if os.path.exists(LEDGER_FILE):
            df = pd.read_csv(LEDGER_FILE)
            mask = (df['Date'] == str(today)) & (df['Symbol'] == symbol)
            if not df[mask].empty: return False
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        else:
            df = pd.DataFrame([new_record])
        df.to_csv(LEDGER_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"存檔失敗: {e}")
        return False

def verify_ledger():
    if not os.path.exists(LEDGER_FILE): return None
    try:
        df = pd.read_csv(LEDGER_FILE)
        df['Target_Date'] = pd.to_datetime(df['Target_Date']).dt.date
        today = datetime.now().date()
        updated = False
        for i, row in df.iterrows():
            if row['Status'] == 'Pending' or 'Run' in row['Status']:
                current_price = get_real_live_price(row['Symbol'])
                if current_price and current_price > 0:
                    entry = row['Entry_Price']
                    ret = (current_price - entry) / entry
                    df.at[i, 'Exit_Price'] = current_price
                    df.at[i, 'Return'] = round(ret * 100, 2)
                    res = "Win" if (row['Direction'] == "Bull" and ret > 0) or (row['Direction'] == "Bear" and ret < 0) else "Loss"
                    if today >= row['Target_Date']: df.at[i, 'Status'] = res
                    else: df.at[i, 'Status'] = f"Run ({res})"
                    updated = True
        if updated: df.to_csv(LEDGER_FILE, index=False)
        return df
    except Exception as e:
        st.error(f"讀取日記失敗: {e}")
        return None

# ==========================================
# ★★★ TSM T+5 主帥版 (含歷史回測繪圖功能) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_tsm_swing_prediction():
    if not HAS_TENSORFLOW: return None, None, 0, None
    try:
        # 1. 下載數據
        tickers = ["TSM", "^SOX", "NVDA", "^TNX", "^VIX"]
        data = yf.download(tickers, period="5y", interval="1d", progress=False, timeout=30)
        
        if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
        else: df = data['Close'].copy()
        df.ffill(inplace=True); df.dropna(inplace=True)

        # 2. 特徵工程
        feat = pd.DataFrame()
        try:
            feat['NVDA_Ret'] = df['NVDA'].pct_change()
            feat['SOX_Ret'] = df['^SOX'].pct_change()
            feat['TNX_Chg'] = df['^TNX'].pct_change()
            feat['VIX'] = df['^VIX']
            feat['TSM_Ret'] = df['TSM'].pct_change()
            feat['RSI'] = ta.rsi(df['TSM'], length=14)
            feat['MACD'] = ta.macd(df['TSM'])['MACD_12_26_9']
        except: return None, None, 0, None
        
        feat.dropna(inplace=True)
        cols = ['NVDA_Ret', 'SOX_Ret', 'TNX_Chg', 'VIX', 'TSM_Ret', 'RSI', 'MACD']
        
        # 3. 標籤
        future_ret = df['TSM'].shift(-5) / df['TSM'] - 1
        feat['Target'] = (future_ret > 0.025).astype(int)
        
        # 4. 訓練
        df_train = feat.iloc[:-90].copy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[cols])
        
        X, y = [], []
        lookback = 60
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(df_train['Target'].iloc[i])
        X, y = np.array(X), np.array(y)
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        from tensorflow.keras.layers import Input, LSTM
        model = Sequential()
        model.add(Input(shape=(lookback, len(cols))))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                  epochs=30, batch_size=32, callbacks=[early], 
                  class_weight=class_weight_dict, verbose=0)
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        
        # 5. ★★★ 新增：產生歷史回測圖表數據 (Backtest Visualization) ★★★
        # 我們預測過去 90 天的訊號，來看看 AI 準不準
        history_days = 90
        # 確保數據夠長
        if len(feat) > lookback + history_days:
            # 抓取最後 N 天的特徵
            recent_data = feat[cols].iloc[-(lookback + history_days):]
            # 必須用同一個 scaler 轉換
            recent_scaled = scaler.transform(recent_data)
            
            pred_history = []
            dates = []
            prices = []
            
            # 逐日滑動預測
            for i in range(lookback, len(recent_scaled)):
                seq = recent_scaled[i-lookback:i]
                # 預測
                prob = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0][0]
                
                # 紀錄日期與收盤價
                idx = recent_data.index[i]
                dates.append(idx)
                # 這裡要對應回原始 df 的價格
                prices.append(df['TSM'].loc[idx])
                pred_history.append(prob)
            
            # 建立 DataFrame 回傳
            df_viz = pd.DataFrame({
                'Date': dates,
                'Price': prices,
                'Prob': pred_history
            })
        else:
            df_viz = None

        # 6. 預測最新一天
        latest_seq = feat[cols].iloc[-lookback:].values
        latest_scaled = scaler.transform(latest_seq)
        prob_latest = model.predict(np.expand_dims(latest_scaled, axis=0), verbose=0)[0][0]
        
        current_price = df['TSM'].iloc[-1]
        
        # 多回傳一個 df_viz
        return prob_latest, acc, current_price, df_viz

    except Exception as e:
        print(f"TSM Model Error: {e}")
        return None, None, 0, None
        
# ==========================================
# ★★★ 修正版：TSM 短線極速預測 (T+3 / 五大因子) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_tsm_short_prediction():
    if not HAS_TENSORFLOW: return None, None
    try:
        # 1. 數據下載
        tickers = ["TSM", "^SOX", "NVDA", "^TNX", "^VIX"]
        data = yf.download(tickers, period="2y", interval="1d", progress=False)
        
        # 兼容 yfinance 新舊版索引
        if isinstance(data.columns, pd.MultiIndex):
            df_main = data['Close'].copy()
        else:
            df_main = data['Close'].copy()
            
        df_main.ffill(inplace=True); df_main.dropna(inplace=True)

        # 2. 特徵工程
        feat_df = pd.DataFrame()
        try:
            feat_df['TSM_Ret'] = df_main['TSM'].pct_change()
            feat_df['SOX_Ret'] = df_main['^SOX'].pct_change()
            feat_df['NVDA_Ret'] = df_main['NVDA'].pct_change()
            feat_df['TSM_RSI'] = ta.rsi(df_main['TSM'], length=14)
            feat_df['TSM_MACD'] = ta.macd(df_main['TSM'])['MACD_12_26_9']
            feat_df['VIX'] = df_main['^VIX']
            feat_df['TNX_Chg'] = df_main['^TNX'].pct_change()
        except: return None, None
        
        feat_df.dropna(inplace=True)
        feature_cols = ['TSM_Ret', 'SOX_Ret', 'NVDA_Ret', 'TSM_RSI', 'TSM_MACD', 'VIX', 'TNX_Chg']
        
        # 標籤：T+3 漲幅 > 1.5%
        future_ret = df_main['TSM'].shift(-3) / df_main['TSM'] - 1
        feat_df['Target'] = (future_ret > 0.015).astype(int)
        
        # 3. 準備數據
        df_train = feat_df.iloc[:-3].copy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[feature_cols])
        
        X, y = [], []
        lookback = 30 
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(df_train['Target'].iloc[i])
            
        X, y = np.array(X), np.array(y)
        
        # ★★★ 訓練/測試集切分 (防止準確率虛高) ★★★
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # --- 模型架構區 ---
        # 這裡必須 import Bidirectional，防止 NameError
        from tensorflow.keras.layers import Input, Bidirectional 
        
        model = Sequential()
        model.add(Input(shape=(lookback, len(feature_cols))))
        
        # ★★★ 這裡就是雙向 LSTM (Layer 1) ★★★
        model.add(Bidirectional(LSTM(64, return_sequences=True))) 
        model.add(Dropout(0.4))
        
        # ★★★ 這裡就是雙向 LSTM (Layer 2) ★★★
        model.add(Bidirectional(LSTM(32))) 
        model.add(Dropout(0.4))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        # 早停機制
        early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # 4. 訓練與驗證
        model.fit(X_train, y_train, 
                  validation_data=(X_test, y_test), 
                  epochs=30, batch_size=32, 
                  callbacks=[early], verbose=0)
        
        # 只看 Test 準確率
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        
        # 5. 預測最新
        latest_seq = feat_df[feature_cols].iloc[-lookback:].values
        latest_scaled = scaler.transform(latest_seq)
        prob = model.predict(np.expand_dims(latest_scaled, axis=0), verbose=0)[0][0]
        
        return prob, acc

    except Exception as e:
        print(f"Short Model Error: {e}")
        return None, None

# --- B. EDZ/Macro ---
@st.cache_resource(ttl=43200)
def get_macro_prediction(target_symbol, features_dict):
    if not HAS_TENSORFLOW: return None, None
    try:
        tickers = { 'Main': target_symbol }
        tickers.update(features_dict)
        data = yf.download(list(tickers.values()), period="3y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close'].copy()
            inv_map = {v: k for k, v in tickers.items()}
            df_close.rename(columns=inv_map, inplace=True)
            df = df_close.copy()
        else: return None, None

        df.ffill(inplace=True); df.bfill(inplace=True)
        feat_cols = []
        df['Main_Ret'] = df['Main'].pct_change()
        feat_cols.append('Main_Ret')
        for name in features_dict.keys():
            df[f'{name}_Ret'] = df[name].pct_change()
            feat_cols.append(f'{name}_Ret')
        df['RSI'] = ta.rsi(df['Main'], length=14)
        feat_cols.append('RSI')
        df.dropna(inplace=True)
        
        days_out = 5
        df['Target'] = ((df['Main'].shift(-days_out) / df['Main'] - 1) > 0.02).astype(int)
        df_train = df.iloc[:-5].copy()
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[feat_cols])
        
        X, y = [], []
        for i in range(20, len(scaled_data)):
            X.append(scaled_data[i-20:i])
            y.append(df_train['Target'].iloc[i])
        
        X, y = np.array(X), np.array(y)
        split = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
            
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(20, len(feat_cols))))
        model.add(Dropout(0.3)); model.add(LSTM(64)); model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        early = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[early])
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        last_seq = df[feat_cols].iloc[-20:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        return prob, acc
    except: return None, None

# --- C. QQQ Scanner ---
@st.cache_resource(ttl=86400)
def train_qqq_brain():
    if not HAS_TENSORFLOW: return None, None, None
    try:
        df = yf.download("QQQ", period="5y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df.ffill(inplace=True)
        df['Return'] = df['Close'].pct_change()
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['MA_Dist'] = (df['Close'] - ta.sma(df['Close'], 20)) / ta.sma(df['Close'], 20)
        df['ATR_Pct'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
        df.dropna(inplace=True)
        
        df['Target'] = ((df['Close'].shift(-5) / df['Close'] - 1) > 0.02).astype(int)
        df_train = df.iloc[:-5].copy()
        
        features = ['Return', 'RSI', 'RVOL', 'MA_Dist', 'ATR_Pct']
        scaler = StandardScaler()
        X, y = [], []
        for i in range(20, len(df_train)):
            X.append(scaler.fit_transform(df_train[features].iloc[i-20:i+1])[:-1]) 
            y.append(df_train['Target'].iloc[i])
            
        model = Sequential()
        model.add(LSTM(64, input_shape=(20, 5))); model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(np.array(X), np.array(y), epochs=40, verbose=0)
        return model, scaler, features
    except: return None, None, None

def scan_tech_stock(symbol, model, scaler, features):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        if len(df) < 60: return None, None, 0
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df.ffill(inplace=True)
        df = df[df['Volume'] > 0].copy()
        df['Return'] = df['Close'].pct_change()
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['MA_Dist'] = (df['Close'] - ta.sma(df['Close'], 20)) / ta.sma(df['Close'], 20)
        df['ATR_Pct'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
        
        df['Target'] = ((df['Close'].shift(-5) / df['Close'] - 1) > 0.02).astype(int)
        df.dropna(inplace=True)
        
        last_seq = df[features].iloc[-20:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        
        test_df = df.iloc[-125:-5] 
        acc = 0.5
        if len(test_df) > 30:
            X_t, y_t = [], []
            for i in range(20, len(test_df)):
                sub = test_df[features].iloc[i-20:i+1]
                X_t.append(scaler.transform(sub)[:-1])
                y_t.append(test_df['Target'].iloc[i])
            if len(y_t) > 0:
                _, acc = model.evaluate(np.array(X_t), np.array(y_t), verbose=0)

        return prob, acc, df['Close'].iloc[-1]
    except: return None, None, 0
        
        # ==========================================
# ★★★ SOXL 最終實戰版：5年數據 + 權重平衡 (F1=0.301) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_soxl_short_prediction():
    if not HAS_TENSORFLOW: return None, None, 0
    try:
        # 1. 下載 5 年數據 (關鍵差異：擴大樣本)
        tickers = ["SOXL", "NVDA", "^TNX", "^VIX"]
        # 注意：這裡 timeout 設長一點，因為 5 年數據量較大
        data = yf.download(tickers, period="5y", interval="1d", progress=False, timeout=30)
        
        if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
        else: df = data['Close'].copy()
        df.ffill(inplace=True); df.dropna(inplace=True)

        # 2. 特徵工程 (使用 Colab 驗證過的 4 大因子)
        feat = pd.DataFrame()
        try:
            # 因子 1: 乖離率 (Mean Reversion)
            ma20 = ta.sma(df['SOXL'], length=20)
            feat['Bias_20'] = (df['SOXL'] - ma20) / ma20
            
            # 因子 2: MACD (動能)
            feat['MACD'] = ta.macd(df['SOXL'])['MACD_12_26_9']
            
            # 因子 3: VIX (恐慌指數)
            feat['VIX'] = df['^VIX']
            
            # 因子 4: NVDA (領頭羊)
            feat['NVDA_Ret'] = df['NVDA'].pct_change()
            
        except: return None, None, 0

        feat.dropna(inplace=True)
        cols = ['Bias_20', 'MACD', 'VIX', 'NVDA_Ret']
        
        # 3. 標籤：T+3 漲幅 > 3%
        future_ret = df['SOXL'].shift(-3) / df['SOXL'] - 1
        feat['Target'] = (future_ret > 0.03).astype(int)
        
        # 準備訓練資料
        df_train = feat.iloc[:-3].copy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[cols])
        
        X, y = [], []
        lookback = 30 
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(df_train['Target'].iloc[i])
        X, y = np.array(X), np.array(y)
        
        # 切分 Test set (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # ★★★ 關鍵：計算類別權重 (Class Weights) ★★★
        # 這一步讓模型敢於預測 "1" (大漲)
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        # 4. 模型架構 (雙向 LSTM)
        from tensorflow.keras.layers import Input, Bidirectional, LSTM
        model = Sequential()
        model.add(Input(shape=(lookback, len(cols))))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.4))
        model.add(LSTM(32))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        
        # 訓練 (帶入 class_weight)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                  epochs=40, batch_size=32, callbacks=[early], 
                  class_weight=class_weight_dict, verbose=0)
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        
        # 5. 預測最新一天
        latest_seq = feat[cols].iloc[-lookback:].values
        latest_scaled = scaler.transform(latest_seq)
        prob = model.predict(np.expand_dims(latest_scaled, axis=0), verbose=0)[0][0]
        
        current_price = df['SOXL'].iloc[-1]
        
        return prob, acc, current_price

    except Exception as e:
        print(f"SOXL Model Error: {e}")
        return None, None, 0

# ==========================================
# 4. 傳統策略分析 (功能模組)
# ==========================================
# ★★★ 優化：加入緩存機制，提升速度並防鎖 IP ★★★
@st.cache_data(ttl=3600)
def get_safe_data(ticker):
    try:
        # 強制單層索引並關閉 auto_adjust
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False, multi_level_index=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.sort_index()
        return df
    except: return None

def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol: return None
        s = yf.Ticker(symbol)
        try:
            info = s.info
        except:
            return None
            
        return {
            "pe": info.get('trailingPE', None),
            "fwd_pe": info.get('forwardPE', None),
            "peg": info.get('pegRatio', None),
            "inst": info.get('heldPercentInstitutions', 0),
            "short": info.get('shortPercentOfFloat', 0),
            "shares_short": info.get('sharesShort', None),
            "shares_short_prev": info.get('sharesShortPriorMonth', None),
            "margin": info.get('grossMargins', 0),
            "eps": info.get('trailingEps', None),
            "rev_growth": info.get('revenueGrowth', None),
            "earn_growth": info.get('earningsGrowth', None)
        }
    except: return None

def clean_text_for_llm(text): return re.sub(r'[^\w\s\u4e00-\u9fff.,:;%()\-]', '', str(text))

# ★★★ 優化：新聞雙軌制 (RSS + YFinance) ★★★
def get_news(symbol):
    news_items = []
    try:
        # 1. 優先嘗試 Google News RSS
        search_query = symbol
        if ".TW" in symbol: search_query = symbol.replace(".TW", " TW stock")
        else: search_query = f"{symbol} stock news"
        
        url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=4) # 縮短 timeout
        
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall('.//item')[:5]:
                title = item.find('title').text
                news_items.append(clean_text_for_llm(title))
    except: pass

    # 2. 如果 RSS 失敗或空的，使用 yfinance 備援
    if not news_items:
        try:
            t = yf.Ticker(symbol)
            for n in t.news[:3]:
                news_items.append(n['title'])
        except: pass
        
    return news_items if news_items else ["暫無新聞數據"]

def calculate_kelly_position(df, capital, win_rate, risk_per_trade, current_signal):
    try:
        if current_signal != 1:
            if current_signal == -1: return "📉 訊號賣出，建議獲利了結/清倉", 0
            else: return "💤 訊號觀望，建議空手等待", 0

        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
        price = df['Close'].iloc[-1]
        stop_loss_dist = 2 * atr
        
        odds = 2.0
        kelly_pct = win_rate - ((1 - win_rate) / odds)
        safe_kelly = max(0, kelly_pct * 0.5) 
        
        risk_money = capital * risk_per_trade
        shares_by_risk = risk_money / stop_loss_dist
        
        if win_rate < 0.45:
            return "⛔ 雖有買訊但勝率過低，建議觀望", 0
            
        shares = int(shares_by_risk)
        cost = shares * price
        
        msg = f"🚀 建議買進 {shares} 股 (約 ${cost:.0f})"
        if safe_kelly > 0.2: msg += " 🔥重倉機會"
        
        return msg, shares
    except: return "計算失敗", 0

def identify_k_pattern(df):
    try:
        if len(df) < 3: return "N/A"
        
        last_3 = df.iloc[-3:].copy()
        c = last_3['Close'].values
        o = last_3['Open'].values
        h = last_3['High'].values
        l = last_3['Low'].values
        
        c2, o2, h2, l2 = c[2], o[2], h[2], l[2]
        c1, o1, h1, l1 = c[1], o[1], h[1], l[1]
        c0, o0, h0, l0 = c[0], o[0], h[0], l[0]
        
        body2 = abs(c2 - o2)
        upper2 = h2 - max(c2, o2)
        lower2 = min(c2, o2) - l2
        body1 = abs(c1 - o1)
        
        # --- 判斷邏輯 ---
        if (c0 < o0) and (abs(c0-o0) > body1 * 2) and \
           (c2 > o2) and (c2 > (o0 + c0)/2) and \
           (c1 < c0 and c1 < c2): 
            return "🌅 晨星轉折 (多)"

        if (c0 > o0) and (abs(c0-o0) > body1 * 2) and \
           (c2 < o2) and (c2 < (o0 + c0)/2) and \
           (c1 > c0 and c1 > c2):
            return "🌃 暮星轉折 (空)"

        if (c0 > o0) and (c1 > o1) and (c2 > o2) and \
           (c1 > c0) and (c2 > c1) and \
           (body2 > 0) and (lower2 < body2 * 0.5):
            return "💂‍♂️ 紅三兵 (強多)"

        if (c0 < o0) and (c1 < o1) and (c2 < o2) and \
           (c1 < c0) and (c2 < c1):
            return "🦅 黑三鴉 (強空)"

        if (c2 > o2) and (c1 < o1) and (c2 > o1) and (o2 < c1):
            return "🔥 多頭吞噬"
        if (c2 < o2) and (c1 > o1) and (c2 < o1) and (o2 > c1):
            return "💀 空頭吞噬"

        if (body1 > body2 * 3) and (max(c2, o2) < max(c1, o1)) and (min(c2, o2) > min(c1, o1)):
            return "🤰 母子變盤線"

        if (lower2 >= body2 * 2) and (upper2 <= body2 * 0.5):
            return "🔨 錘頭/吊人 (測底)"
        
        if (upper2 >= body2 * 2) and (lower2 <= body2 * 0.5):
            return "🌠 流星/倒錘 (測頂)"

        if body2 <= (h2 - l2) * 0.1:
            return "✝️ 十字線 (觀望)"

        return "一般震盪"
    except: return "N/A"

def quick_backtest(df, config, fee=0.0005):
    try:
        close = df['Close']; sigs = pd.Series(0, index=df.index)
        mode = config['mode']
        
        # ★★★ 新增：讀取是否啟用濾網 (預設為 True，即開啟) ★★★
        use_filter = config.get('ma_filter', True)
        
        if mode == "RSI_MA":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            ma_exit = ta.sma(close, length=config['exit_ma'])
            sigs[rsi < config['entry_rsi']] = 1
            sigs[close > ma_exit] = -1
        elif mode == "MA_CROSS":
            f = ta.sma(close, config['fast_ma']); s = ta.sma(close, config['slow_ma'])
            sigs[(f > s) & (f.shift(1) <= s.shift(1))] = 1
            sigs[(f < s) & (f.shift(1) >= s.shift(1))] = -1
        elif mode == "FUSION":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            ma = ta.ema(close, length=config.get('ma_trend', 200))
            if use_filter:
                sigs[(close > ma) & (rsi < config['entry_rsi'])] = 1
            else:
                sigs[rsi < config['entry_rsi']] = 1
            sigs[rsi > config['exit_rsi']] = -1
        elif mode == "BOLL_RSI":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            bb = ta.bbands(close, length=20, std=2)
            lower = bb.iloc[:, 0]; upper = bb.iloc[:, 2]
            sigs[(close < lower) & (rsi < config['entry_rsi'])] = 1
            sigs[close > upper] = -1
        
        # ★★★ 優化重點：RSI_RSI 加入趨勢濾網開關 ★★★
        elif mode == "RSI_RSI":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            # 只有當 1.有設定MA 且 2.濾網開啟 時，才檢查股價是否站上MA
            if config.get('ma_trend', 0) > 0 and use_filter:
                ma_trend = ta.ema(close, length=config['ma_trend'])
                sigs[(rsi < config['entry_rsi']) & (close > ma_trend)] = 1
            else:
                # 沒設定 MA 或 「強制關閉濾網」 -> 只看 RSI 夠不夠低
                sigs[rsi < config['entry_rsi']] = 1
            
            sigs[rsi > config['exit_rsi']] = -1

        elif "RSI" in mode:
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            sigs[rsi < config['entry_rsi']] = 1; sigs[rsi > config['exit_rsi']] = -1
        elif "KD" in mode:
            k = ta.stoch(df['High'], df['Low'], close, k=9, d=3).iloc[:, 0]
            sigs[k < config['entry_k']] = 1; sigs[k > config['exit_k']] = -1
        
        pos=0; ent=0; wins=0; trds=0; rets=[]
        for i in range(len(df)):
            if pos == 0 and sigs.iloc[i] == 1: 
                pos = 1; ent = close.iloc[i]
            elif pos == 1 and sigs.iloc[i] == -1:
                pos = 0; r = (close.iloc[i] - ent) / ent - (fee * 2)
                rets.append(r); trds += 1
                if r > 0: wins += 1
        
        win_rate = float(wins / trds) if trds > 0 else 0.0
        last_sig = sigs.iloc[-1]
        
        stats = {
            "Total_Return": sum(rets)*100, 
            "Win_Rate": win_rate * 100, 
            "Raw_Win_Rate": win_rate,
            "Trades": trds
        }
        return last_sig, stats, sigs
    except Exception as e: return 0, None, None

def plot_chart(df, config, sigs):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02, specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    if config.get('ma_trend', 0) > 0:
        ma = ta.ema(df['Close'], length=config['ma_trend'])
        fig.add_trace(go.Scatter(x=df.index, y=ma, name=f"EMA {config['ma_trend']}", line=dict(color='purple')), row=1, col=1)
    if "RSI" in config['mode']:
        rsi = ta.rsi(df['Close'], length=config.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color='#b39ddb')), row=2, col=1)
        fig.add_hline(y=config.get('entry_rsi', 30), line_dash="dash", row=2, col=1)
    elif "KD" in config['mode']:
        k = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
        fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:, 0], name="K", line=dict(color='yellow')), row=2, col=1)
        fig.add_hline(y=config.get('entry_k', 20), line_dash="dash", row=2, col=1)
    
    # ★★★ 優化：CMF 使用自訂週期 ★★★
    target_len = config.get('cmf_len', 20)
    cmf = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=target_len)
    
    colors = ['#089981' if v >= 0 else '#f23645' for v in cmf]
    fig.add_trace(go.Bar(x=df.index, y=cmf, name=f'CMF ({target_len})', marker_color=colors, opacity=0.5), row=3, col=1, secondary_y=False)
    obv = ta.obv(df['Close'], df['Volume'])
    fig.add_trace(go.Scatter(x=df.index, y=obv, name='OBV', line=dict(color='cyan', width=1)), row=3, col=1, secondary_y=True)
    if sigs is not None:
        buy = df[sigs==1]; sell = df[sigs==-1]
        fig.add_trace(go.Scatter(x=buy.index, y=buy['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell.index, y=sell['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell'), row=1, col=1)
    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    return fig

def get_strategy_desc(cfg, df=None):
    mode = cfg['mode']
    desc = mode; current_val = ""
    # ★★★ 新增：顯示濾網狀態 ★★★
    use_filter = cfg.get('ma_filter', True) 

    if df is not None:
        try:
            close = df['Close']
            if "RSI" in mode or mode == "FUSION":
                rsi = ta.rsi(close, length=cfg.get('rsi_len', 14)).iloc[-1]
                current_val += f" | 🎯 目前 RSI: {rsi:.1f}"
            if "KD" in mode:
                k = ta.stoch(df['High'], df['Low'], close, k=9, d=3).iloc[-1, 0]
                current_val += f" | 🎯 目前 K值: {k:.1f}"
            if mode == "MA_CROSS":
                f = ta.sma(close, cfg['fast_ma']).iloc[-1]; s = ta.sma(close, cfg['slow_ma']).iloc[-1]
                current_val += f" | 🎯 MA{cfg['fast_ma']}: {f:.1f} / MA{cfg['slow_ma']}: {s:.1f}"
            if "BOLL" in mode:
                bb = ta.bbands(close, length=20, std=2)
                lower = bb.iloc[-1, 0]
                current_val += f" | 🎯 下軌: {lower:.1f} (現價: {close.iloc[-1]:.1f})"
        except: pass
    
    if mode == "RSI_RSI": 
        desc = f"RSI 區間 (買 < {cfg['entry_rsi']} / 賣 > {cfg['exit_rsi']})"
        if cfg.get('ma_trend', 0) > 0:
            if use_filter: desc += f" (🛡️ 嚴格模式: 需站上 MA{cfg['ma_trend']})"
            else: desc += f" (🔓 寬鬆模式: 無視 MA{cfg['ma_trend']})"
    elif mode == "RSI_MA": desc = f"RSI + 均線 (RSI < {cfg['entry_rsi']} 買 / 破 MA{cfg['exit_ma']} 賣)"
    elif mode == "KD": desc = f"KD 隨機指標 (K < {cfg['entry_k']} 買 / K > {cfg['exit_k']} 賣)"
    elif mode == "MA_CROSS": desc = f"均線交叉 (MA{cfg['fast_ma']} 穿過 MA{cfg['slow_ma']})"
    elif mode == "FUSION": desc = f"趨勢 + RSI (站上 EMA{cfg['ma_trend']} 且 RSI < {cfg['entry_rsi']})"
    elif mode == "BOLL_RSI": desc = f"布林通道 + RSI (破下軌且 RSI < {cfg['entry_rsi']})"
    return desc + current_val

# ==========================================
# 5. 側邊欄與頁面配置
# ==========================================
st.sidebar.title("🚀 戰情室導航")
app_mode = st.sidebar.radio("選擇功能模組：", ["🤖 AI 深度學習實驗室", "📊 策略分析工具 (單股)", "📒 預測日記 (自動驗證)"])

st.sidebar.divider()
st.sidebar.header("⚙️ 全域設定")
ai_provider = st.sidebar.selectbox("AI 語言模型", ["不使用", "Gemini (User Defined)"])
gemini_key = ""; gemini_model = "models/gemini-3-pro-preview"

if ai_provider == "Gemini (User Defined)":
    gemini_key = st.sidebar.text_input("Gemini Key", type="password")
    gemini_model = st.sidebar.text_input("Model Name", value="models/gemini-3-pro-preview")

st.sidebar.divider()
st.sidebar.header("💰 凱利公式設定")
user_capital = st.sidebar.number_input("總本金 (USD)", value=10000)
user_risk = st.sidebar.number_input("單筆風險 (%)", value=1.0)

if st.sidebar.button("🔄 清除快取 (重置 AI)"):
    st.cache_resource.clear()
    st.rerun()

# ==========================================
# 6. 主畫面邏輯
# ==========================================

# ------------------------------------------
# Mode 1: AI 深度學習實驗室
# ------------------------------------------
if app_mode == "🤖 AI 深度學習實驗室":
    st.header("🤖 AI 深度學習實驗室")
    st.caption("神經網路模型 (LSTM) | T+5 & T+3 雙模預測")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 TSM 雙核心波段", "🐻 EDZ / 宏觀雷達", "⚡ QQQ 科技股通用腦","SOXL 三倍槓桿"])
    
    # === Tab 1: TSM ===
    with tab1:
        st.subheader("TSM 雙核心波段顧問")
        
        # 按鈕：一次觸發兩個模型
        if st.button("🚀 啟動雙模型分析 (T+3 & T+5)", key="btn_tsm") or 'tsm_result_v3' in st.session_state:
            
            # 如果還沒跑過，或是 Session 裡的是舊版資料，就重跑
            if 'tsm_result_v3' not in st.session_state:
                with st.spinner("AI 正在進行雙重驗證 & 歷史回測..."):
                    # 1. 呼叫 T+5 主帥模型 (注意：這裡現在接收 4 個回傳值)
                    p_long, a_long, price, df_viz = get_tsm_swing_prediction()
                    
                    # 2. 呼叫 T+3 短線模型
                    p_short, a_short = get_tsm_short_prediction()
                    
                    # 存入 Session (包含 df_viz)
                    st.session_state['tsm_result_v3'] = (p_long, a_long, p_short, a_short, price, df_viz)
            
            # 從 Session 取出結果
            p_long, a_long, p_short, a_short, price, df_viz = st.session_state['tsm_result_v3']
            
            # --- 顯示即時價格 ---
            st.metric("TSM 即時價格", f"${price:.2f}")
            st.divider()

            col1, col2 = st.columns(2)
            
            # 左邊：T+5 (主帥)
            with col1:
                st.info("🔭 T+5 波段主帥 (宏觀因子)")
                if p_long is not None:
                    # 顯示 F1 Score 或準確度
                    st.write(f"模型戰力 (F1): `0.561` (強)")
                    if p_long > 0.6: 
                        st.success(f"📈 波段看漲 (信心 {p_long*100:.0f}%)")
                    elif p_long < 0.4: 
                        st.warning(f"🐢 動能不足 (信心 {p_long*100:.0f}%)")
                    else: 
                        st.info(f"⚖️ 趨勢不明 (信心 {p_long*100:.0f}%)")
                else:
                    st.error("模型載入失敗")

            # 右邊：T+3 (先鋒)
            with col2:
                st.info("⚡ T+3 短線先鋒 (輔助)")
                if p_short is not None:
                    st.write(f"模型戰力 (F1): `0.455` (中)")
                    if p_short > 0.6: 
                        st.success(f"🚀 短線轉強 (信心 {p_short*100:.0f}%)")
                    elif p_short < 0.4: 
                        st.warning(f"💤 短線整理 (信心 {p_short*100:.0f}%)")
                    else: 
                        st.info(f"⚖️ 震盪 (信心 {p_short*100:.0f}%)")

            # --- 新增：AI 綜合戰略官 (Decision Support) ---
            st.divider()
            st.subheader("🛡️ AI 綜合戰略官")
            
            p5 = p_long if p_long is not None else 0.5
            p3 = p_short if p_short is not None else 0.5
            
            # 判斷訊號
            if p5 > 0.6 and p3 > 0.6:
                signal_msg = "🚀 【強力進攻】趨勢與短線共振，建議積極佈局 (Aggressive Buy)"
                color = "green"
            elif p5 > 0.6 and p3 <= 0.5:
                signal_msg = "📉 【拉回找買點】長線保護短線，等待修正結束再進 (Buy on Dip)"
                color = "blue"
            elif p5 <= 0.5 and p3 > 0.6:
                signal_msg = "🐱 【搶反彈/觀望】逆勢短多，風險較高 (Dead Cat Bounce)"
                color = "orange"
            elif p5 < 0.4 and p3 < 0.4:
                signal_msg = "🛑 【全面防守】趨勢轉空，建議清倉或做空 (Strong Sell)"
                color = "red"
            else:
                signal_msg = "⚖️ 【震盪整理】多看少做 (Hold)"
                color = "gray"

            st.markdown(f"""
            <div style="padding:15px; border-radius:10px; border:2px solid {color}; background-color:rgba(0,0,0,0.2);">
                <h4 style="color:{color}; margin:0;">{signal_msg}</h4>
                <p style="margin-top:10px; color:#ddd;">
                    綜合信心度: <b>{((p5+p3)/2)*100:.0f}%</b> (T+5: {p5*100:.0f}% | T+3: {p3*100:.0f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)

            # --- ★★★ 新增：歷史準度驗證圖 (Interactive Chart) ★★★ ---
            if df_viz is not None:
                st.divider()
                st.caption("📉 AI 歷史預測驗證 (過去 3 個月)")
                
                # 建立雙軸圖表
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 1. 畫股價
                fig.add_trace(
                    go.Scatter(x=df_viz['Date'], y=df_viz['Price'], name="TSM 股價",
                              line=dict(color='gray', width=1)),
                    secondary_y=False
                )
                
                # 2. 標記 AI 看漲點 (Prob > 0.6)
                buy_signals = df_viz[df_viz['Prob'] > 0.6]
                if not buy_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals['Date'], y=buy_signals['Price'],
                            mode='markers', name='AI 喊買 (信心>60%)',
                            marker=dict(color='red', size=8, symbol='triangle-up')
                        ),
                        secondary_y=False
                    )
                    
                # 3. 標記 AI 看跌點 (Prob < 0.4)
                sell_signals = df_viz[df_viz['Prob'] < 0.4]
                if not sell_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals['Date'], y=sell_signals['Price'],
                            mode='markers', name='AI 喊賣 (信心<40%)',
                            marker=dict(color='green', size=8, symbol='triangle-down')
                        ),
                        secondary_y=False
                    )

                # 4. 畫機率曲線
                fig.add_trace(
                    go.Scatter(x=df_viz['Date'], y=df_viz['Prob'], name="看漲機率",
                              line=dict(color='rgba(255, 0, 0, 0.2)', width=1, dash='dot')),
                    secondary_y=True
                )
                
                fig.update_layout(
                    height=350, 
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", y=1.1)
                )
                fig.update_yaxes(title_text="股價", secondary_y=False)
                fig.update_yaxes(title_text="AI 信心度", range=[0, 1], secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 顯示統計
                total_buys = len(buy_signals)
                st.caption(f"💡 近 90 天 AI 共發出 **{total_buys}** 次買進訊號 (紅色三角形)。請觀察三角形出現後，股價是否有波段漲幅？")
    # === Tab 2: EDZ / Macro ===
    with tab2:
        st.subheader("全球風險雷達")
        target_risk = st.selectbox("選擇監測對象", ["EDZ", "GC=F", "CL=F", "HG=F"])
        if st.button(f"分析 {target_risk}", key="btn_macro") or f'macro_{target_risk}' in st.session_state:
            if f'macro_{target_risk}' not in st.session_state:
                with st.spinner("AI 分析宏觀數據..."):
                    feat_map = { 'China': "FXI", 'DXY': "DX-Y.NYB", 'Rates': "^TNX", 'Copper': "HG=F" }
                    prob, acc = get_macro_prediction(target_risk, feat_map)
                    price = get_real_live_price(target_risk) or 0
                    st.session_state[f'macro_{target_risk}'] = (prob, acc, price)
            
            prob, acc, price = st.session_state[f'macro_{target_risk}']
            if prob is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("現價", f"${price:.2f}")
                c2.metric("模型準度", f"{acc*100:.1f}%")
                
                direction = "Bull" if prob > 0.5 else "Bear"
                conf = prob if prob > 0.5 else 1 - prob
                
                if prob > 0.6:
                    c3.metric("趨勢方向", "📈 向上", delta=f"信心 {conf*100:.1f}%")
                    if target_risk == "EDZ": st.error("⚠️ 市場避險情緒高漲！")
                elif prob < 0.4:
                    c3.metric("趨勢方向", "📉 向下", delta=f"信心 {conf*100:.1f}%", delta_color="inverse")
                else:
                    c3.metric("趨勢方向", "💤 震盪", delta=f"信心 {conf*100:.1f}%", delta_color="off")
                
                if st.button("📸 記錄預測 (快照)", key=f"save_{target_risk}"):
                    if save_prediction(target_risk, direction, conf, price):
                        st.success("✅ 已記錄！")
                    else: st.warning("⚠️ 今天已存過")

    # === Tab 3: QQQ Scanner ===
    with tab3:
        st.subheader("QQQ 科技股掃描器")
        tech_list = ["NVDA", "AMD", "AMZN", "MSFT", "GOOGL", "META", "TSLA", "AVGO", "PLTR"]
        if st.button("🚀 掃描科技巨頭", key="btn_scan") or 'scan_result' in st.session_state:
            if 'scan_result' not in st.session_state:
                with st.spinner("AI 正在訓練通用腦..."):
                    model, scaler, feats = train_qqq_brain()
                    if model:
                        res = []
                        prog = st.progress(0)
                        for i, t in enumerate(tech_list):
                            p, acc, pr = scan_tech_stock(t, model, scaler, feats)
                            if p: res.append((t, p, acc, pr))
                            prog.progress((i+1)/len(tech_list))
                        prog.empty()
                        res.sort(key=lambda x: x[1]+x[2], reverse=True)
                        st.session_state['scan_result'] = res
            
            if 'scan_result' in st.session_state:
                for tick, p, acc, pr in st.session_state['scan_result']:
                    mark = "💎" if p > 0.6 and acc > 0.55 else "🛡️" if p < 0.4 and acc > 0.55 else "⚠️"
                    direction = "📈" if p > 0.6 else "📉" if p < 0.4 else "💤"
                    color_str = "green" if p > 0.6 else "red" if p < 0.4 else "gray"
                    col1, col2, col3, col4 = st.columns([2, 2, 3, 2])
                    col1.markdown(f"**{tick}** (${pr:.1f})")
                    col2.markdown(f":{color_str}[{direction} ({p*100:.0f}%)]")
                    col3.caption(f"準度: {acc*100:.0f}% {mark}")
                    if col4.button("💾 存入日記", key=f"save_{tick}"):
                        dir_str = "Bull" if p > 0.5 else "Bear"
                        conf = p if p > 0.5 else 1 - p
                        if save_prediction(tick, dir_str, conf, pr): st.toast(f"✅ {tick} 已存")
                        else: st.toast("⚠️ 已存")

    # === Tab 4 (或新增): SOXL 槓桿戰神 ===
    with tab4: # 假設您想放在第一個分頁
        st.divider()
        st.subheader("🔥 SOXL 槓桿戰神 (T+3)")
        
        if st.button("🚀 啟動 SOXL 預測", key="btn_soxl"):
            with st.spinner("AI 正在分析乖離率與 VIX 恐慌指數..."):
                prob_soxl, acc_soxl, price_soxl = get_soxl_short_prediction()
                
                if prob_soxl is not None:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("SOXL 現價", f"${price_soxl:.2f}")
                    col2.metric("模型戰力 (F1)", "0.301", "高於隨機")
                    
                    # 這裡的邏輯：因為模型加了權重，機率通常會比較極端
                    # > 0.5 就是明確的看漲訊號
                    if prob_soxl > 0.5:
                        col3.success(f"🚀 強力看漲 (信心 {prob_soxl*100:.0f}%)")
                        st.caption("💡 觸發條件：乖離率過大 + VIX 配合 + 輝達動能")
                    else:
                        col3.warning(f"💤 動能不足 (信心 {prob_soxl*100:.0f}%)")
                else:
                    st.error("數據下載失敗，請稍後再試")

# ------------------------------------------
# Mode 2: 策略分析工具 (單股)
# ------------------------------------------
elif app_mode == "📊 策略分析工具 (單股)":
    st.header("📊 單股策略分析")
    
    strategies = {
        # === 📊 指數與外匯 ===
        "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (美元兌台幣匯率)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 25, "exit_k": 70 },
        "QQQ": { "symbol": "QQQ", "name": "QQQ (那斯達克100 ETF)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200, "cmf_len": 30 },
        "QLD": { "symbol": "QLD", "name": "QLD (那斯達克 2倍做多)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200, "cmf_len": 25 },
        "TQQQ": { "symbol": "TQQQ", "name": "TQQQ (那斯達克 3倍做多)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 85, "rsi_len": 2, "ma_trend": 200, "cmf_len": 40 },
        "SOXL_S": { "symbol": "SOXL", "name": "SOXL (費半 3倍做多 - 狙擊)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 10, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 100, "cmf_len": 25 },
        "SOXL_F": { "symbol": "SOXL", "name": "SOXL (費半 3倍做多 - 快攻)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 10, "exit_k": 75, "cmf_len": 25 },
        "EDZ": { "symbol": "EDZ", "name": "EDZ (新興市場 3倍做空 - 避險)", "category": "📊 指數/外匯", "mode": "BOLL_RSI", "entry_rsi": 9, "rsi_len": 2, "ma_trend": 20, "cmf_len": 10 },
        
        # === 🤖 AI 硬體/晶片 ===
        "NVDA": { "symbol": "NVDA", "name": "NVDA (AI 算力之王)", "category": "🤖 AI 硬體/晶片", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200, "vix_max": 32, "rvol_max": 2.5, "cmf_len": 30 },
        "TSM": { "symbol": "TSM", "name": "TSM (台積電 ADR - 晶圓代工)", "category": "🤖 AI 硬體/晶片", "mode": "MA_CROSS", "fast_ma": 5, "slow_ma": 60, "cmf_len": 26 },
        "AVGO": { "symbol": "AVGO", "name": "AVGO (博通 - AI 網通晶片)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 5, "entry_rsi": 55, "exit_rsi": 85, "ma_trend": 200, "cmf_len": 40 },
        "MRVL": { "symbol": "MRVL", "name": "MRVL (邁威爾 - ASIC 客製化晶片)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 20, "exit_rsi": 90, "ma_trend": 100, "ma_filter": False, "cmf_len": 25 }, # ★★★ 寬鬆模式範例 ★★★
        "QCOM": { "symbol": "QCOM", "name": "QCOM (高通 - AI 手機/PC)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 8, "entry_rsi": 30, "exit_rsi": 70, "ma_trend": 100, "cmf_len": 30 },
        "GLW": { "symbol": "GLW", "name": "GLW (康寧 - 玻璃基板/光通訊)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
        "ONTO": { "symbol": "ONTO", "name": "ONTO (安圖 - CoWoS 檢測設備)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 50, "exit_rsi": 65, "ma_trend": 100 },
        "AMD": { "symbol": "AMD", "name": "AMD (超微)", "category": "🤖 AI 硬體/晶片", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "MU": { "symbol": "MU", "name": "MU (美光 - 記憶體)", "category": "🤖 AI 硬體/晶片", "mode": "KD", "entry_k": 20, "exit_k": 80, "cmf_len": 20 },
        "SMCI": { "symbol": "SMCI", "name": "SMCI (美超微 - 伺服器妖股)", "category": "🤖 AI 硬體/晶片", "mode": "BOLL_RSI", "entry_rsi": 15, "rsi_len": 4, "ma_trend": 20, "cmf_len": 10 },
        "ARM": { "symbol": "ARM", "name": "ARM (架構矽智財)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_MA", "entry_rsi": 35, "exit_ma": 20, "rsi_len": 14, "ma_trend": 50, "cmf_len": 20 },

        # === 💻 軟體/巨頭 ===
        "MSFT": { "symbol": "MSFT", "name": "MSFT (微軟)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70, "rsi_len": 14, "ma_trend": 200 },
        "GOOGL": { "symbol": "GOOGL", "name": "GOOGL (谷歌)", "category": "💻 軟體/巨頭", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200, "vix_max": 32, "rvol_max": 2.5 },
        "META": { "symbol": "META", "name": "META (臉書)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 40, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
        "AMZN": { "symbol": "AMZN", "name": "AMZN (亞馬遜)", "category": "💻 軟體/巨頭", "mode": "KD", "entry_k": 20, "exit_k": 85, "cmf_len": 40 },
        "TSLA": { "symbol": "TSLA", "name": "TSLA (特斯拉)", "category": "💻 軟體/巨頭", "mode": "KD", "entry_k": 20, "exit_k": 80, "cmf_len": 10 },
        "AAPL": { "symbol": "AAPL", "name": "AAPL (蘋果)", "category": "💻 軟體/巨頭", "mode": "RSI_MA", "entry_rsi": 30, "exit_ma": 20, "rsi_len": 14, "ma_trend": 200 },
        "PLTR": { "symbol": "PLTR", "name": "PLTR (Palantir - AI國防)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 35, "exit_rsi": 85, "rsi_len": 14, "ma_trend": 50 },
        "CRWD": { "symbol": "CRWD", "name": "CRWD (CrowdStrike - 資安)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 35, "exit_rsi": 90, "rsi_len": 14, "ma_trend": 100, "cmf_len": 20 },
        "PANW": { "symbol": "PANW", "name": "PANW (Palo Alto - 資安)", "category": "💻 軟體/巨頭", "mode": "KD", "entry_k": 20, "exit_k": 80, "cmf_len": 20 },

        # === 💊 生技醫療 (減肥藥) ===
        "LLY": { "symbol": "LLY", "name": "LLY (禮來 - 減肥藥王)", "category": "💊 生技醫療", "mode": "FUSION", "entry_rsi": 60, "exit_rsi": 80, "rsi_len": 14, "ma_trend": 20, "ma_filter": True, "cmf_len": 20 },
        "NVO": { "symbol": "NVO", "name": "NVO (諾和諾德 - 減肥藥)", "category": "💊 生技醫療", "mode": "MA_CROSS", "fast_ma": 10, "slow_ma": 50 },

        # === 🪙 數位資產 (比特幣概念) ===
        "BTC_W": { "symbol": "BTC-USD", "name": "BTC (比特幣 - 波段)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 44, "exit_rsi": 65, "rsi_len": 14, "ma_trend": 200, "cmf_len": 40 },
        "BTC_F": { "symbol": "BTC-USD", "name": "BTC (比特幣 - 閃電)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 50, "rsi_len": 2, "ma_trend": 100, "cmf_len": 40 },
        "MSTR": { "symbol": "MSTR", "name": "MSTR (微策略 - BTC槓桿)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 35, "exit_rsi": 85, "rsi_len": 14, "ma_trend": 20, "cmf_len": 10 },
        "COIN": { "symbol": "COIN", "name": "COIN (Coinbase)", "category": "🪙 數位資產", "mode": "FUSION", "entry_rsi": 30, "exit_rsi": 90, "rsi_len": 14, "ma_trend": 100 },

        # === ⚡ 電力與能源 ===
        "ETN": { "symbol": "ETN", "name": "ETN (伊頓 - 電網)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 40, "exit_rsi": 95, "ma_trend": 200 },
        "VRT": { "symbol": "VRT", "name": "VRT (維諦 - 液冷)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 35, "exit_rsi": 95, "ma_trend": 100 },
        "OKLO": { "symbol": "OKLO", "name": "OKLO (微型核電)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 50, "exit_rsi": 95, "ma_trend": 0 },
        "SMR": { "symbol": "SMR", "name": "SMR (NuScale - 核能)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 45, "exit_rsi": 90, "ma_trend": 0, "cmf_len": 14 },

        # === 🇹🇼 台股 AI 權值 ===
        "CHT": { "symbol": "2412.TW", "name": "中華電", "category": "🇹🇼 台股", "mode": "RSI_RSI", "rsi_len": 14, "entry_rsi": 45, "exit_rsi": 70 },
        "HONHAI": { "symbol": "2317.TW", "name": "鴻海 (AI 伺服器代工)", "category": "🇹🇼 台股", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "QUANTA": { "symbol": "2382.TW", "name": "廣達 (AI 伺服器龍頭)", "category": "🇹🇼 台股", "mode": "RSI_MA", "entry_rsi": 40, "exit_ma": 20, "rsi_len": 14, "ma_trend": 60 },
        "MEDIATEK": { "symbol": "2454.TW", "name": "聯發科 (手機晶片)", "category": "🇹🇼 台股", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 80, "rsi_len": 14, "ma_trend": 0 },

        # === 🛡️ 防禦/傳產/原物料 ===
        "KO": { "symbol": "KO", "name": "KO (可口可樂)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0, "cmf_len": 20 },
        "JNJ": { "symbol": "JNJ", "name": "JNJ (嬌生)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 25, "exit_rsi": 90, "ma_trend": 200, "cmf_len": 20 },
        "PG": { "symbol": "PG", "name": "PG (寶僑)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 20, "exit_rsi": 80, "ma_trend": 0, "cmf_len": 30 },
        "BA": { "symbol": "BA", "name": "BA (波音)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 15, "exit_rsi": 60, "ma_trend": 0, "cmf_len": 25 },
        "JPM": { "symbol": "JPM", "name": "JPM (摩根大通)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 80, "rsi_len": 14, "ma_trend": 200 },
        "COST": { "symbol": "COST", "name": "COST (好市多)", "category": "🛡️ 防禦/傳產", "mode": "MA_CROSS", "fast_ma": 20, "slow_ma": 60 },
        
        "GC": { "symbol": "GC=F", "name": "Gold (黃金期貨)", "category": "⛏️ 原物料", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70, "rsi_len": 14 },
        "CL": { "symbol": "CL=F", "name": "Crude Oil (原油期貨)", "category": "⛏️ 原物料", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "HG": { "symbol": "HG=F", "name": "Copper (銅期貨)", "category": "⛏️ 原物料", "mode": "RSI_MA", "entry_rsi": 30, "exit_ma": 50, "rsi_len": 14 }
    }
    
    # ★★★ 優化重點：兩段式選擇 (分類 -> 股票) ★★★
    all_categories = sorted(list(set(s['category'] for s in strategies.values())))
    selected_cat = st.selectbox("📂 步驟一：選擇板塊分類", all_categories)
    
    cat_strategies = {k: v for k, v in strategies.items() if v['category'] == selected_cat}
    target_key = st.selectbox("📍 步驟二：選擇具體標的", list(cat_strategies.keys()), format_func=lambda x: cat_strategies[x]['name'])
    
    cfg = strategies[target_key]
    
    df = get_safe_data(cfg['symbol'])
    lp = get_real_live_price(cfg['symbol'])
    
    if df is not None and lp:
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else lp
        chg = lp - prev_close
        pct_chg = (chg / prev_close) * 100
        
        current_sig, perf, sigs = quick_backtest(df, cfg)
        win_rate = perf['Raw_Win_Rate'] if perf else 0
        trades_count = perf['Trades'] if perf else 0
        
        kelly_msg, kelly_shares = calculate_kelly_position(df, user_capital, win_rate, user_risk/100, current_sig)
        k_pat = identify_k_pattern(df)
        rsi_val = ta.rsi(df['Close'], 14).iloc[-1]
        fund = get_fundamentals(cfg['symbol'])
        
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("即時價格", f"${lp:.2f}", f"{chg:.2f} ({pct_chg:.2f}%)")
            
            if trades_count > 0:
                c2.metric("策略勝率 (回測)", f"{win_rate*100:.0f}%", delta=f"{trades_count} 次交易")
            else:
                c2.metric("策略勝率 (回測)", "無交易", delta="區間未觸發", delta_color="off")
                
            c3.metric("凱利建議倉位", f"{kelly_shares} 股", delta=kelly_msg.split(' ')[0] if '建議' in kelly_msg else "觀望")
            st.info(f"💡 凱利觀點: {kelly_msg}")

            # ★★★ 補回圖表繪製邏輯 ★★★
            fig = plot_chart(df, cfg, sigs)
            st.plotly_chart(fig, use_container_width=True)

        if fund:
            with st.expander("📊 財報基本面 & 籌碼數據", expanded=False):
                f1, f2, f3, f4, f5 = st.columns(5)
                
                def check_metric(val, high_good=True, low_good=False, threshold_good=0, threshold_bad=0):
                    if val is None: return "N/A", "off"
                    val = float(val)
                    if high_good:
                        if val > threshold_good: return f"{val:.1f}% ✅", "normal"
                        if val < threshold_bad: return f"{val:.1f}% ❌", "inverse"
                        return f"{val:.1f}% ⚠️", "off"
                    if low_good:
                        if val < threshold_good: return f"{val:.1f}% ✅", "normal"
                        if val > threshold_bad: return f"{val:.1f}% ❌", "inverse"
                        return f"{val:.1f}% ⚠️", "off"
                    return f"{val:.1f}", "off"

                pe_val = fund['pe']
                pe_str = "N/A"; pe_delta = "off"
                if pe_val:
                    if pe_val < 25: pe_str, pe_delta = f"{pe_val:.1f} ✅", "normal"
                    elif pe_val > 50: pe_str, pe_delta = f"{pe_val:.1f} ❌", "inverse"
                    else: pe_str, pe_delta = f"{pe_val:.1f} ⚠️", "off"
                f1.metric("本益比 (PE)", pe_str, delta_color=pe_delta)

                eps_val = fund['eps']
                eps_str = "N/A"; eps_delta = "off"
                if eps_val:
                    if eps_val > 0: eps_str, eps_delta = f"${eps_val:.2f} ✅", "normal"
                    else: eps_str, eps_delta = f"${eps_val:.2f} ❌", "inverse"
                f2.metric("EPS", eps_str, delta_color=eps_delta)

                m_str, m_delta = check_metric(fund['margin']*100, high_good=True, threshold_good=40, threshold_bad=10)
                f3.metric("毛利率", m_str, delta_color=m_delta)

                i_str, i_delta = check_metric(fund['inst']*100, high_good=True, threshold_good=60, threshold_bad=20)
                f4.metric("法人持股", i_str, delta_color=i_delta)

                s_str, s_delta = check_metric(fund['short']*100, low_good=True, threshold_good=5, threshold_bad=15)
                f5.metric("空單比例", s_str, delta_color=s_delta)
        else:
            st.warning("⚠️ 暫無財報數據 (API 忙碌中，請稍後再試)")

        # 1. 顯示策略邏輯文字 (這是錨點，請對齊這裡)
        strat_desc = get_strategy_desc(cfg, df)
        st.markdown(f"**🛠️ 當前策略邏輯：** `{strat_desc}`")

        # ==========================================
        # ★★★ 修復點：先初始化變數，防止 NameError ★★★
        # ==========================================
        analyze_btn = False 

        # 2. Gemini 分析區塊 (完整防呆版)
        if ai_provider == "Gemini (User Defined)" and gemini_key:
            st.divider()
            st.subheader("🧠 Gemini 首席分析師")
            
            st.info("ℹ️ 系統將自動抓取 Google News 最新頭條。若您有額外資訊 (如財報細節)，可在下方補充。")

            with st.expander("📝 補充筆記 (選填 / Optional)", expanded=False):
                user_notes = st.text_area("例如：營收創歷史新高、分析師調升評級...", height=68)
            
            # ★★★ 定義按鈕 (注意：這行必須跟上面的 st.info 對齊) ★★★
            analyze_btn = st.button("🚀 啟動 AI 深度分析 (含新聞解讀)")
            
        # ★★★ 檢查按鈕 (現在移到外面也安全了) ★★★
        if analyze_btn and ai_provider == "Gemini (User Defined)":
            with st.spinner("🔍 AI 正在爬取 Google News 並進行大腦運算..."):
                # A. 自動抓新聞
                news_items = get_news(cfg['symbol'])
                
                if news_items:
                    with st.expander(f"📰 AI 已讀取 {len(news_items)} 則最新新聞", expanded=True):
                        for n in news_items:
                            st.caption(f"• {n}")
                else:
                    st.warning("⚠️ 暫時抓不到 Google News，AI 將純以技術面分析。")
                    news_items = []

                # B. 計算策略指標
                strat_rsi_len = cfg.get('rsi_len', 14)
                strat_val_txt = ""
                
                if "RSI" in cfg['mode'] or cfg['mode'] == "FUSION":
                    real_rsi = ta.rsi(df['Close'], length=strat_rsi_len).iloc[-1]
                    strat_val_txt = f"Strategy_RSI({strat_rsi_len}):{real_rsi:.1f}"
                elif "KD" in cfg['mode']:
                    k_val = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3).iloc[-1, 0]
                    strat_val_txt = f"KD_K(9,3):{k_val:.1f}"
                elif cfg['mode'] == "MA_CROSS":
                    ma_fast = ta.sma(df['Close'], cfg['fast_ma']).iloc[-1]
                    ma_slow = ta.sma(df['Close'], cfg['slow_ma']).iloc[-1]
                    dist = (ma_fast - ma_slow) / ma_slow * 100
                    strat_val_txt = f"MA_Gap:{dist:.2f}%"

                base_rsi = ta.rsi(df['Close'], 14).iloc[-1]
                
                sig_map = { 1: "🚀 買進訊號 (Buy)", -1: "📉 賣出訊號 (Sell)", 0: "💤 觀望/無訊號 (Wait)" }
                human_sig = sig_map.get(int(current_sig), "未知")

                # C. 財報數據打包 (含成長率)
                fund_txt = "無財報數據"
                if fund:
                    # 籌碼動態
                    short_trend_str = "N/A"
                    if fund.get('shares_short') and fund.get('shares_short_prev'):
                        change = (fund['shares_short'] - fund['shares_short_prev']) / fund['shares_short_prev']
                        if change > 0.05: short_trend_str = f"🔴 增加 {change*100:.1f}% (空軍集結)"
                        elif change < -0.05: short_trend_str = f"🟢 減少 {abs(change)*100:.1f}% (空軍回補)"
                        else: short_trend_str = f"⚪ 持平 ({change*100:.1f}%)"

                    # 預估 PE
                    pe_trend_str = "N/A"
                    if fund.get('pe') and fund.get('fwd_pe'):
                        if fund['fwd_pe'] < fund['pe']: pe_trend_str = f"↘️ 看好 (預估PE {fund['fwd_pe']:.1f} < 當前)"
                        else: pe_trend_str = f"↗️ 看壞 (預估PE {fund['fwd_pe']:.1f} > 當前)"

                    rev_g = f"{fund.get('rev_growth', 0)*100:.1f}%" if fund.get('rev_growth') is not None else "N/A"
                    earn_g = f"{fund.get('earn_growth', 0)*100:.1f}%" if fund.get('earn_growth') is not None else "N/A"
                    
                    fund_txt = (
                        f"PE評價趨勢:{pe_trend_str} | "
                        f"空單變動(MoM):{short_trend_str} | "
                        f"空單比例:{fund.get('short', 0)*100:.1f}% | "
                        f"營收成長(YoY):{rev_g} | "
                        f"獲利成長(YoY):{earn_g} | "
                        f"毛利率:{fund.get('margin', 0)*100:.1f}%"
                    )

                # D. 組合小抄
                tech_txt = (
                    f"【策略關鍵指標】: {strat_val_txt}\n"
                    f"【籌碼與基本面】: {fund_txt}\n"
                    f"【市場大環境 RSI(14)】: {base_rsi:.1f}\n"
                    f"【回測勝率】: {win_rate*100:.0f}%\n"
                    f"【當前訊號】: {human_sig}"
                )

                # E. 定義與呼叫 (內嵌函數以防變數汙染)
                def analyze_v2(api_key, symbol, news, tech_txt, k_pattern, model_name, user_input=""):
                    if not HAS_GEMINI: return "No Gemini", "⚠️", False
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(model_name)
                        news_str = "\n".join([f"- {n}" for n in news]) if news else "無最新新聞"
                        base_prompt = f"""
                        你是一位華爾街資深操盤手。請根據以下「動態趨勢數據」進行深度分析：
                        【目標標的】：{symbol}
                        【綜合數據面板】：
                        {tech_txt}
                        【K線型態】：{k_pattern}
                        【最新新聞焦點】：
                        {news_str}
                        【用戶補充筆記】：{user_input}
                        請給出分析報告：
                        1. 🎯 核心觀點 (多/空/觀望)
                        2. 📊 籌碼與基本面解讀 (特別關注空單增減與預估PE的變化意義)
                        3. 📰 市場情緒
                        4. 💡 操作建議
                        """
                        return model.generate_content(base_prompt).text, "🧠", True
                    except Exception as e: return str(e), "⚠️", False

                analysis, icon, success = analyze_v2(gemini_key, cfg['symbol'], news_items, tech_txt, k_pat, gemini_model, user_notes)
                
                if success: st.markdown(analysis)
                else: st.error(f"Gemini 連線失敗: {analysis}")

# ------------------------------------------
# Mode 3: 預測日記 (Ledger)
# ------------------------------------------
elif app_mode == "📒 預測日記 (自動驗證)":
    st.header("📒 AI 實戰驗證日記")
    st.caption(f"檔案路徑: {LEDGER_FILE}")
    
    if st.button("🔄 立即刷新並驗證 (Auto-Verify)"):
        with st.spinner("正在檢查最新股價..."):
            df_ledger = verify_ledger()
            if df_ledger is not None: st.success("驗證完成！")
            else: st.info("尚無記錄")
    
    if os.path.exists(LEDGER_FILE):
        df = pd.read_csv(LEDGER_FILE)
        st.dataframe(df, use_container_width=True)
        if not df.empty:
            completed = df[df['Status'].isin(['Win', 'Loss'])]
            if not completed.empty:
                wins = len(completed[completed['Status'] == 'Win'])
                total = len(completed)
                win_rate = wins / total
                st.metric("實戰勝率 (Real Win Rate)", f"{win_rate*100:.1f}%", f"{wins}/{total} 筆")
    else: st.info("目前還沒有日記，請去預測頁面存檔。")










