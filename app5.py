import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time as dt_time
import sys
import re
import importlib.util
import json
import time
import os
import random

# ==========================================
# ★★★ 0. God Mode: 鎖定隨機種子 (確保 AI 穩定) ★★★
# ==========================================
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except: pass

set_seeds(42) # 程式啟動即執行

# ==========================================
# ★★★ 1. 套件安全匯入與設定 ★★★
# ==========================================
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# 深度學習套件 check
try:
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# NLP/LLM 套件 check
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ==========================================
# 2. 頁面設定
# ==========================================
st.set_page_config(
    page_title="2026 量化戰情室 (Ultimate v12.2)",
    page_icon="💎",
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
        .streamlit-expanderHeader { background-color: #1c202a !important; color: #d1d4dc !important; border: 1px solid #2a2e39; }
        .bull-box { background-color: #1a2e1a; padding: 10px; border-left: 5px solid #00ff00; margin-bottom: 5px; border-radius: 5px; }
        .bear-box { background-color: #2e1a1a; padding: 10px; border-left: 5px solid #ff0000; margin-bottom: 5px; border-radius: 5px; }
        .judge-box { background-color: #1a1a2e; padding: 10px; border-left: 5px solid #00aaff; margin-bottom: 5px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("💎 量化戰情室 (Ultimate v12.2)")
st.caption("全配版：包含所有自選股 (AVGO/MRVL/核能) | T+5 波段 AI (含準度) | EDZ 風險雷達")

if st.button('🔄 強制刷新行情 (Clear Cache)'):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

if not HAS_GEMINI:
    st.warning("⚠️ 系統提示：google-generativeai 未安裝，無法使用 Gemini。")
if not HAS_TENSORFLOW:
    st.warning("⚠️ 系統提示：TensorFlow/Keras 未安裝，無法使用波段預測功能。")

# ==========================================
# ★★★ 策略清單 (已恢復您的完整清單) ★★★
# ==========================================
strategies = {
    "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (美元兌台幣匯率)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 25, "exit_k": 70 },
    "QQQ": { "symbol": "QQQ", "name": "QQQ (那斯達克100 ETF)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200 },
    "QLD": { "symbol": "QLD", "name": "QLD (那斯達克 2倍做多)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200 },
    "TQQQ": { "symbol": "TQQQ", "name": "TQQQ (那斯達克 3倍做多)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 85, "rsi_len": 2, "ma_trend": 200 },
    "SOXL_S": { "symbol": "SOXL", "name": "SOXL (費半 3倍做多 - 狙擊)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 10, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 100 },
    "SOXL_F": { "symbol": "SOXL", "name": "SOXL (費半 3倍做多 - 快攻)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 10, "exit_k": 75 },
    "EDZ": { "symbol": "EDZ", "name": "EDZ (新興市場 3倍做空 - 避險)", "category": "📊 指數/外匯", "mode": "BOLL_RSI", "entry_rsi": 9, "rsi_len": 2, "ma_trend": 20 },
    "BTC_W": { "symbol": "BTC-USD", "name": "BTC (比特幣 - 波段)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 44, "exit_rsi": 65, "rsi_len": 14, "ma_trend": 200 },
    "BTC_F": { "symbol": "BTC-USD", "name": "BTC (比特幣 - 閃電)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 50, "rsi_len": 2, "ma_trend": 100 },
    "NVDA": { "symbol": "NVDA", "name": "NVDA (AI 算力之王)", "category": "🤖 AI 硬體/晶片", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200, "vix_max": 32, "rvol_max": 2.5 },
    "TSM": { "symbol": "TSM", "name": "TSM (台積電 ADR - 晶圓代工)", "category": "🤖 AI 硬體/晶片", "mode": "MA_CROSS", "fast_ma": 5, "slow_ma": 60 },
    "AVGO": { "symbol": "AVGO", "name": "AVGO (博通 - AI 網通晶片)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 5, "entry_rsi": 55, "exit_rsi": 85, "ma_trend": 200 },
    "MRVL": { "symbol": "MRVL", "name": "MRVL (邁威爾 - ASIC 客製化晶片)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 20, "exit_rsi": 90, "ma_trend": 100 },
    "QCOM": { "symbol": "QCOM", "name": "QCOM (高通 - AI 手機/PC)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 8, "entry_rsi": 30, "exit_rsi": 70, "ma_trend": 100 },
    "GLW": { "symbol": "GLW", "name": "GLW (康寧 - 玻璃基板/光通訊)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
    "ONTO": { "symbol": "ONTO", "name": "ONTO (安圖 - CoWoS 檢測設備)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 50, "exit_rsi": 65, "ma_trend": 100 },
    "META": { "symbol": "META", "name": "META (臉書 - 廣告與元宇宙)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 40, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
    "GOOGL": { "symbol": "GOOGL", "name": "GOOGL (谷歌 - 搜尋與 Gemini)", "category": "💻 軟體/巨頭", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200, "vix_max": 32, "rvol_max": 2.5 },
    "AMZN": { "symbol": "AMZN", "name": "AMZN (亞馬遜)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 80, "rsi_len": 14, "ma_trend": 200 },
    "TSLA": { "symbol": "TSLA", "name": "TSLA (特斯拉)", "category": "💻 軟體/巨頭", "mode": "KD", "entry_k": 20, "exit_k": 80 },
    "AAPL": { "symbol": "AAPL", "name": "AAPL (蘋果)", "category": "💻 軟體/巨頭", "mode": "RSI_MA", "entry_rsi": 30, "exit_ma": 20, "rsi_len": 14, "ma_trend": 200 },
    "MSFT": { "symbol": "MSFT", "name": "MSFT (微軟)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70, "rsi_len": 14, "ma_trend": 200 },
    "AMD": { "symbol": "AMD", "name": "AMD (超微)", "category": "🤖 AI 硬體/晶片", "mode": "KD", "entry_k": 20, "exit_k": 80 },
    "PLTR": { "symbol": "PLTR", "name": "PLTR (Palantir)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 35, "exit_rsi": 85, "rsi_len": 14, "ma_trend": 50 },
    "ETN": { "symbol": "ETN", "name": "ETN (伊頓 - 電網與電力管理)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 40, "exit_rsi": 95, "ma_trend": 200 },
    "VRT": { "symbol": "VRT", "name": "VRT (維諦 - AI 伺服器液冷)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 35, "exit_rsi": 95, "ma_trend": 100 },
    "OKLO": { "symbol": "OKLO", "name": "OKLO (核能 - 微型反應堆)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 50, "exit_rsi": 95, "ma_trend": 0 },
    "SMR": { "symbol": "SMR", "name": "SMR (NuScale - 模組化核能)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 45, "exit_rsi": 90, "ma_trend": 0 },
    "KO": { "symbol": "KO", "name": "KO (可口可樂 - 消費必需品)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
    "JNJ": { "symbol": "JNJ", "name": "JNJ (嬌生 - 醫療與製藥)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 25, "exit_rsi": 90, "ma_trend": 200 },
    "PG": { "symbol": "PG", "name": "PG (寶僑 - 日用品龍頭)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 20, "exit_rsi": 80, "ma_trend": 0 },
    "BA": { "symbol": "BA", "name": "BA (波音 - 航太製造)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 15, "exit_rsi": 60, "ma_trend": 0 },
    "CHT": { "symbol": "2412.TW", "name": "中華電 (台灣電信龍頭)", "category": "🇹🇼 台股", "mode": "RSI_RSI", "rsi_len": 14, "entry_rsi": 45, "exit_rsi": 70, "ma_trend": 0 },
    "GC": { "symbol": "GC=F", "name": "Gold (黃金期貨)", "category": "⛏️ 原物料", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70, "rsi_len": 14 },
    "CL": { "symbol": "CL=F", "name": "Crude Oil (原油期貨)", "category": "⛏️ 原物料", "mode": "KD", "entry_k": 20, "exit_k": 80 },
    "HG": { "symbol": "HG=F", "name": "Copper (銅期貨)", "category": "⛏️ 原物料", "mode": "RSI_MA", "entry_rsi": 30, "exit_ma": 50, "rsi_len": 14 }
}

# ==========================================
# ★★★ 3. AI 深度學習模組 (LSTM) ★★★
# ==========================================

# --- A. TSM 波段顧問 (含夜盤+利率) ★修正版：回傳準度 ---
@st.cache_resource(ttl=43200)
def get_tsm_swing_prediction(symbol="TSM"):
    if not HAS_TENSORFLOW: return None, None, "TF 未安裝"
    try:
        # 下載數據
        tickers = { 'Main': symbol, 'Night': "EWT", 'Rate': "^TNX", 'AI': 'NVDA' }
        data = yf.download(list(tickers.values()), period="2y", interval="1d", progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close'].copy()
            inv_map = {v: k for k, v in tickers.items()}
            df_close.rename(columns=inv_map, inplace=True)
            df = pd.DataFrame()
            df['Close'] = df_close['Main']
            df['Night_Close'] = df_close['Night']
            df['Rate_Close'] = df_close['Rate']
            df['AI_Close'] = df_close['AI']
        else: return None, None, "Data Error"

        # 特徵工程
        df['Main_Ret'] = df['Close'].pct_change()
        df['Night_Ret'] = df['Night_Close'].pct_change()
        df['Rate_Chg'] = df['Rate_Close'].pct_change()
        df['AI_Ret'] = df['AI_Close'].pct_change()
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['Bias'] = (df['Close'] - ta.sma(df['Close'], 20)) / ta.sma(df['Close'], 20)
        df.dropna(inplace=True)

        # 標籤 (T+5 > 2%)
        days_out = 5; threshold = 0.02
        df['Target'] = ((df['Close'].shift(-days_out) / df['Close'] - 1) > threshold).astype(int)
        df_train = df.iloc[:-days_out].copy()
        
        if len(df_train) < 60: return None, None, "數據不足"

        features = ['Main_Ret', 'Night_Ret', 'Rate_Chg', 'AI_Ret', 'RSI', 'Bias']
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[features])
        
        X, y = [], []
        lookback = 20
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(df_train['Target'].iloc[i])
        
        X, y = np.array(X), np.array(y)
        
        # 切分測試集 (為了計算準度)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # 訓練 (含 EarlyStopping + RestoreBestWeights)
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        # ★ 關鍵：恢復最佳權重
        early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stop])
        
        # ★ 計算準度
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        
        # 預測
        last_seq = df[features].iloc[-lookback:].values
        last_seq_scaled = scaler.transform(last_seq)
        prob = model.predict(np.expand_dims(last_seq_scaled, axis=0), verbose=0)[0][0]
        
        return prob, acc, df['Close'].iloc[-1]
    except Exception as e: return None, None, str(e)

# --- B. EDZ / 原物料 宏觀雷達 (含準度回測) ---
@st.cache_resource(ttl=43200)
def get_macro_prediction(target_symbol, features_dict, threshold=0.02):
    if not HAS_TENSORFLOW: return None, None
    try:
        # 下載
        tickers = { 'Main': target_symbol }
        tickers.update(features_dict)
        data = yf.download(list(tickers.values()), period="3y", interval="1d", progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close'].copy()
            inv_map = {v: k for k, v in tickers.items()}
            df_close.rename(columns=inv_map, inplace=True)
            df = df_close.copy()
        else: return None, None

        # 特徵工程
        feature_cols = []
        df['Main_Ret'] = df['Main'].pct_change()
        feature_cols.append('Main_Ret')
        
        for name in features_dict.keys():
            df[f'{name}_Ret'] = df[name].pct_change()
            feature_cols.append(f'{name}_Ret')
            
        df['RSI'] = ta.rsi(df['Main'], length=14)
        feature_cols.append('RSI')
        df.dropna(inplace=True)
        
        # 標籤
        days_out = 5
        df['Target'] = ((df['Main'].shift(-days_out) / df['Main'] - 1) > threshold).astype(int)
        df_train = df.iloc[:-days_out].copy()
        
        if len(df_train) < 60: return None, None

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[feature_cols])
        
        X, y = [], []
        lookback = 20
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(df_train['Target'].iloc[i])
            
        X, y = np.array(X), np.array(y)
        
        # 切分測試集
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
            
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(lookback, len(feature_cols))))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # 訓練
        early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stop])
        
        # 計算準度
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        
        # 預測
        last_seq = df[feature_cols].iloc[-lookback:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        
        return prob, acc
    except: return None, None

# --- C. QQQ 通用掃描器 ---
@st.cache_resource(ttl=86400)
def train_universal_scanner():
    if not HAS_TENSORFLOW: return None, None, None
    try:
        df = yf.download("QQQ", period="2y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['Return'] = df['Close'].pct_change()
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['MA20_Dist'] = (df['Close'] - ta.sma(df['Close'], 20)) / ta.sma(df['Close'], 20)
        df.dropna(inplace=True)
        
        df['Target'] = ((df['Close'].shift(-5) / df['Close'] - 1) > 0.02).astype(int)
        df_train = df.iloc[:-5].copy()
        
        features = ['Return', 'RSI', 'RVOL', 'MA20_Dist']
        scaler = StandardScaler()
        X, y = [], []
        for i in range(20, len(df_train)):
            X.append(scaler.fit_transform(df_train[features].iloc[i-20:i+1])[:-1]) 
            y.append(df_train['Target'].iloc[i])
            
        model = Sequential()
        model.add(LSTM(64, input_shape=(20, 4))); model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(np.array(X), np.array(y), epochs=30, verbose=0)
        return model, scaler, features
    except: return None, None, None

def scan_stock(symbol, model, scaler, features):
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if len(df) < 30: return None, None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['Return'] = df['Close'].pct_change()
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['MA20_Dist'] = (df['Close'] - ta.sma(df['Close'], 20)) / ta.sma(df['Close'], 20)
        df.dropna(inplace=True)
        
        last_seq = df[features].iloc[-20:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        return prob, df['Close'].iloc[-1]
    except: return None, None

# ==========================================
# 4. 資料與邏輯處理 (保留原功能)
# ==========================================
def get_real_live_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get('last_price')
        if price is None or np.isnan(price) or float(price) <= 0:
            if symbol.endswith(".TW"): df_rt = yf.download(symbol, period="5d", interval="1m", progress=False)
            elif "-USD" in symbol or "=X" in symbol: df_rt = yf.download(symbol, period="1d", interval="1m", progress=False)
            else: df_rt = yf.download(symbol, period="5d", interval="1m", prepost=True, progress=False)
            if df_rt.empty: return None
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            return float(df_rt['Close'].iloc[-1])
        return float(price)
    except: return None

def get_safe_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        df.index = pd.to_datetime(df.index)
        return df
    except: return None

def clean_text_for_llm(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^\w\s\u4e00-\u9fff.,:;%()\-]', '', text)

def get_news_content(symbol):
    try:
        if "=" in symbol or "^" in symbol: return []
        stock = yf.Ticker(symbol)
        news = stock.news
        if not news: return []
        clean_news = []
        for n in news[:5]:
            title = n.get('title', n.get('content', {}).get('title', ''))
            if title: clean_news.append(clean_text_for_llm(title))
        return clean_news
    except: return []

# 基本面與 FinBERT
@st.cache_data(ttl=86400)
def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol or "-USD" in symbol: return None 
        stock = yf.Ticker(symbol)
        return {
            "pe": stock.info.get('trailingPE', None), 
            "inst": stock.info.get('heldPercentInstitutions', 0),
            "short": stock.info.get('shortPercentOfFloat', 0)
        }
    except: return None

@st.cache_resource
def load_finbert_model():
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
    except: return None

def analyze_sentiment_finbert(symbol):
    if not HAS_TRANSFORMERS: return 0, "套件未安裝", []
    try:
        if "=" in symbol: return 0, "Skip", []
        stock = yf.Ticker(symbol); news_list = stock.news
        if not news_list: return 0, "無新聞", []
        
        classifier = load_finbert_model()
        if not classifier: return 0, "Load Error", []
        
        texts = [clean_text_for_llm(n.get('title','')) for n in news_list[:5] if n.get('title')]
        if not texts: return 0, "No Text", []
        
        results = classifier(texts)
        total = sum([1 if r['label']=='positive' else -1 if r['label']=='negative' else 0 for r in results])
        logs = [f"{r['label']} ({r['score']:.2f}): {texts[i][:30]}..." for i, r in enumerate(results)]
        return total/len(texts), texts[0], logs
    except: return 0, "Error", []

# AI 邏輯與辯論
def analyze_logic_gemini(api_key, symbol, news_titles, tech_ctx, k_pattern, model_name):
    if not HAS_GEMINI: return "Gemini 未安裝", "⚠️", False
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"分析 {symbol}。技術: {tech_ctx}。型態: {k_pattern}。新聞: {news_titles}。給出操作建議。"
        return model.generate_content(prompt).text, "⚡", True
    except Exception as e: return f"Error: {e}", "⚠️", False

def run_ai_debate(api_key, symbol, news_titles, tech_ctx, k_pattern, model_name):
    if not HAS_GEMINI: return "Gemini 未安裝", "⚠️", False, None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"""
        針對 {symbol} 進行多空辯論。
        技術面: {tech_ctx}
        型態: {k_pattern}
        新聞: {news_titles}
        請輸出 JSON: {{ "bull": "多方觀點", "bear": "空方觀點", "judge": "總結裁決" }}
        """
        res = model.generate_content(prompt).text
        if "```json" in res: res = res.replace("```json", "").replace("```", "")
        return json.loads(res).get("judge"), "⚖️", True, json.loads(res)
    except Exception as e: return f"Error: {e}", "⚠️", False, None

# 籌碼與倉位
def analyze_chips_volume(df, inst, short):
    try:
        cmf = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20).iloc[-1]
        msg = "🔴 主力買進" if cmf > 0.05 else "🟢 主力賣出" if cmf < -0.05 else "⚪ 中性"
        return msg, {"cmf": cmf, "inst": inst, "short": short}
    except: return "N/A", None

def calculate_position_size(price, df, capital, risk_pct):
    try:
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
        shares = (capital * (risk_pct/100)) / (2 * atr)
        return f"{int(shares)}股"
    except: return "N/A"

# 技術指標與繪圖
def quick_backtest(df, config, fee=0.0005):
    try:
        close = df['Close']; sigs = pd.Series(0, index=df.index)
        if "RSI" in config['mode']:
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            sigs[rsi < config['entry_rsi']] = 1; sigs[rsi > config['exit_rsi']] = -1
        elif "KD" in config['mode']:
            k = ta.stoch(df['High'], df['Low'], close, k=9, d=3).iloc[:, 0]
            sigs[k < config['entry_k']] = 1; sigs[k > config['exit_k']] = -1
        
        pos=0; ent=0; wins=0; trds=0; rets=[]
        for i in range(len(df)):
            if pos==0 and sigs.iloc[i]==1: pos=1; ent=close.iloc[i]
            elif pos==1 and sigs.iloc[i]==-1:
                pos=0; r = (close.iloc[i]-ent)/ent - fee*2
                rets.append(r); trds+=1; wins += 1 if r>0 else 0
        return sigs, {"Total_Return": sum(rets)*100, "Win_Rate": (wins/trds*100) if trds else 0}
    except: return None, None

def plot_chart(df, config, sigs, show):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    if "RSI" in config['mode']:
        rsi = ta.rsi(df['Close'], length=config.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI"), row=2, col=1)
        fig.add_hline(y=config.get('entry_rsi', 30), line_dash="dash", row=2, col=1)
    
    if show and sigs is not None:
        buy = df[sigs==1]; sell = df[sigs==-1]
        fig.add_trace(go.Scatter(x=buy.index, y=buy['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell.index, y=sell['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', color='red')), row=1, col=1)
    
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
    return fig

# 主分析流程
def analyze_ticker(config, ai_provider, groq_key, gemini_key, model_name, debate):
    symbol = config['symbol']
    df = get_safe_data(symbol)
    if df is None: return None
    
    lp = get_real_live_price(symbol) or df['Close'].iloc[-1]
    
    # 簡單技術訊號
    rsi = ta.rsi(df['Close'], length=14).iloc[-1]
    sig = "BUY" if rsi < config.get('entry_rsi', 30) else "SELL" if rsi > config.get('exit_rsi', 70) else "WAIT"
    
    # AI 分析
    llm_res = "N/A"; debate_res = None; is_llm = False
    news = get_news_content(symbol)
    tech_txt = f"現價 {lp:.2f}, RSI {rsi:.1f}, 訊號 {sig}"
    
    if ai_provider == "Gemini (User Defined)" and gemini_key:
        if debate:
            llm_res, _, is_llm, debate_res = run_ai_debate(gemini_key, symbol, news, tech_txt, "N/A", model_name)
        else:
            llm_res, _, is_llm = analyze_logic_gemini(gemini_key, symbol, news, tech_txt, "N/A", model_name)
            
    fund = get_fundamentals(symbol)
    chip_msg, _ = analyze_chips_volume(df, fund['inst'] if fund else 0, fund['short'] if fund else 0)
    pos_msg = calculate_position_size(lp, df, st.session_state.get('user_capital', 10000), st.session_state.get('user_risk', 1))
    
    return {
        "Symbol": symbol, "Name": config['name'], "Price": lp, "Prev_Close": df['Close'].iloc[-2],
        "Signal": sig, "Action": f"RSI:{rsi:.1f}", "Raw_DF": df, "Strat_Desc": config['mode'],
        "Is_LLM": is_llm, "LLM_Analysis": llm_res, "Debate": debate_res, "Chip": chip_msg, "Position": pos_msg
    }

# ==========================================
# 5. 側邊欄 UI
# ==========================================
with st.sidebar:
    st.header("⚙️ 設定")
    ai_provider = st.selectbox("AI 模型", ["不使用", "Gemini (User Defined)"])
    gemini_key = ""; gemini_model = "models/gemini-2.0-flash"; debate_mode = False
    
    if ai_provider == "Gemini (User Defined)":
        gemini_key = st.text_input("Gemini Key", type="password")
        gemini_model = st.text_input("Model Name", value="models/gemini-2.0-flash")
        debate_mode = st.checkbox("啟用 AI 辯論模式", value=False)

    st.divider()
    
    # ★★★ 全市場掃描區 ★★★
    st.header("⚡ AI 資金流向")
    st.caption("預測 T+5 漲幅 > 2%")
    scan_list = ["AMZN", "NVDA", "AAPL", "MSFT", "GOOGL", "AMD", "TSM", "TSLA", "PLTR", "GC=F", "CL=F"]
    
    if st.button("🚀 掃描全市場"):
        with st.spinner("AI 正在訓練通用腦..."):
            model, scaler, feats = train_universal_scanner()
            if model:
                res = []
                bar = st.progress(0)
                for i, tick in enumerate(scan_list):
                    p, price = scan_stock(tick, model, scaler, feats)
                    if p: res.append((tick, p, price))
                    bar.progress((i+1)/len(scan_list))
                
                res.sort(key=lambda x: x[1], reverse=True)
                bar.empty()
                for tick, p, pr in res:
                    color = "green" if p > 0.6 else "red" if p < 0.4 else "gray"
                    icon = "🔥" if p > 0.6 else "❄️"
                    st.markdown(f"**{tick}**: :{color}[{p*100:.0f}%] ${pr:.1f} {icon}")
            else: st.error("TF Error")

    st.divider()
    st.header("💰 資金管理")
    st.session_state['user_capital'] = st.number_input("本金 (USD)", value=10000)
    st.session_state['user_risk'] = st.number_input("風險 (%)", value=1.0)
    
    st.divider()
    target_key = st.selectbox("選擇標的", list(strategies.keys()), format_func=lambda x: strategies[x]['name'])
    target_config = strategies[target_key]
    
    show_signals = st.checkbox("顯示買賣訊號", value=True)
    st.session_state['tx_fee'] = st.number_input("手續費", value=0.0005)

# ==========================================
# 6. 主畫面 Dashboard
# ==========================================

# ★★★ 儀表板區域 ★★★
c1, c2 = st.columns(2)

# EDZ 風險雷達
with c1.container(border=True):
    st.subheader("🐻 EDZ / 原物料風險雷達 (T+5)")
    st.caption("因子: 利率 + 銅價 + 中國 + 美元")
    target_risk = st.selectbox("選擇監測對象", ["EDZ", "GC=F", "CL=F", "HG=F"])
    
    if st.button("檢測風險 / 趨勢"):
        with st.spinner("AI 分析宏觀數據 (含準度回測)..."):
            # 設定對應的特徵因子
            feat_map = { 'China': "FXI", 'DXY': "DX-Y.NYB", 'Rates': "^TNX", 'Copper': "HG=F" }
            # 修正：接收兩個回傳值 (prob, acc)
            prob, acc = get_macro_prediction(target_risk, feat_map)
            
        if prob is not None:
            conf = prob if prob > 0.5 else 1 - prob
            
            # 新增：顯示準度
            st.metric("模型歷史準度", f"{acc*100:.1f}%", delta="可信" if acc > 0.6 else "普通")
            
            if prob > 0.6:
                st.error(f"📈 看漲訊號 (信心 {conf*100:.1f}%)")
                st.markdown(f"**{target_risk}** 趨勢向上。若為 EDZ 則代表市場風險高。")
            elif prob < 0.4:
                st.success(f"📉 看跌訊號 (信心 {conf*100:.1f}%)")
                st.markdown(f"**{target_risk}** 趨勢向下/盤整。")
            else:
                st.warning(f"💤 盤整震盪 (信心 {conf*100:.1f}%)")
        else: st.info("需 TensorFlow")

# TSM 波段顧問
with c2.container(border=True):
    st.subheader("📈 TSM 波段顧問 (T+5)")
    st.caption("因子: 夜盤 EWT + 利率 + 供應鏈")
    
    if st.button("AI 判讀 TSM"):
        with st.spinner("AI 運算中 (含準度回測)..."):
            # 呼叫修正後的函數，接收三個返回值
            prob, acc, price = get_tsm_swing_prediction("TSM")
            
        if prob:
            conf = prob if prob > 0.5 else 1 - prob
            
            # 使用三欄位顯示：現價、準度、建議
            m1, m2, m3 = st.columns(3)
            m1.metric("TSM 現價", f"${price:.2f}")
            
            # 顯示準度
            m2.metric("回測準度", f"{acc*100:.1f}%", delta="表現優異" if acc>0.58 else "表現尚可")

            # 顯示建議
            if prob > 0.6:
                m3.metric("AI 建議", "看漲 🚀")
                st.success(f"信心度 {conf*100:.1f}%：預期 5 天後漲幅 > 2%。**建議拉回佈局。**")
            elif prob < 0.4:
                m3.metric("AI 建議", "看跌/盤 📉")
                st.error(f"信心度 {conf*100:.1f}%：上漲空間有限。**建議獲利了結或觀望。**")
            else:
                m3.metric("AI 建議", "震盪 ⚖️")
                st.info(f"信心度 {conf*100:.1f}%：多空不明，建議空手。")
        else: st.info("需 TensorFlow")

st.divider()

# ★★★ 單股深度分析 ★★★
if target_key:
    st.subheader(f"📊 {target_config['name']} 深度分析")
    
    row = analyze_ticker(target_config, ai_provider, "", gemini_key, gemini_model, debate_mode)
    if row:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.metric("價格", f"${row['Price']:.2f}", f"{row['Price']-row['Prev_Close']:.2f}")
            c2.caption(f"策略: {row['Strat_Desc']}")
            st.markdown(f"#### {row['Signal']} | {row['Action']}")
            st.warning(f"建議倉位: {row['Position']}")
            
            if row.get('Debate'):
                with st.expander("⚖️ AI 辯論", expanded=True):
                    st.write(f"多方: {row['Debate'].get('bull')}")
                    st.write(f"空方: {row['Debate'].get('bear')}")
                    st.success(f"裁決: {row['Debate'].get('judge')}")
            elif row['Is_LLM']:
                st.info(f"AI 分析: {row['LLM_Analysis']}")
            
            if row['Raw_DF'] is not None:
                sigs, perf = quick_backtest(row['Raw_DF'], target_config, st.session_state['tx_fee'])
                st.plotly_chart(plot_chart(row['Raw_DF'], target_config, sigs, show_signals), use_container_width=True)
                if perf: st.caption(f"回測: {perf['Total_Return']:.1f}% (勝率 {perf['Win_Rate']:.0f}%)")
