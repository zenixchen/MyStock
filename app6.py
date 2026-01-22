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
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except: HAS_GEMINI = False

# ==========================================
# 2. 頁面設定
# ==========================================
st.set_page_config(
    page_title="2026 量化戰情室 (Dual-Core v23.0)",
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
# ★★★ 3. AI 模型核心 (T+5 與 T+3 雙模並存) ★★★
# ==========================================
# ==========================================
# ★★★ 1. 舊版模型：T+5 波段 (趨勢 / 誠實驗證版) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_tsm_swing_prediction():
    if not HAS_TENSORFLOW: return None, None, "TF缺"
    try:
        # --- 內建手動計算函式 (避開 pandas_ta 錯誤) ---
        def manual_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        def manual_sma(series, period=20):
            return series.rolling(window=period).mean()

        # 1. 定義原始四大因子
        tickers = { 'Main': 'TSM', 'Night': "EWT", 'Rate': "^TNX", 'AI': 'NVDA' }
        data = yf.download(list(tickers.values()), period="3y", interval="1d", progress=False, auto_adjust=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close'].copy()
            df = pd.DataFrame()
            for key, symbol in tickers.items():
                if symbol in df_close.columns:
                    df[f'{key}_Close'] = df_close[symbol]
                else:
                    # 容錯
                    if len(tickers) == 1: df[f'{key}_Close'] = df_close
                    else: df[f'{key}_Close'] = 0 
        else:
             return None, None, "DataFmt"

        df.ffill(inplace=True); df.bfill(inplace=True); df.fillna(0, inplace=True)

        # 2. 特徵工程 (手動計算)
        df['Main_Ret'] = df['Main_Close'].pct_change()
        df['Night_Ret'] = df['Night_Close'].pct_change()
        df['Rate_Chg'] = df['Rate_Close'].pct_change()
        df['AI_Ret'] = df['AI_Close'].pct_change()
        
        # 手算 RSI 和 Bias
        df['RSI'] = manual_rsi(df['Main_Close'], period=14)
        sma_20 = manual_sma(df['Main_Close'], period=20)
        df['Bias'] = (df['Main_Close'] - sma_20) / sma_20
        
        df.dropna(inplace=True)

        # T+5 標籤
        days_out = 5; threshold = 0.02
        df['Target'] = ((df['Main_Close'].shift(-days_out) / df['Main_Close'] - 1) > threshold).astype(int)
        
        # 移除最後 5 天 (無答案) 用於訓練
        df_train = df.iloc[:-days_out].copy()
        features = ['Main_Ret', 'Night_Ret', 'Rate_Chg', 'AI_Ret', 'RSI', 'Bias']
        
        # 3. 準備數據
        scaler = StandardScaler()
        scaler.fit(df_train[features]) # Fit 全體
        scaled_data = scaler.transform(df_train[features])
        
        X, y = [], []
        lookback = 20
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(df_train['Target'].iloc[i])
        
        X, y = np.array(X), np.array(y)
        
        # ★★★ 關鍵修正：切分驗證集 (80/20) ★★★
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        # 4. 訓練 LSTM
        from tensorflow.keras.layers import Input
        model = Sequential()
        model.add(Input(shape=(lookback, len(features))))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        # 只用 Train 訓練
        model.fit(X_train, y_train, epochs=40, batch_size=16, verbose=0)
        
        # ★★★ 用 Validation 評估 (誠實分數) ★★★
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        
        # 5. 預測未來 (用最新數據)
        last_seq = df[features].iloc[-lookback:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        
        return prob, acc, df['Main_Close'].iloc[-1]
    except Exception as e: return None, None, str(e)


# ==========================================
# ★★★ 2. 新版模型：T+3 極速 (短線 / 誠實驗證版) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_tsm_short_prediction():
    if not HAS_TENSORFLOW: return None, None
    try:
        # --- 內建手動計算函式 ---
        def manual_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        def manual_macd(series, fast=12, slow=26, signal=9):
            exp1 = series.ewm(span=fast, adjust=False).mean()
            exp2 = series.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            return macd

        # 1. 五大護法因子
        tickers = ["TSM", "^SOX", "NVDA", "^TNX", "^VIX"]
        data = yf.download(tickers, period="2y", interval="1d", progress=False, auto_adjust=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close'].copy()
            try: df_close = df_close[tickers] 
            except: pass
            df = df_close.copy()
        else:
            df = data['Close'].copy()

        df.ffill(inplace=True); df.dropna(inplace=True)

        # 2. 特徵工程 (手動計算)
        feat_df = pd.DataFrame()
        feat_df['TSM_Ret'] = df['TSM'].pct_change()
        feat_df['SOX_Ret'] = df['^SOX'].pct_change()
        feat_df['NVDA_Ret'] = df['NVDA'].pct_change()
        feat_df['TSM_RSI'] = manual_rsi(df['TSM'], period=14)
        feat_df['TSM_MACD'] = manual_macd(df['TSM'])
        feat_df['VIX'] = df['^VIX']
        feat_df['TNX_Chg'] = df['^TNX'].pct_change()
        
        feat_df.dropna(inplace=True)
        feature_cols = ['TSM_Ret', 'SOX_Ret', 'NVDA_Ret', 'TSM_RSI', 'TSM_MACD', 'VIX', 'TNX_Chg']
        
        # T+3 標籤
        future_ret = df['TSM'].shift(-3) / df['TSM'] - 1
        feat_df['Target'] = (future_ret > 0.015).astype(int)
        
        df_train = feat_df.iloc[:-3].copy()
        
        # 3. 準備數據
        scaler = StandardScaler()
        scaler.fit(df_train[feature_cols])
        scaled_data = scaler.transform(df_train[feature_cols])
        
        X, y = [], []
        lookback = 30
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(df_train['Target'].iloc[i])
            
        X, y = np.array(X), np.array(y)

        # ★★★ 關鍵修正：切分驗證集 (80/20) ★★★
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        # 4. 訓練雙向 LSTM
        from tensorflow.keras.layers import Input, Bidirectional
        model = Sequential()
        model.add(Input(shape=(lookback, len(feature_cols))))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        # 只用 Train 訓練
        model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0)
        
        # ★★★ 用 Validation 評估 (誠實分數) ★★★
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        
        # 5. 預測未來
        latest_seq = feat_df[feature_cols].iloc[-lookback:].values
        latest_scaled = scaler.transform(latest_seq)
        latest_input = latest_scaled.reshape(1, lookback, len(feature_cols))
        
        prob = model.predict(latest_input, verbose=0)[0][0]
        
        return prob, acc
    except Exception as e:
        print(f"Error in Short Model: {e}")
        return None, None

# --- C. EDZ/Macro ---
@st.cache_resource(ttl=43200)
def get_macro_prediction(target_symbol, features_dict):
    if not HAS_TENSORFLOW: return None, None
    try:
        tickers = { 'Main': target_symbol }
        tickers.update(features_dict)
        data = yf.download(list(tickers.values()), period="3y", interval="1d", progress=False, auto_adjust=False)
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
        
        model = Sequential()
        model.add(Input(shape=(20, len(feat_cols))))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3)); model.add(LSTM(64)); model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        early = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
        model.fit(X, y, epochs=40, batch_size=32, verbose=0)
        
        loss, acc = model.evaluate(X[int(len(X)*0.8):], y[int(len(X)*0.8):], verbose=0)
        last_seq = df[feat_cols].iloc[-20:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        return prob, acc
    except: return None, None

# --- D. QQQ Scanner ---
@st.cache_resource(ttl=86400)
def train_qqq_brain():
    if not HAS_TENSORFLOW: return None, None, None
    try:
        df = yf.download("QQQ", period="5y", interval="1d", progress=False, auto_adjust=False)
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
        model.add(Input(shape=(20, 5))); model.add(LSTM(64)); model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(np.array(X), np.array(y), epochs=40, verbose=0)
        return model, scaler, features
    except: return None, None, None

def scan_tech_stock(symbol, model, scaler, features):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=False)
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
        return prob, 0.6, df['Close'].iloc[-1]
    except: return None, None, 0

# ==========================================
# 4. 傳統策略分析
# ==========================================
def get_safe_data(ticker):
    try:
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
        try: info = s.info
        except: return None
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

def get_news(symbol):
    try:
        search_query = symbol
        if ".TW" in symbol: search_query = symbol.replace(".TW", " TW stock")
        else: search_query = f"{symbol} stock news"
        url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            news_items = []
            for item in root.findall('.//item')[:5]:
                title = item.find('title').text
                if len(title) > 10: news_items.append(clean_text_for_llm(title))
            return news_items
        return []
    except Exception as e: return [f"News Error: {str(e)}"]

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
        risk_money = capital * risk_per_trade
        shares_by_risk = risk_money / stop_loss_dist
        if win_rate < 0.45: return "⛔ 雖有買訊但勝率過低，建議觀望", 0
        shares = int(shares_by_risk)
        cost = shares * price
        msg = f"🚀 建議買進 {shares} 股 (約 ${cost:.0f})"
        return msg, shares
    except: return "計算失敗", 0

def identify_k_pattern(df):
    try:
        if len(df) < 3: return "N/A"
        last_3 = df.iloc[-3:].copy()
        c, o = last_3['Close'].values, last_3['Open'].values
        c2, o2 = c[2], o[2]
        c1, o1 = c[1], o[1]
        c0, o0 = c[0], o[0]
        body2, body1 = abs(c2 - o2), abs(c1 - o1)
        if (c2 > o2) and (c1 < o1) and (c2 > o1) and (o2 < c1): return "🔥 多頭吞噬"
        if (c2 < o2) and (c1 > o1) and (c2 < o1) and (o2 > c1): return "💀 空頭吞噬"
        if (c0 < o0) and (abs(c0-o0) > body1 * 2) and (c2 > o2) and (c1 < c0 and c1 < c2): return "🌅 晨星轉折"
        if body2 <= (last_3['High'].values[2] - last_3['Low'].values[2]) * 0.1: return "✝️ 十字線"
        return "一般震盪"
    except: return "N/A"

def quick_backtest(df, config, fee=0.0005):
    try:
        close = df['Close']; sigs = pd.Series(0, index=df.index)
        mode = config['mode']
        
        if mode == "RSI_MA":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            ma_exit = ta.sma(close, length=config['exit_ma'])
            sigs[rsi < config['entry_rsi']] = 1
            sigs[close > ma_exit] = -1
        elif mode == "RSI_RSI":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            if config.get('ma_trend', 0) > 0:
                ma_trend = ta.ema(close, length=config['ma_trend'])
                sigs[(rsi < config['entry_rsi']) & (close > ma_trend)] = 1
            else:
                sigs[rsi < config['entry_rsi']] = 1
            sigs[rsi > config['exit_rsi']] = -1
        elif mode == "MA_CROSS":
            f = ta.sma(close, config['fast_ma']); s = ta.sma(close, config['slow_ma'])
            sigs[(f > s) & (f.shift(1) <= s.shift(1))] = 1
            sigs[(f < s) & (f.shift(1) >= s.shift(1))] = -1
        elif mode == "FUSION":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            ma = ta.ema(close, length=config.get('ma_trend', 200))
            sigs[(close > ma) & (rsi < config['entry_rsi'])] = 1
            sigs[rsi > config['exit_rsi']] = -1
        elif mode == "BOLL_RSI":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            bb = ta.bbands(close, length=20, std=2)
            lower = bb.iloc[:, 0]; upper = bb.iloc[:, 2]
            sigs[(close < lower) & (rsi < config['entry_rsi'])] = 1
            sigs[close > upper] = -1
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
        stats = { "Total_Return": sum(rets)*100, "Win_Rate": win_rate * 100, "Raw_Win_Rate": win_rate, "Trades": trds }
        return sigs.iloc[-1], stats, sigs
    except Exception as e: return 0, None, None

def plot_chart(df, config, sigs):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02, specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    if config.get('ma_trend', 0) > 0:
        ma = ta.ema(df['Close'], length=config['ma_trend'])
        fig.add_trace(go.Scatter(x=df.index, y=ma, name=f"EMA {config['ma_trend']}", line=dict(color='purple')), row=1, col=1)
    
    # 動態 CMF
    cmf_len = config.get('cmf_len', 20)
    cmf = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=cmf_len)
    
    if "RSI" in config['mode']:
        rsi = ta.rsi(df['Close'], length=config.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color='#b39ddb')), row=2, col=1)
        fig.add_hline(y=config.get('entry_rsi', 30), line_dash="dash", row=2, col=1)
    elif "KD" in config['mode']:
        k = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
        fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:, 0], name="K", line=dict(color='yellow')), row=2, col=1)
        fig.add_hline(y=config.get('entry_k', 20), line_dash="dash", row=2, col=1)
    
    colors = ['#089981' if v >= 0 else '#f23645' for v in cmf]
    fig.add_trace(go.Bar(x=df.index, y=cmf, name=f'CMF({cmf_len})', marker_color=colors, opacity=0.5), row=3, col=1, secondary_y=False)
    
    if sigs is not None:
        buy = df[sigs==1]; sell = df[sigs==-1]
        fig.add_trace(go.Scatter(x=buy.index, y=buy['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell.index, y=sell['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell'), row=1, col=1)
    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    return fig

def get_strategy_desc(cfg, df=None):
    mode = cfg['mode']
    desc = mode
    if mode == "RSI_RSI": desc = f"RSI({cfg.get('rsi_len',14)}) < {cfg['entry_rsi']} (需站上MA{cfg.get('ma_trend',0)})"
    return desc

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
    st.caption("神經網路模型 (LSTM) | T+5 & T+3 雙模並存")
    
    tab1, tab2, tab3 = st.tabs(["📈 TSM 雙核心波段", "🐻 EDZ / 宏觀雷達", "⚡ QQQ 科技股通用腦"])
    
    # === Tab 1: TSM ===
    with tab1:
        st.subheader("TSM 雙核心波段顧問")
        
        # 按鈕：一次觸發兩個模型
        if st.button("🚀 啟動雙模型分析 (T+3 & T+5)", key="btn_tsm") or 'tsm_result_v2' in st.session_state:
            
            if 'tsm_result_v2' not in st.session_state:
                with st.spinner("AI 正在進行雙重驗證..."):
                    # 1. 呼叫舊模型 (T+5)
                    prob_long, acc_long, price = get_tsm_swing_prediction()
                    # 2. 呼叫新模型 (T+3)
                    prob_short, acc_short = get_tsm_short_prediction()
                    
                    st.session_state['tsm_result_v2'] = (prob_long, acc_long, prob_short, acc_short, price)
            
            # 取出結果
            p_long, a_long, p_short, a_short, price = st.session_state['tsm_result_v2']
            
            # --- 顯示介面 ---
            st.metric("TSM 即時價格", f"${price:.2f}")
            st.divider()

            col1, col2 = st.columns(2)
            
            # 左邊：T+5 (趨勢)
            with col1:
                st.info("🔭 T+5 趨勢模型 (舊版)")
                if p_long is not None:
                    st.write(f"準確率: `{a_long*100:.1f}%`")
                    if p_long > 0.6: st.success(f"看漲 (機率 {p_long*100:.0f}%)")
                    elif p_long < 0.4: st.error(f"看跌 (機率 {p_long*100:.0f}%)")
                    else: st.warning(f"震盪 (機率 {p_long*100:.0f}%)")
                else:
                    st.error("模型載入失敗")

            # 右邊：T+3 (短線)
            with col2:
                st.info("⚡ T+3 極速模型 (新版)")
                if p_short is not None:
                    st.write(f"準確率: `{a_short*100:.1f}%`")
                    if p_short > 0.5: st.success(f"短多 (機率 {p_short*100:.0f}%)")
                    elif p_short < 0.4: st.error(f"短空 (機率 {p_short*100:.0f}%)")
                    else: st.warning(f"盤整 (機率 {p_short*100:.0f}%)")
                else:
                    st.error("模型載入失敗")

            # --- 綜合建議 (主從架構優化版) ---
            st.subheader("🤖 AI 總結")
            
            # 防呆：確保數值存在
            if p_long is not None and p_short is not None:
                
                # ★★★ 核心邏輯：T+5 (p_long) 權重 80%，T+3 (p_short) 權重 20% ★★★
                
                # 情況 1: 主帥 (T+5) 看漲
                if p_long > 0.6: 
                    if p_short > 0.5:
                        st.success("🔥🔥 強力進攻 (主升段確認！T+5趨勢向上 + T+3短線點火)")
                        final_dir = "Bull_Strong"
                        final_conf = 0.9  # 信心爆棚
                    else:
                        st.info("📈 逢低佈局 (趨勢向上，但短線有雜訊。建議分批買進，不要追高)")
                        final_dir = "Bull_Dip"
                        final_conf = 0.7
                
                # 情況 2: 主帥 (T+5) 看跌/震盪
                elif p_long < 0.4:
                    if p_short > 0.6:
                        st.warning("⚠️ 短線反彈逃命波 (T+5看空，T+3看反彈。建議趁反彈減碼)")
                        final_dir = "Bear_Bounce"
                        final_conf = 0.6
                    else:
                        st.error("❄️❄️ 全面撤退 (長短線共振看空，現金為王)")
                        final_dir = "Bear_Strong"
                        final_conf = 0.9
                
                # 情況 3: 主帥看不懂 (盤整)
                else:
                    st.write("💤 趨勢不明，依照短線 T+3 輕倉操作")
                    if p_short > 0.6:
                        st.success("⚡ 短線嘗試做多 (快進快出)")
                        final_dir = "Neutral_Bull"
                        final_conf = 0.55
                    else:
                        st.warning("💤 觀望為主")
                        final_dir = "Neutral"
                        final_conf = 0.5

                # 顯示信心分數 (加權計算)
                # T+5 佔 70%, T+3 佔 30%
                weighted_conf = (p_long * 0.7) + (p_short * 0.3)
                st.caption(f"綜合信心指數: {weighted_conf*100:.1f}% (T+5權重70% / T+3權重30%)")

                # 存檔按鈕
                if st.button("📸 記錄綜合預測", key="save_tsm_dual"):
                    if save_prediction("TSM", final_dir, weighted_conf, price):
                        st.success("✅ 已記錄！")
                    else: st.warning("⚠️ 今天已存過")

    # === Tab 2: Macro ===
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
                elif prob < 0.4:
                    c3.metric("趨勢方向", "📉 向下", delta=f"信心 {conf*100:.1f}%", delta_color="inverse")
                else:
                    c3.metric("趨勢方向", "💤 震盪", delta=f"信心 {conf*100:.1f}%", delta_color="off")
                
                if st.button("📸 記錄預測", key=f"save_{target_risk}"):
                    save_prediction(target_risk, direction, conf, price)

    # === Tab 3: QQQ ===
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
                    mark = "💎" if p > 0.6 else "🛡️"
                    direction = "📈" if p > 0.6 else "📉"
                    col1, col2, col3, col4 = st.columns([2, 2, 3, 2])
                    col1.markdown(f"**{tick}** (${pr:.1f})")
                    col2.markdown(f"{direction} ({p*100:.0f}%)")
                    if col4.button("💾 存", key=f"save_{tick}"):
                        save_prediction(tick, "Bull" if p>0.5 else "Bear", p if p>0.5 else 1-p, pr)

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
