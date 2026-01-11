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
    from groq import Groq
    HAS_GROQ = True
except: HAS_GROQ = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except: HAS_GEMINI = False

# ==========================================
# 2. 頁面設定
# ==========================================
st.set_page_config(
    page_title="2026 量化戰情室 (Ultimate v16.0)",
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
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { background-color: #1c202a; border-radius: 4px 4px 0 0; color: #d1d4dc; }
        .stTabs [aria-selected="true"] { background-color: #2962ff; color: white; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# ★★★ 3. AI 模型核心 ★★★
# ==========================================

# --- Module A: TSM 專用波段 AI ---
@st.cache_resource(ttl=43200)
def get_tsm_swing_prediction():
    if not HAS_TENSORFLOW: return None, None, "TF缺"
    try:
        tickers = { 'Main': 'TSM', 'Night': "EWT", 'Rate': "^TNX", 'AI': 'NVDA' }
        data = yf.download(list(tickers.values()), period="2y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close'].copy()
            inv_map = {v: k for k, v in tickers.items()}
            df_close.rename(columns=inv_map, inplace=True)
            df = pd.DataFrame()
            for col in tickers.keys(): df[f'{col}_Close'] = df_close[col]
        else: return None, None, "DataErr"

        df['Main_Ret'] = df['Main_Close'].pct_change()
        df['Night_Ret'] = df['Night_Close'].pct_change()
        df['Rate_Chg'] = df['Rate_Close'].pct_change()
        df['AI_Ret'] = df['AI_Close'].pct_change()
        df['RSI'] = ta.rsi(df['Main_Close'], length=14)
        df['Bias'] = (df['Main_Close'] - ta.sma(df['Main_Close'], 20)) / ta.sma(df['Main_Close'], 20)
        df.dropna(inplace=True)

        days_out = 5; threshold = 0.02
        df['Target'] = ((df['Main_Close'].shift(-days_out) / df['Main_Close'] - 1) > threshold).astype(int)
        df_train = df.iloc[:-days_out].copy()
        
        features = ['Main_Ret', 'Night_Ret', 'Rate_Chg', 'AI_Ret', 'RSI', 'Bias']
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[features])
        
        X, y = [], []
        lookback = 20
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(df_train['Target'].iloc[i])
        
        X, y = np.array(X), np.array(y)
        split = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
        
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stop])
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        last_seq = df[features].iloc[-lookback:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        return prob, acc, df['Main_Close'].iloc[-1]
    except Exception as e: return None, None, str(e)

# --- Module B: EDZ / 宏觀風險 AI ---
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
        df_train = df.iloc[:-days_out].copy()
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[feat_cols])
        
        X, y = [], []
        lookback = 20
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(df_train['Target'].iloc[i])
        
        X, y = np.array(X), np.array(y)
        split = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
            
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(lookback, len(feat_cols))))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stop])
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        last_seq = df[feat_cols].iloc[-lookback:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        return prob, acc
    except: return None, None

# --- Module C: QQQ 通用腦 (5年版) ---
@st.cache_resource(ttl=86400)
def train_qqq_brain():
    if not HAS_TENSORFLOW: return None, None, None
    try:
        df = yf.download("QQQ", period="5y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
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
# 4. 傳統策略分析 (功能模組 - 修復版)
# ==========================================
def get_safe_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except: return None

# ★★★ 財報基本面獲取 ★★★
def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol: return None
        s = yf.Ticker(symbol)
        info = s.info
        return {
            "pe": info.get('trailingPE', None),
            "fwd_pe": info.get('forwardPE', None),
            "peg": info.get('pegRatio', None),
            "inst": info.get('heldPercentInstitutions', 0),
            "short": info.get('shortPercentOfFloat', 0),
            "margin": info.get('grossMargins', 0),
            "eps": info.get('trailingEps', None)
        }
    except: return None

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

def clean_text_for_llm(text): return re.sub(r'[^\w\s\u4e00-\u9fff.,:;%()\-]', '', str(text))

def get_news(symbol):
    try:
        if "=" in symbol or "^" in symbol: return []
        news = yf.Ticker(symbol).news
        return [clean_text_for_llm(n['title']) for n in news[:3]]
    except: return []

# ★★★ 凱利公式 + 方向判斷 ★★★
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

# ★★★ Gemini 大腦 (含財報分析) ★★★
def analyze_logic_gemini_full(api_key, symbol, news, tech_txt, k_pattern, model_name, user_input=""):
    if not HAS_GEMINI: return "No Gemini", "⚠️", False
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        base_prompt = f"""
        華爾街資深操盤手分析 {symbol}。
        【技術面】：{tech_txt}
        【K線型態】：{k_pattern}
        【新聞焦點】：{news}
        """
        
        if user_input:
            base_prompt += f"\n【用戶補充財報/筆記】：{user_input}\n請針對用戶補充的內容進行深度解讀，並結合技術面給出建議。"
        else:
            base_prompt += "\n請給出：1.趨勢判斷 2.操作建議 3.風險提示"
            
        response = model.generate_content(base_prompt)
        return response.text, "🧠", True
    except Exception as e: return str(e), "⚠️", False

def identify_k_pattern(df):
    try:
        c, o = df['Close'].iloc[-1], df['Open'].iloc[-1]
        c_prev, o_prev = df['Close'].iloc[-2], df['Open'].iloc[-2]
        pat = "一般震盪"
        if c > o and c_prev < o_prev and c > o_prev and o < c_prev: pat = "🔥 多頭吞噬"
        elif c < o and c_prev > o_prev and c < o_prev and o > c_prev: pat = "💀 空頭吞噬"
        return pat
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
                pos = 0; 
                r = (close.iloc[i] - ent) / ent - (fee * 2)
                rets.append(r); trds += 1
                if r > 0: wins += 1
        
        win_rate = float(wins / trds) if trds > 0 else 0.0
        last_sig = sigs.iloc[-1]
        
        return last_sig, {"Total_Return": sum(rets)*100, "Win_Rate": win_rate * 100, "Raw_Win_Rate": win_rate}, sigs
    except Exception as e: 
        return 0, None, None

# ★★★ 繪圖函數：雙軸籌碼圖 (CMF + OBV) ★★★
def plot_chart(df, config, sigs):
    # 創建 3 行的子圖
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.02,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]] # 第三行開啟雙軸
    )
    
    # 1. K線圖
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    if config.get('ma_trend', 0) > 0:
        ma = ta.ema(df['Close'], length=config['ma_trend'])
        fig.add_trace(go.Scatter(x=df.index, y=ma, name=f"EMA {config['ma_trend']}", line=dict(color='purple')), row=1, col=1)

    # 2. 技術指標 (RSI/KD)
    if "RSI" in config['mode']:
        rsi = ta.rsi(df['Close'], length=config.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color='#b39ddb')), row=2, col=1)
        fig.add_hline(y=config.get('entry_rsi', 30), line_dash="dash", row=2, col=1)
    elif "KD" in config['mode']:
        k = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
        fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:, 0], name="K", line=dict(color='yellow')), row=2, col=1)
        fig.add_hline(y=config.get('entry_k', 20), line_dash="dash", row=2, col=1)

    # 3. 籌碼面 (CMF + OBV 雙軸)
    # CMF (柱狀圖) - 左軸
    cmf = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    colors = ['#089981' if v >= 0 else '#f23645' for v in cmf]
    fig.add_trace(go.Bar(x=df.index, y=cmf, name='CMF', marker_color=colors, opacity=0.5), row=3, col=1, secondary_y=False)
    
    # OBV (線圖) - 右軸
    obv = ta.obv(df['Close'], df['Volume'])
    fig.add_trace(go.Scatter(x=df.index, y=obv, name='OBV', line=dict(color='cyan', width=1)), row=3, col=1, secondary_y=True)

    # 4. 買賣訊號點
    if sigs is not None:
        buy = df[sigs==1]; sell = df[sigs==-1]
        fig.add_trace(go.Scatter(x=buy.index, y=buy['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell.index, y=sell['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell'), row=1, col=1)
    
    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    return fig

def get_strategy_desc(cfg):
    mode = cfg['mode']
    if mode == "RSI_RSI": return f"RSI 區間 (買 < {cfg['entry_rsi']} / 賣 > {cfg['exit_rsi']})"
    elif mode == "RSI_MA": return f"RSI + 均線 (RSI < {cfg['entry_rsi']} 買 / 破 MA{cfg['exit_ma']} 賣)"
    elif mode == "KD": return f"KD 隨機指標 (K < {cfg['entry_k']} 買 / K > {cfg['exit_k']} 賣)"
    elif mode == "MA_CROSS": return f"均線交叉 (MA{cfg['fast_ma']} 穿過 MA{cfg['slow_ma']})"
    elif mode == "FUSION": return f"趨勢 + RSI (站上 EMA{cfg['ma_trend']} 且 RSI < {cfg['entry_rsi']})"
    elif mode == "BOLL_RSI": return f"布林通道 + RSI (破下軌且 RSI < {cfg['entry_rsi']})"
    return mode

# ==========================================
# 5. 側邊欄與頁面配置
# ==========================================
st.sidebar.title("🚀 戰情室導航")
app_mode = st.sidebar.radio("選擇功能模組：", ["🤖 AI 深度學習實驗室", "📊 策略分析工具 (單股)"])

st.sidebar.divider()
st.sidebar.header("⚙️ 全域設定")
ai_provider = st.sidebar.selectbox("AI 語言模型", ["不使用", "Gemini (User Defined)"])
gemini_key = ""; gemini_model = "models/gemini-2.0-flash"

if ai_provider == "Gemini (User Defined)":
    gemini_key = st.sidebar.text_input("Gemini Key", type="password")
    gemini_model = st.sidebar.text_input("Model Name", value="models/gemini-2.0-flash")

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
    st.caption("神經網路模型 (LSTM) | T+5 波段預測")
    
    tab1, tab2, tab3 = st.tabs(["📈 TSM 專用波段", "🐻 EDZ / 宏觀雷達", "⚡ QQQ 科技股通用腦"])
    
    with tab1:
        st.subheader("TSM 專屬波段顧問 (T+5)")
        if st.button("開始分析 TSM", key="btn_tsm"):
            with st.spinner("AI 正在運算..."):
                prob, acc, price = get_tsm_swing_prediction()
            if prob is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("TSM 現價", f"${price:.2f}")
                c2.metric("模型準度", f"{acc*100:.1f}%", delta="可信" if acc>0.58 else "普通")
                conf = prob if prob > 0.5 else 1 - prob
                if prob > 0.6:
                    c3.metric("AI 建議", "🚀 看漲", delta=f"信心 {conf*100:.1f}%")
                elif prob < 0.4:
                    c3.metric("AI 建議", "📉 看跌/盤", delta=f"信心 {conf*100:.1f}%", delta_color="inverse")
                else:
                    c3.metric("AI 建議", "⚖️ 震盪")
            else: st.error("TF Error")

    with tab2:
        st.subheader("全球風險雷達")
        target_risk = st.selectbox("選擇監測對象", ["EDZ", "GC=F", "CL=F", "HG=F"])
        if st.button(f"分析 {target_risk}", key="btn_macro"):
            with st.spinner("AI 分析宏觀數據..."):
                feat_map = { 'China': "FXI", 'DXY': "DX-Y.NYB", 'Rates': "^TNX", 'Copper': "HG=F" }
                prob, acc = get_macro_prediction(target_risk, feat_map)
            if prob is not None:
                c1, c2 = st.columns(2)
                c1.metric("模型準度", f"{acc*100:.1f}%")
                conf = prob if prob > 0.5 else 1 - prob
                if prob > 0.6:
                    c2.metric("趨勢方向", "📈 向上", delta=f"信心 {conf*100:.1f}%")
                    if target_risk == "EDZ": st.error("⚠️ 市場避險情緒高漲！")
                elif prob < 0.4:
                    c2.metric("趨勢方向", "📉 向下", delta=f"信心 {conf*100:.1f}%", delta_color="inverse")
                else:
                    c2.metric("趨勢方向", "💤 震盪")

    with tab3:
        st.subheader("QQQ 科技股掃描器")
        tech_list = ["NVDA", "AMD", "AMZN", "MSFT", "GOOGL", "META", "TSLA", "AVGO", "PLTR"]
        if st.button("🚀 掃描科技巨頭", key="btn_scan"):
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
                    for tick, p, acc, pr in res:
                        mark = "💎" if p > 0.6 and acc > 0.55 else "🛡️" if p < 0.4 and acc > 0.55 else "⚠️"
                        direction = "📈 看漲" if p > 0.6 else "📉 看跌" if p < 0.4 else "💤 盤整"
                        color_str = "green" if p > 0.6 else "red" if p < 0.4 else "gray"
                        with st.container(border=True):
                            c1, c2, c3 = st.columns([2, 3, 3])
                            c1.markdown(f"**{tick}** (${pr:.1f})")
                            c2.markdown(f":{color_str}[{direction} ({p*100:.0f}%)]")
                            c3.caption(f"準度: {acc*100:.0f}% {mark}")

# ------------------------------------------
# Mode 2: 策略分析工具 (單股)
# ------------------------------------------
elif app_mode == "📊 策略分析工具 (單股)":
    st.header("📊 單股策略分析")
    
    # 全配策略清單
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
    
    target_key = st.selectbox("選擇標的", list(strategies.keys()), format_func=lambda x: strategies[x]['name'])
    cfg = strategies[target_key]
    
    # 1. 抓取數據與即時價格
    df = get_safe_data(cfg['symbol'])
    lp = get_real_live_price(cfg['symbol'])
    
    if df is not None and lp:
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else lp
        chg = lp - prev_close
        pct_chg = (chg / prev_close) * 100
        
        # 2. 執行回測
        current_sig, perf, sigs = quick_backtest(df, cfg)
        win_rate = perf['Raw_Win_Rate'] if perf else 0
        
        # 3. 凱利公式
        kelly_msg, kelly_shares = calculate_kelly_position(df, user_capital, win_rate, user_risk/100, current_sig)
        
        # 4. K線與訊號
        k_pat = identify_k_pattern(df)
        rsi_val = ta.rsi(df['Close'], 14).iloc[-1]
        
        # 5. 基本面與籌碼數據
        fund = get_fundamentals(cfg['symbol'])
        
        # 6. UI 顯示 (恢復大字體儀表板)
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("即時價格", f"${lp:.2f}", f"{chg:.2f} ({pct_chg:.2f}%)")
            c2.metric("策略勝率 (回測)", f"{win_rate*100:.0f}%", delta="凱利參考" if win_rate > 0.5 else "偏低")
            c3.metric("凱利建議倉位", f"{kelly_shares} 股", delta=kelly_msg.split(' ')[0] if '建議' in kelly_msg else "觀望")
            
            st.info(f"💡 凱利觀點: {kelly_msg}")

        # 7. 財報基本面區塊 (新增)
        if fund:
            with st.expander("📊 財報基本面 & 籌碼數據", expanded=False):
                f1, f2, f3, f4, f5 = st.columns(5)
                f1.metric("本益比 (PE)", f"{fund['pe']:.1f}" if fund['pe'] else "N/A")
                f2.metric("EPS", f"${fund['eps']:.2f}" if fund['eps'] else "N/A")
                f3.metric("毛利率", f"{fund['margin']*100:.1f}%" if fund['margin'] else "N/A")
                f4.metric("法人持股", f"{fund['inst']*100:.1f}%" if fund['inst'] else "N/A")
                f5.metric("空單比例", f"{fund['short']*100:.1f}%" if fund['short'] else "N/A")

        # 8. 策略詳細說明
        strat_desc = get_strategy_desc(cfg)
        st.markdown(f"**🛠️ 當前策略邏輯：** `{strat_desc}`")

        # 9. Gemini 大腦分析 (恢復詳細版 + 用戶輸入)
        if ai_provider == "Gemini (User Defined)" and gemini_key:
            st.subheader("🧠 Gemini 首席分析師")
            
            with st.expander("📝 輸入財報筆記 / 新聞重點 (AI 幫你解讀)", expanded=False):
                user_notes = st.text_area("貼上新聞或財報數據...", height=100)
                analyze_btn = st.button("🚀 開始深度分析")
            
            if analyze_btn or user_notes: # 有按鈕或有文字都觸發
                with st.spinner("AI 正在深度解讀中..."):
                    news = get_news(cfg['symbol'])
                    tech_txt = f"RSI:{rsi_val:.1f} | 策略勝率:{win_rate*100:.0f}% | 訊號:{current_sig}"
                    analysis, _, _ = analyze_logic_gemini_full(gemini_key, cfg['symbol'], news, tech_txt, k_pat, gemini_model, user_notes)
                    st.markdown(analysis)
        
        # 10. 圖表 (含籌碼面雙軸)
        st.plotly_chart(plot_chart(df, cfg, sigs), use_container_width=True)

    else:
        st.error("無法取得數據")
