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
    page_title="2026 量化戰情室 (Ultimate v20.0)",
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
# ★★★ 3. AI 模型核心 ★★★
# ==========================================

# --- A. TSM ---
@st.cache_resource(ttl=43200)
def get_tsm_swing_prediction():
    if not HAS_TENSORFLOW: return None, None, "TF缺"
    try:
        tickers = { 'Main': 'TSM', 'Night': "EWT", 'Rate': "^TNX", 'AI': 'NVDA' }
        data = yf.download(list(tickers.values()), period="5y", interval="1d", progress=False)
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

        df['Target'] = ((df['Main_Close'].shift(-5) / df['Main_Close'] - 1) > 0.02).astype(int)
        df_train = df.iloc[:-5].copy()
        
        features = ['Main_Ret', 'Night_Ret', 'Rate_Chg', 'AI_Ret', 'RSI', 'Bias']
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[features])
        
        X, y = [], []
        for i in range(20, len(scaled_data)):
            X.append(scaled_data[i-20:i])
            y.append(df_train['Target'].iloc[i])
        
        X, y = np.array(X), np.array(y)
        split = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
        
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.3)); model.add(LSTM(64)); model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        early = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[early])
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        last_seq = df[features].iloc[-20:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        return prob, acc, df['Main_Close'].iloc[-1]
    except Exception as e: return None, None, str(e)

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

        feat_cols = []
        df['Main_Ret'] = df['Main'].pct_change()
        feat_cols.append('Main_Ret')
        for name in features_dict.keys():
            df[f'{name}_Ret'] = df[name].pct_change()
            feat_cols.append(f'{name}_Ret')
        df['RSI'] = ta.rsi(df['Main'], length=14)
        feat_cols.append('RSI')
        df.dropna(inplace=True)
        
        df['Target'] = ((df['Main'].shift(-5) / df['Main'] - 1) > 0.02).astype(int)
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
# 4. 傳統策略分析 (功能模組)
# ==========================================
def get_safe_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except: return None

# ★★★ 財報數據獲取 (含容錯機制) ★★★
def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol: return None
        s = yf.Ticker(symbol)
        # 使用 info 可能會被檔，增加 try-except
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

def analyze_logic_gemini_full(api_key, symbol, news, tech_txt, k_pattern, model_name, user_input=""):
    if not HAS_GEMINI: return "No Gemini", "⚠️", False
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        base_prompt = f"華爾街資深操盤手分析 {symbol}。\n【技術面】：{tech_txt}\n【K線型態】：{k_pattern}\n【新聞焦點】：{news}"
        if user_input: base_prompt += f"\n【用戶筆記】：{user_input}\n請深度結合分析。"
        else: base_prompt += "\n請給出：1.趨勢判斷 2.操作建議 3.風險提示"
        return model.generate_content(base_prompt).text, "🧠", True
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

# ★★★ 修正版回測 (無訊號處理) ★★★
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
                pos = 0; r = (close.iloc[i] - ent) / ent - (fee * 2)
                rets.append(r); trds += 1
                if r > 0: wins += 1
        
        # ★★★ 無交易防護 ★★★
        win_rate = float(wins / trds) if trds > 0 else 0.0
        last_sig = sigs.iloc[-1]
        
        stats = {
            "Total_Return": sum(rets)*100, 
            "Win_Rate": win_rate * 100, 
            "Raw_Win_Rate": win_rate,
            "Trades": trds # 回傳交易次數
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
    cmf = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    colors = ['#089981' if v >= 0 else '#f23645' for v in cmf]
    fig.add_trace(go.Bar(x=df.index, y=cmf, name='CMF', marker_color=colors, opacity=0.5), row=3, col=1, secondary_y=False)
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
    if mode == "RSI_RSI": desc = f"RSI 區間 (買 < {cfg['entry_rsi']} / 賣 > {cfg['exit_rsi']})"
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
        # 使用 Session State 防止按鈕刷新後消失
        if st.button("開始分析 TSM", key="btn_tsm") or 'tsm_result' in st.session_state:
            if 'tsm_result' not in st.session_state:
                with st.spinner("AI 正在運算..."):
                    prob, acc, price = get_tsm_swing_prediction()
                    st.session_state['tsm_result'] = (prob, acc, price)
            
            prob, acc, price = st.session_state['tsm_result']
            
            if prob is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("TSM 現價", f"${price:.2f}")
                c2.metric("模型準度", f"{acc*100:.1f}%", delta="可信" if acc>0.58 else "普通")
                
                direction = "Bull" if prob > 0.5 else "Bear"
                conf = prob if prob > 0.5 else 1 - prob
                
                if prob > 0.6:
                    c3.metric("AI 建議", "🚀 看漲", delta=f"信心 {conf*100:.1f}%")
                elif prob < 0.4:
                    c3.metric("AI 建議", "📉 看跌/盤", delta=f"信心 {conf*100:.1f}%", delta_color="inverse")
                else:
                    c3.metric("AI 建議", "⚖️ 震盪", delta=f"信心 {conf*100:.1f}%", delta_color="off")
                
                if st.button("📸 記錄預測 (快照)", key="save_tsm"):
                    if save_prediction("TSM", direction, conf, price):
                        st.success("✅ 已記錄！")
                    else: st.warning("⚠️ 今天已存過")
            else: st.error("TF Error")

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

# ------------------------------------------
# Mode 2: 策略分析工具 (單股)
# ------------------------------------------
elif app_mode == "📊 策略分析工具 (單股)":
    st.header("📊 單股策略分析")
    
    # ★★★ 微調過的策略清單 (讓 AMZN 容易有訊號) ★★★
    strategies = {
        "NVDA": { "symbol": "NVDA", "name": "NVDA (輝達)", "category": "🤖 AI 硬體/晶片", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
        "TSM": { "symbol": "TSM", "name": "TSM (台積電 ADR)", "category": "🤖 AI 硬體/晶片", "mode": "MA_CROSS", "fast_ma": 5, "slow_ma": 60 },
        "AMZN": { "symbol": "AMZN", "name": "AMZN (亞馬遜)", "category": "💻 軟體/巨頭", "mode": "KD", "entry_k": 20, "exit_k": 85 },
        "AMD": { "symbol": "AMD", "name": "AMD (超微)", "category": "🤖 AI 硬體/晶片", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "TSLA": { "symbol": "TSLA", "name": "TSLA (特斯拉)", "category": "💻 軟體/巨頭", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "AAPL": { "symbol": "AAPL", "name": "AAPL (蘋果)", "category": "💻 軟體/巨頭", "mode": "RSI_MA", "entry_rsi": 30, "exit_ma": 20, "rsi_len": 14 },
        "MSFT": { "symbol": "MSFT", "name": "MSFT (微軟)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70, "rsi_len": 14 },
        "GOOGL": { "symbol": "GOOGL", "name": "GOOGL (谷歌)", "category": "💻 軟體/巨頭", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2 },
        "QQQ": { "symbol": "QQQ", "name": "QQQ (那斯達克)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2 },
        "SOXL": { "symbol": "SOXL", "name": "SOXL (費半三倍)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 10, "exit_rsi": 90, "rsi_len": 2 },
        "EDZ": { "symbol": "EDZ", "name": "EDZ (避險)", "category": "📊 指數/外匯", "mode": "BOLL_RSI", "entry_rsi": 9, "rsi_len": 2 },
        "BTC": { "symbol": "BTC-USD", "name": "BTC (比特幣)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 44, "exit_rsi": 65, "rsi_len": 14 },
        "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (匯率)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 25, "exit_k": 70 },
        "QLD": { "symbol": "QLD", "name": "QLD (那斯達克 2倍)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2 },
        "TQQQ": { "symbol": "TQQQ", "name": "TQQQ (那斯達克 3倍)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 85, "rsi_len": 2 },
        "SOXL_F": { "symbol": "SOXL", "name": "SOXL (費半 3倍 - 快攻)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 10, "exit_k": 75 },
        "BTC_F": { "symbol": "BTC-USD", "name": "BTC (比特幣 - 閃電)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 50, "rsi_len": 2 },
        "AVGO": { "symbol": "AVGO", "name": "AVGO (博通)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 5, "entry_rsi": 55, "exit_rsi": 85 },
        "MRVL": { "symbol": "MRVL", "name": "MRVL (邁威爾)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 20, "exit_rsi": 90 },
        "QCOM": { "symbol": "QCOM", "name": "QCOM (高通)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 8, "entry_rsi": 30, "exit_rsi": 70 },
        "GLW": { "symbol": "GLW", "name": "GLW (康寧)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 30, "exit_rsi": 90 },
        "ONTO": { "symbol": "ONTO", "name": "ONTO (安圖)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 50, "exit_rsi": 65 },
        "META": { "symbol": "META", "name": "META (臉書)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 40, "exit_rsi": 90, "rsi_len": 2 },
        "PLTR": { "symbol": "PLTR", "name": "PLTR (Palantir)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 35, "exit_rsi": 85, "rsi_len": 14 },
        "ETN": { "symbol": "ETN", "name": "ETN (伊頓)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 40, "exit_rsi": 95 },
        "VRT": { "symbol": "VRT", "name": "VRT (維諦)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 35, "exit_rsi": 95 },
        "OKLO": { "symbol": "OKLO", "name": "OKLO (核能)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 50, "exit_rsi": 95 },
        "SMR": { "symbol": "SMR", "name": "SMR (NuScale)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 45, "exit_rsi": 90 },
        "KO": { "symbol": "KO", "name": "KO (可口可樂)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90 },
        "JNJ": { "symbol": "JNJ", "name": "JNJ (嬌生)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 25, "exit_rsi": 90 },
        "PG": { "symbol": "PG", "name": "PG (寶僑)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 20, "exit_rsi": 80 },
        "BA": { "symbol": "BA", "name": "BA (波音)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 15, "exit_rsi": 60 },
        "CHT": { "symbol": "2412.TW", "name": "中華電", "category": "🇹🇼 台股", "mode": "RSI_RSI", "rsi_len": 14, "entry_rsi": 45, "exit_rsi": 70 },
        "GC": { "symbol": "GC=F", "name": "Gold (黃金)", "category": "⛏️ 原物料", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70, "rsi_len": 14 },
        "CL": { "symbol": "CL=F", "name": "Crude Oil (原油)", "category": "⛏️ 原物料", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "HG": { "symbol": "HG=F", "name": "Copper (銅)", "category": "⛏️ 原物料", "mode": "RSI_MA", "entry_rsi": 30, "exit_ma": 50, "rsi_len": 14 }
    }
    
    target_key = st.selectbox("選擇標的", list(strategies.keys()), format_func=lambda x: strategies[x]['name'])
    cfg = strategies[target_key]
    
    df = get_safe_data(cfg['symbol'])
    lp = get_real_live_price(cfg['symbol'])
    
    if df is not None and lp:
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else lp
        chg = lp - prev_close
        pct_chg = (chg / prev_close) * 100
        
        current_sig, perf, sigs = quick_backtest(df, cfg)
        # ★★★ 修正點：若無交易，回傳0但顯示提示 ★★★
        win_rate = perf['Raw_Win_Rate'] if perf else 0
        trades_count = perf['Trades'] if perf else 0
        
        kelly_msg, kelly_shares = calculate_kelly_position(df, user_capital, win_rate, user_risk/100, current_sig)
        k_pat = identify_k_pattern(df)
        rsi_val = ta.rsi(df['Close'], 14).iloc[-1]
        fund = get_fundamentals(cfg['symbol'])
        
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("即時價格", f"${lp:.2f}", f"{chg:.2f} ({pct_chg:.2f}%)")
            
            # ★★★ 修正點：回測結果顯示優化 ★★★
            if trades_count > 0:
                c2.metric("策略勝率 (回測)", f"{win_rate*100:.0f}%", delta=f"{trades_count} 次交易")
            else:
                c2.metric("策略勝率 (回測)", "無交易", delta="區間未觸發", delta_color="off")
                
            c3.metric("凱利建議倉位", f"{kelly_shares} 股", delta=kelly_msg.split(' ')[0] if '建議' in kelly_msg else "觀望")
            st.info(f"💡 凱利觀點: {kelly_msg}")

        # ★★★ 修正點：基本面顯示優化 (含失敗提示) ★★★
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

        strat_desc = get_strategy_desc(cfg, df)
        st.markdown(f"**🛠️ 當前策略邏輯：** `{strat_desc}`")

        if ai_provider == "Gemini (User Defined)" and gemini_key:
            st.subheader("🧠 Gemini 首席分析師")
            with st.expander("📝 輸入財報筆記 / 新聞重點", expanded=False):
                user_notes = st.text_area("貼上新聞或財報數據...", height=100)
                analyze_btn = st.button("🚀 開始深度分析")
            if analyze_btn or user_notes:
                with st.spinner("AI 正在深度解讀中..."):
                    news = get_news(cfg['symbol'])
                    tech_txt = f"RSI:{rsi_val:.1f} | 策略勝率:{win_rate*100:.0f}% | 訊號:{current_sig}"
                    analysis, _, _ = analyze_logic_gemini_full(gemini_key, cfg['symbol'], news, tech_txt, k_pat, gemini_model, user_notes)
                    st.markdown(analysis)
        
        st.plotly_chart(plot_chart(df, cfg, sigs), use_container_width=True)
    else: st.error("無法取得數據")

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

