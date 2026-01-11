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
    page_title="2026 量化戰情室 (Ultimate v14.0)",
    page_icon="🧠",
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
        .bull-box { background-color: #1a2e1a; padding: 10px; border-left: 5px solid #00ff00; margin-bottom: 5px; border-radius: 5px; }
        .bear-box { background-color: #2e1a1a; padding: 10px; border-left: 5px solid #ff0000; margin-bottom: 5px; border-radius: 5px; }
        .judge-box { background-color: #1a1a2e; padding: 10px; border-left: 5px solid #00aaff; margin-bottom: 5px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# ★★★ 3. AI 模型核心 (獨立模組) ★★★
# ==========================================

# --- Module A: TSM 專用波段 AI ---
@st.cache_resource(ttl=43200)
def get_tsm_swing_prediction():
    if not HAS_TENSORFLOW: return None, None, "TF缺"
    try:
        # TSM 專屬因子: 夜盤(EWT) + 利率(TNX) + 供應鏈(NVDA)
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

        # 預測 T+5 > 2%
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

# --- Module C: QQQ 通用腦 (針對科技股) ---
@st.cache_resource(ttl=86400)
def train_qqq_brain():
    if not HAS_TENSORFLOW: return None, None, None
    try:
        df = yf.download("QQQ", period="2y", interval="1d", progress=False)
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
        model.fit(np.array(X), np.array(y), epochs=30, verbose=0)
        return model, scaler, features
    except: return None, None, None

def scan_tech_stock(symbol, model, scaler, features):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        if len(df) < 60: return None, None, 0
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['Return'] = df['Close'].pct_change()
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['MA_Dist'] = (df['Close'] - ta.sma(df['Close'], 20)) / ta.sma(df['Close'], 20)
        df['ATR_Pct'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
        
        # 回測標籤
        df['Target'] = ((df['Close'].shift(-5) / df['Close'] - 1) > 0.02).astype(int)
        df.dropna(inplace=True)
        
        # 1. 預測未來
        last_seq = df[features].iloc[-20:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        
        # 2. 準度回測 (適配度)
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
# 4. 傳統策略分析 (資料與指標)
# ==========================================
def get_safe_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except: return None

def clean_text_for_llm(text): return re.sub(r'[^\w\s\u4e00-\u9fff.,:;%()\-]', '', str(text))

def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol: return None
        s = yf.Ticker(symbol)
        return { "pe": s.info.get('trailingPE'), "inst": s.info.get('heldPercentInstitutions'), "short": s.info.get('shortPercentOfFloat') }
    except: return None

def analyze_logic_gemini(api_key, symbol, news, tech, pattern, model_name):
    if not HAS_GEMINI: return "No Gemini", "⚠️", False
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"分析 {symbol}。技術: {tech}。型態: {pattern}。新聞: {news}。給出操作建議。"
        return model.generate_content(prompt).text, "⚡", True
    except Exception as e: return str(e), "⚠️", False

def run_ai_debate(api_key, symbol, news, tech, pattern, model_name):
    if not HAS_GEMINI: return "No Gemini", "⚠️", False, None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"對 {symbol} 進行多空辯論。\n技術: {tech}\n型態: {pattern}\n新聞: {news}\n回傳 JSON: {{'bull': '...', 'bear': '...', 'judge': '...'}}"
        res = model.generate_content(prompt).text
        if "```json" in res: res = res.replace("```json", "").replace("```", "")
        return json.loads(res).get("judge"), "⚖️", True, json.loads(res)
    except: return "Error", "⚠️", False, None

def analyze_ticker_strategy(config, ai_provider, gemini_key, model_name, debate):
    symbol = config['symbol']
    df = get_safe_data(symbol)
    if df is None: return None
    
    lp = df['Close'].iloc[-1]
    rsi = ta.rsi(df['Close'], 14).iloc[-1]
    sig = "BUY" if rsi < config.get('entry_rsi', 30) else "SELL" if rsi > config.get('exit_rsi', 70) else "WAIT"
    
    llm_res = "N/A"; debate_res = None; is_llm = False
    
    if ai_provider == "Gemini (User Defined)" and gemini_key:
        news = [clean_text_for_llm(n['title']) for n in yf.Ticker(symbol).news[:3]]
        if debate:
            llm_res, _, is_llm, debate_res = run_ai_debate(gemini_key, symbol, news, f"RSI:{rsi}", "N/A", model_name)
        else:
            llm_res, _, is_llm = analyze_logic_gemini(gemini_key, symbol, news, f"RSI:{rsi}", "N/A", model_name)
            
    return {
        "Symbol": symbol, "Name": config['name'], "Price": lp, "Prev_Close": df['Close'].iloc[-2],
        "Signal": sig, "Action": f"RSI:{rsi:.1f}", "Raw_DF": df, "Strat_Desc": config['mode'],
        "Is_LLM": is_llm, "LLM_Analysis": llm_res, "Debate": debate_res
    }

def quick_backtest(df, config):
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
                pos=0; r = (close.iloc[i]-ent)/ent; rets.append(r); trds+=1; wins += 1 if r>0 else 0
        return sigs, {"Total_Return": sum(rets)*100, "Win_Rate": (wins/trds*100) if trds else 0}
    except: return None, None

def plot_chart(df, config, sigs):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    if "RSI" in config['mode']:
        rsi = ta.rsi(df['Close'], length=config.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI"), row=2, col=1)
        fig.add_hline(y=config.get('entry_rsi', 30), line_dash="dash", row=2, col=1)
    if sigs is not None:
        buy = df[sigs==1]; sell = df[sigs==-1]
        fig.add_trace(go.Scatter(x=buy.index, y=buy['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell.index, y=sell['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', color='red')), row=1, col=1)
    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
    return fig

# ==========================================
# 5. 側邊欄與頁面配置
# ==========================================
st.sidebar.title("🚀 戰情室導航")
app_mode = st.sidebar.radio("選擇功能模組：", ["🤖 AI 深度學習實驗室", "📊 策略分析工具 (舊版)"])

st.sidebar.divider()
st.sidebar.header("⚙️ 全域設定")
ai_provider = st.sidebar.selectbox("AI 語言模型", ["不使用", "Gemini (User Defined)"])
gemini_key = ""; gemini_model = "models/gemini-2.0-flash"; debate_mode = False

if ai_provider == "Gemini (User Defined)":
    gemini_key = st.sidebar.text_input("Gemini Key", type="password")
    gemini_model = st.sidebar.text_input("Model Name", value="models/gemini-2.0-flash")
    debate_mode = st.sidebar.checkbox("啟用 AI 辯論模式", value=False)

if st.sidebar.button("🔄 清除快取 (重置 AI)"):
    st.cache_resource.clear()
    st.rerun()

# ==========================================
# 6. 主畫面邏輯 (分流)
# ==========================================

# ------------------------------------------
# Mode 1: AI 深度學習實驗室 (TSM / EDZ / QQQ)
# ------------------------------------------
if app_mode == "🤖 AI 深度學習實驗室":
    st.header("🤖 AI 深度學習實驗室")
    st.caption("神經網路模型 (LSTM) | T+5 波段預測 | 鎖定最佳權重")
    
    tab1, tab2, tab3 = st.tabs(["📈 TSM 專用波段", "🐻 EDZ / 宏觀雷達", "⚡ QQQ 科技股通用腦"])
    
    # [Tab 1] TSM 專用
    with tab1:
        st.subheader("TSM 專屬波段顧問 (T+5)")
        st.info("因子：台積電 + EWT (夜盤) + ^TNX (利率) + NVDA (供應鏈)")
        if st.button("開始分析 TSM", key="btn_tsm"):
            with st.spinner("AI 正在運算 (含準度回測)..."):
                prob, acc, price = get_tsm_swing_prediction()
            
            if prob is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("TSM 現價", f"${price:.2f}")
                c2.metric("模型準度", f"{acc*100:.1f}%", delta="可信" if acc>0.58 else "普通")
                
                conf = prob if prob > 0.5 else 1 - prob
                if prob > 0.6:
                    c3.metric("AI 建議", "🚀 看漲")
                    st.success(f"信心度 {conf*100:.1f}%：預期 5 天後漲幅 > 2%。建議拉回佈局。")
                elif prob < 0.4:
                    c3.metric("AI 建議", "📉 看跌/盤")
                    st.error(f"信心度 {conf*100:.1f}%：上漲空間有限。建議觀望。")
                else:
                    c3.metric("AI 建議", "⚖️ 震盪")
                    st.warning("多空不明，建議空手。")
            else: st.error("運算失敗，請檢查 TensorFlow")

    # [Tab 2] EDZ / 宏觀
    with tab2:
        st.subheader("全球風險與原物料雷達")
        st.info("因子：標的 + 利率 + 銅價 + 中國股市 + 美元")
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
                    c2.metric("趨勢方向", "📈 向上/風險高")
                    st.error(f"信心 {conf*100:.1f}%：{target_risk} 趨勢向上。(若是 EDZ 代表市場風險高)")
                elif prob < 0.4:
                    c2.metric("趨勢方向", "📉 向下/盤整")
                    st.success(f"信心 {conf*100:.1f}%：{target_risk} 趨勢向下或盤整。")
                else:
                    c2.metric("趨勢方向", "💤 震盪")
                    st.warning("無明顯趨勢。")

    # [Tab 3] QQQ 通用腦
    with tab3:
        st.subheader("QQQ 科技股掃描器")
        st.info("原理：用 QQQ 學會的邏輯，去檢視個股是否具備「科技股上漲型態」。")
        # 您指定的觀察清單
        tech_list = ["NVDA", "AMD", "AMZN", "MSFT", "GOOGL", "META", "TSLA", "AVGO", "PLTR"]
        
        if st.button("🚀 掃描科技巨頭", key="btn_scan"):
            with st.spinner("AI 正在訓練通用腦並掃描..."):
                model, scaler, feats = train_qqq_brain()
                if model:
                    res = []
                    prog = st.progress(0)
                    for i, t in enumerate(tech_list):
                        p, acc, pr = scan_tech_stock(t, model, scaler, feats)
                        if p: res.append((t, p, acc, pr))
                        prog.progress((i+1)/len(tech_list))
                    
                    prog.empty()
                    # 排序：準度+信心 高者在先
                    res.sort(key=lambda x: x[1]+x[2], reverse=True)
                    
                    for tick, p, acc, pr in res:
                        mark = ""
                        if p > 0.6 and acc > 0.55: mark = "💎 鑽石機會"
                        elif p < 0.4 and acc > 0.55: mark = "🛡️ 建議避開"
                        elif acc < 0.5: mark = "⚠️ QQQ不懂它"
                        
                        col = "green" if p > 0.5 else "red"
                        with st.container(border=True):
                            c1, c2, c3 = st.columns([2, 2, 3])
                            c1.markdown(f"**{tick}** (${pr:.1f})")
                            c2.markdown(f":{col}[信心 {p*100:.0f}%]")
                            c3.caption(f"適配準度: {acc*100:.0f}%  {mark}")

# ------------------------------------------
# Mode 2: 策略分析工具 (舊版功能)
# ------------------------------------------
elif app_mode == "📊 策略分析工具 (舊版)":
    st.header("📊 單股策略分析")
    
    # 這裡放回您原本完整的所有策略清單
    strategies = {
        "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (匯率)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 25, "exit_k": 70 },
        "QQQ": { "symbol": "QQQ", "name": "QQQ (那斯達克)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200 },
        "QLD": { "symbol": "QLD", "name": "QLD (那斯達克2倍)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200 },
        "TQQQ": { "symbol": "TQQQ", "name": "TQQQ (那斯達克3倍)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 85, "rsi_len": 2, "ma_trend": 200 },
        "SOXL_S": { "symbol": "SOXL", "name": "SOXL (費半3倍-狙擊)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 10, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 100 },
        "SOXL_F": { "symbol": "SOXL", "name": "SOXL (費半3倍-快攻)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 10, "exit_k": 75 },
        "EDZ": { "symbol": "EDZ", "name": "EDZ (新興空-避險)", "category": "📊 指數/外匯", "mode": "BOLL_RSI", "entry_rsi": 9, "rsi_len": 2, "ma_trend": 20 },
        "BTC_W": { "symbol": "BTC-USD", "name": "BTC (比特幣-波段)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 44, "exit_rsi": 65, "rsi_len": 14, "ma_trend": 200 },
        "BTC_F": { "symbol": "BTC-USD", "name": "BTC (比特幣-閃電)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 50, "rsi_len": 2, "ma_trend": 100 },
        "NVDA": { "symbol": "NVDA", "name": "NVDA (輝達)", "category": "🤖 AI 硬體/晶片", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
        "TSM": { "symbol": "TSM", "name": "TSM (台積電)", "category": "🤖 AI 硬體/晶片", "mode": "MA_CROSS", "fast_ma": 5, "slow_ma": 60 },
        "AVGO": { "symbol": "AVGO", "name": "AVGO (博通)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 5, "entry_rsi": 55, "exit_rsi": 85, "ma_trend": 200 },
        "MRVL": { "symbol": "MRVL", "name": "MRVL (邁威爾)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 20, "exit_rsi": 90, "ma_trend": 100 },
        "QCOM": { "symbol": "QCOM", "name": "QCOM (高通)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 8, "entry_rsi": 30, "exit_rsi": 70, "ma_trend": 100 },
        "GLW": { "symbol": "GLW", "name": "GLW (康寧)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
        "ONTO": { "symbol": "ONTO", "name": "ONTO (安圖)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 50, "exit_rsi": 65, "ma_trend": 100 },
        "META": { "symbol": "META", "name": "META (臉書)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 40, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
        "GOOGL": { "symbol": "GOOGL", "name": "GOOGL (谷歌)", "category": "💻 軟體/巨頭", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
        "AMZN": { "symbol": "AMZN", "name": "AMZN (亞馬遜)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 80, "rsi_len": 14, "ma_trend": 200 },
        "TSLA": { "symbol": "TSLA", "name": "TSLA (特斯拉)", "category": "💻 軟體/巨頭", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "AAPL": { "symbol": "AAPL", "name": "AAPL (蘋果)", "category": "💻 軟體/巨頭", "mode": "RSI_MA", "entry_rsi": 30, "exit_ma": 20, "rsi_len": 14, "ma_trend": 200 },
        "MSFT": { "symbol": "MSFT", "name": "MSFT (微軟)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70, "rsi_len": 14, "ma_trend": 200 },
        "AMD": { "symbol": "AMD", "name": "AMD (超微)", "category": "🤖 AI 硬體/晶片", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "PLTR": { "symbol": "PLTR", "name": "PLTR (Palantir)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 35, "exit_rsi": 85, "rsi_len": 14, "ma_trend": 50 },
        "ETN": { "symbol": "ETN", "name": "ETN (伊頓)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 40, "exit_rsi": 95, "ma_trend": 200 },
        "VRT": { "symbol": "VRT", "name": "VRT (維諦)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 35, "exit_rsi": 95, "ma_trend": 100 },
        "OKLO": { "symbol": "OKLO", "name": "OKLO (核能)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 50, "exit_rsi": 95, "ma_trend": 0 },
        "SMR": { "symbol": "SMR", "name": "SMR (NuScale)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 45, "exit_rsi": 90, "ma_trend": 0 },
        "KO": { "symbol": "KO", "name": "KO (可口可樂)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
        "JNJ": { "symbol": "JNJ", "name": "JNJ (嬌生)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 25, "exit_rsi": 90, "ma_trend": 200 },
        "PG": { "symbol": "PG", "name": "PG (寶僑)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 20, "exit_rsi": 80, "ma_trend": 0 },
        "BA": { "symbol": "BA", "name": "BA (波音)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 15, "exit_rsi": 60, "ma_trend": 0 },
        "CHT": { "symbol": "2412.TW", "name": "中華電", "category": "🇹🇼 台股", "mode": "RSI_RSI", "rsi_len": 14, "entry_rsi": 45, "exit_rsi": 70, "ma_trend": 0 },
        "GC": { "symbol": "GC=F", "name": "Gold (黃金)", "category": "⛏️ 原物料", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70, "rsi_len": 14 },
        "CL": { "symbol": "CL=F", "name": "Crude Oil (原油)", "category": "⛏️ 原物料", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "HG": { "symbol": "HG=F", "name": "Copper (銅)", "category": "⛏️ 原物料", "mode": "RSI_MA", "entry_rsi": 30, "exit_ma": 50, "rsi_len": 14 }
    }
    
    target_key = st.selectbox("選擇標的", list(strategies.keys()), format_func=lambda x: strategies[x]['name'])
    cfg = strategies[target_key]
    
    row = analyze_ticker_strategy(cfg, ai_provider, gemini_key, gemini_model, debate_mode)
    if row:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.metric("價格", f"${row['Price']:.2f}", f"{row['Price']-row['Prev_Close']:.2f}")
            c2.caption(f"策略: {row['Strat_Desc']}")
            st.markdown(f"#### {row['Signal']} | {row['Action']}")
            
            if row.get('Debate'):
                with st.expander("⚖️ AI 辯論", expanded=True):
                    st.write(f"🐂 多方: {row['Debate'].get('bull')}")
                    st.write(f"🐻 空方: {row['Debate'].get('bear')}")
                    st.success(f"⚖️ 裁決: {row['Debate'].get('judge')}")
            elif row['Is_LLM']:
                st.info(f"AI 分析: {row['LLM_Analysis']}")
            
            if row['Raw_DF'] is not None:
                sigs, perf = quick_backtest(row['Raw_DF'], cfg)
                st.plotly_chart(plot_chart(row['Raw_DF'], cfg, sigs), use_container_width=True)
                if perf: st.caption(f"回測績效: {perf['Total_Return']:.1f}% (勝率 {perf['Win_Rate']:.0f}%)")
