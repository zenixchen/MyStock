import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
import importlib.util

# ==========================================
# ★★★ 0. 2026 系統設定 (高頻訊號版) ★★★
# ==========================================
st.set_page_config(
    page_title="2026 量子戰情室 (Quantum Command)",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except: pass

st.markdown("""
    <style>
        .stApp { background-color: #0b0e11; }
        h1, h2, h3 { color: #00f2ff !important; font-family: 'Orbitron', sans-serif; }
        .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
        .stMetric label { color: #8b949e !important; }
        .stMetric div[data-testid="stMetricValue"] { color: #e6edf3 !important; }
    </style>
""", unsafe_allow_html=True)

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# ==========================================
# 1. 數據核心 (★修復重點區★)
# ==========================================
def get_live_price_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        try:
            current_price = ticker.fast_info.get('last_price')
        except:
            current_price = None

        # 下載數據
        period = "2y"
        df = yf.download(symbol, period=period, interval="1d", progress=False, timeout=10)
        
        if df is None or df.empty: return None, 0, 0
        
        # ★修復 1: 強制處理 MultiIndex，確保欄位是乾淨的 'Close' 而不是 ('Close', 'NVDA')
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        # 取得最後收盤價
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else last_close
        
        if current_price is None or np.isnan(current_price):
            current_price = last_close
        
        # ★修復 2: 模擬 K 線並確保時區一致性，避免 concat 後變成 object
        if df.index[-1].date() != datetime.now().date():
            # 建立新的一行
            new_index = pd.Timestamp.now()
            # 如果原來的 index 有時區，新的一行也要有 (或者把原來的拿掉)
            if df.index.tz is not None:
                new_index = new_index.tz_localize(df.index.tz)
                
            new_row = pd.DataFrame({
                'Open': [current_price], 'High': [current_price], 
                'Low': [current_price], 'Close': [current_price], 
                'Volume': [0.0] # 用浮點數 0.0
            }, index=[new_index])
            
            df = pd.concat([df, new_row])

        # ★修復 3 (關鍵): 強制將所有數據轉為 float，防止 'Object' 類型導致運算崩潰
        cols_to_fix = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        # 清除 NaN (防止運算錯誤)
        df.dropna(subset=['Close'], inplace=True)

        return df, float(current_price), float(prev_close)
    except Exception as e:
        print(f"Data Error: {e}")
        return None, 0, 0

@st.cache_data(ttl=3600)
def get_fundamentals_2026(symbol):
    if "=" in symbol or "-USD" in symbol: return None
    try:
        t = yf.Ticker(symbol)
        info = t.info
        return {
            "pe": info.get('trailingPE'),
            "inst": info.get('heldPercentInstitutions', 0),
            "short": info.get('shortPercentOfFloat', 0)
        }
    except: return None

# ==========================================
# 2. 鯨魚偵測 (僅作參考，不影響訊號)
# ==========================================
def analyze_smc_whale(df):
    if df is None or len(df) < 50: return "N/A", 0
    
    # 確保輸入是 Series
    try:
        cmf = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
        if cmf is None: return "N/A", 50
        cmf_val = cmf.iloc[-1]
        
        mfi = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        mfi_val = mfi.iloc[-1] if mfi is not None else 50
        
        whale_score = 50
        if cmf_val > 0.15: whale_score += 20
        elif cmf_val < -0.15: whale_score -= 20
        if mfi_val > 60 and cmf_val > 0: whale_score += 10
        
        whale_status = "⚪ 散戶盤"
        if whale_score >= 75: whale_status = "🐳 巨鯨大買"
        elif whale_score >= 60: whale_status = "🔴 主力進駐"
        elif whale_score <= 30: whale_status = "🟢 主力倒貨"
        
        return whale_status, whale_score
    except:
        return "N/A", 50

# ==========================================
# 3. 策略引擎 (純技術指標，無濾網)
# ==========================================
def run_strategy(df, config):
    c = df['Close']; h = df['High']; l = df['Low']
    lp = c.iloc[-1]
    
    sig = "WAIT"; color = "gray"; desc = ""
    mode = config['mode']

    # --- 1. RSI_RSI (區間) ---
    if mode == "RSI_RSI":
        rsi = ta.rsi(c, length=config.get('rsi_len', 14))
        if rsi is not None:
            curr_rsi = rsi.iloc[-1]
            desc = f"RSI: {curr_rsi:.1f}"
            if curr_rsi < config['entry_rsi']:
                sig = "BUY"; color = "green"; desc += " (超賣)"
            elif curr_rsi > config['exit_rsi']:
                sig = "SELL"; color = "red"; desc += " (超買)"

    # --- 2. RSI_MA (趨勢回檔) ---
    elif mode == "RSI_MA":
        rsi = ta.rsi(c, length=config.get('rsi_len', 14))
        ma_trend = ta.ema(c, length=config.get('ma_trend', 200))
        exit_ma = ta.sma(c, length=config['exit_ma'])
        
        if rsi is not None and ma_trend is not None:
            curr_rsi = rsi.iloc[-1]
            curr_ma = ma_trend.iloc[-1]
            curr_exit = exit_ma.iloc[-1]
            
            desc = f"RSI:{curr_rsi:.1f} | 趨勢:{'多' if lp>curr_ma else '空'}"
            
            if lp > curr_ma and curr_rsi < config['entry_rsi']:
                sig = "BUY"; color = "green"; desc += " (順勢回檔)"
            elif lp > curr_exit and curr_rsi > 70:
                sig = "SELL"; color = "red"

    # --- 3. FUSION (混合模式) ---
    elif mode == "FUSION" or mode == "FUSION_SMC":
        rsi = ta.rsi(c, config.get('rsi_len', 14))
        ma = ta.ema(c, config.get('ma_trend', 200))
        
        if rsi is not None and ma is not None:
            curr_rsi = rsi.iloc[-1]
            curr_ma = ma.iloc[-1]
            desc = f"RSI:{curr_rsi:.0f} (MA200之上)"
            
            if lp > curr_ma and curr_rsi < config['entry_rsi']:
                sig = "STRONG BUY"; color = "green"; desc += " (黃金坑)"
            elif curr_rsi > config['exit_rsi']:
                sig = "SELL"; color = "red"; desc += " (過熱)"

    # --- 4. SUPERTREND ---
    elif mode == "SUPERTREND":
        st_val = ta.supertrend(h, l, c, length=config['period'], multiplier=config['multiplier'])
        if st_val is not None:
            dr = st_val.iloc[-1, 1]; prev_dr = st_val.iloc[-2, 1]
            desc = "趨勢多頭" if dr == 1 else "趨勢空頭"
            if prev_dr == -1 and dr == 1: sig = "BUY"; color = "green"; desc = "趨勢翻多"
            elif prev_dr == 1 and dr == -1: sig = "SELL"; color = "red"; desc = "趨勢翻空"
            elif dr == 1: sig = "HOLD"; color = "#00f2ff"

    # --- 5. KD ---
    elif mode == "KD":
        k = ta.stoch(h, l, c, k=9, d=3)
        if k is not None:
            curr_k = k.iloc[-1, 0]
            desc = f"K值: {curr_k:.1f}"
            if curr_k < config['entry_k']: sig = "BUY"; color = "green"; desc += " (低檔)"
            elif curr_k > config['exit_k']: sig = "SELL"; color = "red"; desc += " (高檔)"

    # --- 6. MA_CROSS ---
    elif mode == "MA_CROSS":
        f = ta.sma(c, config['fast_ma']); s = ta.sma(c, config['slow_ma'])
        if f is not None and s is not None:
            curr_f, prev_f = f.iloc[-1], f.iloc[-2]
            curr_s, prev_s = s.iloc[-1], s.iloc[-2]
            desc = f"MA{config['fast_ma']} v MA{config['slow_ma']}"
            if prev_f <= prev_s and curr_f > curr_s: sig = "BUY"; color = "green"; desc = "黃金交叉"
            elif prev_f >= prev_s and curr_f < curr_s: sig = "SELL"; color = "red"; desc = "死亡交叉"
            elif curr_f > curr_s: sig = "HOLD"; color = "#00f2ff"

    # --- 7. BOLL_RSI ---
    elif mode == "BOLL_RSI":
        rsi = ta.rsi(c, config.get('rsi_len', 14))
        bb = ta.bbands(c, length=20, std=2)
        if rsi is not None and bb is not None:
            curr_rsi = rsi.iloc[-1]
            lower = bb.iloc[-1, 0]; upper = bb.iloc[-1, 2]
            desc = f"Boll+RSI({curr_rsi:.0f})"
            if lp < lower and curr_rsi < config['entry_rsi']: sig = "BUY"; color = "green"; desc += " (破底反彈)"
            elif lp >= upper: sig = "SELL"; color = "red"; desc += " (觸頂)"

    return sig, color, desc

# ==========================================
# 4. 圖表引擎 (視覺化)
# ==========================================
def plot_pro_chart(df, symbol):
    # ★修復 4: 再次確保運算前數據是 float (Plotly 也怕 object)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)

    # 計算指標 (使用 try-except 包裹以防萬一)
    try:
        df['EMA50'] = ta.ema(df['Close'], 50)
        df['EMA200'] = ta.ema(df['Close'], 200)
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], 20)
    except Exception as e:
        print(f"Indicator Error: {e}")
        # 如果計算失敗，填入 0 或 NaN 避免畫圖崩潰
        df['CMF'] = 0

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#00f2ff', decreasing_line_color='#ff007a'), row=1, col=1)
    
    if 'EMA50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA 50', line=dict(color='#ffeb3b', width=1)), row=1, col=1)
    if 'EMA200' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], name='EMA 200', line=dict(color='#9c27b0', width=2)), row=1, col=1)

    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#b39ddb')), row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)

    if 'CMF' in df.columns:
        # 處理 CMF 可能有的 NaN
        cmf_vals = df['CMF'].fillna(0)
        colors = ['#00f2ff' if v >= 0 else '#ff007a' for v in cmf_vals]
        fig.add_trace(go.Bar(x=df.index, y=cmf_vals, name='Whale Flow', marker_color=colors), row=3, col=1)

    fig.update_layout(height=550, margin=dict(t=10, b=0, l=0, r=0), paper_bgcolor='#161b22', plot_bgcolor='#0d1117', font=dict(color='#8b949e'), showlegend=False, xaxis_rangeslider_visible=False)
    fig.update_xaxes(showgrid=True, gridcolor='#30363d'); fig.update_yaxes(showgrid=True, gridcolor='#30363d')
    return fig

# ==========================================
# 5. AI 分析
# ==========================================
def get_ai_analysis(symbol, price, signal, whale_score, client=None):
    context = f"{symbol} @ {price}, Sig: {signal}, Whale: {whale_score}"
    
    if client:
        try:
            prompt = f"Analyst style (Traditional Chinese): Analyze {context}. concise."
            chat = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile", temperature=0.3)
            return chat.choices[0].message.content
        except: pass

    divergence_msg = ""
    if "BUY" in signal and whale_score < 40:
        divergence_msg = "⚠️ 注意：技術面買進，但主力籌碼尚未跟上 (背離)。"
    elif "SELL" in signal and whale_score > 60:
        divergence_msg = "⚠️ 注意：技術面賣出，但主力仍在吸籌 (背離)。"
    else:
        divergence_msg = "✅ 技術與籌碼方向一致。"

    return f"""
    🤖 **AI 戰術分析 (Rule-Based)**:
    * **訊號判定**: 觸發 {signal} 訊號 (純技術指標)。
    * **籌碼參考**: 巨鯨指數 {whale_score} ({'主力盤' if whale_score>50 else '散戶盤'})。
    * **綜合解讀**: {divergence_msg}
    """

# ==========================================
# 6. 主程式 (核心策略庫)
# ==========================================
st.title("🌌 2026 Quantum Command Center")

strategies = {
    "NVDA": { "symbol": "NVDA", "name": "NVDA (聖杯)", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
    "TSM": { "symbol": "TSM", "name": "TSM (趨勢)", "mode": "MA_CROSS", "fast_ma": 5, "slow_ma": 60 },
    "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (美元)", "mode": "KD", "entry_k": 25, "exit_k": 70 },
    "KO": { "symbol": "KO", "name": "KO (可樂)", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
    "BA": { "symbol": "BA", "name": "BA (波音)", "mode": "SUPERTREND", "period": 15, "multiplier": 1.0 },
    "META": { "symbol": "META", "name": "META (暴力反彈)", "mode": "RSI_RSI", "entry_rsi": 40, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
    "GOOGL": { "symbol": "GOOGL", "name": "GOOGL (聖杯)", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
    "QQQ": { "symbol": "QQQ", "name": "QQQ (穩健)", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200 },
    "QLD": { "symbol": "QLD", "name": "QLD (2倍)", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200 },
    "TQQQ": { "symbol": "TQQQ", "name": "TQQQ (3倍)", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 85, "rsi_len": 2, "ma_trend": 200 },
    "EDZ": { "symbol": "EDZ", "name": "EDZ (救援)", "mode": "BOLL_RSI", "entry_rsi": 9, "rsi_len": 2, "ma_trend": 20 },
    "SOXL_S": { "symbol": "SOXL", "name": "SOXL (狙擊)", "mode": "RSI_RSI", "entry_rsi": 10, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 100 },
    "SOXL_F": { "symbol": "SOXL", "name": "SOXL (快攻)", "mode": "KD", "entry_k": 10, "exit_k": 75 },
    "BTC_W": { "symbol": "BTC-USD", "name": "BTC (波段)", "mode": "RSI_RSI", "entry_rsi": 44, "exit_rsi": 65, "rsi_len": 14, "ma_trend": 200 },
    "BTC_F": { "symbol": "BTC-USD", "name": "BTC (閃電)", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 50, "rsi_len": 2, "ma_trend": 100 },
}

with st.sidebar:
    st.header("⚙️ 戰情室設定")
    groq_key = st.text_input("Groq API Key (選填)", type="password")
    st.divider()
    option_list = list(strategies.keys())
    selected_key = st.selectbox("選擇監控目標", option_list, index=0)
    if st.button("🔄 刷新數據"): st.cache_data.clear(); st.rerun()

groq_client = Groq(api_key=groq_key) if HAS_GROQ and groq_key else None
cfg = strategies[selected_key]
symbol = cfg['symbol']

with st.spinner(f"正在連線量子衛星獲取 {symbol} ..."):
    df, price, prev = get_live_price_data(symbol)

if df is not None:
    whale_status, whale_score = analyze_smc_whale(df)
    signal, sig_color, sig_desc = run_strategy(df, cfg)
    fund = get_fundamentals_2026(symbol)
    
    pct = (price - prev) / prev * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("即時價格", f"${price:,.2f}", f"{pct:+.2f}%")
    
    c2.markdown(f"""
    <div style="text-align: center; border: 1px solid {sig_color}; padding: 10px; border-radius: 5px; background: #0d1117;">
        <span style="color: gray; font-size: 12px;">SYSTEM SIGNAL</span><br>
        <strong style="color: {sig_color}; font-size: 22px;">{signal}</strong>
    </div>""", unsafe_allow_html=True)
    
    score_col = "#00f2ff" if whale_score >= 50 else "#ff007a"
    c3.markdown(f"""
    <div style="text-align: center; border: 1px solid #30363d; padding: 10px; border-radius: 5px; background: #0d1117;">
        <span style="color: gray; font-size: 12px;">WHALE REF</span><br>
        <strong style="color: {score_col}; font-size: 22px;">{whale_score}</strong>
    </div>""", unsafe_allow_html=True)

    c4.metric("策略模式", cfg['mode'], f"{whale_status}")

    st.plotly_chart(plot_pro_chart(df, symbol), use_container_width=True)

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("🧬 數據細節")
        if fund:
            if fund['short'] > 0.2: st.error(f"⚠️ 軋空警戒: 空單 {fund['short']*100:.1f}%")
            if fund['inst'] > 0.6: st.success(f"🏦 機構控盤: {fund['inst']*100:.0f}%")
        st.info(f"技術描述: {sig_desc}")

    with col_r:
        st.subheader("🧠 戰術分析")
        ai_res = get_ai_analysis(symbol, price, signal, whale_score, groq_client)
        st.markdown(ai_res)

else:
    st.error("Data Error: 無法獲取行情。")
