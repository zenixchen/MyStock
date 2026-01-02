import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
import re
import importlib.util

# ==========================================
# ★ 0. 系統設定
# ==========================================
st.set_page_config(
    page_title="2026 量化戰情室 (WT發明版)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except: pass

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# CSS 美化
st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            height: 40px; white-space: pre-wrap; background-color: #1c202a; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] { background-color: #2962ff; color: white; }
        div[data-testid="stMetricValue"] { font-size: 20px; color: #e0e0e0; }
        h4 { color: #8b949e; font-weight: 300; font-size: 14px; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 2026 量化戰情室 (WT 獨家發明)")
st.caption("AI 原創指標：WT (Whale Thrust) 巨鯨推力 = (價差/波動) × 資金流")

if st.button('🔄 更新行情'):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 1. 數據核心
# ==========================================
def get_safe_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
        
        try:
            t = yf.Ticker(ticker)
            live_price = t.fast_info.get('last_price')
            if live_price and not np.isnan(live_price):
                if df.index[-1].date() != datetime.now().date():
                    new_idx = pd.Timestamp.now()
                    if df.index.tz is not None: new_idx = new_idx.tz_localize(df.index.tz)
                    new_row = pd.DataFrame({
                        'Open': [live_price], 'High': [live_price], 
                        'Low': [live_price], 'Close': [live_price], 
                        'Volume': [0.0]
                    }, index=[new_idx])
                    df = pd.concat([df, new_row])
                    df.loc[df.index[-1], 'Close'] = float(live_price)
        except: pass
        return df
    except Exception as e:
        print(f"Data Error {ticker}: {e}")
        return None

# ==========================================
# 2. 輔助功能 & AI
# ==========================================
@st.cache_data(ttl=86400)
def get_fundamentals(symbol):
    if "=" in symbol or "^" in symbol: return None
    try:
        info = yf.Ticker(symbol).info
        return {
            "pe": info.get('trailingPE'),
            "inst": info.get('heldPercentInstitutions', 0),
            "short": info.get('shortPercentOfFloat', 0)
        }
    except: return None

def clean_text(text):
    return re.sub(r'[^\w\s\u4e00-\u9fff.,:;%()\-]', '', str(text)) if text else ""

def get_news(symbol):
    try:
        news = yf.Ticker(symbol).news
        return [clean_text(n.get('title','')) for n in news[:5]] if news else []
    except: return []

def analyze_deep_logic_2026(client, symbol, news_list, signal, action, price_context):
    if not client or not news_list: return None
    news_text = "\n".join([f"- {n}" for n in news_list[:5]])
    prompt = f"""
    You are a sophisticated AI Investment Committee.
    Target: {symbol}
    Signal: {signal} ({action})
    Context: {price_context}
    News: {news_text}
    Output in Traditional Chinese Markdown:
    ### 🏛️ AI 投資委員會 ({symbol})
    **🐂 多頭觀點**: ...
    **🐻 空頭警示**: ...
    **⚖️ 風險評估**: ...
    ---
    **🎯 最終指令**: [Strong Buy/Buy/Wait/Sell/Strong Sell]
    **💡 關鍵洞察**: [One sentence insight]
    """
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.4, max_tokens=1000
        )
        return resp.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# ==========================================
# 3. 策略運算核心 (WT 邏輯植入)
# ==========================================
def find_rsi_price(df, target_rsi, rsi_len):
    if df is None or len(df)<20: return 0
    lc = df['Close'].iloc[-1]; l, h = lc*0.5, lc*1.5
    for _ in range(10):
        mid = (l+h)/2
        sim = pd.concat([df['Close'], pd.Series([mid])], ignore_index=True)
        r = ta.rsi(sim, length=rsi_len).iloc[-1]
        if r > target_rsi: h = mid
        else: l = mid
    return mid

def run_strategy(df, cfg):
    if df is None: return "ERR", "無數據", "ERR", "---", "---"
    
    c = df['Close']; h = df['High']; l = df['Low']
    lp = c.iloc[-1]
    sig="WAIT"; act="觀望"; s_type="WAIT"; b_at="---"; s_at="---"
    mode = cfg['mode']
    
    # ★★★ 計算 WT 巨鯨推力 ★★★
    # 1. 機構成本 (VWAP)
    vwap = ta.vwma(c, df['Volume'], length=20)
    # 2. 真實波動 (ATR)
    atr = ta.atr(h, l, c, length=14)
    # 3. 資金流 (MFI)
    mfi = ta.mfi(h, l, c, df['Volume'], length=14)
    
    # 4. WT 公式 = ((Close - VWAP) / ATR) * (MFI / 50)
    # 防止 ATR 為 0 或 NaN
    atr_safe = atr.replace(0, 1).fillna(1)
    wt = ((c - vwap) / atr_safe) * (mfi / 50)
    curr_wt = wt.iloc[-1]

    # WT 輔助判斷
    wt_status = ""
    if curr_wt > 2.0: wt_status = " | 🚀WT噴射(強多)"
    elif curr_wt < -2.0: wt_status = " | 💀WT墜毀(強空)"
    elif curr_wt > 0: wt_status = " | 🟢WT多方控盤"
    else: wt_status = " | 🔴WT空方控盤"

    # 1. RSI / FUSION
    if mode == "RSI_RSI" or mode == "FUSION":
        rsi_len = cfg.get('rsi_len', 14)
        rsi = ta.rsi(c, length=rsi_len)
        curr_rsi = rsi.iloc[-1]
        entry_rsi = cfg.get('entry_rsi', 30)
        exit_rsi = cfg.get('exit_rsi', 70)
        b_at = f"${find_rsi_price(df, entry_rsi, rsi_len):.2f}"
        s_at = f"${find_rsi_price(df, exit_rsi, rsi_len):.2f}"
        
        trend_ok = True
        if cfg.get('ma_trend', 0) > 0:
            ma = ta.ema(c, length=cfg['ma_trend']).iloc[-1]
            if lp < ma: trend_ok = False
        
        if curr_rsi < entry_rsi:
            if trend_ok: sig="🔥 BUY"; act="低檔順勢"; s_type="BUY"
            else: sig="✋ WAIT"; act="低檔逆勢"; s_type="WAIT"
        elif curr_rsi > exit_rsi:
            sig="💰 SELL"; act="高檔過熱"; s_type="SELL"
        else:
            act = f"震盪中 (RSI:{curr_rsi:.1f})"

    # 2. RSI_MA
    elif mode == "RSI_MA":
        rsi = ta.rsi(c, length=cfg.get('rsi_len', 14))
        curr_rsi = rsi.iloc[-1]
        exit_ma_val = ta.sma(c, length=cfg.get('exit_ma', 20)).iloc[-1]
        entry_rsi = cfg.get('entry_rsi', 30)
        b_at = f"${find_rsi_price(df, entry_rsi, 14):.2f}"; s_at = f"${exit_ma_val:.2f} (MA)"
        
        if curr_rsi < entry_rsi: sig="🔥 BUY"; act="RSI低檔佈局"; s_type="BUY"
        elif lp > exit_ma_val:
            if curr_rsi > 80: sig="💰 SELL"; act="突破均線且過熱"; s_type="SELL"
            else: act="持有 (均線之上)"
        else: act = f"等待 (RSI:{curr_rsi:.1f})"

    # 3. SUPERTREND
    elif mode == "SUPERTREND":
        st_val = ta.supertrend(h, l, c, length=cfg['period'], multiplier=cfg['multiplier'])
        if st_val is not None:
            curr_dir = st_val.iloc[-1, 1]; prev_dir = st_val.iloc[-2, 1]
            s_line = st_val.iloc[-1, 0]
            s_at = f"${s_line:.2f}"
            if prev_dir == -1 and curr_dir == 1: sig="🚀 BUY"; act="趨勢翻多"; s_type="BUY"
            elif prev_dir == 1 and curr_dir == -1: sig="📉 SELL"; act="趨勢翻空"; s_type="SELL"
            elif curr_dir == 1: sig="✊ HOLD"; act="多頭續抱"; s_type="HOLD"
            else: sig="☁️ EMPTY"; act="空頭觀望"; s_type="EMPTY"

    # 4. KD
    elif mode == "KD":
        k = ta.stoch(h, l, c, k=9, d=3).iloc[-1, 0]
        b_at = f"K<{cfg['entry_k']}"; s_at = f"K>{cfg['exit_k']}"
        if k < cfg['entry_k']: sig="🚀 BUY"; act=f"KD低檔({k:.1f})"; s_type="BUY"
        elif k > cfg['exit_k']: sig="💀 SELL"; act=f"KD高檔({k:.1f})"; s_type="SELL"
        else: act = f"K值 {k:.1f}"

    # 5. MA_CROSS
    elif mode == "MA_CROSS":
        f = ta.sma(c, cfg['fast_ma']); s = ta.sma(c, cfg['slow_ma'])
        cf, pf = f.iloc[-1], f.iloc[-2]; cs, ps = s.iloc[-1], s.iloc[-2]
        if pf<=ps and cf>cs: sig="🔥 BUY"; act="黃金交叉"; s_type="BUY"
        elif pf>=ps and cf<cs: sig="📉 SELL"; act="死亡交叉"; s_type="SELL"
        elif cf>cs: sig="✊ HOLD"; act="多頭排列"; s_type="HOLD"
        else: sig="☁️ EMPTY"; act="空頭排列"; s_type="EMPTY"

    # 6. BOLL_RSI
    elif mode == "BOLL_RSI":
        rsi = ta.rsi(c, length=cfg.get('rsi_len', 14))
        curr_rsi = rsi.iloc[-1]
        bb = ta.bbands(c, length=20, std=2)
        low_b = bb.iloc[-1, 0]; up_b = bb.iloc[-1, 2]
        b_at = f"${low_b:.2f}"; s_at = f"${up_b:.2f}"
        if lp < low_b and curr_rsi < cfg['entry_rsi']: sig="🚑 BUY"; act="破底搶反彈"; s_type="BUY"
        elif lp >= up_b: sig="💀 SELL"; act="觸頂回調"; s_type="SELL"
        else: act="通道震盪"

    # 加入 WT 診斷
    act += wt_status
    return sig, act, s_type, b_at, s_at

# ==========================================
# 4. 視覺化 (★ WT 獨家指標可視化)
# ==========================================
def plot_chart(df, cfg, signals=None):
    if df is None: return None
    
    # 計算 WT
    df['VWAP'] = ta.vwma(df['Close'], df['Volume'], length=20)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).fillna(1)
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14).fillna(50)
    
    # WT = ((Close - VWAP) / ATR) * (MFI / 50)
    df['WT'] = ((df['Close'] - df['VWAP']) / df['ATR']) * (df['MFI'] / 50)
    
    # 顏色判斷
    wt_colors = []
    for val in df['WT']:
        if val > 2.0: wt_colors.append('#ff1744') # 紅色 (噴出/過熱)
        elif val < -2.0: wt_colors.append('#00e676') # 綠色 (超跌/機會)
        elif val > 0: wt_colors.append('#ef5350') # 淺紅 (多方)
        else: wt_colors.append('#66bb6a') # 淺綠 (空方)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Row 1: K線 + VWAP
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', increasing_line_color='#ef5350', decreasing_line_color='#00e676'
    ), row=1, col=1)
    
    # VWAP 線 (機構成本)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP (成本)', line=dict(color='#FFD700', width=2)), row=1, col=1)

    if cfg.get('ma_trend', 0) > 0:
        ma = ta.ema(df['Close'], length=cfg['ma_trend'])
        fig.add_trace(go.Scatter(x=df.index, y=ma, name=f'EMA{cfg["ma_trend"]}', line=dict(color='orange', width=1)), row=1, col=1)

    # Row 2: RSI
    if "RSI" in cfg['mode'] or cfg['mode'] in ["FUSION", "BOLL_RSI"]:
        rsi = ta.rsi(df['Close'], length=cfg.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='#b39ddb', width=2)), row=2, col=1)
        fig.add_hline(y=cfg.get('entry_rsi', 30), line_dash="dash", line_color='green', row=2, col=1)
        fig.add_hline(y=cfg.get('exit_rsi', 70), line_dash="dash", line_color='red', row=2, col=1)
    elif cfg['mode'] == "KD":
        k = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
        fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:,0], name='K', line=dict(color='yellow')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:,1], name='D', line=dict(color='blue')), row=2, col=1)

    # Row 3: WT 獨家指標 (能量柱)
    fig.add_trace(go.Bar(
        x=df.index, y=df['WT'], name='Whale Thrust', marker_color=wt_colors
    ), row=3, col=1)
    
    # 畫 0 軸和警戒線
    fig.add_hline(y=2.0, line_dash="dot", line_color='red', annotation_text="噴出區", row=3, col=1)
    fig.add_hline(y=-2.0, line_dash="dot", line_color='green', annotation_text="超跌區", row=3, col=1)

    fig.update_layout(
        height=600, margin=dict(t=10, b=0, l=0, r=0),
        paper_bgcolor='#161b22', plot_bgcolor='#161b22',
        font=dict(color='#d1d4dc'), showlegend=True, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False
    )
    return fig

# ==========================================
# 5. 監控名單
# ==========================================
strategies = {
    "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (美元)", "mode": "KD", "entry_k": 25, "exit_k": 70 },
    "KO": { "symbol": "KO", "name": "KO (可樂)", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
    "BA": { "symbol": "BA", "name": "BA (波音)", "mode": "RSI_RSI", "rsi_len": 14, "entry_rsi": 25, "exit_rsi": 65, "ma_trend": 0 },
    "META": { "symbol": "META", "name": "META (暴力反彈)", "mode": "RSI_RSI", "entry_rsi": 40, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
    "NVDA": { "symbol": "NVDA", "name": "NVDA (聖杯)", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
    "GOOGL": { "symbol": "GOOGL", "name": "GOOGL (聖杯)", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
    "QQQ": { "symbol": "QQQ", "name": "QQQ (穩健)", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200 },
    "QLD": { "symbol": "QLD", "name": "QLD (2倍)", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200 },
    "TQQQ": { "symbol": "TQQQ", "name": "TQQQ (3倍)", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 85, "rsi_len": 2, "ma_trend": 200 },
    "EDZ": { "symbol": "EDZ", "name": "EDZ (救援)", "mode": "BOLL_RSI", "entry_rsi": 9, "rsi_len": 2, "ma_trend": 20 },
    "SOXL_S": { "symbol": "SOXL", "name": "SOXL (狙擊)", "mode": "RSI_RSI", "entry_rsi": 10, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 100 },
    "SOXL_F": { "symbol": "SOXL", "name": "SOXL (快攻)", "mode": "KD", "entry_k": 10, "exit_k": 75 },
    "BTC_W": { "symbol": "BTC-USD", "name": "BTC (波段)", "mode": "RSI_RSI", "entry_rsi": 44, "exit_rsi": 65, "rsi_len": 14, "ma_trend": 200 },
    "BTC_F": { "symbol": "BTC-USD", "name": "BTC (閃電)", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 50, "rsi_len": 2, "ma_trend": 100 },
    "TSM": { "symbol": "TSM", "name": "TSM (趨勢)", "mode": "MA_CROSS", "fast_ma": 5, "slow_ma": 60 },
}

# ==========================================
# 6. 主執行介面
# ==========================================
with st.sidebar:
    st.header("⚙️ 設定")
    groq_key = st.text_input("Groq API Key (選填)", type="password")
    st.divider()
    option_list = list(strategies.keys())
    selected_keys = st.multiselect("選擇監控目標", option_list, default=option_list)
    if st.button("🚀 開始掃描"): st.rerun()

groq_client = None
if HAS_GROQ and groq_key: 
    try: groq_client = Groq(api_key=groq_key)
    except: pass

cols = st.columns(2)
for i, key in enumerate(selected_keys):
    cfg = strategies[key]
    col = cols[i % 2]
    with col.container(border=True):
        c1, c2 = st.columns([2, 1])
        c1.subheader(f"{cfg['name']}")
        df = get_safe_data(cfg['symbol'])
        sig, act, s_type, b_at, s_at = run_strategy(df, cfg)
        fund = get_fundamentals(cfg['symbol'])
        
        price = df['Close'].iloc[-1] if df is not None else 0
        chg = price - df['Close'].iloc[-2] if df is not None and len(df)>1 else 0
        c2.metric("Price", f"{price:,.2f}", f"{chg:+.2f}")
        
        sig_color = "green" if "BUY" in sig else "red" if "SELL" in sig else "gray"
        st.markdown(f"#### :{sig_color}[{sig}]")
        st.caption(f"策略: {act} | 掛買: {b_at} | 掛賣: {s_at}")

        tab1, tab2, tab3 = st.tabs(["🧪 WT 獨家發明", "🧬 基本面", "🤖 AI 委員會"])
        
        with tab1:
            if df is not None:
                st.plotly_chart(plot_chart(df, cfg), use_container_width=True)
                st.info("💡 WT (巨鯨推力)：>2 噴出(紅) / <-2 超跌(綠)。柱狀越高代表脫離成本越遠+資金越強。")
        
        with tab2:
            if fund:
                f1, f2, f3 = st.columns(3)
                f1.metric("PE", f"{fund['pe']:.1f}" if fund['pe'] else "-")
                f2.metric("機構持股", f"{fund['inst']*100:.0f}%")
                f3.metric("空單比", f"{fund['short']*100:.1f}%")
        
        with tab3:
            news = get_news(cfg['symbol'])
            if news:
                if st.button(f"🗳️ 召開 AI 投資委員會 ({cfg['symbol']})", key=f"btn_{key}"):
                    if groq_client:
                        with st.spinner("委員會辯論中..."):
                            price_ctx = f"Price: {price:.2f}, Signal: {sig}, Act: {act}"
                            res = analyze_deep_logic_2026(groq_client, cfg['symbol'], news, sig, act, price_ctx)
                            if res: st.markdown(res)
                    else:
                        st.warning("請先輸入 Groq API Key")
            else:
                st.info("無近期新聞")

st.caption("Auto-generated by 2026 Quant (WT Invention)")
