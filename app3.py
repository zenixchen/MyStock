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
    page_title="2026 量化戰情室 (VWAP+MFI版)",
    page_icon="🦅",
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

st.title("🦅 2026 量化戰情室 (VWAP 機構操盤版)")
st.caption("核心指標升級：VWAP (機構成本) + MFI (量能RSI) + NVI (聰明錢)")

if st.button('🔄 更新行情'):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 1. 數據核心
# ==========================================
def get_safe_data(ticker):
    try:
        # 下載數據
        df = yf.download(ticker, period="2y", interval="1d", progress=False, timeout=10)
        
        if df is None or df.empty: return None
        
        # 處理 MultiIndex
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        # 強制轉型
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
        
        # 取得即時價格
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
    News:
    {news_text}
    
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
    except Exception as e:
        return f"AI Error: {e}"

# ==========================================
# 3. 策略運算核心 (邏輯微調)
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
    
    # 計算 VWAP (做為全域參考)
    # 使用 Pandas TA 的 VWMA (Volume Weighted Moving Average) 來模擬日線級別的 VWAP 概念
    vwap_val = ta.vwma(c, df['Volume'], length=20).iloc[-1]
    vwap_status = "多方控盤" if lp > vwap_val else "空方控盤"

    # 1. RSI / FUSION
    if mode == "RSI_RSI" or mode == "FUSION":
        rsi_len = cfg.get('rsi_len', 14)
        rsi = ta.rsi(c, length=rsi_len).iloc[-1]
        entry_rsi = cfg.get('entry_rsi', 30)
        exit_rsi = cfg.get('exit_rsi', 70)
        
        b_price = find_rsi_price(df, entry_rsi, rsi_len)
        s_price = find_rsi_price(df, exit_rsi, rsi_len)
        b_at = f"${b_price:.2f}"; s_at = f"${s_price:.2f}"
        
        trend_ok = True
        if cfg.get('ma_trend', 0) > 0:
            ma = ta.ema(c, length=cfg['ma_trend']).iloc[-1]
            if lp < ma: trend_ok = False
        
        if rsi < entry_rsi:
            if trend_ok: sig="🔥 BUY"; act="低檔順勢"; s_type="BUY"
            else: sig="✋ WAIT"; act="低檔逆勢"; s_type="WAIT"
        elif rsi > exit_rsi:
            sig="💰 SELL"; act="高檔過熱"; s_type="SELL"
        else:
            act = f"震盪中 (RSI:{rsi:.1f})"

    # 2. RSI_MA
    elif mode == "RSI_MA":
        rsi_len = cfg.get('rsi_len', 14)
        rsi = ta.rsi(c, length=rsi_len).iloc[-1]
        exit_ma_val = ta.sma(c, length=cfg.get('exit_ma', 20)).iloc[-1]
        entry_rsi = cfg.get('entry_rsi', 30)
        
        b_price = find_rsi_price(df, entry_rsi, rsi_len)
        b_at = f"${b_price:.2f}"; s_at = f"${exit_ma_val:.2f} (MA)"
        
        if rsi < entry_rsi: sig="🔥 BUY"; act="RSI低檔佈局"; s_type="BUY"
        elif lp > exit_ma_val:
            if rsi > 80: sig="💰 SELL"; act="突破均線且過熱"; s_type="SELL"
            else: act="持有 (均線之上)"
        else: act = f"等待 (RSI:{rsi:.1f})"

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
        rsi = ta.rsi(c, length=cfg.get('rsi_len', 14)).iloc[-1]
        bb = ta.bbands(c, length=20, std=2)
        low_b = bb.iloc[-1, 0]; up_b = bb.iloc[-1, 2]
        b_at = f"${low_b:.2f}"; s_at = f"${up_b:.2f}"
        if lp < low_b and rsi < cfg['entry_rsi']: sig="🚑 BUY"; act="破底搶反彈"; s_type="BUY"
        elif lp >= up_b: sig="💀 SELL"; act="觸頂回調"; s_type="SELL"
        else: act="通道震盪"

    # 補充 VWAP 狀態
    act += f" | {vwap_status}"

    return sig, act, s_type, b_at, s_at

# ==========================================
# 4. 視覺化 (★ 新增 VWAP & MFI & NVI)
# ==========================================
def plot_chart(df, cfg, signals=None):
    if df is None: return None
    
    # 1. 計算 VWAP (日線級別使用 VWMA 代替，或計算 Anchor VWAP)
    # 這裡使用 20日 VWMA 作為月線級別的機構成本參考
    df['VWAP'] = ta.vwma(df['Close'], df['Volume'], length=20)
    
    # 2. 計算 MFI (Money Flow Index)
    # MFI 是 "量化的 RSI"，0-100，比 CMF 更直觀
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    
    # 3. 巨鯨成交量配色 (維持 Z-Score 邏輯)
    cmf = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20).fillna(0)
    rolling_mean = cmf.rolling(window=60).mean()
    rolling_std = cmf.rolling(window=60).std()
    z_score = (cmf - rolling_mean) / (rolling_std + 1e-9)
    
    vol_colors = []
    for i in range(len(df)):
        z = z_score.iloc[i]
        val = cmf.iloc[i]
        is_up = df['Close'].iloc[i] >= df['Open'].iloc[i]
        if z > 2.0 and val > 0: c = '#ffd700' # 金色巨鯨
        elif z < -2.0 and val < 0: c = '#9c27b0' # 紫色拋售
        else: c = '#089981' if is_up else '#f23645'
        vol_colors.append(c)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Row 1: K線 + VWAP
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', increasing_line_color='#089981', decreasing_line_color='#f23645'
    ), row=1, col=1)
    
    # 繪製 VWAP (機構成本線) - 黃色粗線
    if 'VWAP' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP (機構成本)', line=dict(color='#FFD700', width=2)), row=1, col=1)
    
    if cfg.get('ma_trend', 0) > 0:
        ma = ta.ema(df['Close'], length=cfg['ma_trend'])
        fig.add_trace(go.Scatter(x=df.index, y=ma, name=f'EMA{cfg["ma_trend"]}', line=dict(color='orange', width=1)), row=1, col=1)

    # Row 2: MFI (取代原本的 RSI/KD)
    # MFI 比 RSI 多了成交量資訊，更難鈍化
    if 'MFI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], name='MFI (資金流)', line=dict(color='#00e676', width=1.5)), row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color='green', annotation_text="超賣 (吸籌)", row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color='red', annotation_text="過熱 (倒貨)", row=2, col=1)
    
    # 輔助顯示 RSI (淡色) 供參考
    if "RSI" in cfg['mode']:
        rsi = ta.rsi(df['Close'], length=cfg.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='rgba(255, 255, 255, 0.3)', width=1)), row=2, col=1)

    # Row 3: 巨鯨成交量
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume (Whale)', marker_color=vol_colors
    ), row=3, col=1)

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

        tab1, tab2, tab3 = st.tabs(["📊 圖表 (VWAP+MFI)", "🧬 聰明錢 NVI", "🤖 AI 委員會"])
        
        with tab1:
            if df is not None:
                st.plotly_chart(plot_chart(df, cfg), use_container_width=True)
                st.caption("🟡 黃線 = VWAP (機構成本) | 🟢 綠線 = MFI (資金流 RSI)")
        
        with tab2:
            # 計算 NVI (聰明錢指標)
            if df is not None:
                nvi = ta.nvi(df['Close'], df['Volume'])
                nvi_ema = ta.ema(nvi, length=255) # NVI 年線
                
                # 簡單繪圖
                nvi_df = pd.DataFrame({'NVI': nvi, 'NVI_EMA': nvi_ema}).dropna()
                st.line_chart(nvi_df.tail(100))
                
                curr_nvi = nvi.iloc[-1]
                curr_ema = nvi_ema.iloc[-1] if not nvi_ema.isna().all() else 0
                
                if curr_nvi > curr_ema:
                    st.success(f"✅ 聰明錢 (NVI) 趨勢向上：主力吸籌中")
                else:
                    st.error(f"⚠️ 聰明錢 (NVI) 趨勢向下：主力觀望或出貨")

            if fund:
                st.divider()
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

st.caption("Auto-generated by 2026 Quant (VWAP+MFI Enhanced)")
