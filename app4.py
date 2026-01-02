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
import time

# ==========================================
# ★ 0. 系統設定
# ==========================================
st.set_page_config(
    page_title="2026 量化戰情室 (WT進化版)",
    page_icon="🐋",
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

# CSS 美化 (TradingView 風格)
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
        .big-font { font-size:24px !important; font-weight: bold; color: #2962ff; }
    </style>
""", unsafe_allow_html=True)

st.title("🐋 2026 量化戰情室 (WT 進化版)")
st.caption("v2.0 新增功能：歷史回測系統 | 自動掃描器 | AI 深度解讀")

if st.button('🔄 更新全市場行情'):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 1. 數據核心
# ==========================================
def get_safe_data(ticker):
    try:
        # 下載 1.5 年數據以確保回測樣本足夠
        df = yf.download(ticker, period="2y", interval="1d", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
        
        # 嘗試補上最新即時盤 (如果是盤中)
        try:
            t = yf.Ticker(ticker)
            live_price = t.fast_info.get('last_price')
            if live_price and not np.isnan(live_price):
                last_date = df.index[-1].date()
                now_date = datetime.now().date()
                # 簡單判斷：如果最後一筆不是今天，且有即時價，就補一根 K 棒
                if last_date != now_date:
                    new_idx = pd.Timestamp.now()
                    if df.index.tz is not None: new_idx = new_idx.tz_localize(df.index.tz)
                    new_row = pd.DataFrame({
                        'Open': [live_price], 'High': [live_price], 
                        'Low': [live_price], 'Close': [live_price], 
                        'Volume': [0.0]
                    }, index=[new_idx])
                    df = pd.concat([df, new_row])
                else:
                    # 如果今天是最後一筆，直接更新收盤價
                    df.loc[df.index[-1], 'Close'] = float(live_price)
        except: pass
        return df
    except Exception as e:
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
        return [clean_text(n.get('title','')) for n in news[:3]] if news else []
    except: return []

def analyze_deep_logic_2026(client, symbol, news_list, signal, action, price_context, wt_val):
    if not client: return None
    news_text = "\n".join([f"- {n}" for n in news_list])
    
    # 讓 AI 理解 WT 指標
    wt_desc = ""
    if wt_val > 2: wt_desc = "WT指標顯示『極度噴出』，注意乖離過大風險，但也代表動能極強。"
    elif wt_val < -2: wt_desc = "WT指標顯示『極度超跌』，恐慌殺盤，可能是反彈契機。"
    elif wt_val > 0: wt_desc = "WT指標 > 0，多方控盤中。"
    else: wt_desc = "WT指標 < 0，空方控盤中。"

    prompt = f"""
    You are a Hedge Fund AI Analyst using the 'Whale Thrust (WT)' indicator.
    Target: {symbol}
    Tech Signal: {signal} ({action})
    WT Indicator: {wt_val:.2f} ({wt_desc})
    Context: {price_context}
    Recent News: 
    {news_text}
    
    Output in Traditional Chinese Markdown:
    ### 🐋 巨鯨 AI 投資報告 ({symbol})
    **📊 WT 動能解讀**: {wt_desc} (解釋這對股價意味著什麼)
    **📰 新聞與基本面**: (結合新聞分析)
    **🛡️ 風控建議**: (止損或加碼建議)
    ---
    **🎯 最終決策**: [Strong Buy / Buy / Hold / Sell]
    """
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3, max_tokens=800
        )
        return resp.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# ==========================================
# 3. 策略運算核心 (WT + 回測)
# ==========================================
def calculate_wt(df):
    # WT 公式 = ((Close - VWAP) / ATR) * (MFI / 50)
    vwap = ta.vwma(df['Close'], df['Volume'], length=20)
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).fillna(1) # 防止除以0
    mfi = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14).fillna(50)
    
    # 避免 ATR 為 0
    atr = atr.replace(0, 1)
    
    wt = ((df['Close'] - vwap) / atr) * (mfi / 50)
    return wt

# ==========================================
# ★★★ v2.1 改良版回測系統 ★★★
# ==========================================
def backtest_wt_strategy(df):
    """
    改良策略：
    1. 進場：WT > 0 且 股價站上 20MA (趨勢確認)
    2. 出場：股價跌破 20MA (趨勢反轉) 或 WT < -1 (動能衰退)
    """
    if df is None or len(df) < 50: return None
    
    # 計算指標
    wt = calculate_wt(df)
    close = df['Close']
    ma20 = ta.sma(close, length=20)
    
    signals = pd.Series(0, index=df.index)
    
    # 進場條件：WT翻正 + 站上均線 (雙重確認，過濾假訊號)
    buy_cond = (wt > 0) & (close > ma20)
    
    # 出場條件：跌破均線 (趨勢結束)
    sell_cond = (close < ma20)
    
    # 生成訊號
    signals[buy_cond] = 1
    signals[sell_cond] = -1
    
    # 執行回測 (向量化邏輯轉為逐日模擬)
    pos = 0; ent = 0; wins = 0; trds = 0; rets = []
    
    for i in range(1, len(df)):
        # 空手 -> 買進
        if pos == 0 and signals.iloc[i] == 1:
            pos = 1; ent = close.iloc[i]
        
        # 持倉 -> 賣出
        elif pos == 1 and signals.iloc[i] == -1:
            pos = 0
            r = (close.iloc[i] - ent) / ent
            # 扣除手續費滑價成本 (假設單邊 0.1%)
            r = r - 0.002 
            rets.append(r); trds += 1
            if r > 0: wins += 1
            
    # 如果最後還持有，以最後一根收盤價結算
    if pos == 1:
        r = (close.iloc[-1] - ent) / ent
        rets.append(r); trds += 1
        if r > 0: wins += 1

    total_ret = sum(rets) * 100
    win_rate = (wins / trds * 100) if trds > 0 else 0
    
    # 計算最大回撤 (MDD) - 加分題
    cum_ret = np.cumsum(rets)
    try:
        peak = np.maximum.accumulate(cum_ret)
        drawdown = peak - cum_ret
        mdd = drawdown.max() if len(drawdown) > 0 else 0
    except: mdd = 0

    return {"Return": total_ret, "WinRate": win_rate, "Trades": trds, "MDD": mdd}

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
    if df is None: return "ERR", "無數據", "ERR", "---", "---", 0
    
    c = df['Close']; lp = c.iloc[-1]
    sig="WAIT"; act="觀望"; s_type="WAIT"; b_at="---"; s_at="---"
    mode = cfg['mode']
    
    # 計算 WT 用於診斷
    wt_series = calculate_wt(df)
    curr_wt = wt_series.iloc[-1]

    # WT 狀態描述
    wt_status = ""
    if curr_wt > 2.0: wt_status = " | 🚀WT噴出"
    elif curr_wt < -2.0: wt_status = " | 💎WT超跌"
    elif curr_wt > 0: wt_status = " | 🟢多方"
    else: wt_status = " | 🔴空方"

    # --- 傳統策略邏輯 ---
    if mode == "RSI_RSI" or mode == "FUSION":
        rsi = ta.rsi(c, length=cfg.get('rsi_len', 14))
        curr_rsi = rsi.iloc[-1]
        b_at = f"${find_rsi_price(df, cfg.get('entry_rsi', 30), 14):.2f}"
        
        if curr_rsi < cfg.get('entry_rsi', 30):
            sig="🔥 BUY"; act="RSI低檔"; s_type="BUY"
        elif curr_rsi > cfg.get('exit_rsi', 70):
            sig="💰 SELL"; act="RSI過熱"; s_type="SELL"
        else: act = f"RSI:{curr_rsi:.1f}"

    elif mode == "KD":
        k = ta.stoch(df['High'], df['Low'], c, k=9, d=3).iloc[-1, 0]
        b_at = f"K<{cfg['entry_k']}"
        if k < cfg['entry_k']: sig="🚀 BUY"; act=f"KD低檔({k:.1f})"; s_type="BUY"
        elif k > cfg['exit_k']: sig="💀 SELL"; act=f"KD高檔({k:.1f})"; s_type="SELL"
        else: act = f"K:{k:.1f}"
    
    elif mode == "SUPERTREND":
        st_val = ta.supertrend(df['High'], df['Low'], c, length=cfg['period'], multiplier=cfg['multiplier'])
        if st_val is not None:
            dr = st_val.iloc[-1, 1]
            if dr == 1: sig="✊ HOLD"; act="多頭續抱"; s_type="HOLD"
            else: sig="☁️ EMPTY"; act="空頭觀望"; s_type="EMPTY"

    elif mode == "MA_CROSS":
        f = ta.sma(c, cfg['fast_ma']); s = ta.sma(c, cfg['slow_ma'])
        if f.iloc[-1] > s.iloc[-1]: sig="✊ HOLD"; act="多頭排列"; s_type="HOLD"
        else: sig="☁️ EMPTY"; act="空頭排列"; s_type="EMPTY"

    # 疊加 WT 狀態
    act += wt_status
    return sig, act, s_type, b_at, s_at, curr_wt

# ==========================================
# 4. 視覺化
# ==========================================
def plot_chart(df, cfg):
    if df is None: return None
    
    wt = calculate_wt(df)
    colors = ['#ff1744' if v > 2 else '#00e676' if v < -2 else '#ef5350' if v > 0 else '#66bb6a' for v in wt]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Row 1
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    vwap = ta.vwma(df['Close'], df['Volume'], length=20)
    fig.add_trace(go.Scatter(x=df.index, y=vwap, name='VWAP', line=dict(color='#FFD700', width=1)), row=1, col=1)

    # Row 2 (Indicators)
    if "RSI" in cfg['mode'] or cfg['mode'] == "FUSION":
        rsi = ta.rsi(df['Close'], length=14)
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color='green', row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color='red', row=2, col=1)
    elif cfg['mode'] == "KD":
        k = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
        fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:,0], name='K', line=dict(color='yellow')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:,1], name='D', line=dict(color='blue')), row=2, col=1)

    # Row 3 (WT)
    fig.add_trace(go.Bar(x=df.index, y=wt, name='WT', marker_color=colors), row=3, col=1)
    fig.add_hline(y=2.0, line_dash="dot", line_color='red', row=3, col=1)
    fig.add_hline(y=-2.0, line_dash="dot", line_color='green', row=3, col=1)
    fig.add_hline(y=0, line_color='gray', row=3, col=1)

    fig.update_layout(height=600, margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor='#161b22', plot_bgcolor='#161b22', font=dict(color='#d1d4dc'), showlegend=False, xaxis_rangeslider_visible=False)
    return fig

# ==========================================
# 5. 監控名單
# ==========================================
strategies = {
    "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (美元)", "mode": "KD", "entry_k": 25, "exit_k": 70 },
    "KO": { "symbol": "KO", "name": "KO (可樂)", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90 },
    "BA": { "symbol": "BA", "name": "BA (波音)", "mode": "RSI_RSI", "rsi_len": 14, "entry_rsi": 25, "exit_rsi": 65 },
    "META": { "symbol": "META", "name": "META (暴力反彈)", "mode": "RSI_RSI", "entry_rsi": 40, "exit_rsi": 90 },
    "NVDA": { "symbol": "NVDA", "name": "NVDA (聖杯)", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90 },
    "GOOGL": { "symbol": "GOOGL", "name": "GOOGL (聖杯)", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90 },
    "QQQ": { "symbol": "QQQ", "name": "QQQ (穩健)", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20 },
    "QLD": { "symbol": "QLD", "name": "QLD (2倍)", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20 },
    "TQQQ": { "symbol": "TQQQ", "name": "TQQQ (3倍)", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 85 },
    "EDZ": { "symbol": "EDZ", "name": "EDZ (救援)", "mode": "BOLL_RSI", "entry_rsi": 9, "rsi_len": 2 },
    "SOXL_S": { "symbol": "SOXL", "name": "SOXL (狙擊)", "mode": "RSI_RSI", "entry_rsi": 10, "exit_rsi": 90 },
    "SOXL_F": { "symbol": "SOXL", "name": "SOXL (快攻)", "mode": "KD", "entry_k": 10, "exit_k": 75 },
    "BTC_W": { "symbol": "BTC-USD", "name": "BTC (波段)", "mode": "RSI_RSI", "entry_rsi": 44, "exit_rsi": 65 },
    "TSM": { "symbol": "TSM", "name": "TSM (趨勢)", "mode": "MA_CROSS", "fast_ma": 5, "slow_ma": 60 },
}

# ==========================================
# 6. 主執行介面
# ==========================================
with st.sidebar:
    st.header("⚙️ 控制台")
    groq_key = st.text_input("Groq API Key (選填)", type="password")
    
    st.divider()
    st.markdown("### 🕵️‍♂️ WT 自動掃描器")
    if st.button("🚀 掃描全市場 (WT訊號)"):
        st.write("掃描中...")
        scan_results = []
        progress_bar = st.progress(0)
        total = len(strategies)
        for i, (key, cfg) in enumerate(strategies.items()):
            df = get_safe_data(cfg['symbol'])
            if df is not None:
                wt = calculate_wt(df).iloc[-1]
                if wt > 2.0: scan_results.append((cfg['name'], wt, "🚀 噴出"))
                elif wt < -2.0: scan_results.append((cfg['name'], wt, "💎 超跌機遇"))
            progress_bar.progress((i+1)/total)
        
        if scan_results:
            st.success(f"發現 {len(scan_results)} 個機會！")
            for res in scan_results:
                st.write(f"**{res[0]}**: WT={res[1]:.2f} ({res[2]})")
        else:
            st.info("目前無極端 WT 訊號")
    
    st.divider()
    selected_keys = st.multiselect("監控清單", list(strategies.keys()), default=list(strategies.keys()))

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
        
        # 取得 WT 與策略訊號
        sig, act, s_type, b_at, s_at, curr_wt = run_strategy(df, cfg)
        
        # 價格顯示
        price = df['Close'].iloc[-1] if df is not None else 0
        chg = price - df['Close'].iloc[-2] if df is not None and len(df)>1 else 0
        c2.metric("Price", f"{price:,.2f}", f"{chg:+.2f}")
        
        # 訊號顯示
        sig_color = "green" if "BUY" in sig else "red" if "SELL" in sig else "gray"
        st.markdown(f"#### :{sig_color}[{sig}]")
        st.caption(f"{act}")

        tab1, tab2, tab3 = st.tabs(["🧪 WT 圖表", "📊 歷史回測", "🤖 AI 決策"])
        
        with tab1:
            if df is not None:
                st.plotly_chart(plot_chart(df, cfg), use_container_width=True)
                if curr_wt > 2: st.warning("⚠️ WT > 2：動能極強但需防乖離，適合移動停利。")
                elif curr_wt < -2: st.success("💎 WT < -2：恐慌殺盤區，注意反彈機會。")
        
        with tab2:
            if st.button("執行回測 (v2.1 改良版)", key=f"bt_{key}"):
                res = backtest_wt_strategy(df)
                if res:
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("總報酬", f"{res['Return']:.1f}%", delta_color="normal")
                    b2.metric("勝率", f"{res['WinRate']:.0f}%")
                    b3.metric("最大回撤", f"{res['MDD']:.1f}%", delta_color="inverse") # 越小越好
                    b4.metric("交易次數", res['Trades'])
                    st.caption("策略邏輯：WT > 0 且 站上20MA 買進；跌破 20MA 賣出 (含手續費模擬)。")
                else: st.error("數據不足無法回測")
        
        with tab3:
            if st.button(f"🗳️ AI 委員會分析", key=f"ai_{key}"):
                if groq_client:
                    news = get_news(cfg['symbol'])
                    with st.spinner("AI 正在解讀 WT 指標與新聞..."):
                        price_ctx = f"Price: {price:.2f}, Signal: {sig}"
                        res = analyze_deep_logic_2026(groq_client, cfg['symbol'], news, sig, act, price_ctx, curr_wt)
                        if res: st.markdown(res)
                else: st.warning("請先輸入 API Key")

st.caption("Auto-generated by 2026 Quant (Evolution v2.0)")
