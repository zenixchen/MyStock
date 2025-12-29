import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
# ★ 深度學習 NLP 套件 (備用)
from transformers import pipeline

# ==========================================
# ★★★ LLM 設定區 (Groq) ★★★
# ==========================================
try:
    from groq import Groq
    # 預設不填，讓使用者在側邊欄填入
    GROQ_API_KEY_DEFAULT = "" 
except ImportError:
    GROQ_API_KEY_DEFAULT = ""

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(
    page_title="2025 量化戰情室 (LLM 邏輯版)",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        h1, h2, h3, h4, h5, h6, span, div { color: #e0e0e0; font-family: 'Roboto', sans-serif; }
        div[data-testid="stMetric"] { background-color: #1c202a; border: 1px solid #2d3342; border-radius: 8px; }
        section[data-testid="stSidebar"] { background-color: #161920; }
        .stButton > button { background-color: #2962ff; color: white; border: none; font-weight: bold; }
        .stButton > button:hover { background-color: #1e4bd1; }
        .streamlit-expanderHeader { background-color: #1c202a; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("💎 2025 全明星量化戰情室 (LLM 邏輯版)")
st.caption("15檔核心持股 (原始策略) + LLM 新聞邏輯推演 | 不更動任何參數")

# ==========================================
# 1. 核心函數 (資料獲取)
# ==========================================
def get_real_live_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get('last_price')
        if price is None or np.isnan(price):
            suffix = "1d" if "-USD" in symbol else "5d"
            df_rt = yf.download(symbol, period=suffix, interval="1m", progress=False, timeout=5)
            if df_rt.empty: return None
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            return float(df_rt['Close'].iloc[-1])
        return float(price)
    except: return None

def get_safe_data(ticker):
    try:
        # 下載 5 年數據 (維持您原本的設定)
        df = yf.download(ticker, period="5y", interval="1d", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except: return None

def get_news_content(symbol):
    """抓取新聞標題供 LLM 分析"""
    try:
        if "=" in symbol or "^" in symbol: return []
        stock = yf.Ticker(symbol)
        news = stock.news
        if not news: return []
        return [n.get('title', n.get('content', {}).get('title', '')) for n in news[:3]]
    except: return []

# ==========================================
# 2. 基本面與 FinBERT (保留原本功能)
# ==========================================
@st.cache_data(ttl=86400)
def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol or "-USD" in symbol: return None 
        stock = yf.Ticker(symbol)
        info = stock.info
        if info.get('quoteType', '').upper() != 'EQUITY': return None
        return {
            "growth": info.get('revenueGrowth', 0), 
            "pe": info.get('trailingPE', None), 
            "eps": info.get('trailingEps', None), 
            "inst": info.get('heldPercentInstitutions', 0),
            "short": info.get('shortPercentOfFloat', 0)
        }
    except: return None

@st.cache_resource
def load_finbert_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment_finbert(symbol):
    try:
        if "=" in symbol or "^" in symbol: return 0, "無新聞", []
        stock = yf.Ticker(symbol)
        news_list = stock.news
        if not news_list: return 0, "無新聞", []
        
        classifier = load_finbert_model()
        texts = [n.get('title', '') for n in news_list[:5] if n.get('title')]
        if not texts: return 0, "無新聞", []

        results = classifier(texts)
        total_score = 0
        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        debug_logs = []
        
        for i, res in enumerate(results):
            val = score_map[res['label']] * res['score']
            total_score += val
            icon = "🔥" if res['label']=="positive" else "❄️" if res['label']=="negative" else "⚪"
            debug_logs.append(f"{icon} {res['label'][:3]} {res['score']:.2f}: {texts[i]}")
            
        return total_score/len(texts), texts[0], debug_logs
    except Exception as e: return 0, str(e), []

# ==========================================
# 3. LLM 邏輯分析 (Groq) - 新增功能
# ==========================================
def analyze_logic_llm(client, symbol, news_titles, tech_signal):
    if not client or not news_titles: return "無 AI 分析 (未連線或無新聞)", "⚪", False
    try:
        news_text = "\n".join([f"- {t}" for t in news_titles])
        prompt = f"""
        你是專業操盤手。分析 {symbol}。
        新聞：
        {news_text}
        
        技術面訊號：
        {tech_signal}
        
        請用繁體中文回答：
        1. 一句話總結多空邏輯 (50字內)。
        2. 情緒評分 (-10悲觀 ~ +10樂觀)。
        3. 操作建議 (做多/觀望/做空)。
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", temperature=0.3,
        )
        return chat_completion.choices[0].message.content, "🤖", True
    except Exception as e: return f"LLM Error: {str(e)}", "⚠️", False

# ==========================================
# 4. 技術指標與優化 (保留您原本的 Grid Search)
# ==========================================
def optimize_rsi_strategy(df, symbol):
    """(保留) 針對指定股票，暴力測試 RSI 參數組合"""
    if df is None or df.empty: return None
    rsi_lengths = [6, 12, 14, 20]; entries = [20, 25, 30, 40]; exits = [60, 70, 75, 85]
    results = []
    
    prog_text = f"AI 正在優化 {symbol}..."
    my_bar = st.progress(0, text=prog_text)
    total = len(rsi_lengths)*len(entries)*len(exits); count=0
    
    close = df['Close'].values
    for l in rsi_lengths:
        rsi = ta.rsi(df['Close'], length=l)
        if rsi is None: continue
        rsi_val = rsi.values
        for ent in entries:
            for ext in exits:
                count+=1; my_bar.progress(count/total)
                sig = np.zeros(len(close)); pos=0; entry=0; wins=0; trds=0; ret_tot=0
                
                # Numpy 加速回測
                sig[rsi_val < ent] = 1; sig[rsi_val > ext] = -1
                for i in range(len(close)):
                    if pos==0 and sig[i]==1: pos=1; entry=close[i]
                    elif pos==1 and sig[i]==-1:
                        pos=0; r=(close[i]-entry)/entry; ret_tot+=r; trds+=1
                        if r>0: wins+=1
                
                if trds>0:
                    results.append({"Length": l, "Buy": ent, "Sell": ext, "Return": ret_tot*100, "WinRate": wins/trds*100, "Trades": trds})
    
    my_bar.empty()
    return pd.DataFrame(results)

def find_price_for_rsi(df, target_rsi, length=2):
    if df is None or df.empty: return 0
    last_close = df['Close'].iloc[-1]
    low, high = last_close * 0.4, last_close * 1.6
    temp_df = df.copy()
    for _ in range(10): 
        mid = (low + high) / 2
        new_row = pd.DataFrame({'Close': [mid]}, index=[df.index[-1] + pd.Timedelta(days=1)])
        sim_series = pd.concat([temp_df['Close'], new_row['Close']])
        rsi = ta.rsi(sim_series, length=length).iloc[-1]
        if rsi > target_rsi: high = mid
        else: low = mid
    return round(mid, 2)

def predict_volatility(df):
    try:
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        if atr is None: return None, None
        return df['Close'].iloc[-1] + atr.iloc[-1], df['Close'].iloc[-1] - atr.iloc[-1]
    except: return None, None

def analyze_chips_volume(df, inst_percent, short_percent):
    try:
        obv = ta.obv(df['Close'], df['Volume'])
        if obv is None or len(obv) < 20: return "無數據"
        msg = "🔴 籌碼流入" if obv.iloc[-1] > ta.sma(obv, length=20).iloc[-1] else "🟢 籌碼渙散"
        if short_percent and short_percent > 0.2: msg += f" | ⚠️ 軋空警戒 ({short_percent*100:.1f}%)"
        return msg
    except: return "計算錯誤"

# ==========================================
# 5. 主分析邏輯 (100% 原始邏輯 + LLM)
# ==========================================
def analyze_ticker(config, groq_client=None):
    symbol = config['symbol']
    df = get_safe_data(symbol)
    if df is None: return {"Symbol": symbol, "Name": config['name'], "Signal": "ERR", "Price": 0, "Raw_DF": None, "Type": "ERR"}

    lp = get_real_live_price(symbol) or df['Close'].iloc[-1]
    prev_c = df['Close'].iloc[-1]
    
    # 模擬今日 K 線
    new_row = pd.DataFrame({'Close': [lp], 'High': [max(lp, df['High'].iloc[-1])], 'Low': [min(lp, df['Low'].iloc[-1])], 'Open': [lp], 'Volume': [0]}, index=[pd.Timestamp.now()])
    calc_df = pd.concat([df.copy(), new_row])
    c, h, l = calc_df['Close'], calc_df['High'], calc_df['Low']
    
    sig = "WAIT"; act = "觀望"; buy_at = "---"; sell_at = "---"; sig_type = "WAIT"
    
    # ★★★ 策略邏輯 (完全保留您的原始判斷) ★★★
    if config['mode'] == "SUPERTREND":
        st = ta.supertrend(h, l, c, length=config['period'], multiplier=config['multiplier'])
        if st is not None:
            dr = st.iloc[-1, 1]; p_dr = st.iloc[-2, 1]; st_val = st.iloc[-1, 0]
            sell_at = f"${st_val:.2f}"
            if p_dr == -1 and dr == 1: sig = "🚀 BUY"; act = "趨勢翻多"; sig_type="BUY"
            elif p_dr == 1 and dr == -1: sig = "📉 SELL"; act = "趨勢翻空"; sig_type="SELL"
            elif dr == 1: sig = "✊ HOLD"; act = f"多頭續抱 (損{st_val:.1f})"; sig_type="HOLD"
            else: sig = "☁️ EMPTY"; act = "空頭觀望"; sig_type="EMPTY"

    elif config['mode'] == "FUSION":
        rsi = ta.rsi(c, length=config['rsi_len']).iloc[-1]
        ma = ta.ema(c, length=config['ma_trend']).iloc[-1]
        buy_at = f"${find_price_for_rsi(df, config['entry_rsi'], config['rsi_len'])}"
        sell_at = f"${find_price_for_rsi(df, config['exit_rsi'], config['rsi_len'])}"
        
        if lp > ma and rsi < config['entry_rsi']: sig = "🔥 BUY"; act = "趨勢回檔超跌"; sig_type="BUY"
        elif rsi > config['exit_rsi']: sig = "💰 SELL"; act = "RSI過熱獲利"; sig_type="SELL"
        else: act = f"趨勢多頭 (RSI:{rsi:.1f})"

    elif config['mode'] in ["RSI_RSI", "RSI_MA"]:
        rsi = ta.rsi(c, length=config.get('rsi_len', 14)).iloc[-1]
        # RSI_MA / RSI_RSI 邏輯
        buy_at = f"${find_price_for_rsi(df, config['entry_rsi'], config.get('rsi_len', 14))}"
        
        if config['mode'] == "RSI_RSI":
            sell_at = f"${find_price_for_rsi(df, config['exit_rsi'], config.get('rsi_len', 14))}"
            if rsi < config['entry_rsi']: sig = "🔥 BUY"; act = f"RSI低檔 ({rsi:.1f})"; sig_type="BUY"
            elif rsi > config['exit_rsi']: sig = "💰 SELL"; act = f"RSI高檔 ({rsi:.1f})"; sig_type="SELL"
            else: act = f"區間震盪 (RSI:{rsi:.1f})"
        else:
            s_val = ta.sma(c, length=config['exit_ma']).iloc[-1]
            sell_at = f"${s_val:.2f}"
            if rsi < config['entry_rsi']: sig = "🔥 BUY"; act = "短線超賣"; sig_type="BUY"
            elif lp > s_val: sig = "💰 SELL"; act = "觸及均線壓力"; sig_type="SELL"

    elif config['mode'] == "KD":
        k = ta.stoch(h, l, c, k=9, d=3).iloc[-1, 0]
        buy_at = f"K<{config['entry_k']}"; sell_at = f"K>{config['exit_k']}"
        if k < config['entry_k']: sig = "🚀 BUY"; act = f"KD低檔 ({k:.1f})"; sig_type="BUY"
        elif k > config['exit_k']: sig = "💀 SELL"; act = f"KD高檔 ({k:.1f})"; sig_type="SELL"
        else: act = f"盤整中 (K:{k:.1f})"

    elif config['mode'] == "MA_CROSS":
        f, s = ta.sma(c, config['fast_ma']), ta.sma(c, config['slow_ma'])
        curr_f, prev_f = f.iloc[-1], f.iloc[-2]; curr_s, prev_s = s.iloc[-1], s.iloc[-2]
        if prev_f <= prev_s and curr_f > curr_s: sig = "🔥 BUY"; act = "黃金交叉"; sig_type="BUY"
        elif prev_f >= prev_s and curr_f < curr_s: sig = "📉 SELL"; act = "死亡交叉"; sig_type="SELL"
        elif curr_f > curr_s: sig = "✊ HOLD"; act = "多頭排列"; sig_type="HOLD"
        else: sig = "☁️ EMPTY"; act = "空頭排列"; sig_type="EMPTY"
        
    elif config['mode'] == "BOLL_RSI":
        rsi = ta.rsi(c, length=config.get('rsi_len', 2)).iloc[-1]
        bb = ta.bbands(c, length=20, std=2)
        lower = bb.iloc[-1, 0]; mid = bb.iloc[-1, 1]; upper = bb.iloc[-1, 2]
        buy_at = f"${lower:.2f}"; sell_at = f"${mid:.2f}"
        if lp < lower and rsi < config['entry_rsi']: sig = "🚑 BUY"; act = "破底搶反彈"; sig_type="BUY"
        elif lp >= upper: sig = "💀 SELL"; act = "觸上軌快逃"; sig_type="SELL"
        elif lp >= mid: sig = "⚠️ HOLD"; act = "中軸震盪"; sig_type="HOLD"

    # 基本面與其他
    fund = get_fundamentals(symbol)
    fund_msg = f"PE: {fund['pe']:.1f}" if fund and fund['pe'] else "N/A"
    
    # LLM 分析
    llm_res = "未啟用 LLM"; is_llm = False
    if groq_client:
        news = get_news_content(symbol)
        tech_ctx = f"目前 ${lp:.2f}。訊號: {sig} ({act})。"
        llm_res, _, is_llm = analyze_logic_llm(groq_client, symbol, news, tech_ctx)
    else:
        # 降級使用 FinBERT
        news = get_news_content(symbol)
        score, _, logs = analyze_sentiment_finbert(symbol)
        llm_res = f"情緒分: {score:.2f} (無 Groq Key)"; is_llm = False

    p_high, p_low = predict_volatility(df)
    pred_msg = f"${p_low:.2f}~${p_high:.2f}" if p_high else ""
    chip_msg = analyze_chips_volume(df, fund['inst'] if fund else 0, fund['short'] if fund else 0)

    return {
        "Symbol": symbol, "Name": config['name'], "Price": lp, "Prev_Close": prev_c,
        "Signal": sig, "Action": act, "Type": sig_type, "Buy_At": buy_at, "Sell_At": sell_at,
        "Fund": fund_msg, "LLM_Analysis": llm_res, "Is_LLM": is_llm, "Raw_DF": df,
        "Pred": pred_msg, "Chip": chip_msg
    }

# ==========================================
# 6. 視覺化
# ==========================================
def plot_chart(df, config, signals=None):
    if df is None: return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25])
    
    # K線
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    
    # 策略指標
    if config['mode'] == "SUPERTREND":
        st = ta.supertrend(df['High'], df['Low'], df['Close'], length=config['period'], multiplier=config['multiplier'])
        if st is not None: fig.add_trace(go.Scatter(x=df.index, y=st[st.columns[0]], name='SuperTrend', line=dict(color='orange')), row=1, col=1)
    elif config['mode'] == "MA_CROSS":
        f = ta.sma(df['Close'], config['fast_ma']); s = ta.sma(df['Close'], config['slow_ma'])
        fig.add_trace(go.Scatter(x=df.index, y=f, line=dict(color='yellow')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=s, line=dict(color='blue')), row=1, col=1)
        
    # 副圖
    if "RSI" in config['mode'] or config['mode'] == "FUSION" or config['mode'] == "BOLL_RSI":
        rsi = ta.rsi(df['Close'], length=config.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='#b39ddb')), row=2, col=1)
        fig.add_hline(y=config.get('entry_rsi', 30), line_color='green', row=2, col=1)
        fig.add_hline(y=config.get('exit_rsi', 70), line_color='red', row=2, col=1)
    elif config['mode'] == "KD":
        k = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
        if k is not None:
            fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:, 0], name='K', line=dict(color='yellow')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:, 1], name='D', line=dict(color='blue')), row=2, col=1)

    fig.update_layout(height=450, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='#131722', plot_bgcolor='#131722', font=dict(color='white'), showlegend=False)
    fig.update_xaxes(rangeslider=dict(visible=False))
    return fig

def quick_backtest(df, config):
    if df is None or len(df) < 50: return None, None
    close = df['Close']; signals = pd.Series(0, index=df.index)
    try:
        # 重現簡單回測邏輯
        if "RSI" in config['mode'] or config['mode'] == "FUSION":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            signals[rsi < config['entry_rsi']] = 1; signals[rsi > config['exit_rsi']] = -1
        elif config['mode'] == "KD":
            k = ta.stoch(df['High'], df['Low'], close, k=9, d=3).iloc[:, 0]
            signals[k < config['entry_k']] = 1; signals[k > config['exit_k']] = -1
        elif config['mode'] == "SUPERTREND":
            st = ta.supertrend(df['High'], df['Low'], close, length=config['period'], multiplier=config['multiplier'])
            dr = st.iloc[:, 1]
            signals[(dr == 1) & (dr.shift(1) == -1)] = 1; signals[(dr == -1) & (dr.shift(1) == 1)] = -1
        elif config['mode'] == "MA_CROSS":
            f, s = ta.sma(close, config['fast_ma']), ta.sma(close, config['slow_ma'])
            signals[(f > s) & (f.shift(1) <= s.shift(1))] = 1; signals[(f < s) & (f.shift(1) >= s.shift(1))] = -1
            
        pos = 0; ent = 0; trd = 0; wins = 0; rets = []
        for i in range(len(df)):
            if pos == 0 and signals.iloc[i] == 1: pos = 1; ent = close.iloc[i]
            elif pos == 1 and signals.iloc[i] == -1:
                pos = 0; r = (close.iloc[i] - ent) / ent; rets.append(r); trd += 1
                if r > 0: wins += 1
        return signals, {"Total_Return": sum(rets)*100, "Win_Rate": (wins/trd*100) if trd else 0, "Trades": trd}
    except: return None, None

def display_card(placeholder, row, config):
    with placeholder.container(border=True):
        st.subheader(f"{row['Name']}")
        c1, c2 = st.columns(2)
        c1.metric("Price", f"${row['Price']:,.2f}", f"{row['Price']-row['Prev_Close']:.2f}")
        
        sig_col = "green" if "BUY" in row['Signal'] else "red" if "SELL" in row['Signal'] else "gray"
        c2.markdown(f":{sig_col}[**{row['Signal']}**] | {row['Action']}")
        
        if row['Is_LLM']:
            with st.expander("🧠 AI 觀點 (LLM)", expanded=True):
                st.markdown(row['LLM_Analysis'])
        else:
            st.caption(f"FinBERT/Info: {row['LLM_Analysis']}")

        if row['Raw_DF'] is not None:
            with st.expander("📊 K線與回測", expanded=False):
                sig, perf = quick_backtest(row['Raw_DF'], config)
                st.plotly_chart(plot_chart(row['Raw_DF'], config, sig), use_container_width=True)
                if perf: st.caption(f"模擬績效: 報酬 {perf['Total_Return']:.1f}% | 勝率 {perf['Win_Rate']:.0f}%")
        
        st.text(f"籌碼: {row['Chip']} | 波動: {row['Pred']}")

# ==========================================
# 7. 執行區
# ==========================================
with st.sidebar:
    st.header("⚙️ 設定")
    user_key = st.text_input("Groq API Key (選填)", value=GROQ_API_KEY_DEFAULT, type="password")
    
    st.divider()
    st.header("🕵️‍♀️ 隱藏寶石掃描")
    custom_input = st.text_area("代碼 (逗號分隔)", placeholder="PLTR, AMD, SOFI, 2603.TW")
    enable_opt = st.checkbox("🧪 執行 Grid Search 優化 (慢)", value=False)
    run_scan = st.button("🚀 掃描自選股")

groq_client = None
if user_key: 
    try: groq_client = Groq(api_key=user_key)
    except: st.sidebar.error("API Key 無效")

# A. 自選股掃描
if run_scan and custom_input:
    st.subheader("🔍 自選股掃描結果")
    tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    cols = st.columns(2) if len(tickers) > 1 else [st.container()]
    
    for i, sym in enumerate(tickers):
        with cols[i % 2]:
            st.text(f"⏳ 分析 {sym}...")
            def_cfg = {"symbol": sym, "name": sym, "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70}
            row = analyze_ticker(def_cfg, groq_client)
            display_card(st.empty(), row, def_cfg)
            
            # 您原本的 Grid Search 功能
            if enable_opt and row['Raw_DF'] is not None:
                with st.expander(f"🧪 {sym} 最佳參數"):
                    opt_res = optimize_rsi_strategy(row['Raw_DF'], sym)
                    if opt_res is not None and not opt_res.empty:
                        best = opt_res.sort_values(by="Return", ascending=False).iloc[0]
                        st.write(f"最佳回報參數: RSI {int(best['Length'])} ({int(best['Buy'])}/{int(best['Sell'])}) -> 報酬 {best['Return']:.1f}%")

# B. 核心持股清單 (100% 您的原始參數)
strategies = {
    "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (美元)", "mode": "KD", "entry_k": 25, "exit_k": 70 },
    "KO": { "symbol": "KO", "name": "KO (可樂)", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
    "BA": { "symbol": "BA", "name": "BA (波音)", "mode": "RSI_RSI", "rsi_len": 14, "entry_rsi": 25, "exit_rsi": 65, "ma_trend": 0 },
    "META": { "symbol": "META", "name": "META (暴力反彈)", "mode": "RSI_RSI", "entry_rsi": 40, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
    "NVDA": { "symbol": "NVDA", "name": "NVDA (聖杯)", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200, "vix_max": 32, "rvol_max": 2.5 },
    "GOOGL": { "symbol": "GOOGL", "name": "GOOGL (聖杯)", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200, "vix_max": 32, "rvol_max": 2.5 },
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

st.divider()
st.subheader("📋 核心持股監控")
if st.button("🔄 刷新全市場"): st.cache_data.clear(); st.rerun()

col1, col2 = st.columns(2)
holders = [col1.empty() if i % 2 == 0 else col2.empty() for i in range(len(strategies))]

for i, (k, cfg) in enumerate(strategies.items()):
    with holders[i].container(): st.caption(f"Analyzing {cfg['name']}...")
    row = analyze_ticker(cfg, groq_client)
    holders[i].empty()
    display_card(holders[i], row, cfg)

st.success("✅ 全市場掃描完成")
