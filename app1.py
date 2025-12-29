import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
# ★ 深度學習 NLP 套件
from transformers import pipeline

# ==========================================
# 0. 頁面設定 & UI 優化 (TradingView 風格)
# ==========================================
st.set_page_config(
    page_title="2025 量化戰情室 (旗艦版)",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ★★★ CSS 美化區 ★★★
st.markdown("""
    <style>
        /* 全域背景：改為深灰藍 (TradingView Dark) */
        .stApp {
            background-color: #0e1117;
        }
        
        /* 調整標題文字顏色 */
        h1, h2, h3, h4, h5, h6, span, div {
            color: #e0e0e0;
            font-family: 'Roboto', sans-serif;
        }
        
        /* 讓 Metric 數據卡片有立體感 */
        div[data-testid="stMetric"] {
            background-color: #1c202a;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #2d3342;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        }
        div[data-testid="stMetricLabel"] > div {
            color: #9db2bf !important; /* 標籤顏色 */
        }
        div[data-testid="stMetricValue"] > div {
            color: #ffffff !important; /* 數值顏色 */
        }
        
        /* 側邊欄優化 */
        section[data-testid="stSidebar"] {
            background-color: #161920;
        }
        
        /* 按鈕優化 */
        .stButton > button {
            background-color: #2962ff;
            color: white;
            border-radius: 6px;
            border: none;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #1e4bd1;
        }
        
        /* Expander 邊框 */
        .streamlit-expanderHeader {
            background-color: #1c202a;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📱 2025 全明星量化戰情室 (旗艦版)")
st.caption("五維分析: 技術 + 財報 + FinBERT情緒 + ATR波動 + 籌碼(OBV/空單) | RSI 修正版")

if st.button('🔄 立即更新行情'):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 1. 核心函數 (資料獲取)
# ==========================================

def get_real_live_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get('last_price')
        
        if price is None or np.isnan(price):
            if "-USD" in symbol:
                df_rt = yf.download(symbol, period="1d", interval="1m", progress=False, timeout=5)
            else:
                df_rt = yf.download(symbol, period="5d", interval="1m", prepost=True, progress=False, timeout=5)
                
            if df_rt.empty: return None
            
            if isinstance(df_rt.columns, pd.MultiIndex): 
                df_rt.columns = df_rt.columns.get_level_values(0)
                
            return float(df_rt['Close'].iloc[-1])
            
        return float(price)
    except: 
        return None

def get_safe_data(ticker):
    try:
        # 下載 5 年數據以獲得更完整的 RSI 回測
        df = yf.download(ticker, period="5y", interval="1d", progress=False, timeout=10)
        
        if df is None or df.empty: return None
        
        # 處理 yfinance 新版 MultiIndex 問題
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        # 確保索引是 Datetime
        df.index = pd.to_datetime(df.index)
        return df
    except: return None

# ==========================================
# ★ 模組 1: 財報基本面
# ==========================================
@st.cache_data(ttl=86400)
def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol or "-USD" in symbol: return None 
        stock = yf.Ticker(symbol)
        info = stock.info
        
        quote_type = info.get('quoteType', '').upper()
        if quote_type != 'EQUITY': return None
        
        return {
            "growth": info.get('revenueGrowth', 0), 
            "pe": info.get('trailingPE', None), 
            "eps": info.get('trailingEps', None), 
            "inst": info.get('heldPercentInstitutions', 0),
            "short": info.get('shortPercentOfFloat', 0)
        }
    except:
        return None

# ==========================================
# ★ 模組 2: FinBERT 情緒分析
# ==========================================
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
        texts_to_analyze = []
        display_titles = []
        
        for item in news_list[:5]:
            title = item.get('title')
            if not title and 'content' in item:
                title = item['content'].get('title')
            summary = item.get('summary', '')
            if title:
                full_text = f"{title}. {summary}"
                texts_to_analyze.append(full_text[:512])
                display_titles.append(title)
            
        if not texts_to_analyze: return 0, "無新聞 (格式不符)", []

        results = classifier(texts_to_analyze)
        total_score = 0
        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        debug_logs = []
        
        for i, res in enumerate(results):
            sentiment = res['label']
            confidence = res['score']
            title = display_titles[i]
            total_score += score_map[sentiment] * confidence
            icon = "🔥" if sentiment == "positive" else "❄️" if sentiment == "negative" else "⚪"
            debug_logs.append(f"{icon} {sentiment.upper()} ({confidence:.2f}): {title}")
            
        avg_score = total_score / len(texts_to_analyze)
        return avg_score, display_titles[0], debug_logs
    except Exception as e:
        return 0, f"AI 分析失敗: {str(e)[:20]}...", []

# ==========================================
# ★ 模組 3: ATR 波動預測
# ==========================================
def predict_volatility(df):
    try:
        if df is None or df.empty: return None, None
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        if atr is None: return None, None
        return df['Close'].iloc[-1] + atr.iloc[-1], df['Close'].iloc[-1] - atr.iloc[-1]
    except: return None, None

# ==========================================
# ★ 模組 4: 籌碼量能分析
# ==========================================
def analyze_chips_volume(df, inst_percent, short_percent):
    try:
        if df is None or df.empty: return "資料不足"
        obv = ta.obv(df['Close'], df['Volume'])
        if obv is None or len(obv) < 20: return "量能計算失敗"
        
        chip_msg = "🔴 籌碼流入 (OBV上升)" if obv.iloc[-1] > ta.sma(obv, length=20).iloc[-1] else "🟢 籌碼渙散 (OBV下降)"
        if inst_percent and inst_percent > 0: chip_msg += f" | 機構: {inst_percent*100:.0f}%"
        if short_percent and short_percent > 0:
            sp = short_percent * 100
            if sp > 20: chip_msg += f" | ⚠️ 軋空警戒 ({sp:.1f}%)"
            elif sp > 10: chip_msg += f" | 空單偏高 ({sp:.1f}%)"
        return chip_msg
    except Exception as e: return f"籌碼錯誤: {str(e)}"

# ==========================================
# ★ 模組 5: 視覺化與輕量回測 & 參數優化
# ==========================================
def plot_interactive_chart(df, config, signals=None):
    if df is None or df.empty: return None

    # 配色方案 (TradingView 風格)
    COLOR_UP = '#089981'     # 漲：薄荷綠
    COLOR_DOWN = '#f23645'   # 跌：珊瑚紅
    COLOR_BG = '#131722'     # 背景：深藍灰
    COLOR_GRID = '#2a2e39'   # 網格：淡灰
    COLOR_TEXT = '#d1d4dc'   # 文字：柔白

    # 建立子圖
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.75, 0.25],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # --- 主圖 (K線) ---
    fig.add_trace(go.Candlestick(
        x=df.index, 
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name='Price',
        increasing_line_color=COLOR_UP, increasing_fillcolor=COLOR_UP,
        decreasing_line_color=COLOR_DOWN, decreasing_fillcolor=COLOR_DOWN
    ), row=1, col=1)

    # ==========================================
    # ★ 策略指標線
    # ==========================================
    
    # 1. 雙均線交叉 (MA_CROSS)
    if config['mode'] == "MA_CROSS":
        fast_ma = ta.sma(df['Close'], length=config['fast_ma'])
        slow_ma = ta.sma(df['Close'], length=config['slow_ma'])
        fig.add_trace(go.Scatter(x=df.index, y=fast_ma, mode='lines', name=f'MA {config["fast_ma"]}', line=dict(color='#ffeb3b', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=slow_ma, mode='lines', name=f'MA {config["slow_ma"]}', line=dict(color='#2962ff', width=2)), row=1, col=1)

    # 2. 超級趨勢 (SuperTrend)
    elif config['mode'] == "SUPERTREND":
        st_data = ta.supertrend(df['High'], df['Low'], df['Close'], length=config['period'], multiplier=config['multiplier'])
        if st_data is not None:
            fig.add_trace(go.Scatter(x=df.index, y=st_data[st_data.columns[0]], mode='lines', name='SuperTrend', line=dict(color='#ff9800', width=2)), row=1, col=1)
    
    # 3. 一般趨勢濾網 (單條 EMA)
    elif config.get('ma_trend'):
        ma = ta.ema(df['Close'], length=config['ma_trend'])
        fig.add_trace(go.Scatter(x=df.index, y=ma, mode='lines', name=f'EMA {config["ma_trend"]}', line=dict(color='#2962ff', width=1.5)), row=1, col=1)

    # --- 副圖 (RSI / KD / Volume) ---
    if "RSI" in config['mode'] or config['mode'] == "FUSION" or config['mode'] == "BOLL_RSI":
        rsi = ta.rsi(df['Close'], length=config.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI', line=dict(color='#b39ddb', width=1.5)), row=2, col=1)
        fig.add_hrect(y0=config.get('entry_rsi', 30), y1=config.get('exit_rsi', 70), fillcolor="rgba(255, 255, 255, 0.05)", line_width=0, row=2, col=1)
        fig.add_hline(y=config.get('entry_rsi', 30), line_dash="solid", line_color=COLOR_UP, row=2, col=1, opacity=0.5)
        fig.add_hline(y=config.get('exit_rsi', 70), line_dash="solid", line_color=COLOR_DOWN, row=2, col=1, opacity=0.5)

    elif config['mode'] == "KD":
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
        if stoch is not None:
            fig.add_trace(go.Scatter(x=df.index, y=stoch.iloc[:, 0], name='K', line=dict(color='#ffeb3b', width=1)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=stoch.iloc[:, 1], name='D', line=dict(color='#2962ff', width=1)), row=2, col=1)

    else: # 預設顯示成交量
        colors = [COLOR_UP if c >= o else COLOR_DOWN for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.5), row=2, col=1)

    # --- 買賣點標記 ---
    if signals is not None:
        buy_pts = df.loc[signals == 1]
        sell_pts = df.loc[signals == -1]
        if not buy_pts.empty: 
            fig.add_trace(go.Scatter(
                x=buy_pts.index, y=buy_pts['Low']*0.98, mode='markers', 
                marker=dict(symbol='triangle-up', size=10, color='#00e676', line=dict(width=1, color='black')), name='Buy'
            ), row=1, col=1)
        if not sell_pts.empty: 
            fig.add_trace(go.Scatter(
                x=sell_pts.index, y=sell_pts['High']*1.02, mode='markers', 
                marker=dict(symbol='triangle-down', size=10, color='#ff1744', line=dict(width=1, color='black')), name='Sell'
            ), row=1, col=1)

    # --- Layout 美化 ---
    fig.update_layout(
        height=550,
        margin=dict(t=40, b=0, l=10, r=10),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT, family="Roboto"),
        showlegend=True, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True, gridcolor=COLOR_GRID, gridwidth=1,
            type="date"
        ),
        yaxis=dict(showgrid=True, gridcolor=COLOR_GRID, gridwidth=1),
        xaxis2=dict(showgrid=True, gridcolor=COLOR_GRID, gridwidth=1),
        yaxis2=dict(showgrid=True, gridcolor=COLOR_GRID, gridwidth=1)
    )

    # 時間軸按鈕
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="#2a2e39",
            activecolor="#2962ff",
            font=dict(color="white")
        ),
        range=[df.index[-min(120, len(df))], df.index[-1]]
    )

    return fig

def quick_backtest(df, config):
    if df is None or len(df) < 50: return None, None
    bt_df = df.copy()
    close = bt_df['Close']
    signals = pd.Series(0, index=bt_df.index)
    
    try:
        if config['mode'] in ["RSI_RSI", "FUSION", "RSI_MA", "BOLL_RSI"]:
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            signals[rsi < config['entry_rsi']] = 1
            signals[rsi > config['exit_rsi']] = -1
        elif config['mode'] == "KD":
            stoch = ta.stoch(bt_df['High'], bt_df['Low'], close, k=9, d=3)
            signals[stoch.iloc[:, 0] < config['entry_k']] = 1
            signals[stoch.iloc[:, 0] > config['exit_k']] = -1
        elif config['mode'] == "SUPERTREND":
            st = ta.supertrend(bt_df['High'], bt_df['Low'], close, length=config['period'], multiplier=config['multiplier'])
            direction = st.iloc[:, 1]
            signals[(direction == 1) & (direction.shift(1) == -1)] = 1
            signals[(direction == -1) & (direction.shift(1) == 1)] = -1
        elif config['mode'] == "MA_CROSS":
            fast = ta.sma(close, length=config['fast_ma'])
            slow = ta.sma(close, length=config['slow_ma'])
            signals[(fast > slow) & (fast.shift(1) <= slow.shift(1))] = 1
            signals[(fast < slow) & (fast.shift(1) >= slow.shift(1))] = -1
            
        position = 0; entry = 0; trades = 0; wins = 0; returns = []
        for i in range(len(bt_df)):
            sig = signals.iloc[i]
            price = close.iloc[i]
            if position == 0 and sig == 1:
                position = 1; entry = price
            elif position == 1 and sig == -1:
                position = 0; ret = (price - entry) / entry
                returns.append(ret); trades += 1
                if ret > 0: wins += 1
        
        return signals, {"Total_Return": sum(returns)*100, "Win_Rate": (wins/trades*100) if trades else 0, "Trades": trades}
    except: return None, None

# ==========================================
# ★ 模組化顯示函數
# ==========================================
def display_stock_card(placeholder, row, config):
    """
    將單一股票的顯示邏輯封裝，供主迴圈和自選掃描共用
    """
    with placeholder.container(border=True):
        st.subheader(f"{row['Name']}")
        
        # 價格區塊
        if row['Price'] > 0: 
            kp1, kp2 = st.columns(2)
            kp1.metric("昨日收盤", f"${row['Prev_Close']:,.2f}")
            kp2.metric("目前價格", f"${row['Price']:,.2f}", f"{row['Price'] - row['Prev_Close']:.2f}")

        # 訊號區塊
        if "STRONG BUY" in row['Signal']: st.success(f"💎 {row['Signal']}")
        elif "BUY" in row['Signal']: st.success(f"{row['Signal']}")
        elif "SELL" in row['Signal']: st.error(f"{row['Signal']}")
        elif "HOLD" in row['Signal']: st.info(f"{row['Signal']}")
        elif "ERR" in row['Type']: st.error(f"錯誤: {row['Action']}")
        else: st.write(f"⚪ {row['Signal']}")
        
        st.caption(f"建議: {row['Action']}")
        
        # 數據摘要區塊
        if any([row.get(k) for k in ['Fund', 'Sent', 'Pred', 'Chip']]):
            c1, c2 = st.columns(2)
            c1.markdown(f"**財報:** {row.get('Fund', '--')}\n\n**籌碼:** {row.get('Chip', '--')}")
            c2.markdown(f"**情緒:** {row.get('Sent', '--')}\n\n**預測:** {row.get('Pred', '--')}")

        # 圖表與回測區塊
        raw_df = row.get("Raw_DF")
        if raw_df is not None and not raw_df.empty:
            with st.expander("📊 查看 K線圖與回測績效", expanded=False):
                t1, t2 = st.tabs(["📈 K線圖", "🚀 回測"])
                signals, perf = quick_backtest(raw_df, config)
                with t1:
                    fig = plot_interactive_chart(raw_df, config, signals)
                    if fig: st.plotly_chart(fig, use_container_width=True)
                with t2:
                    if perf:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("交易", perf['Trades'])
                        m2.metric("勝率", f"{perf['Win_Rate']:.0f}%")
                        m3.metric("報酬", f"{perf['Total_Return']:.1f}%", delta_color="normal" if perf['Total_Return']>0 else "inverse")
                    else: st.info("無法回測")
        else:
            if row['Type'] != "ERR": st.warning("⚠️ 無法顯示圖表 (Raw_DF 缺失)")

        # AI 新聞思考區塊
        if row.get('News') and row['News'] != "無新聞":
            with st.expander("🧐 AI 思考過程"):
                for log in row.get('Logs', []): st.text(log)
        
        st.divider()
        # 顯示策略名稱
        strat_map = {
            "RSI_RSI": "RSI區間", "KD": "KD震盪", "SUPERTREND": "超級趨勢", 
            "MA_CROSS": "均線交叉", "FUSION": "AI融合", "BOLL_RSI": "布林極限",
            "RSI_MA": "RSI+均線"
        }
        strat_name = strat_map.get(config['mode'], config['mode'])
        st.text(f"🛠 策略: {strat_name} | 掛買: {row['Buy_At']} | 掛賣: {row['Sell_At']}")

# ==========================================
# ★ 新增模組: 參數優化器 (Grid Search)
# ==========================================
def optimize_rsi_strategy(df, symbol):
    """
    針對指定股票，暴力測試 RSI 參數組合，尋找最佳解
    """
    if df is None or df.empty: return None

    # 設定測試範圍 (為了速度，不要設太密)
    rsi_lengths = [6, 12, 14, 20]
    entries = [20, 25, 30, 40]
    exits = [60, 70, 75, 85]
    
    results = []
    
    # 建立進度條
    progress_text = f"AI 正在為 {symbol} 尋找最佳參數..."
    my_bar = st.progress(0, text=progress_text)
    total_loops = len(rsi_lengths) * len(entries) * len(exits)
    counter = 0

    for length in rsi_lengths:
        rsi_series = ta.rsi(df['Close'], length=length)
        for ent in entries:
            for ext in exits:
                counter += 1
                if counter % 10 == 0:
                    my_bar.progress(counter / total_loops, text=f"正在測試 {symbol}: RSI({length}) {ent}/{ext}")

                signals = pd.Series(0, index=df.index)
                signals[rsi_series < ent] = 1
                signals[rsi_series > ext] = -1
                
                trades = 0; wins = 0; position = 0; entry_price = 0; total_ret = 0
                close_prices = df['Close'].values
                sig_values = signals.values
                
                for i in range(len(close_prices)):
                    s = sig_values[i]
                    p = close_prices[i]
                    if position == 0 and s == 1:
                        position = 1; entry_price = p
                    elif position == 1 and s == -1:
                        position = 0; ret = (p - entry_price) / entry_price
                        total_ret += ret; trades += 1
                        if ret > 0: wins += 1
                
                win_rate = (wins / trades * 100) if trades > 0 else 0
                results.append({
                    "Length": length, "Buy": ent, "Sell": ext,
                    "Return": total_ret * 100, "WinRate": win_rate, "Trades": trades
                })
    
    my_bar.empty()
    return pd.DataFrame(results)

# ==========================================
# 2. 技術指標與決策邏輯
# ==========================================
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

def analyze_ticker(config):
    symbol = config['symbol']
    # 初始化變數
    signal, action_msg, signal_type = "💤 WAIT", "觀望中", "WAIT"
    buy_at, sell_at = "---", "---"
    df_daily = None
    
    try:
        df_daily = get_safe_data(symbol)
        if df_daily is None: raise Exception("數據下載失敗")
        
        prev_close = df_daily['Close'].iloc[-1]
        live_price = get_real_live_price(symbol)
        if live_price is None or np.isnan(live_price): live_price = prev_close
        
        # ★★★ 修正開始：防止重複疊加今日 K 線 (修復 RSI 異常) ★★★
        # 邏輯：檢查最後一筆資料日期。如果是今天，則更新；如果是過去，則新增。
        calc_df = df_daily.copy()
        
        # 取得最後一筆的日期
        last_date = calc_df.index[-1].date()
        today_date = pd.Timestamp.now().date()
        
        if last_date == today_date:
            # 已有今日資料，執行更新 (High取最大，Low取最小)
            calc_df.iloc[-1, calc_df.columns.get_loc('Close')] = live_price
            calc_df.iloc[-1, calc_df.columns.get_loc('High')] = max(live_price, calc_df.iloc[-1]['High'])
            calc_df.iloc[-1, calc_df.columns.get_loc('Low')] = min(live_price, calc_df.iloc[-1]['Low'])
        else:
            # 資料停留在過去，新增今日資料 (High/Low/Open 初始值均為 live_price)
            new_row = pd.DataFrame({
                'Open': [live_price], 'High': [live_price], 'Low': [live_price], 'Close': [live_price], 'Volume': [0]
            }, index=[pd.Timestamp.now()])
            calc_df = pd.concat([calc_df, new_row])
        # ★★★ 修正結束 ★★★

        close, high, low = calc_df['Close'], calc_df['High'], calc_df['Low']
        curr_price = live_price

        # --- 策略判斷 ---
        if config['mode'] == "SUPERTREND":
            st_data = ta.supertrend(high, low, close, length=config['period'], multiplier=config['multiplier'])
            if st_data is not None:
                curr_dir, prev_dir, st_value = st_data.iloc[-1, 1], st_data.iloc[-2, 1], st_data.iloc[-1, 0]
                sell_at = f"${st_value:.2f}"
                if prev_dir == -1 and curr_dir == 1: 
                    signal, action_msg, signal_type = "🚀 BUY", "突破壓力線，趨勢翻多", "BUY"
                elif prev_dir == 1 and curr_dir == -1: 
                    signal, action_msg, signal_type = "📉 SELL", "跌破支撐線，趨勢翻空", "SELL"
                elif curr_dir == 1: 
                    signal, action_msg, signal_type = "✊ HOLD", f"多頭趨勢中 (停損價 {st_value:.2f})", "HOLD"
                else: 
                    signal, action_msg, signal_type = "☁️ EMPTY", f"空頭排列，等待突破 {st_value:.2f}", "EMPTY"

        elif config['mode'] == "FUSION":
            curr_rsi = ta.rsi(close, length=config['rsi_len']).iloc[-1]
            trend_ma = ta.ema(close, length=config['ma_trend']).iloc[-1]
            b_price = find_price_for_rsi(df_daily, config['entry_rsi'], length=config['rsi_len'])
            s_price = find_price_for_rsi(df_daily, config['exit_rsi'], length=config['rsi_len'])
            buy_at, sell_at = f"${b_price:.2f}", f"${s_price:.2f}"
            is_buy = (curr_price > trend_ma) and (curr_rsi < config['entry_rsi'])
            if is_buy: 
                signal, action_msg, signal_type = "🔥 BUY", "趨勢向上且短線超跌，強力買進", "BUY"
            elif curr_rsi > config['exit_rsi']: 
                signal, action_msg, signal_type = "💰 SELL", "RSI過熱 (超買)，建議獲利了結", "SELL"
            else: 
                action_msg = f"趨勢多頭，等待回檔 (RSI: {curr_rsi:.1f})"

        elif config['mode'] in ["RSI_RSI", "RSI_MA"]:
            rsi_len = config.get('rsi_len', 14)
            curr_rsi = ta.rsi(close, length=rsi_len).iloc[-1]
            use_trend = config.get('ma_trend', 0) > 0
            is_trend_ok = (curr_price > ta.ema(close, length=config['ma_trend']).iloc[-1]) if use_trend else True
            b_price = find_price_for_rsi(df_daily, config['entry_rsi'], length=rsi_len)
            buy_at = f"${b_price:.2f}"
            s_val = 0
            if config['mode'] == "RSI_RSI": 
                s_val = find_price_for_rsi(df_daily, config['exit_rsi'], length=rsi_len)
                sell_at = f"${s_val:.2f}"
                if is_trend_ok and curr_rsi < config['entry_rsi']: 
                    signal, action_msg, signal_type = "🔥 BUY", f"RSI低檔 ({curr_rsi:.1f})，甜蜜點浮現", "BUY"
                elif curr_rsi > config['exit_rsi']: 
                    signal, action_msg, signal_type = "💰 SELL", f"RSI高檔 ({curr_rsi:.1f})，建議賣出", "SELL"
                else: 
                    action_msg = f"區間震盪，等待兩端 (RSI: {curr_rsi:.1f})"
            else: 
                s_val = ta.sma(close, length=config['exit_ma']).iloc[-1]
                sell_at = f"${s_val:.2f} (MA)"
                if is_trend_ok and curr_rsi < config['entry_rsi']: 
                    signal, action_msg, signal_type = "🔥 BUY", f"短線超賣 (RSI<{config['entry_rsi']})，進場布局", "BUY"
                elif curr_price > s_val: 
                    signal, action_msg, signal_type = "💰 SELL", f"反彈至均線壓力 ({config['exit_ma']}MA)，獲利了結", "SELL"
                else: 
                    action_msg = f"等待機會 (RSI: {curr_rsi:.1f})"

        elif config['mode'] == "KD":
            stoch = ta.stoch(high, low, close, k=9, d=3, smooth_k=3)
            curr_k = stoch.iloc[:, 0].iloc[-1]
            buy_at, sell_at = f"K<{config['entry_k']}", f"K>{config['exit_k']}"
            if curr_k < config['entry_k']: 
                if "TWD" in symbol:
                    signal, action_msg, signal_type = "💵 BUY", "美元超跌 (便宜)，分批換匯", "BUY"
                else:
                    signal, action_msg, signal_type = "🚀 BUY", f"KD低檔黃金交叉區，進場", "BUY"
            elif curr_k > config['exit_k']: 
                if "TWD" in symbol:
                    signal, action_msg, signal_type = "📉 SELL", "美元過熱 (太貴)，暫停買進", "SELL"
                else:
                    signal, action_msg, signal_type = "💀 SELL", f"KD高檔鈍化，建議賣出", "SELL"
            else: 
                action_msg = f"盤整中 (K值: {curr_k:.1f})"

        elif config['mode'] == "BOLL_RSI":
            rsi_len = config.get('rsi_len', 14)
            rsi_val = ta.rsi(close, length=rsi_len).iloc[-1]
            bb = ta.bbands(close, length=20, std=2)
            lower, mid, upper = bb.iloc[:, 0].iloc[-1], bb.iloc[:, 1].iloc[-1], bb.iloc[:, 2].iloc[-1]
            buy_at, sell_at = f"${lower:.2f}", f"${mid:.2f}"
            if curr_price < lower and rsi_val < config['entry_rsi']: 
                signal, action_msg, signal_type = "🚑 BUY", "嚴重超跌 (破下軌)，搶反彈", "BUY"
            elif curr_price >= upper or rsi_val > 90: 
                signal, action_msg, signal_type = "💀 SELL", "嚴重超買 (觸上軌)，快逃", "SELL"
            elif curr_price >= mid: 
                signal, action_msg, signal_type = "⚠️ HOLD", "反彈至中軸，減碼觀望", "HOLD"
            else: 
                action_msg = f"布林通道震盪中 (RSI: {rsi_val:.1f})"

        elif config['mode'] == "MA_CROSS":
             fast_series = ta.sma(close, length=config['fast_ma'])
             slow_series = ta.sma(close, length=config['slow_ma'])
             curr_fast, prev_fast = fast_series.iloc[-1], fast_series.iloc[-2]
             curr_slow, prev_slow = slow_series.iloc[-1], slow_series.iloc[-2]
             
             if prev_fast <= prev_slow and curr_fast > curr_slow:
                 signal, action_msg, signal_type = "🔥 BUY", "黃金交叉 (突破均線)！", "BUY"
             elif prev_fast >= prev_slow and curr_fast < curr_slow:
                 signal, action_msg, signal_type = "📉 SELL", "死亡交叉 (跌破均線)！", "SELL"
             elif curr_fast > curr_slow:
                 signal, action_msg, signal_type = "✊ HOLD", "均線多頭排列，續抱", "HOLD"
             else:
                 signal, action_msg, signal_type = "☁️ EMPTY", "均線空頭排列，觀望", "EMPTY"

        # 基本面/情緒整合
        fund_data = get_fundamentals(symbol)
        fund_msg = ""
        is_growth = False; is_cheap = False; inst_pct = 0; short_pct = 0 
        
        if fund_data:
            g = fund_data['growth'] if fund_data['growth'] else 0
            pe = fund_data['pe']
            eps = fund_data['eps']
            inst_pct = fund_data['inst']; short_pct = fund_data['short']
            
            growth_str = f"💎高成長" if g > 0.2 else (f"🟢穩健" if g > 0 else f"⚠️衰退")
            
            pe_str = ""
            if pe is not None:
                if pe < 0: pe_str = "虧損無PE"
                elif pe < 15: 
                    pe_str = f"🟢低估(PE {pe:.1f})"; is_cheap = True
                elif pe < 30: pe_str = f"⚪適中(PE {pe:.1f})"
                else: pe_str = f"🔴太貴(PE {pe:.1f})"
            else:
                pe_str = f"💀虧損(EPS {eps:.2f})" if eps and eps < 0 else "無PE"
            fund_msg = f"{growth_str} | {pe_str}"

        score, news_title, debug_logs = analyze_sentiment_finbert(symbol)
        sent_msg = ""
        if score > 0.5: sent_msg = f"🔥 極度樂觀 (+{score:.2f})"
        elif score > 0.1: sent_msg = f"🙂 偏樂觀 (+{score:.2f})"
        elif score < -0.5: sent_msg = f"❄️ 極度悲觀 ({score:.2f})"
        elif score < -0.1: sent_msg = f"😨 偏悲觀 ({score:.2f})"
        else: sent_msg = f"⚪ 中立事實 ({score:.2f})"

        p_high, p_low = predict_volatility(df_daily)
        pred_msg = f"區間: ${p_low:.2f} ~ ${p_high:.2f} (波動 {(p_high-p_low)/live_price*100:.1f}%)" if p_high else ""
        chip_msg = analyze_chips_volume(df_daily, inst_pct, short_pct)

        final_signal = signal
        if "BUY" in signal and is_growth: final_signal = "💎 STRONG BUY"; action_msg += " (財報護體)"
        elif "BUY" in signal and is_cheap: final_signal = "💰 VALUE BUY"; action_msg += " (估值便宜)"
        if "BUY" in signal and score < -0.5: action_msg += " ⚠️ 但新聞極度悲觀"

        return {
            "Symbol": symbol, "Name": config['name'], "Price": live_price, "Prev_Close": prev_close, 
            "Signal": final_signal, "Action": action_msg, "Buy_At": buy_at, "Sell_At": sell_at, "Type": signal_type,
            "Fund": fund_msg, "Sent": sent_msg, "News": news_title, "Pred": pred_msg, "Chip": chip_msg, "Logs": debug_logs,
            "Raw_DF": df_daily  
        }
    except Exception as e:
        return {"Symbol": symbol, "Name": config['name'], "Price": 0, "Prev_Close": 0, "Signal": "ERR", "Action": str(e), "Type": "ERR", "Logs": [], "Raw_DF": None}

# ==========================================
# 3. 執行區 (確保變數已初始化，防止 NameError)
# ==========================================

# 定義核心持股監控名單 (BA 已更新)
strategies = {
    "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (美元)", "mode": "KD", "entry_k": 25, "exit_k": 70 },
    "KO": { "symbol": "KO", "name": "KO (可樂)", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
    
    # ★★★ 已更新: BA 改用「勝率王」參數 (Buy<25, Sell>65) ★★★
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

# 初始化變數
run_custom_scan = False
custom_tickers_input = ""
enable_opt = False

with st.sidebar:
    st.header("🇹🇼 台股雷達")
    def get_fast_info(ticker_symbol):
        try:
            t = yf.Ticker(ticker_symbol)
            return t.fast_info['last_price'], t.fast_info['previous_close']
        except: return None, None

    try:
        with st.spinner('更新台股數據中...'):
            twii_now, twii_prev = get_fast_info("^TWII")
            tsm_tw_now, _ = get_fast_info("2330.TW")
            tsm_us_now, _ = get_fast_info("TSM")
            usd_now, _ = get_fast_info("TWD=X")

        if twii_now:
            st.metric("台股加權指數", f"{twii_now:,.0f}", f"{(twii_now - twii_prev) / twii_prev * 100:+.2f}%")
        
        if tsm_tw_now and tsm_us_now and usd_now:
            premium = ((tsm_us_now - (tsm_tw_now * 5) / usd_now) / ((tsm_tw_now * 5) / usd_now) * 100)
            st.metric("TSM ADR 溢價率", f"{premium:+.2f}%", delta="美股 vs 台股", delta_color="inverse")
    except Exception as e: st.error(f"異常: {e}")
    
    st.divider()
    # ★★★ 隱藏寶石掃描功能 ★★★
    st.header("🕵️‍♀️ 隱藏寶石掃描")
    st.caption("輸入代碼 (逗號分隔) 以搜尋其他潛力股")
    custom_tickers_input = st.text_area("代碼", placeholder="PLTR, AMD, SOFI, 2603.TW")
    # ★ 優化開關
    enable_opt = st.checkbox("🧪 同步尋找最佳策略 (會比較慢)", value=False)
    run_custom_scan = st.button("🚀 開始掃描自選股")

    st.divider()
    with st.expander("📚 指標說明", expanded=True):
        st.markdown("""
        **FinBERT 情緒 AI**: 🔥/❄️ 代表新聞利多/利空程度。
        **ATR 波動**: 預測明日股價震盪區間。
        **籌碼**: OBV 能量潮 + 機構持股比例。
        """)

# ==========================================
# 4. 主畫面邏輯
# ==========================================

# A. 自選掃描模式 (當按下按鈕時執行)
if run_custom_scan and custom_tickers_input:
    st.subheader(f"🔍 自選股深度掃描結果")
    ticker_list = [t.strip().upper() for t in custom_tickers_input.split(",") if t.strip()]
    
    if not ticker_list:
        st.warning("請輸入有效的代碼")
    else:
        # 動態建立 Layout
        cols = st.columns(2) if len(ticker_list) > 1 else [st.container()]
        
        for i, symbol in enumerate(ticker_list):
            with cols[i % 2]:
                st.text(f"⏳ 正在分析 {symbol} ...")
                
                # 1. 預設策略跑一次
                default_config = {
                    "symbol": symbol, "name": f"{symbol} (預設)",
                    "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70, "rsi_len": 14, "ma_trend": 60
                }
                row = analyze_ticker(default_config)
                display_stock_card(st.empty(), row, default_config)
                
                # 2. ★ 如果開啟優化，執行暴力搜索
                if enable_opt:
                    with st.expander(f"🧪 {symbol} 最佳參數報告", expanded=True):
                        st.info("正在進行歷史回測優化 (Grid Search)...")
                        opt_df = optimize_rsi_strategy(row.get('Raw_DF'), symbol)
                        
                        if opt_df is not None and not opt_df.empty:
                            # 找出 報酬率冠軍
                            best_ret = opt_df.sort_values(by="Return", ascending=False).iloc[0]
                            # 找出 勝率冠軍 (且交易次數 > 3)
                            safe_df = opt_df[opt_df['Trades'] >= 3]
                            best_win = safe_df.sort_values(by="WinRate", ascending=False).iloc[0] if not safe_df.empty else best_ret

                            # ★ 強化顯示交易次數
                            st.markdown(f"""
                            **🏆 報酬率冠軍參數:**
                            - RSI長度: `{int(best_ret['Length'])}` | 買進: `<{int(best_ret['Buy'])}` | 賣出: `>{int(best_ret['Sell'])}`
                            - 績效: 報酬 **{best_ret['Return']:.1f}%** | 交易 **{int(best_ret['Trades'])}** 次
                            
                            **🎯 高勝率參數:**
                            - RSI長度: `{int(best_win['Length'])}` | 買進: `<{int(best_win['Buy'])}` | 賣出: `>{int(best_win['Sell'])}`
                            - 績效: 勝率 **{best_win['WinRate']:.0f}%** | 交易 **{int(best_win['Trades'])}** 次
                            """)
                            
                            st.caption("💡 提示: 上述參數為歷史最佳，未來不一定保證獲利 (Overfitting 風險)")
                        else:
                            st.warning("數據不足，無法優化")

    st.divider()
    st.subheader("📋 核心持股清單")

# B. 核心持股監控 (預設執行)
st.info("📡 核心市場掃描中... (AI 模型載入中，第一次請稍候)")
col1, col2 = st.columns(2)
placeholder_list = [col1.empty() if i % 2 == 0 else col2.empty() for i in range(len(strategies))]

for i, (key, config) in enumerate(strategies.items()):
    with placeholder_list[i].container(): st.text(f"⏳ 分析 {config['name']}...")
    row = analyze_ticker(config)
    placeholder_list[i].empty()
    
    # 呼叫共用的顯示函數
    display_stock_card(placeholder_list[i], row, config)

st.caption("✅ 掃描完成 | Auto-generated by Gemini AI")
