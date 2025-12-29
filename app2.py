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
# 0. 頁面設定 & UI 優化
# ==========================================
st.set_page_config(
    page_title="2025 量化戰情室 (AI 自適應版)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ★★★ CSS 美化區 ★★★
st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        h1, h2, h3, h4, h5, h6, span, div { color: #e0e0e0; font-family: 'Roboto', sans-serif; }
        div[data-testid="stMetric"] {
            background-color: #1c202a; padding: 15px; border-radius: 10px;
            border: 1px solid #2d3342; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        }
        div[data-testid="stMetricLabel"] > div { color: #9db2bf !important; }
        div[data-testid="stMetricValue"] > div { color: #ffffff !important; }
        section[data-testid="stSidebar"] { background-color: #161920; }
        .stButton > button { background-color: #2962ff; color: white; border-radius: 6px; border: none; font-weight: bold; }
        .stButton > button:hover { background-color: #1e4bd1; }
        .streamlit-expanderHeader { background-color: #1c202a; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 2025 全明星量化戰情室 (AI 自適應版)")
st.caption("五維分析 + AI 市場體制識別 (Trend/Range) | 自動微調策略參數")

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
    except: return None

def get_safe_data(ticker):
    try:
        # 下載長一點的數據以計算 200MA
        df = yf.download(ticker, period="2y", interval="1d", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except: return None

# ==========================================
# ★ 新增核心模組: 自適應市場體制識別 (AI Brain)
# ==========================================
def detect_market_regime(df):
    """
    判斷市場狀態:
    1. Bull Trend (多頭趨勢): 價格 > 200MA 且 ADX > 25
    2. Bear Trend (空頭趨勢): 價格 < 200MA 且 ADX > 25
    3. Ranging (盤整震盪): ADX < 25 (無論價格位置)
    """
    if df is None or len(df) < 200: return "UNKNOWN", 0

    # 1. 計算 ADX (趨勢強度)
    try:
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx_df is None or adx_df.empty: return "UNKNOWN", 0
        current_adx = adx_df['ADX_14'].iloc[-1]

        # 2. 計算長期均線 (趨勢方向)
        ma200 = ta.ema(df['Close'], length=200).iloc[-1]
        close = df['Close'].iloc[-1]

        # 3. 判定邏輯
        regime = ""
        if current_adx < 25:
            regime = "RANGING" # 震盪盤
        else:
            if close > ma200:
                regime = "BULL_TREND" # 多頭趨勢
            else:
                regime = "BEAR_TREND" # 空頭趨勢
        return regime, current_adx
    except:
        return "UNKNOWN", 0

def get_adaptive_config(df, original_config):
    """
    根據市場狀態，自動微調策略與參數，保持原策略邏輯但不被市場雙巴
    """
    regime, adx_val = detect_market_regime(df)
    new_config = original_config.copy()
    
    # 記錄體制供顯示用
    new_config['regime'] = regime
    new_config['adx'] = adx_val
    new_config['adaptive_msg'] = "維持原始設定" # 預設

    # 如果是美元匯率或特定商品，可能不適用通用邏輯，可在此排除
    if "TWD" in new_config['symbol']:
        return new_config

    # ★ 自適應邏輯核心 ★
    if regime == "BULL_TREND":
        # === 多頭趨勢 (Lion) ===
        if original_config['mode'] in ["KD", "BOLL_RSI"]:
            # 震盪指標在趨勢盤容易賣飛，改為 SuperTrend 或放寬 RSI
            new_config['mode'] = "SUPERTREND"
            new_config['period'] = 10
            new_config['multiplier'] = 3.0
            new_config['adaptive_msg'] = "趨勢強勁，轉為 SuperTrend 趨勢跟隨"
        elif "RSI" in original_config['mode']:
            # RSI 強勢微調：回檔 45 即買，賣點延後
            new_config['entry_rsi'] = max(original_config.get('entry_rsi', 30), 45)
            new_config['exit_rsi'] = 90
            new_config['adaptive_msg'] = "多頭強勢：放寬買點 (RSI<45)，讓利潤奔跑"

    elif regime == "BEAR_TREND":
        # === 空頭趨勢 (Bear) ===
        if "RSI" in original_config['mode']:
            # 嚴格抄底
            new_config['entry_rsi'] = 20
            new_config['exit_rsi'] = 50
            new_config['adaptive_msg'] = "空頭趨勢：嚴格抄底 (RSI<20)，反彈快逃"
        else:
            # 強制轉為保守 RSI 策略
            new_config['mode'] = "RSI_RSI"
            new_config['entry_rsi'] = 20
            new_config['exit_rsi'] = 45
            new_config['adaptive_msg'] = "空頭保護：強制轉為深跌反彈策略"

    elif regime == "RANGING":
        # === 盤整震盪 (Crab) ===
        if original_config['mode'] in ["SUPERTREND", "MA_CROSS"]:
            # 趨勢策略在盤整會被雙巴，轉為 KD
            new_config['mode'] = "KD"
            new_config['entry_k'] = 20
            new_config['exit_k'] = 80
            new_config['adaptive_msg'] = "盤整震盪：轉為 KD 區間操作避免雙巴"
        elif "RSI" in original_config['mode']:
            # 標準區間
            new_config['entry_rsi'] = 30
            new_config['exit_rsi'] = 70
            new_config['adaptive_msg'] = "盤整震盪：回歸 RSI 標準區間 (30-70)"

    return new_config

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
    except: return None

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
            if not title and 'content' in item: title = item['content'].get('title')
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
            total_score += score_map[sentiment] * confidence
            icon = "🔥" if sentiment == "positive" else "❄️" if sentiment == "negative" else "⚪"
            debug_logs.append(f"{icon} {sentiment.upper()} ({confidence:.2f}): {display_titles[i]}")
            
        avg_score = total_score / len(texts_to_analyze)
        return avg_score, display_titles[0], debug_logs
    except Exception as e:
        return 0, f"AI 分析失敗: {str(e)[:20]}...", []

# ==========================================
# ★ 模組 3 & 4: 波動與籌碼
# ==========================================
def predict_volatility(df):
    try:
        if df is None or df.empty: return None, None
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        if atr is None: return None, None
        return df['Close'].iloc[-1] + atr.iloc[-1], df['Close'].iloc[-1] - atr.iloc[-1]
    except: return None, None

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
# ★ 模組 5: 視覺化與輕量回測
# ==========================================
def plot_interactive_chart(df, config, signals=None):
    if df is None or df.empty: return None
    COLOR_UP, COLOR_DOWN = '#089981', '#f23645'
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
        row_heights=[0.75, 0.25], specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    # K線圖
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name='Price', increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN
    ), row=1, col=1)

    # 顯示策略指標
    if config['mode'] == "MA_CROSS":
        fast_ma = ta.sma(df['Close'], length=config['fast_ma'])
        slow_ma = ta.sma(df['Close'], length=config['slow_ma'])
        fig.add_trace(go.Scatter(x=df.index, y=fast_ma, mode='lines', name=f'MA {config["fast_ma"]}', line=dict(color='#ffeb3b')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=slow_ma, mode='lines', name=f'MA {config["slow_ma"]}', line=dict(color='#2962ff')), row=1, col=1)
    elif config['mode'] == "SUPERTREND":
        st_data = ta.supertrend(df['High'], df['Low'], df['Close'], length=config['period'], multiplier=config['multiplier'])
        if st_data is not None:
            fig.add_trace(go.Scatter(x=df.index, y=st_data[st_data.columns[0]], mode='lines', name='SuperTrend', line=dict(color='#ff9800')), row=1, col=1)

    # 副圖
    if "RSI" in config['mode'] or config['mode'] == "FUSION" or config['mode'] == "BOLL_RSI":
        rsi = ta.rsi(df['Close'], length=config.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI', line=dict(color='#b39ddb')), row=2, col=1)
        fig.add_hrect(y0=config.get('entry_rsi', 30), y1=config.get('exit_rsi', 70), fillcolor="rgba(255, 255, 255, 0.05)", line_width=0, row=2, col=1)
        fig.add_hline(y=config.get('entry_rsi', 30), line_dash="solid", line_color=COLOR_UP, row=2, col=1)
        fig.add_hline(y=config.get('exit_rsi', 70), line_dash="solid", line_color=COLOR_DOWN, row=2, col=1)
    elif config['mode'] == "KD":
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
        if stoch is not None:
            fig.add_trace(go.Scatter(x=df.index, y=stoch.iloc[:, 0], name='K', line=dict(color='#ffeb3b')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=stoch.iloc[:, 1], name='D', line=dict(color='#2962ff')), row=2, col=1)

    # 買賣訊號
    if signals is not None:
        buy_pts = df.loc[signals == 1]; sell_pts = df.loc[signals == -1]
        if not buy_pts.empty: 
            fig.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00e676'), name='Buy'), row=1, col=1)
        if not sell_pts.empty: 
            fig.add_trace(go.Scatter(x=sell_pts.index, y=sell_pts['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ff1744'), name='Sell'), row=1, col=1)

    fig.update_layout(height=500, margin=dict(t=30, b=0, l=10, r=10), paper_bgcolor='#131722', plot_bgcolor='#131722', font=dict(color='#d1d4dc'), showlegend=True)
    return fig

def quick_backtest(df, config):
    if df is None or len(df) < 50: return None, None
    bt_df = df.copy(); close = bt_df['Close']; signals = pd.Series(0, index=bt_df.index)
    
    try:
        # 根據動態 Config 進行回測
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
            fast = ta.sma(close, length=config['fast_ma']); slow = ta.sma(close, length=config['slow_ma'])
            signals[(fast > slow) & (fast.shift(1) <= slow.shift(1))] = 1
            signals[(fast < slow) & (fast.shift(1) >= slow.shift(1))] = -1
            
        position = 0; entry = 0; trades = 0; wins = 0; returns = []
        for i in range(len(bt_df)):
            sig = signals.iloc[i]; price = close.iloc[i]
            if position == 0 and sig == 1: position = 1; entry = price
            elif position == 1 and sig == -1: position = 0; ret = (price - entry) / entry; returns.append(ret); trades += 1; wins += 1 if ret > 0 else 0
        
        return signals, {"Total_Return": sum(returns)*100, "Win_Rate": (wins/trades*100) if trades else 0, "Trades": trades}
    except: return None, None

def display_stock_card(placeholder, row, config):
    with placeholder.container(border=True):
        # 標題區：名稱 + 市場體制 (AI Brain 結果)
        regime_icon = "🦁" if config.get('regime') == "BULL_TREND" else "🐻" if config.get('regime') == "BEAR_TREND" else "🦀"
        regime_text = "多頭" if config.get('regime') == "BULL_TREND" else "空頭" if config.get('regime') == "BEAR_TREND" else "盤整"
        
        st.subheader(f"{row['Name']}")
        st.markdown(f"**市場狀態:** {regime_icon} {regime_text} (ADX:{config.get('adx',0):.0f})")
        if config.get('adaptive_msg'):
            st.caption(f"🤖 AI調整: {config['adaptive_msg']}")

        # 價格與訊號
        if row['Price'] > 0: 
            kp1, kp2 = st.columns(2)
            kp1.metric("昨日收盤", f"${row['Prev_Close']:,.2f}")
            kp2.metric("目前價格", f"${row['Price']:,.2f}", f"{row['Price'] - row['Prev_Close']:.2f}")

        if "STRONG BUY" in row['Signal']: st.success(f"💎 {row['Signal']}")
        elif "BUY" in row['Signal']: st.success(f"{row['Signal']}")
        elif "SELL" in row['Signal']: st.error(f"{row['Signal']}")
        elif "HOLD" in row['Signal']: st.info(f"{row['Signal']}")
        else: st.write(f"⚪ {row['Signal']}")
        
        st.caption(f"建議: {row['Action']}")
        
        # 數據區
        c1, c2 = st.columns(2)
        c1.markdown(f"**財報:** {row.get('Fund', '--')}\n\n**籌碼:** {row.get('Chip', '--')}")
        c2.markdown(f"**情緒:** {row.get('Sent', '--')}\n\n**預測:** {row.get('Pred', '--')}")

        # 圖表區
        if row.get("Raw_DF") is not None:
            with st.expander("📊 K線圖與 AI 修正後回測", expanded=False):
                signals, perf = quick_backtest(row["Raw_DF"], config)
                st.plotly_chart(plot_interactive_chart(row["Raw_DF"], config, signals), use_container_width=True)
                if perf: st.write(f"目前策略模擬績效: 報酬 {perf['Total_Return']:.1f}% | 勝率 {perf['Win_Rate']:.0f}%")
        
        st.divider()
        st.text(f"🛠 當前策略: {config['mode']} | 掛買: {row['Buy_At']} | 掛賣: {row['Sell_At']}")

# ==========================================
# 2. 策略決策邏輯 (含工具函式)
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

def analyze_ticker(base_config):
    symbol = base_config['symbol']
    try:
        df_daily = get_safe_data(symbol)
        if df_daily is None: raise Exception("數據下載失敗")
        
        # ★★★ 關鍵修改：獲取自適應配置 ★★★
        config = get_adaptive_config(df_daily, base_config)

        prev_close = df_daily['Close'].iloc[-1]
        live_price = get_real_live_price(symbol)
        if live_price is None or np.isnan(live_price): live_price = prev_close
        
        # 模擬今日 K 線
        calc_df = df_daily.copy()
        new_row = pd.DataFrame({'Close': [live_price], 'High': [max(live_price, df_daily['High'].iloc[-1])], 'Low': [min(live_price, df_daily['Low'].iloc[-1])], 'Open': [live_price], 'Volume': [0]}, index=[pd.Timestamp.now()])
        calc_df = pd.concat([calc_df, new_row])
        close, high, low = calc_df['Close'], calc_df['High'], calc_df['Low']
        curr_price = live_price
        
        signal, action_msg, signal_type = "💤 WAIT", "觀望中", "WAIT"
        buy_at, sell_at = "---", "---"

        # --- 使用 config (已是自適應後的) 進行判斷 ---
        if config['mode'] == "SUPERTREND":
            st_data = ta.supertrend(high, low, close, length=config['period'], multiplier=config['multiplier'])
            if st_data is not None:
                curr_dir, prev_dir, st_value = st_data.iloc[-1, 1], st_data.iloc[-2, 1], st_data.iloc[-1, 0]
                sell_at = f"${st_value:.2f}"
                if prev_dir == -1 and curr_dir == 1: 
                    signal, action_msg, signal_type = "🚀 BUY", "趨勢翻多 (Breakout)", "BUY"
                elif prev_dir == 1 and curr_dir == -1: 
                    signal, action_msg, signal_type = "📉 SELL", "趨勢翻空 (Breakdown)", "SELL"
                elif curr_dir == 1: 
                    signal, action_msg, signal_type = "✊ HOLD", f"趨勢多頭 (止損 {st_value:.2f})", "HOLD"
                else: 
                    signal, action_msg, signal_type = "☁️ EMPTY", f"趨勢空頭", "EMPTY"

        elif config['mode'] == "FUSION":
            curr_rsi = ta.rsi(close, length=config['rsi_len']).iloc[-1]
            trend_ma = ta.ema(close, length=config['ma_trend']).iloc[-1]
            b_price = find_price_for_rsi(df_daily, config['entry_rsi'], length=config['rsi_len'])
            s_price = find_price_for_rsi(df_daily, config['exit_rsi'], length=config['rsi_len'])
            buy_at, sell_at = f"${b_price:.2f}", f"${s_price:.2f}"
            is_buy = (curr_price > trend_ma) and (curr_rsi < config['entry_rsi'])
            if is_buy: 
                signal, action_msg, signal_type = "🔥 BUY", "趨勢向上+短線超跌", "BUY"
            elif curr_rsi > config['exit_rsi']: 
                signal, action_msg, signal_type = "💰 SELL", "RSI過熱", "SELL"
            else: action_msg = f"等待 (RSI: {curr_rsi:.1f})"

        elif config['mode'] in ["RSI_RSI", "RSI_MA"]:
            rsi_len = config.get('rsi_len', 14)
            curr_rsi = ta.rsi(close, length=rsi_len).iloc[-1]
            use_trend = config.get('ma_trend', 0) > 0
            is_trend_ok = (curr_price > ta.ema(close, length=config['ma_trend']).iloc[-1]) if use_trend else True
            b_price = find_price_for_rsi(df_daily, config['entry_rsi'], length=rsi_len)
            buy_at = f"${b_price:.2f}"
            
            if config['mode'] == "RSI_RSI": 
                s_val = find_price_for_rsi(df_daily, config['exit_rsi'], length=rsi_len)
                sell_at = f"${s_val:.2f}"
                if is_trend_ok and curr_rsi < config['entry_rsi']: 
                    signal, action_msg, signal_type = "🔥 BUY", f"RSI低檔 ({curr_rsi:.1f})", "BUY"
                elif curr_rsi > config['exit_rsi']: 
                    signal, action_msg, signal_type = "💰 SELL", f"RSI高檔 ({curr_rsi:.1f})", "SELL"
                else: action_msg = f"區間震盪 (RSI: {curr_rsi:.1f})"
            else: 
                s_val = ta.sma(close, length=config['exit_ma']).iloc[-1]
                sell_at = f"${s_val:.2f} (MA)"
                if is_trend_ok and curr_rsi < config['entry_rsi']: 
                    signal, action_msg, signal_type = "🔥 BUY", f"短線超賣", "BUY"
                elif curr_price > s_val: 
                    signal, action_msg, signal_type = "💰 SELL", f"觸及均線壓力", "SELL"

        elif config['mode'] == "KD":
            stoch = ta.stoch(high, low, close, k=9, d=3, smooth_k=3)
            curr_k = stoch.iloc[:, 0].iloc[-1]
            buy_at, sell_at = f"K<{config['entry_k']}", f"K>{config['exit_k']}"
            if curr_k < config['entry_k']: 
                signal, action_msg, signal_type = "🚀 BUY", f"KD低檔交叉", "BUY"
            elif curr_k > config['exit_k']: 
                signal, action_msg, signal_type = "💀 SELL", f"KD高檔鈍化", "SELL"
            else: action_msg = f"K值: {curr_k:.1f}"

        elif config['mode'] == "BOLL_RSI":
            rsi_val = ta.rsi(close, length=config.get('rsi_len', 14)).iloc[-1]
            bb = ta.bbands(close, length=20, std=2)
            lower, mid, upper = bb.iloc[:, 0].iloc[-1], bb.iloc[:, 1].iloc[-1], bb.iloc[:, 2].iloc[-1]
            buy_at, sell_at = f"${lower:.2f}", f"${mid:.2f}"
            if curr_price < lower and rsi_val < config['entry_rsi']: 
                signal, action_msg, signal_type = "🚑 BUY", "破下軌+超跌", "BUY"
            elif curr_price >= upper: 
                signal, action_msg, signal_type = "💀 SELL", "觸上軌", "SELL"
            elif curr_price >= mid: 
                signal, action_msg, signal_type = "⚠️ HOLD", "中軸震盪", "HOLD"

        elif config['mode'] == "MA_CROSS":
             fast = ta.sma(close, length=config['fast_ma']); slow = ta.sma(close, length=config['slow_ma'])
             curr_fast, prev_fast = fast.iloc[-1], fast.iloc[-2]
             curr_slow, prev_slow = slow.iloc[-1], slow.iloc[-2]
             if prev_fast <= prev_slow and curr_fast > curr_slow:
                 signal, action_msg, signal_type = "🔥 BUY", "黃金交叉", "BUY"
             elif prev_fast >= prev_slow and curr_fast < curr_slow:
                 signal, action_msg, signal_type = "📉 SELL", "死亡交叉", "SELL"
             elif curr_fast > curr_slow:
                 signal, action_msg, signal_type = "✊ HOLD", "多頭排列", "HOLD"
             else:
                 signal, action_msg, signal_type = "☁️ EMPTY", "空頭排列", "EMPTY"

        # 基本面/情緒整合
        fund_data = get_fundamentals(symbol)
        fund_msg = ""
        is_growth = False; is_cheap = False
        inst_pct = 0; short_pct = 0
        if fund_data:
            g = fund_data['growth'] if fund_data['growth'] else 0
            pe = fund_data['pe']
            inst_pct = fund_data['inst']; short_pct = fund_data['short']
            growth_str = f"💎高成長" if g > 0.2 else (f"🟢穩健" if g > 0 else f"⚠️衰退")
            pe_str = f"🟢低估" if pe and pe < 15 else (f"🔴貴" if pe and pe > 30 else "⚪")
            if g > 0.2: is_growth = True
            if pe and pe < 15: is_cheap = True
            fund_msg = f"{growth_str} | {pe_str}"

        score, news_title, debug_logs = analyze_sentiment_finbert(symbol)
        sent_msg = f"🔥 樂觀" if score > 0.1 else (f"❄️ 悲觀" if score < -0.1 else "⚪ 中立")
        p_high, p_low = predict_volatility(df_daily)
        pred_msg = f"區間: ${p_low:.2f}~${p_high:.2f}" if p_high else ""
        chip_msg = analyze_chips_volume(df_daily, inst_pct, short_pct)

        final_signal = signal
        if "BUY" in signal and is_growth: final_signal = "💎 STRONG BUY"
        elif "BUY" in signal and is_cheap: final_signal = "💰 VALUE BUY"
        if "BUY" in signal and score < -0.5: action_msg += " (⚠️新聞悲觀)"

        return {
            "Symbol": symbol, "Name": base_config['name'], "Price": live_price, "Prev_Close": prev_close, 
            "Signal": final_signal, "Action": action_msg, "Buy_At": buy_at, "Sell_At": sell_at, "Type": signal_type,
            "Fund": fund_msg, "Sent": sent_msg, "News": news_title, "Pred": pred_msg, "Chip": chip_msg, "Logs": debug_logs,
            "Raw_DF": df_daily  
        }
    except Exception as e:
        return {"Symbol": symbol, "Name": base_config['name'], "Price": 0, "Prev_Close": 0, "Signal": "ERR", "Action": str(e), "Type": "ERR", "Logs": [], "Raw_DF": None}

# ==========================================
# 3. 執行區 (設定清單)
# ==========================================
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
    st.info("💡 系統小提示: 策略現在會自動偵測『趨勢』或『盤整』，並微調參數以提高勝率。")

# 主畫面執行
st.subheader("📋 核心持股清單 (AI Auto-Adaptive)")
col1, col2 = st.columns(2)
placeholder_list = [col1.empty() if i % 2 == 0 else col2.empty() for i in range(len(strategies))]

for i, (key, config) in enumerate(strategies.items()):
    with placeholder_list[i].container(): st.text(f"⏳ AI 分析 {config['name']} 中...")
    row = analyze_ticker(config)
    placeholder_list[i].empty()
    display_stock_card(placeholder_list[i], row, get_adaptive_config(row.get('Raw_DF'), config))
    
st.success("✅ 掃描完成 | AI Brain Active")
