pip install plotly
import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
# ★ 深度學習 NLP 套件
from transformers import pipeline

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(
    page_title="2025 量化戰情室 (旗艦版)",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📱 2025 全明星量化戰情室 (旗艦版)")
st.caption("五維分析: 技術 + 財報 + FinBERT情緒 + ATR波動 + 籌碼(OBV/空單)")

if st.button('🔄 立即更新行情'):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 1. 核心函數 (資料獲取)
# ==========================================

def get_real_live_price(symbol):
    try:
        # 優先嘗試使用 fast_info (速度快，且通常是最新的 Last Price)
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get('last_price')
        
        # 如果抓不到 (例如某些特殊指數)，才退回到原本的 download 方法
        if price is None or np.isnan(price):
            if "-USD" in symbol:
                df_rt = yf.download(symbol, period="1d", interval="1m", progress=False, timeout=5)
            else:
                df_rt = yf.download(symbol, period="5d", interval="1m", prepost=True, progress=False, timeout=5)
                
            if df_rt.empty: return None
            
            # 處理 yfinance 新版 MultiIndex 問題
            if isinstance(df_rt.columns, pd.MultiIndex): 
                df_rt.columns = df_rt.columns.get_level_values(0)
                
            return float(df_rt['Close'].iloc[-1])
            
        return float(price)
    except: 
        return None

def get_safe_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, timeout=10)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

# ==========================================
# ★ 模組 1: 財報基本面 (含空單數據)
# ==========================================
@st.cache_data(ttl=86400)
def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol or "-USD" in symbol: return None 
        stock = yf.Ticker(symbol)
        info = stock.info
        
        quote_type = info.get('quoteType', '').upper()
        if quote_type != 'EQUITY': return None
        
        rev_growth = info.get('revenueGrowth', 0)
        pe_ratio = info.get('trailingPE', None)
        eps = info.get('trailingEps', None)
        
        # ★ 籌碼數據
        inst_hold = info.get('heldPercentInstitutions', 0) # 機構持股
        short_float = info.get('shortPercentOfFloat', 0)   # 空單比例 (美股專用)
        
        return {
            "growth": rev_growth, 
            "pe": pe_ratio, 
            "eps": eps, 
            "inst": inst_hold,
            "short": short_float
        }
    except:
        return None

# ==========================================
# ★ 模組 2: FinBERT 情緒分析 (標題+摘要)
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
            
            icon = "⚪"
            if sentiment == "positive": icon = "🔥"
            elif sentiment == "negative": icon = "❄️"
            
            log_entry = f"{icon} {sentiment.upper()} ({confidence:.2f}): {title}"
            debug_logs.append(log_entry)
            
        avg_score = total_score / len(texts_to_analyze)
        latest_news = display_titles[0]
        
        return avg_score, latest_news, debug_logs
        
    except Exception as e:
        return 0, f"AI 分析失敗: {str(e)[:20]}...", []

# ==========================================
# ★ 模組 3: ATR 波動預測
# ==========================================
def predict_volatility(df):
    try:
        if df is None or df.empty: return None, None
        high = df['High']; low = df['Low']; close = df['Close']
        atr = ta.atr(high, low, close, length=14)
        if atr is None or np.isnan(atr.iloc[-1]): return None, None
        current_atr = atr.iloc[-1]
        last_close = close.iloc[-1]
        return last_close + current_atr, last_close - current_atr
    except:
        return None, None

# ==========================================
# ★ 模組 4: 籌碼量能分析 (OBV + 機構 + 軋空)
# ==========================================
def analyze_chips_volume(df, inst_percent, short_percent):
    try:
        if df is None or df.empty: return "資料不足"
        
        # 1. OBV (能量潮)
        close = df['Close']
        volume = df['Volume']
        obv = ta.obv(close, volume)
        
        if obv is None or len(obv) < 20: return "量能計算失敗"
        
        curr_obv = obv.iloc[-1]
        obv_ma = ta.sma(obv, length=20).iloc[-1]
        
        chip_msg = ""
        
        # 判斷 OBV
        if curr_obv > obv_ma:
            chip_msg = "🔴 籌碼流入 (OBV上升)"
        else:
            chip_msg = "🟢 籌碼渙散 (OBV下降)"
            
        # 2. 機構持股
        if inst_percent and inst_percent > 0:
            chip_msg += f" | 機構: {inst_percent*100:.0f}%"
            
        # 3. ★ 空單比例 (軋空判斷)
        if short_percent and short_percent > 0:
            sp = short_percent * 100
            if sp > 20:
                chip_msg += f" | ⚠️ 軋空警戒 ({sp:.1f}%)"
            elif sp > 10:
                chip_msg += f" | 空單偏高 ({sp:.1f}%)"
            
        return chip_msg
    except Exception as e:
        return f"籌碼錯誤: {str(e)}"

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# ★ 模組 5: 視覺化與輕量回測
# ==========================================

def plot_interactive_chart(df, config, signals=None):
    """
    繪製互動式 K 線圖，包含策略指標與買賣訊號
    """
    if df is None or df.empty: return None

    # 建立副圖 (Subplots): 上面是 K 線，下面是副指標 (RSI/KD/Vol)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # 1. 主圖：K 線
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='K線'
    ), row=1, col=1)

    # 2. 根據策略繪製主圖指標 (MA / SuperTrend / BBands)
    if config['mode'] == "SUPERTREND":
        # 重新計算一次 SuperTrend 用於繪圖
        st_data = ta.supertrend(df['High'], df['Low'], df['Close'], length=config['period'], multiplier=config['multiplier'])
        if st_data is not None:
            # SuperTrend 的欄位名稱通常是 SUPERT_7_3.0
            st_col = st_data.columns[0] 
            fig.add_trace(go.Scatter(x=df.index, y=st_data[st_col], mode='lines', name='SuperTrend', line=dict(color='orange', width=1)), row=1, col=1)

    elif config['mode'] in ["MA_CROSS", "RSI_MA", "FUSION", "META", "NVDA", "QQQ"]:
        # 繪製均線
        if config.get('ma_trend'):
            ma = ta.ema(df['Close'], length=config['ma_trend'])
            fig.add_trace(go.Scatter(x=df.index, y=ma, mode='lines', name=f'EMA {config["ma_trend"]}', line=dict(color='blue', width=1)), row=1, col=1)
        if config.get('fast_ma'):
            ma_f = ta.sma(df['Close'], length=config['fast_ma'])
            fig.add_trace(go.Scatter(x=df.index, y=ma_f, mode='lines', name=f'MA {config["fast_ma"]}', line=dict(color='cyan', width=1)), row=1, col=1)
        if config.get('slow_ma'):
            ma_s = ta.sma(df['Close'], length=config['slow_ma'])
            fig.add_trace(go.Scatter(x=df.index, y=ma_s, mode='lines', name=f'MA {config["slow_ma"]}', line=dict(color='purple', width=1)), row=1, col=1)
            
    elif config['mode'] == "BOLL_RSI":
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None:
            fig.add_trace(go.Scatter(x=df.index, y=bb.iloc[:, 0], mode='lines', name='Lower', line=dict(color='gray', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb.iloc[:, 2], mode='lines', name='Upper', line=dict(color='gray', dash='dot')), row=1, col=1)

    # 3. 副圖：指標 (RSI / KD / Volume)
    if "RSI" in config['mode'] or config['mode'] == "FUSION":
        rsi_len = config.get('rsi_len', 14)
        rsi = ta.rsi(df['Close'], length=rsi_len)
        fig.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name=f'RSI {rsi_len}', line=dict(color='purple')), row=2, col=1)
        # 畫超買超賣線
        fig.add_hline(y=config.get('entry_rsi', 30), line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=config.get('exit_rsi', 70), line_dash="dash", line_color="red", row=2, col=1)
        
    elif config['mode'] == "KD":
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3, smooth_k=3)
        if stoch is not None:
            k = stoch.iloc[:, 0]
            d = stoch.iloc[:, 1]
            fig.add_trace(go.Scatter(x=df.index, y=k, mode='lines', name='K%', line=dict(color='orange')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=d, mode='lines', name='D%', line=dict(color='blue')), row=2, col=1)
            fig.add_hline(y=config.get('entry_k', 20), line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=config.get('exit_k', 80), line_dash="dash", line_color="red", row=2, col=1)
    else:
        # 預設畫成交量
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(100, 100, 100, 0.5)'), row=2, col=1)

    # 4. 標記買賣訊號點 (如果有回測產生的 signals)
    if signals is not None and not signals.empty:
        # 買點 (Buy)
        buy_points = df.loc[signals == 1]
        if not buy_points.empty:
            fig.add_trace(go.Scatter(
                x=buy_points.index, y=buy_points['Low'] * 0.98,
                mode='markers', marker=dict(symbol='triangle-up', size=10, color='lime'),
                name='買進訊號'
            ), row=1, col=1)
        
        # 賣點 (Sell)
        sell_points = df.loc[signals == -1]
        if not sell_points.empty:
            fig.add_trace(go.Scatter(
                x=sell_points.index, y=sell_points['High'] * 1.02,
                mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                name='賣出訊號'
            ), row=1, col=1)

    fig.update_layout(
        height=500, 
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def quick_backtest(df, config):
    """
    快速回測：計算過去 1 年的簡單績效 (不含滑價與手續費，純訊號測試)
    回傳: 訊號 Series, 績效 Dict
    """
    if df is None or len(df) < 50: return None, None
    
    # 複製一份以免影響原始資料
    bt_df = df.copy()
    close = bt_df['Close']
    signals = pd.Series(0, index=bt_df.index) # 0:無, 1:買, -1:賣
    
    # --- 根據策略邏輯產生訊號 (簡化版邏輯) ---
    try:
        if config['mode'] == "RSI_RSI" or config['mode'] == "FUSION":
            rsi = ta.rsi(close, length=config['rsi_len'])
            # 簡單邏輯：RSI < Entry 買，RSI > Exit 賣
            signals[rsi < config['entry_rsi']] = 1
            signals[rsi > config['exit_rsi']] = -1
            
        elif config['mode'] == "KD":
            stoch = ta.stoch(bt_df['High'], bt_df['Low'], close, k=9, d=3)
            k = stoch.iloc[:, 0]
            signals[k < config['entry_k']] = 1
            signals[k > config['exit_k']] = -1
            
        elif config['mode'] == "MA_CROSS":
            fast = ta.sma(close, length=config['fast_ma'])
            slow = ta.sma(close, length=config['slow_ma'])
            # 黃金交叉買，死亡交叉賣
            signals[(fast > slow) & (fast.shift(1) <= slow.shift(1))] = 1
            signals[(fast < slow) & (fast.shift(1) >= slow.shift(1))] = -1
            
        elif config['mode'] == "SUPERTREND":
            st_data = ta.supertrend(bt_df['High'], bt_df['Low'], close, length=config['period'], multiplier=config['multiplier'])
            direction = st_data.iloc[:, 1] # 1: Up, -1: Down
            # 方向轉變時產生訊號
            signals[(direction == 1) & (direction.shift(1) == -1)] = 1
            signals[(direction == -1) & (direction.shift(1) == 1)] = -1

        # --- 計算績效 (Vectorized Backtest) ---
        # 假設：買入後持有直到出現賣出訊號 (簡化)
        # 持倉狀態: 1=持有, 0=空手
        position = 0
        returns = []
        entry_price = 0
        trade_count = 0
        win_count = 0
        
        for i in range(len(bt_df)):
            sig = signals.iloc[i]
            price = close.iloc[i]
            
            if position == 0 and sig == 1: # 進場
                position = 1
                entry_price = price
            elif position == 1 and sig == -1: # 出場
                position = 0
                ret = (price - entry_price) / entry_price
                returns.append(ret)
                trade_count += 1
                if ret > 0: win_count += 1
                
        total_ret = sum(returns)
        win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
        
        perf = {
            "Total_Return": total_ret * 100,
            "Win_Rate": win_rate,
            "Trades": trade_count
        }
        return signals, perf

    except Exception as e:
        return None, None

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
    try:
        df_daily = get_safe_data(symbol)
        if df_daily is None: raise Exception("數據下載失敗")
        
        prev_close = df_daily['Close'].iloc[-1]
        
        live_price = get_real_live_price(symbol)
        if live_price is None or np.isnan(live_price): live_price = prev_close
        
        calc_df = df_daily.copy()
        new_row = pd.DataFrame({'Close': [live_price], 'High': [max(live_price, df_daily['High'].iloc[-1])], 'Low': [min(live_price, df_daily['Low'].iloc[-1])], 'Open': [live_price], 'Volume': [0]}, index=[pd.Timestamp.now()])
        calc_df = pd.concat([calc_df, new_row])
        close, high, low = calc_df['Close'], calc_df['High'], calc_df['Low']
        curr_price = live_price

        signal, action_msg, signal_type = "💤 WAIT", "觀望中", "WAIT"
        buy_at, sell_at = "---", "---"

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

        # ★ 修正重點：MA_CROSS 邏輯升級 (同步 App1)
        elif config['mode'] == "MA_CROSS":
             fast_series = ta.sma(close, length=config['fast_ma'])
             slow_series = ta.sma(close, length=config['slow_ma'])
             
             # 抓今天和昨天
             curr_fast, prev_fast = fast_series.iloc[-1], fast_series.iloc[-2]
             curr_slow, prev_slow = slow_series.iloc[-1], slow_series.iloc[-2]
             
             # 1. 黃金交叉 (昨天在下，今天在上)
             if prev_fast <= prev_slow and curr_fast > curr_slow:
                 signal, action_msg, signal_type = "🔥 BUY", "黃金交叉 (突破均線)！", "BUY"
             # 2. 死亡交叉 (昨天在上，今天在下)
             elif prev_fast >= prev_slow and curr_fast < curr_slow:
                 signal, action_msg, signal_type = "📉 SELL", "死亡交叉 (跌破均線)！", "SELL"
             # 3. 多頭排列
             elif curr_fast > curr_slow:
                 signal, action_msg, signal_type = "✊ HOLD", "均線多頭排列，續抱", "HOLD"
             # 4. 空頭排列
             else:
                 signal, action_msg, signal_type = "☁️ EMPTY", "均線空頭排列，觀望", "EMPTY"

        # ==========================
        # 3. 整合：財報 + 情緒 + ATR + 籌碼
        # ==========================
        fund_data = get_fundamentals(symbol)
        fund_msg = ""
        is_growth = False
        is_cheap = False
        inst_pct = 0 
        short_pct = 0 
        
        if fund_data:
            g = fund_data['growth'] if fund_data['growth'] else 0
            pe = fund_data['pe']
            eps = fund_data['eps']
            inst_pct = fund_data['inst'] 
            short_pct = fund_data['short']
            
            growth_str = ""
            if g > 0.2: 
                growth_str = f"💎高成長"
                is_growth = True
            elif g > 0: growth_str = f"🟢穩健"
            else: growth_str = f"⚠️衰退"

            pe_str = ""
            if pe is not None:
                if pe < 0: pe_str = "虧損無PE"
                elif pe < 15: 
                    pe_str = f"🟢低估(PE {pe:.1f})"
                    is_cheap = True
                elif pe < 30: pe_str = f"⚪適中(PE {pe:.1f})"
                elif pe >= 30:
                    if is_growth: pe_str = f"🟠偏高(PE {pe:.1f})"
                    else: pe_str = f"🔴太貴(PE {pe:.1f})"
            else:
                if eps is not None and eps < 0:
                     pe_str = f"💀虧損(EPS {eps:.2f})"
                else:
                     pe_str = "無PE"
            fund_msg = f"{growth_str} | {pe_str}"

        # FinBERT
        score, news_title, debug_logs = analyze_sentiment_finbert(symbol)
        
        sent_msg = ""
        if score > 0.5: sent_msg = f"🔥 極度樂觀 (+{score:.2f})"
        elif score > 0.1: sent_msg = f"🙂 偏樂觀 (+{score:.2f})"
        elif score < -0.5: sent_msg = f"❄️ 極度悲觀 ({score:.2f})"
        elif score < -0.1: sent_msg = f"😨 偏悲觀 ({score:.2f})"
        else: sent_msg = f"⚪ 中立事實 ({score:.2f})"

        # ATR
        p_high, p_low = predict_volatility(df_daily)
        pred_msg = ""
        if p_high and p_low:
             vol_pct = (p_high - p_low) / live_price * 100
             pred_msg = f"區間: ${p_low:.2f} ~ ${p_high:.2f} (波動 {vol_pct:.1f}%)"

        # 籌碼
        chip_msg = analyze_chips_volume(df_daily, inst_pct, short_pct)

        # 訊號整合
        final_signal = signal
        if "BUY" in signal and is_growth:
            final_signal = "💎 STRONG BUY"
            action_msg += " (財報護體)"
        elif "BUY" in signal and is_cheap:
            final_signal = "💰 VALUE BUY"
            action_msg += " (估值便宜)"
        
        if "BUY" in signal and score < -0.5:
             action_msg += " ⚠️ 但新聞極度悲觀"

        return {
            "Symbol": symbol,
            "Name": config['name'],
            "Price": live_price,
            "Prev_Close": prev_close, 
            "Signal": final_signal,
            "Action": action_msg,
            "Buy_At": buy_at,
            "Sell_At": sell_at,
            "Type": signal_type,
            "Fund": fund_msg,
            "Sent": sent_msg,
            "News": news_title,
            "Pred": pred_msg,
            "Chip": chip_msg,
            "Logs": debug_logs
            "Raw_DF": df_daily  # <--- ★★★ 請務必加入這一行，把數據傳出來繪圖！
        }
    except Exception as e:
        return {"Symbol": symbol, "Name": config['name'], "Price": 0, "Prev_Close": 0, "Signal": "ERR", "Action": str(e), "Type": "ERR", "Logs": []}

# ==========================================
# 3. 執行區
# ==========================================
with st.sidebar:
    st.header("🇹🇼 台股雷達")
    def get_fast_info(ticker_symbol):
        try:
            t = yf.Ticker(ticker_symbol)
            curr = t.fast_info['last_price']
            prev = t.fast_info['previous_close']
            return curr, prev
        except: return None, None

    try:
        with st.spinner('更新台股數據中...'):
            twii_now, twii_prev = get_fast_info("^TWII")
            tsm_tw_now, _ = get_fast_info("2330.TW")
            tsm_us_now, _ = get_fast_info("TSM")
            usd_now, _ = get_fast_info("TWD=X")

        if twii_now and twii_prev:
            change_pct = (twii_now - twii_prev) / twii_prev * 100
            st.metric("台股加權指數", f"{twii_now:,.0f}", f"{change_pct:+.2f}%")
        else: st.error("無法取得大盤數據")

        if tsm_tw_now and tsm_us_now and usd_now:
            fair_adr = (tsm_tw_now * 5) / usd_now
            premium = ((tsm_us_now - fair_adr) / fair_adr * 100)
            st.metric("TSM ADR 溢價率", f"{premium:+.2f}%", delta="美股 vs 台股", delta_color="inverse")
            if premium > 5: st.warning("⚠️ 溢價過高")
            elif premium < -2: st.success("🚀 折價")
            else: st.info("✅ 價格合理")
        else: st.warning("數據連線中...")

    except Exception as e: st.error(f"異常: {e}")
    
    st.divider()
    with st.expander("📚 指標說明", expanded=True):
        st.markdown("""
        **FinBERT 情緒 AI**
        🔥 > 0.5: 強烈利多新聞
        ❄️ < -0.5: 強烈利空新聞
        
        **ATR 波動預測**
        預測明日股價的安全活動範圍。
        
        **籌碼分析 (Chip)**
        🔴 OBV上升: 籌碼流入 (健康)
        ⚠️ 軋空警戒: 空單比例 > 20%
        """)

strategies = {
    "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (美元)", "mode": "KD", "entry_k": 25, "exit_k": 70 },
    "KO": { "symbol": "KO", "name": "KO (可樂)", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
    "BA": { "symbol": "BA", "name": "BA (波音)", "mode": "SUPERTREND", "period": 15, "multiplier": 1.0 },
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

st.info("📡 市場掃描中... (AI 模型載入中，第一次請稍候)")

col1, col2 = st.columns(2)
placeholder_list = []

for i in range(len(strategies)):
    with (col1 if i % 2 == 0 else col2):
        placeholder_list.append(st.empty())

# ==========================================
# 3. 執行區 (請替換原本的迴圈)
# ==========================================

# ... (前面的 sidebar 和市場掃描與 placeholder_list 建立不用動) ...

# ★★★ 請將原本的 for i, (key, config) in enumerate(strategies.items()): 迴圈替換為以下內容 ★★★

for i, (key, config) in enumerate(strategies.items()):
    with placeholder_list[i].container():
        st.text(f"⏳ 分析 {config['name']}...")
    
    # 執行分析
    row = analyze_ticker(config)
    
    # 清空並重新繪製容器
    placeholder_list[i].empty()
    with placeholder_list[i].container(border=True):
        
        # --- 區塊 A: 標題與價格 (保持不變) ---
        st.subheader(f"{row['Name']}")
        
        if row['Price'] > 0: 
            kp1, kp2 = st.columns(2)
            with kp1:
                st.caption("昨日收盤")
                st.write(f"**${row['Prev_Close']:,.2f}**")
            with kp2:
                st.caption("目前價格")
                chg = row['Price'] - row['Prev_Close']
                color_str = "green" if chg >= 0 else "red"
                st.markdown(f":{color_str}[**${row['Price']:,.2f}**]")
        else: 
            st.write("**Data Error**")

        # --- 區塊 B: 訊號顯示 (保持不變) ---
        if "STRONG BUY" in row['Signal']: st.success(f"💎 {row['Signal']}")
        elif "BUY" in row['Signal']: st.success(f"{row['Signal']}")
        elif "SELL" in row['Signal']: st.error(f"{row['Signal']}")
        elif "HOLD" in row['Signal']: st.info(f"{row['Signal']}")
        elif "ERR" in row['Type']: st.error(f"錯誤: {row['Action']}")
        else: st.write(f"⚪ {row['Signal']}")
        
        st.caption(f"建議: {row['Action']}")
        
        # --- 區塊 C: 五維分析數據 (保持不變) ---
        if row.get('Fund') or row.get('Sent') or row.get('Pred') or row.get('Chip'):
            c1, c2 = st.columns(2)
            with c1: 
                if row.get('Fund'): st.markdown(f"**財報:** {row['Fund']}")
                if row.get('Chip'): st.markdown(f"**籌碼:** {row['Chip']}")
            with c2: 
                if row.get('Sent'): st.markdown(f"**情緒:** {row['Sent']}")
            
            if row.get('Pred'):
                st.markdown(f"**🔮 明日預測:** {row['Pred']}")

        # =========================================================
        # ★★★ 新增區塊: 視覺化圖表與回測系統 ★★★
        # =========================================================
        # 檢查是否有數據可以繪圖 (Raw_DF 是否存在)
        if row.get("Raw_DF") is not None and not row["Raw_DF"].empty:
            
            # 使用 expander 摺疊起來，以免畫面太長
            with st.expander("📊 查看 K線圖與回測績效", expanded=False):
                
                # 建立兩個標籤頁
                tab_chart, tab_backtest = st.tabs(["📈 技術分析圖", "🚀 歷史回測 (1年)"])
                
                # 先執行回測，取得買賣點訊號 (signals) 和績效 (perf)
                bt_signals, perf = quick_backtest(row["Raw_DF"], config)
                
                # --- Tab 1: 繪製互動圖表 ---
                with tab_chart:
                    # 呼叫你剛剛寫好的繪圖函數，並傳入回測訊號來標記買賣點
                    fig = plot_interactive_chart(row["Raw_DF"], config, bt_signals)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # --- Tab 2: 顯示回測結果 ---
                with tab_backtest:
                    if perf:
                        # 顯示三個關鍵指標
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("交易次數", f"{perf['Trades']} 次")
                        
                        # 勝率顏色
                        win_color = "normal"
                        if perf['Win_Rate'] > 60: win_color = "normal" # Streamlit metric 預設綠色就是 normal
                        mc2.metric("勝率", f"{perf['Win_Rate']:.1f}%")
                        
                        # 報酬率顏色
                        ret_color = "normal"
                        if perf['Total_Return'] > 0: ret_color = "normal"
                        else: ret_color = "inverse" # 虧損顯示紅色
                        
                        mc3.metric("總報酬率", f"{perf['Total_Return']:.2f}%", delta="近一年策略表現", delta_color=ret_color)
                        
                        # 簡單的評語
                        if perf['Total_Return'] > 20:
                            st.success("🔥 此策略過去一年表現優異！")
                        elif perf['Total_Return'] < -10:
                            st.warning("⚠️ 此策略近期表現不佳，請小心使用。")
                        else:
                            st.info("💡 表現平穩。")
                    else:
                        st.info("資料不足，無法進行有效回測。")

        # --- 區塊 D: 底部掛單資訊與新聞 (保持不變) ---
        if row.get('News') and row['News'] != "無新聞":
            with st.expander("🧐 AI 思考過程 (新聞分析)"):
                if row.get('Logs'):
                    for log in row['Logs']:
                        st.text(log)
                else:
                    st.text(f"最新頭條: {row['News']}")
        
        st.divider()
        st.text(f"掛買: {row['Buy_At']} | 掛賣: {row['Sell_At']}")

st.caption("✅ 掃描完成 | Auto-generated by Gemini AI")


