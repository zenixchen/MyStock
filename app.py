import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(
    page_title="2025 量化戰情室 (Pro)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📱 2025 全明星量化戰情室 (Pro)")
st.caption("特色: 白話文操作建議 + 財報估值 + NLP情緒分析 + ATR波動預測")

if st.button('🔄 立即更新行情'):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 1. 核心函數 (資料獲取)
# ==========================================
def get_real_live_price(symbol):
    try:
        if "-USD" in symbol:
            df_rt = yf.download(symbol, period="1d", interval="1m", progress=False, timeout=5)
        else:
            df_rt = yf.download(symbol, period="5d", interval="1m", prepost=True, progress=False, timeout=5)
        if df_rt.empty: return None
        if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
        return float(df_rt['Close'].iloc[-1])
    except: return None

def get_safe_data(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, timeout=10)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

# ==========================================
# ★ 模組 1: 財報基本面 (含 ETF 過濾 & 虧損判斷)
# ==========================================
@st.cache_data(ttl=86400)
def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol or "-USD" in symbol: return None 
        
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # 嚴格檢查 quoteType
        quote_type = info.get('quoteType', '').upper()
        if quote_type != 'EQUITY':
            return None
        
        rev_growth = info.get('revenueGrowth', 0)
        pe_ratio = info.get('trailingPE', None)
        eps = info.get('trailingEps', None)
        
        return {"growth": rev_growth, "pe": pe_ratio, "eps": eps}
    except:
        return None

# ==========================================
# ★ 模組 2: NLP 情緒分析
# ==========================================
def analyze_sentiment_basic(symbol):
    try:
        if "=" in symbol or "^" in symbol: return 0, "無新聞"
        stock = yf.Ticker(symbol)
        news_list = stock.news
        
        score = 0
        headlines = []
        pos_words = ["soar", "jump", "surge", "beat", "buy", "gain", "high", "growth", "strong", "up", "record"]
        neg_words = ["drop", "fall", "miss", "cut", "sell", "loss", "weak", "down", "crash", "plunge", "concern"]
        
        if not news_list: return 0, "無新聞"

        for item in news_list[:3]:
            title = item['title'].lower()
            headlines.append(item['title'])
            for w in pos_words:
                if w in title: score += 1
            for w in neg_words:
                if w in title: score -= 1
        
        latest_news = headlines[0] if headlines else "無新聞"      
        return score, latest_news
    except:
        return 0, "分析失敗"

# ==========================================
# ★ 模組 3: ATR 波動預測 (新增功能)
# ==========================================
def predict_volatility(df):
    try:
        if df is None or df.empty: return None, None
        
        # 計算 ATR (14天平均真實波幅)
        high = df['High']
        low = df['Low']
        close = df['Close']
        atr = ta.atr(high, low, close, length=14)
        
        if atr is None or np.isnan(atr.iloc[-1]): return None, None
        
        current_atr = atr.iloc[-1]
        last_close = close.iloc[-1]
        
        # 預測區間 (收盤價 +/- 1倍ATR)
        pred_high = last_close + current_atr
        pred_low = last_close - current_atr
        
        return pred_high, pred_low
    except:
        return None, None

# ==========================================
# 2. 技術指標與「白話文」決策邏輯
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

        # --- 策略判斷 (維持原樣) ---
        
        if config['mode'] == "SUPERTREND":
            # 波音策略
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
            # NVDA/GOOGL 策略
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
            # KO, QQQ, QLD 策略
            rsi_len = config.get('rsi_len', 14)
            curr_rsi = ta.rsi(close, length=rsi_len).iloc[-1]
            use_trend = config.get('ma_trend', 0) > 0
            is_trend_ok = (curr_price > ta.ema(close, length=config['ma_trend']).iloc[-1]) if use_trend else True
            
            b_price = find_price_for_rsi(df_daily, config['entry_rsi'], length=rsi_len)
            buy_at = f"${b_price:.2f}"
            
            s_val = 0
            if config['mode'] == "RSI_RSI": # 純 RSI 策略 (KO, TQQQ)
                s_val = find_price_for_rsi(df_daily, config['exit_rsi'], length=rsi_len)
                sell_at = f"${s_val:.2f}"
                
                if is_trend_ok and curr_rsi < config['entry_rsi']: 
                    signal, action_msg, signal_type = "🔥 BUY", f"RSI低檔 ({curr_rsi:.1f})，甜蜜點浮現", "BUY"
                elif curr_rsi > config['exit_rsi']: 
                    signal, action_msg, signal_type = "💰 SELL", f"RSI高檔 ({curr_rsi:.1f})，建議賣出", "SELL"
                else: 
                    action_msg = f"區間震盪，等待兩端 (RSI: {curr_rsi:.1f})"

            else: # RSI + MA 策略 (QQQ, QLD)
                s_val = ta.sma(close, length=config['exit_ma']).iloc[-1]
                sell_at = f"${s_val:.2f} (MA)"
                
                if is_trend_ok and curr_rsi < config['entry_rsi']: 
                    signal, action_msg, signal_type = "🔥 BUY", f"短線超賣 (RSI<{config['entry_rsi']})，進場布局", "BUY"
                elif curr_price > s_val: 
                    # 這裡就是您原本看不懂的地方，改為白話文
                    signal, action_msg, signal_type = "💰 SELL", f"反彈至均線壓力 ({config['exit_ma']}MA)，獲利了結", "SELL"
                else: 
                    action_msg = f"等待機會 (RSI: {curr_rsi:.1f})"

        elif config['mode'] == "KD":
            # 匯率 / SOXL_F 策略
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
            # EDZ 策略
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
             fast = ta.sma(close, length=config['fast_ma']).iloc[-1]
             slow = ta.sma(close, length=config['slow_ma']).iloc[-1]
             if fast > slow: 
                 signal, action_msg, signal_type = "✊ HOLD", "均線多頭排列，續抱", "HOLD"
             else: 
                 signal, action_msg, signal_type = "☁️ EMPTY", "均線空頭排列，空手觀望", "EMPTY"

        # ==========================
        # 3. 整合：財報(成長+PE) + 情緒 + ATR預測
        # ==========================
        fund_data = get_fundamentals(symbol)
        fund_msg = ""
        is_growth =
