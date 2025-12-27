import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
# 若不使用情緒分析可註解下行
from transformers import pipeline

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(
    page_title="2025 量化戰情室 (聖杯旗艦版)",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📱 2025 全明星量化戰情室 (聖杯旗艦版)")
st.caption("FUSION 策略 (含 VIX/RVOL 濾網) | 財報基本面 | 即時行情")

if st.button('🔄 立即更新行情'):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 1. 核心函數 (資料獲取)
# ==========================================
def get_real_live_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # 美股優先用 history 抓含盤前盤後 (解決盤後價格不準問題)
        if ".TW" not in symbol:
            df = ticker.history(period="1d", interval="1m", prepost=True)
            if not df.empty: return float(df['Close'].iloc[-1])
        
        # 台股或 history 抓不到，退回使用 fast_info
        price = ticker.fast_info.get('last_price')
        if price and not np.isnan(price): return float(price)
        return None
    except: return None

def get_real_volume(symbol):
    # 取得當日累積成交量 (用於計算 RVOL)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="1m", prepost=True) 
        if not df.empty:
            # 近似計算：使用當日最後一筆的 Volume 往往不準，改用當日累計 volume
            # 但 yfinance history period=1d 給的是分鐘線，我們抓 daily 比較準
            df_day = ticker.history(period="1d")
            if not df_day.empty: return float(df_day['Volume'].iloc[-1])
        return 0
    except: return 0

def get_safe_data(ticker):
    try:
        # 抓取 2 年日線供技術指標計算
        df = yf.download(ticker, period="2y", interval="1d", progress=False, timeout=10)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

# ★ 新增：取得 VIX 恐慌指數
@st.cache_data(ttl=300) # 5分鐘更新一次 VIX 即可
def get_vix_now():
    try:
        vix = yf.Ticker("^VIX")
        price = vix.fast_info.get('last_price')
        # 如果 fast_info 抓不到，試試 history
        if not price or np.isnan(price):
            df = vix.history(period="1d")
            if not df.empty: price = df['Close'].iloc[-1]
        return float(price) if price else 0
    except: return 0

# ==========================================
# ★ 模組 1: 財報基本面
# ==========================================
@st.cache_data(ttl=3600)
def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol: return None 
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # 抓取關鍵欄位
        return {
            "growth": info.get('revenueGrowth', 0), 
            "pe": info.get('trailingPE') if info.get('trailingPE') else info.get('forwardPE'),
            "eps": info.get('trailingEps'), 
            "inst": info.get('heldPercentInstitutions', 0),
            "short": info.get('shortPercentOfFloat', 0)
        }
    except: return None

# ==========================================
# ★ 模組 2: 情緒分析
# ==========================================
@st.cache_resource
def load_finbert_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment_finbert(symbol):
    try:
        if "=" in symbol or "^" in symbol: return 0, "無新聞"
        stock = yf.Ticker(symbol)
        news = stock.news
        if not news: return 0, "無新聞"
        
        classifier = load_finbert_model()
        # 只抓前 3 則標題分析
        texts = [i.get('title')[:512] for i in news[:3] if i.get('title')]
        if not texts: return 0, "無新聞"
        
        results = classifier(texts)
        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        total = sum(score_map[r['label']] * r['score'] for r in results)
        
        return total / len(texts), texts[0]
    except: return 0, "分析略過"

def analyze_chips_volume(df, inst_pct, short_pct):
    try:
        obv = ta.obv(df['Close'], df['Volume'])
        msg = []
        if obv is not None and len(obv) > 20:
            if obv.iloc[-1] > ta.sma(obv, length=20).iloc[-1]: msg.append("🔴OBV升")
            else: msg.append("🟢OBV降")
        if inst_pct and inst_pct > 0: msg.append(f"機構:{inst_pct*100:.0f}%")
        if short_pct and short_pct > 0.1: msg.append(f"⚠️空單:{short_pct*100:.1f}%")
        return " | ".join(msg)
    except: return ""

# ==========================================
# 2. 技術指標與決策邏輯 (含 VIX/RVOL 判斷)
# ==========================================
def analyze_ticker(config):
    symbol = config['symbol']
    try:
        # 1. 數據準備
        df_daily = get_safe_data(symbol)
        if df_daily is None: return None
        prev_close = df_daily['Close'].iloc[-1]
        
        live_price = get_real_live_price(symbol)
        if live_price is None: live_price = prev_close
        
        # 抓即時量 (為了算 RVOL)
        live_vol = get_real_volume(symbol)
        if live_vol == 0: live_vol = df_daily['Volume'].iloc[-1]

        # 合併 K 線計算指標
        new_row = pd.DataFrame({
            'Close': [live_price], 'High': [max(live_price, df_daily['High'].iloc[-1])],
            'Low': [min(live_price, df_daily['Low'].iloc[-1])], 'Open': [live_price], 'Volume': [live_vol]
        }, index=[pd.Timestamp.now()])
        calc_df = pd.concat([df_daily, new_row])
        
        close = calc_df['Close']
        high, low = calc_df['High'], calc_df['Low']
        
        signal, action_msg = "⚪ WAIT", "觀望"
        mode = config['mode']

        # --- 策略邏輯區 ---

        # ★ FUSION 模式 (聖杯策略：含 VIX + RVOL 濾網)
        if mode == "FUSION":
            curr_rsi = ta.rsi(close, length=config['rsi_len']).iloc[-1]
            trend_ma = ta.ema(close, length=config['ma_trend']).iloc[-1]
            
            # 計算 RVOL (相對成交量)
            # 簡單定義：今日預估量 / 過去 20 日均量
            avg_vol = df_daily['Volume'].rolling(window=20).mean().iloc[-1]
            curr_rvol = (live_vol / avg_vol) if avg_vol > 0 else 1.0
            
            # 取得 VIX
            curr_vix = get_vix_now()
            
            # 讀取參數 (如果沒有設定，給寬鬆預設值)
            vix_limit = config.get('vix_max', 100)
            rvol_limit = config.get('rvol_max', 10)
            
            is_trend_up = live_price > trend_ma
            is_oversold = curr_rsi < config['entry_rsi']
            is_vix_safe = curr_vix < vix_limit
            is_rvol_safe = curr_rvol < rvol_limit
            
            if is_trend_up and is_oversold:
                if is_vix_safe and is_rvol_safe:
                    signal, action_msg = "🏆 BUY", f"聖杯浮現 (RSI:{curr_rsi:.1f} | VIX:{curr_vix:.1f})"
                else:
                    reasons = []
                    if not is_vix_safe: reasons.append(f"VIX過高({curr_vix:.1f})")
                    if not is_rvol_safe: reasons.append(f"爆量({curr_rvol:.1f}倍)")
                    action_msg = f"等待安全 (過濾: {' '.join(reasons)})"
                    
            elif curr_rsi > config['exit_rsi']:
                signal, action_msg = "💰 SELL", f"RSI過熱 ({curr_rsi:.1f})"
            else:
                action_msg = f"趨勢等待 (RSI:{curr_rsi:.1f})"

        # SUPERTREND
        elif mode == "SUPERTREND":
            st_data = ta.supertrend(high, low, close, length=config['period'], multiplier=config['multiplier'])
            if st_data is not None:
                if st_data.iloc[-1, 1] == 1: signal, action_msg = "🚀 BUY", "趨勢向上"
                else: signal, action_msg = "📉 SELL", "趨勢向下"

        # RSI 相關策略
        elif mode in ["RSI_RSI", "RSI_MA"]:
            curr_rsi = ta.rsi(close, length=config['rsi_len']).iloc[-1]
            use_trend = config.get('ma_trend', 0) > 0
            is_trend_ok = (live_price > ta.ema(close, length=config['ma_trend']).iloc[-1]) if use_trend else True
            
            if is_trend_ok and curr_rsi < config['entry_rsi']:
                signal, action_msg = "🔥 BUY", f"RSI低檔 ({curr_rsi:.1f})"
            elif mode == "RSI_RSI" and curr_rsi > config['exit_rsi']:
                signal, action_msg = "💰 SELL", f"RSI過熱 ({curr_rsi:.1f})"
            elif mode == "RSI_MA" and live_price > ta.sma(close, length=config['exit_ma']).iloc[-1]:
                signal, action_msg = "💰 SELL", "觸及均線壓力"
            else:
                action_msg = f"RSI: {curr_rsi:.1f}"

        # KD
        elif mode == "KD":
            k = ta.stoch(high, low, close).iloc[:, 0].iloc[-1]
            if k < config['entry_k']: signal, action_msg = "🚀 BUY", f"KD低檔 ({k:.1f})"
            elif k > config['exit_k']: signal, action_msg = "💀 SELL", f"KD高檔 ({k:.1f})"
            else: action_msg = f"KD值: {k:.1f}"

        # BOLL_RSI
        elif mode == "BOLL_RSI":
            curr_rsi = ta.rsi(close, length=config['rsi_len']).iloc[-1]
            bb = ta.bbands(close, length=20, std=2)
            lower, upper = bb.iloc[:, 0].iloc[-1], bb.iloc[:, 2].iloc[-1]
            if live_price < lower and curr_rsi < config['entry_rsi']:
                signal, action_msg = "🚑 BUY", "破底+超跌 (搶反彈)"
            elif live_price >= upper:
                signal, action_msg = "💀 SELL", "觸及布林上軌"
            else:
                action_msg = f"通道震盪 (RSI: {curr_rsi:.1f})"

        # MA_CROSS
        elif mode == "MA_CROSS":
             fast = ta.sma(close, length=config['fast_ma']).iloc[-1]
             slow = ta.sma(close, length=config['slow_ma']).iloc[-1]
             if fast > slow: signal, action_msg = "🔥 BUY", "均線多頭"
             else: signal, action_msg = "☁️ SELL", "均線空頭"

        # --- 整合財報 ---
        fund_data = get_fundamentals(symbol)
        fund_msg = "N/A"
        is_cheap, is_growth = False, False
        inst_pct, short_pct = 0, 0
        if fund_data:
            g, pe = fund_data['growth'], fund_data['pe']
            inst_pct, short_pct = fund_data['inst'], fund_data['short']
            p_str = f"PE {pe:.1f}" if pe else "No PE"
            g_str = f"成長 {g:.1%}" if g else "成長未知"
            fund_msg = f"{p_str} | {g_str}"
            if pe and pe < 20: is_cheap = True
            if g and g > 0.15: is_growth = True

        # 情緒與籌碼
        score, news = analyze_sentiment_finbert(symbol)
        sent_msg = f"🙂樂觀({score:.2f})" if score > 0.2 else (f"😨悲觀({score:.2f})" if score < -0.2 else "中立")
        chip_msg = analyze_chips_volume(df_daily, inst_pct, short_pct)

        # 訊號加權
        if "BUY" in signal and is_cheap: signal = "💰 VALUE BUY"
        if "BUY" in signal and is_growth: signal = "💎 GROWTH BUY"
        
        return {
            "Symbol": symbol, "Name": config['name'], "Price": live_price,
            "Change": live_price - prev_close, "Signal": signal, "Action": action_msg,
            "Fund": fund_msg, "Sent": sent_msg, "Chip": chip_msg, "News": news
        }
    except Exception as e:
        return {"Symbol": symbol, "Name": config['name'], "Price": 0, "Signal": "ERR", "Action": str(e)}

# ==========================================
# 3. 執行與顯示
# ==========================================
st.sidebar.header("監控面板")

# ★ 用戶原始策略設定 (含 NVDA/GOOGL 的 VIX 與 RVOL 濾網)
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

col1, col2 = st.columns(2)
cols = [col1, col2]

for i, (key, config) in enumerate(strategies.items()):
    with cols[i % 2]:
        res = analyze_ticker(config)
        if res and res['Price'] > 0:
            with st.container(border=True):
                # 標題與價格
                c1, c2 = st.columns([2, 1])
                c1.subheader(res['Name'])
                chg_color = "green" if res['Change'] >= 0 else "red"
                c2.markdown(f"**${res['Price']:.2f}** (:{chg_color}[{res['Change']:.2f}])")
                
                # 訊號與建議
                if "BUY" in res['Signal']: st.success(f"{res['Signal']} | {res['Action']}")
                elif "SELL" in res['Signal']: st.error(f"{res['Signal']} | {res['Action']}")
                else: st.info(f"{res['Signal']} | {res['Action']}")
                
                # 詳細資訊
                st.markdown(f"**📊 財報:** {res.get('Fund', 'N/A')}")
                st.markdown(f"**🧠 情緒:** {res.get('Sent', 'N/A')}")
                st.markdown(f"**🎰 籌碼:** {res.get('Chip', 'N/A')}")
                
                if res.get('News') and res['News'] != "無新聞":
                    st.caption(f"📰 {res['News']}")
        else:
            st.error(f"{config['name']} 讀取失敗")

st.caption("✅ 聖杯版載入完成 | Gemini AI Assistant")
