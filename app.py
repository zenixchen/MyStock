import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
# ★ 新增：深度學習 NLP 套件
from transformers import pipeline

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(
    page_title="2025 量化戰情室 (FinBERT版)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📱 2025 全明星量化戰情室 (AI 旗艦版)")
st.caption("特色: FinBERT金融情緒分析 + 財報估值 + ATR波動預測")

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
        
        rev_growth = info.get('revenueGrowth', 0)
        pe_ratio = info.get('trailingPE', None)
        eps = info.get('trailingEps', None)
        
        return {"growth": rev_growth, "pe": pe_ratio, "eps": eps}
    except:
        return None

# ==========================================
# ★ 模組 2: Level 3 FinBERT 情緒分析 (核心升級)
# ==========================================

# 使用 cache_resource 確保模型只載入一次 (省時間/省記憶體)
@st.cache_resource
def load_finbert_model():
    # 下載並快取 ProsusAI/finbert 模型
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment_finbert(symbol):
    try:
        if "=" in symbol or "^" in symbol: return 0, "無新聞"
        stock = yf.Ticker(symbol)
        news_list = stock.news
        
        if not news_list: return 0, "無新聞"
        
        # 載入模型 (第一次會很久)
        classifier = load_finbert_model()
        
        headlines = []
        for item in news_list[:3]: # 分析最新的 3 則
            headlines.append(item['title'])
            
        if not headlines: return 0, "無新聞"

        # AI 開始閱讀新聞
        results = classifier(headlines)
        
        # 計算分數 (Positive=1, Negative=-1, Neutral=0)
        total_score = 0
        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        
        for res in results:
            # res 格式: {'label': 'positive', 'score': 0.95}
            sentiment = res['label']
            confidence = res['score']
            
            # 分數 = 方向 * 信心度 (例如非常確定的利多 = 1 * 0.99)
            total_score += score_map[sentiment] * confidence
            
        # 平均分數
        avg_score = total_score / len(headlines)
        latest_news = headlines[0]
        
        return avg_score, latest_news
        
    except Exception as e:
        # 如果出錯 (例如網路連不到 HuggingFace)，回傳錯誤
        return 0, f"AI 分析失敗: {str(e)[:20]}..."

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

        elif config['mode'] == "MA_CROSS":
             fast = ta.sma(close, length=config['fast_ma']).iloc[-1]
             slow = ta.sma(close, length=config['slow_ma']).iloc[-1]
             if fast > slow: 
                 signal, action_msg, signal_type = "✊ HOLD", "均線多頭排列，續抱", "HOLD"
             else: 
                 signal, action_msg, signal_type = "☁️ EMPTY", "均線空頭排列，空手觀望", "EMPTY"

        # ==========================
        # 3. 整合：財報 + FinBERT情緒 + ATR
        # ==========================
        fund_data = get_fundamentals(symbol)
        fund_msg = ""
        is_growth = False
        is_cheap = False
        
        if fund_data:
            g = fund_data['growth'] if fund_data['growth'] else 0
            pe = fund_data['pe']
            eps = fund_data['eps']
            
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

        # ★ FinBERT 情緒分析
        # 分數範圍在 -1 到 1 之間
        score, news_title = analyze_sentiment_finbert(symbol)
        sent_msg = ""
        if score > 0.5: sent_msg = f"🔥 極度樂觀 (+{score:.2f})"
        elif score > 0.1: sent_msg = f"🙂 偏樂觀 (+{score:.2f})"
        elif score < -0.5: sent_msg = f"❄️ 極度悲觀 ({score:.2f})"
        elif score < -0.1: sent_msg = f"😨 偏悲觀 ({score:.2f})"
        else: sent_msg = f"⚪ 中立/無感 ({score:.2f})"

        # ATR 預測
        p_high, p_low = predict_volatility(df_daily)
        pred_msg = ""
        if p_high and p_low:
             vol_pct = (p_high - p_low) / live_price * 100
             pred_msg = f"區間: ${p_low:.2f} ~ ${p_high:.2f} (波動 {vol_pct:.1f}%)"

        final_signal = signal
        if "BUY" in signal and is_growth:
            final_signal = "💎 STRONG BUY"
            action_msg += " (財報護體)"
        elif "BUY" in signal and is_cheap:
            final_signal = "💰 VALUE BUY"
            action_msg += " (估值便宜)"
        # 增加一個情緒濾網：如果技術面買進，但AI讀新聞覺得很不妙
        if "BUY" in signal and score < -0.5:
             action_msg += " ⚠️ 但新聞極度悲觀"

        return {
            "Symbol": symbol,
            "Name": config['name'],
            "Price": live_price,
            "Signal": final_signal,
            "Action": action_msg,
            "Buy_At": buy_at,
            "Sell_At": sell_at,
            "Type": signal_type,
            "Fund": fund_msg,
            "Sent": sent_msg,
            "News": news_title,
            "Pred": pred_msg
        }
    except Exception as e:
        return {"Symbol": symbol, "Name": config['name'], "Price": 0, "Signal": "ERR", "Action": str(e), "Type": "ERR"}

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
        """)

strategies = {
    "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (美元)", "mode": "KD", "entry_k": 25, "exit_k": 70 },
    "KO": { "symbol": "KO", "name": "KO (可樂)", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
    "BA": { "symbol": "BA", "name": "BA (波音)", "mode": "SUPERTREND", "period": 15, "multiplier": 1.0 },
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

for i, (key, config) in enumerate(strategies.items()):
    with placeholder_list[i].container():
        st.text(f"⏳ 分析 {config['name']}...")
    
    row = analyze_ticker(config)
    
    placeholder_list[i].empty()
    with placeholder_list[i].container(border=True):
        st.subheader(f"{row['Name']}")
        
        if row['Price'] > 0: st.write(f"**${row['Price']:,.2f}**")
        else: st.write("**Data Error**")

        if "STRONG BUY" in row['Signal']: st.success(f"💎 {row['Signal']}")
        elif "BUY" in row['Signal']: st.success(f"{row['Signal']}")
        elif "SELL" in row['Signal']: st.error(f"{row['Signal']}")
        elif "HOLD" in row['Signal']: st.info(f"{row['Signal']}")
        elif "ERR" in row['Type']: st.error(f"錯誤: {row['Action']}")
        else: st.write(f"⚪ {row['Signal']}")
        
        st.caption(f"建議: {row['Action']}")
        
        if row.get('Fund') or row.get('Sent') or row.get('Pred'):
            c1, c2 = st.columns(2)
            with c1: 
                if row.get('Fund'): st.markdown(f"**財報:** {row['Fund']}")
            with c2: 
                if row.get('Sent'): st.markdown(f"**情緒:** {row['Sent']}") # 這裡現在顯示的是 AI 判斷結果
            
            if row.get('Pred'):
                st.markdown(f"**🔮 明日預測:** {row['Pred']}")
            
            if row.get('News') and row['News'] != "無新聞":
                with st.expander("AI 閱讀頭條"):
                    st.caption(row['News'])
        
        st.divider()
        st.text(f"掛買: {row['Buy_At']} | 掛賣: {row['Sell_At']}")

st.caption("✅ 掃描完成 | Auto-generated by Gemini AI")
