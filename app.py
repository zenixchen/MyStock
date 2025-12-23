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
st.caption("含: 技術指標 + 財報估值(P/E) + NLP情緒分析")

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
# ★ 新增模組 1: 財報基本面 (含 ETF 過濾 & 虧損判斷)
# ==========================================
@st.cache_data(ttl=86400) # 財報一天抓一次就好
def get_fundamentals(symbol):
    try:
        # 1. 排除明顯非個股
        if "=" in symbol or "^" in symbol or "-USD" in symbol: return None 
        
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # 2. ★ 關鍵修改：嚴格檢查 quoteType
        # 如果不是 'EQUITY' (個股)，就直接回傳 None (ETF/基金/期貨不顯示財報)
        quote_type = info.get('quoteType', '').upper()
        if quote_type != 'EQUITY':
            return None
        
        # 3. 抓取數據 (含 EPS)
        rev_growth = info.get('revenueGrowth', 0)
        pe_ratio = info.get('trailingPE', None) # 本益比
        eps = info.get('trailingEps', None)     # 每股盈餘 (判斷虧損用)
        
        return {"growth": rev_growth, "pe": pe_ratio, "eps": eps}
    except:
        return None

# ==========================================
# ★ 新增模組 2: NLP 情緒分析
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
# 2. 技術指標計算邏輯
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

        signal, action_msg, signal_type = "💤 WAIT", "觀望", "WAIT"
        buy_at, sell_at = "---", "---"

        # --- 策略判斷 ---
        if config['mode'] == "SUPERTREND":
            st_data = ta.supertrend(high, low, close, length=config['period'], multiplier=config['multiplier'])
            if st_data is not None:
                curr_dir, prev_dir, st_value = st_data.iloc[-1, 1], st_data.iloc[-2, 1], st_data.iloc[-1, 0]
                if prev_dir == -1 and curr_dir == 1: signal, action_msg, signal_type = "🚀 BUY", "趨勢翻多", "BUY"
                elif prev_dir == 1 and curr_dir == -1: signal, action_msg, signal_type = "📉 SELL", "趨勢翻空", "SELL"
                elif curr_dir == 1: signal, action_msg, signal_type = "✊ HOLD", f"停利: {st_value:.2f}", "HOLD"
                else: signal, action_msg, signal_type = "☁️ EMPTY", f"突破 {st_value:.2f} 買", "EMPTY"
                sell_at = f"${st_value:.2f}"

        elif config['mode'] == "FUSION":
            curr_rsi = ta.rsi(close, length=config['rsi_len']).iloc[-1]
            trend_ma = ta.ema(close, length=config['ma_trend']).iloc[-1]
            b_price = find_price_for_rsi(df_daily, config['entry_rsi'], length=config['rsi_len'])
            s_price = find_price_for_rsi(df_daily, config['exit_rsi'], length=config['rsi_len'])
            buy_at, sell_at = f"${b_price:.2f}", f"${s_price:.2f}"
            is_buy = (curr_price > trend_ma) and (curr_rsi < config['entry_rsi'])
            if is_buy: signal, action_msg, signal_type = "🔥 BUY", "RSI低+趨勢安", "BUY"
            elif curr_rsi > config['exit_rsi']: signal, action_msg, signal_type = "💰 SELL", "RSI過熱", "SELL"
            else: action_msg = f"RSI: {curr_rsi:.1f}"

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
            else:
                s_val = ta.sma(close, length=config['exit_ma']).iloc[-1]
                sell_at = f"${s_val:.2f} (MA)"
            if is_trend_ok and curr_rsi < config['entry_rsi']: signal, action_msg, signal_type = "🔥 BUY", f"RSI<{config['entry_rsi']}", "BUY"
            elif config['mode']=="RSI_RSI" and curr_rsi > config['exit_rsi']: signal, action_msg, signal_type = "💰 SELL", f"RSI>{config['exit_rsi']}", "SELL"
            elif config['mode']=="RSI_MA" and curr_price > s_val: signal, action_msg, signal_type = "💰 SELL", "站上均線", "SELL"
            else: action_msg = f"RSI: {curr_rsi:.1f}"

        elif config['mode'] == "KD":
            stoch = ta.stoch(high, low, close, k=9, d=3, smooth_k=3)
            curr_k = stoch.iloc[:, 0].iloc[-1]
            buy_at, sell_at = f"K<{config['entry_k']}", f"K>{config['exit_k']}"
            if curr_k < config['entry_k']: signal, action_msg, signal_type = "🚀 BUY", f"K值{curr_k:.1f}低", "BUY"
            elif curr_k > config['exit_k']: signal, action_msg, signal_type = "💀 SELL", f"K值{curr_k:.1f}高", "SELL"
            else: action_msg = f"K值: {curr_k:.1f}"

        elif config['mode'] == "BOLL_RSI":
            rsi_len = config.get('rsi_len', 14)
            rsi_val = ta.rsi(close, length=rsi_len).iloc[-1]
            bb = ta.bbands(close, length=20, std=2)
            lower, mid, upper = bb.iloc[:, 0].iloc[-1], bb.iloc[:, 1].iloc[-1], bb.iloc[:, 2].iloc[-1]
            buy_at, sell_at = f"${lower:.2f}", f"${mid:.2f}"
            if "TWD" in symbol: 
                if curr_price < lower and rsi_val < config['entry_rsi']: signal, action_msg, signal_type = "💵 BUY", "超跌+破下軌", "BUY"
                elif curr_price >= upper: signal, action_msg, signal_type = "📉 SELL", "太貴(上軌)", "SELL"
                else: action_msg = f"RSI: {rsi_val:.1f}"
            else:
                if curr_price < lower and rsi_val < config['entry_rsi']: signal, action_msg, signal_type = "🚑 BUY", "救援機會", "BUY"
                elif curr_price >= upper or rsi_val > 90: signal, action_msg, signal_type = "💀 SELL", "過熱出場", "SELL"
                elif curr_price >= mid: signal, action_msg, signal_type = "⚠️ HOLD", "減碼觀望", "HOLD"
                else: action_msg = f"RSI: {rsi_val:.1f}"

        elif config['mode'] == "MA_CROSS":
             fast = ta.sma(close, length=config['fast_ma']).iloc[-1]
             slow = ta.sma(close, length=config['slow_ma']).iloc[-1]
             if fast > slow: signal, action_msg, signal_type = "✊ HOLD", "多頭排列", "HOLD"
             else: signal, action_msg, signal_type = "☁️ EMPTY", "空頭排列", "EMPTY"

        # ==========================
        # ★ 整合：財報(成長+PE) + 情緒
        # ==========================
        fund_data = get_fundamentals(symbol)
        fund_msg = ""
        is_growth = False
        is_cheap = False
        
        # 只有當 get_fundamentals 回傳非 None 時 (即個股)，才處理財報字串
        if fund_data:
            g = fund_data['growth'] if fund_data['growth'] else 0
            pe = fund_data['pe']
            eps = fund_data['eps'] # 取得 EPS
            
            # 1. 成長判斷
            growth_str = ""
            if g > 0.2: 
                growth_str = f"💎高成長"
                is_growth = True
            elif g > 0: growth_str = f"🟢穩健"
            else: growth_str = f"⚠️衰退"

            # 2. P/E 判斷 (含虧損邏輯)
            pe_str = ""
            if pe is not None:
                if pe < 15: 
                    pe_str = f"🟢低估(PE {pe:.1f})"
                    is_cheap = True
                elif pe < 30: pe_str = f"⚪適中(PE {pe:.1f})"
                elif pe >= 30:
                    if is_growth: pe_str = f"🟠偏高(PE {pe:.1f})"
                    else: pe_str = f"🔴太貴(PE {pe:.1f})"
            else:
                # 如果沒有 PE，檢查是否因為賠錢 (EPS < 0)
                if eps is not None and eps < 0:
                     pe_str = f"💀虧損(EPS {eps:.2f})"
                else:
                     pe_str = "無PE"
            
            fund_msg = f"{growth_str} | {pe_str}"

        # 情緒分析
        score, news_title = analyze_sentiment_basic(symbol)
        sent_msg = ""
        if score > 0: sent_msg = f"🔥 樂觀 (+{score})"
        elif score < 0: sent_msg = f"❄️ 悲觀 ({score})"
        else: sent_msg = "⚪ 中立"

        # 訊號升級邏輯
        final_signal = signal
        if "BUY" in signal and is_growth:
            final_signal = "💎 STRONG BUY"
            action_msg += " + 財報優"
        elif "BUY" in signal and is_cheap:
            final_signal = "💰 VALUE BUY"
            action_msg += " + 估值低"

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
            "News": news_title
        }
    except Exception as e:
        return {"Symbol": symbol, "Name": config['name'], "Price": 0, "Signal": "ERR", "Action": str(e), "Type": "ERR"}

# ==========================================
# 3. 執行區
# ==========================================

# A. 側邊欄 (含 P/E 指南)
with st.sidebar:
    st.header("🇹🇼 台股雷達")
    try:
        with st.spinner('連線台股中...'):
            df_2330 = get_safe_data("2330.TW")
            df_twii = get_safe_data("^TWII")
            df_usdtwd = get_safe_data("TWD=X")
            df_tsm = get_safe_data("TSM")
        
        if df_2330 is not None and df_twii is not None:
            tw_price = df_2330['Close'].iloc[-1]
            idx_price = df_twii['Close'].iloc[-1]
            idx_change = (idx_price - df_twii['Close'].iloc[-2]) / df_twii['Close'].iloc[-2] * 100
            st.metric("台股加權", f"{idx_price:.0f}", f"{idx_change:.2f}%")
            
            usd = df_usdtwd['Close'].iloc[-1] if df_usdtwd is not None else 32.5
            us_tsm = df_tsm['Close'].iloc[-1] if df_tsm is not None else 0
            fair_adr = (tw_price * 5) / usd
            premium = ((us_tsm - fair_adr) / fair_adr * 100) if us_tsm > 0 else 0
            
            st.metric("TSM 溢價率", f"{premium:.2f}%", delta_color="inverse")
            if premium > 2: st.warning("⚠️ 美股太貴")
            elif premium < -2: st.success("🚀 美股便宜")
            else: st.info("✅ 價格合理")
        else:
            st.error("台股連線逾時")
    except:
        st.error("台股數據異常")
    
    st.divider()
    # ★ 新增：P/E 判讀指南
    with st.expander("📚 P/E (本益比) 判讀指南", expanded=True):
        st.markdown("""
        **P/E = 股價 / 每股盈餘**
        *(越低越便宜，但也可能代表公司爛)*
        
        🟢 **< 15 (低估/便宜)**
        適合價值投資，撿便宜。
        
        ⚪ **15 ~ 30 (適中/合理)**
        市場平均，KO、QQQ多在此區間。
        
        🟠 **> 30 (偏高/昂貴)**
        除非是 NVDA 這種高成長股，否則需小心追高。
        
        💀 **虧損 (負值)**
        公司正在賠錢 (如 BA)，風險較高。
        """)

# B. 策略掃描 (完整 14 支)
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

st.info("📡 市場掃描中... (含基本面 P/E 分析)")

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
        
        # 只有在有財報 或 有情緒數據時才顯示
        if row.get('Fund') or row.get('Sent'):
            c1, c2 = st.columns(2)
            with c1: 
                # 如果是 ETF，Fund 會是空的，就不會顯示
                if row.get('Fund'): st.markdown(f"**財報:** {row['Fund']}")
            with c2: 
                if row.get('Sent'): st.markdown(f"**情緒:** {row['Sent']}")
            
            if row.get('News') and row['News'] != "無新聞":
                with st.expander("最新頭條"):
                    st.caption(row['News'])
        
        st.divider()
        st.text(f"掛買: {row['Buy_At']} | 掛賣: {row['Sell_At']}")

st.caption("✅ 掃描完成 | Auto-generated by Gemini AI")
