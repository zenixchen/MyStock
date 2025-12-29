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
    page_title="2025 量化戰情室 (AI進化版)",
    page_icon="🧬",
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

st.title("🧬 2025 全明星量化戰情室 (AI 進化版)")
st.caption("五維分析 + 體制識別 (Trend/Range) + 參數自我進化 (Walk-Forward Opt)")

if st.button('🔄 立即進化並更新行情'):
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
        # 下載 2 年數據以計算 200MA 與長期回測
        df = yf.download(ticker, period="2y", interval="1d", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except: return None

# ==========================================
# ★ 模組 A: 策略自我進化 (Walk-Forward Optimization)
# ==========================================

# 1. 獨立的 SuperTrend 優化函數 (供 BULL_TREND 使用)
def optimize_supertrend(df):
    """
    針對趨勢盤，暴力測試最佳的 SuperTrend 參數
    """
    if df is None or len(df) < 100: return 10, 3.0, "預設"

    # 定義測試範圍 (週期, 倍數)
    params_grid = [
        (10, 3.0), # 標準 (穩健)
        (7, 3.0),  # 敏感 (適合飆股)
        (14, 2.0), # 寬鬆 (適合大波動且不想被洗)
        (20, 2.0)  # 長線 (大波段)
    ]
    
    best_score = -999
    best_params = (10, 3.0)
    
    # 用最近半年數據回測
    train_df = df.iloc[-126:].copy()
    high = train_df['High']; low = train_df['Low']; close = train_df['Close']

    for p, m in params_grid:
        try:
            st_data = ta.supertrend(high, low, close, length=p, multiplier=m)
            if st_data is None: continue
            
            # 計算訊號
            direction = st_data.iloc[:, 1]
            signals = pd.Series(0, index=train_df.index)
            signals[(direction == 1) & (direction.shift(1) == -1)] = 1  # 轉多
            signals[(direction == -1) & (direction.shift(1) == 1)] = -1 # 轉空
            
            # 簡易回測
            trades = 0; wins = 0; total_ret = 0; position = 0; entry = 0
            prices = close.values; sig_vals = signals.values
            
            for i in range(len(prices)):
                if position == 0 and sig_vals[i] == 1:
                    position = 1; entry = prices[i]
                elif position == 1 and sig_vals[i] == -1:
                    position = 0; ret = (prices[i] - entry) / entry
                    total_ret += ret; trades += 1
                    if ret > 0: wins += 1
            
            # 評分: 報酬優先，勝率為輔
            if trades > 0:
                score = total_ret * 100 + (wins/trades * 10)
                if score > best_score:
                    best_score = score
                    best_params = (p, m)
        except: continue
        
    return best_params[0], best_params[1], f"最佳化 ({best_params[0]}/{best_params[1]})"


# 2. 獨立的 RSI 進化函數 (供 analyze_ticker 使用)
def evolve_strategy(df, symbol):
    """
    進化邏輯：
    暴力測試過去 6 個月 (約 120 K棒) 的參數組合，
    找出「勝率 + 報酬率」綜合分數最高的設定。
    """
    if df is None or len(df) < 150: return None, None

    # 定義基因庫 (參數範圍)
    param_grid = [
        {'rsi_len': 6,  'entry': 20, 'exit': 70, 'desc': '極短線 (RSI 6)'}, 
        {'rsi_len': 6,  'entry': 30, 'exit': 80, 'desc': '短線積極 (RSI 6)'},
        {'rsi_len': 14, 'entry': 30, 'exit': 70, 'desc': '標準 (RSI 14)'},
        {'rsi_len': 14, 'entry': 25, 'exit': 75, 'desc': '標準寬鬆 (RSI 14)'},
        {'rsi_len': 24, 'entry': 40, 'exit': 60, 'desc': '長線平穩 (RSI 24)'},
    ]

    best_score = -999
    best_config = None
    best_perf = ""

    # 使用最近 126 天 (約半年) 來訓練
    train_df = df.iloc[-126:].copy()
    close = train_df['Close']

    for params in param_grid:
        # 模擬策略訊號
        rsi = ta.rsi(close, length=params['rsi_len'])
        if rsi is None: continue
        
        signals = pd.Series(0, index=train_df.index)
        signals[rsi < params['entry']] = 1
        signals[rsi > params['exit']] = -1
        
        # 快速向量回測 (簡化版)
        prices = close.values
        sig_vals = signals.values
        trades = 0; wins = 0; total_ret = 0; position = 0; entry_price = 0
        
        for i in range(len(prices)):
            if position == 0 and sig_vals[i] == 1:
                position = 1; entry_price = prices[i]
            elif position == 1 and sig_vals[i] == -1:
                position = 0; ret = (prices[i] - entry_price) / entry_price
                total_ret += ret
                trades += 1
                if ret > 0: wins += 1
        
        # 評分標準：總報酬 + (勝率加權)
        if trades > 0:
            win_rate = wins / trades
            score = (total_ret * 100) + (win_rate * 20) # 勝率權重較高，偏好穩定
            
            if score > best_score:
                best_score = score
                best_config = params
                best_perf = f"半年回測: 報酬 {total_ret*100:.1f}% | 勝率 {win_rate*100:.0f}% ({trades}趟)"

    return best_config, best_perf

# ==========================================
# ★ 模組 B: 自適應市場體制識別 (Regime Detection)
# ==========================================
# ==========================================
# ★ 優化後的體制識別：使用 DI 交叉判斷方向
# ==========================================
def detect_market_regime(df, threshold=25):
    """
    判斷市場狀態 (DI 交叉版):
    1. Ranging (盤整): ADX < threshold
    2. Bull Trend (多頭): ADX > threshold 且 +DI > -DI
    3. Bear Trend (空頭): ADX > threshold 且 -DI > +DI
    """
    if df is None or len(df) < 100: return "UNKNOWN", 0

    try:
        # 1. 計算 ADX 完整數據 (包含 ADX, DMP, DMN)
        # pandas_ta 的 adx 函數會返回三列數據
        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        
        if adx_data is None or adx_data.empty: return "UNKNOWN", 0
        
        # 取得最新一筆數據
        # 注意：pandas_ta 的欄位命名預設為 ADX_14, DMP_14 (+DI), DMN_14 (-DI)
        current_adx = adx_data['ADX_14'].iloc[-1]
        plus_di = adx_data['DMP_14'].iloc[-1]   # 多方力道
        minus_di = adx_data['DMN_14'].iloc[-1]  # 空方力道

        # 2. 判定邏輯
        regime = ""
        
        # 先看戰況激不激烈 (趨勢強度)
        if current_adx < threshold:
            regime = "RANGING" # 盤整震盪
        else:
            # 再看誰贏 (趨勢方向) - 這是您要的修改
            if plus_di > minus_di:
                regime = "BULL_TREND" # 多方勝
            else:
                regime = "BEAR_TREND" # 空方勝
                
        return regime, current_adx
    except Exception as e:
        # print(f"Error: {e}") # 除錯用
        return "UNKNOWN", 0

def get_adaptive_config(df, original_config):
    regime, adx_val = detect_market_regime(df)
    new_config = original_config.copy()
    
    new_config['regime'] = regime
    new_config['adx'] = adx_val
    if 'adaptive_msg' not in new_config: new_config['adaptive_msg'] = "維持原始設定"
    
    if "TWD" in new_config['symbol']: return new_config

    # ★ 體制覆蓋邏輯 (Regime Override) ★
    
    if regime == "BULL_TREND":
        # === 多頭趨勢 ===
        if original_config['mode'] in ["KD", "BOLL_RSI"]:
            # ★★★ 關鍵修改：不只切換，還執行 SuperTrend 優化 ★★★
            best_p, best_m, opt_msg = optimize_supertrend(df)
            
            new_config['mode'] = "SUPERTREND"
            new_config['period'] = best_p
            new_config['multiplier'] = best_m
            new_config['adaptive_msg'] += f" ➔ 強力趨勢，轉為 SuperTrend {opt_msg}"
            
        elif "RSI" in original_config['mode']:
            new_config['entry_rsi'] = max(new_config.get('entry_rsi', 30), 45)
            new_config['exit_rsi'] = 90
            new_config['adaptive_msg'] += " (多頭修正: 放寬買點)"

    elif regime == "BEAR_TREND":
        # === 空頭趨勢 ===
        if "RSI" in original_config['mode']:
            new_config['entry_rsi'] = 20
            new_config['exit_rsi'] = 50
            new_config['adaptive_msg'] += " (空頭修正: 嚴格抄底)"
        else:
            new_config['mode'] = "RSI_RSI"
            new_config['entry_rsi'] = 20
            new_config['exit_rsi'] = 45
            new_config['adaptive_msg'] = "空頭保護：強制轉為深跌反彈策略"

    elif regime == "RANGING":
        # === 盤整震盪 ===
        if original_config['mode'] in ["SUPERTREND", "MA_CROSS"]:
            new_config['mode'] = "KD"
            new_config['entry_k'] = 20
            new_config['exit_k'] = 80
            new_config['adaptive_msg'] = "盤整震盪：轉為 KD 區間操作"

    return new_config

# ==========================================
# ★ 模組 C: 財報 / 情緒 / 籌碼
# ==========================================
@st.cache_data(ttl=86400)
def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol or "-USD" in symbol: return None 
        stock = yf.Ticker(symbol)
        info = stock.info
        if info.get('quoteType', '').upper() != 'EQUITY': return None
        return {
            "growth": info.get('revenueGrowth', 0), "pe": info.get('trailingPE', None), 
            "eps": info.get('trailingEps', None), "inst": info.get('heldPercentInstitutions', 0),
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
        news = stock.news
        if not news: return 0, "無新聞", []
        
        classifier = load_finbert_model()
        texts = [f"{item.get('title', '')}. {item.get('summary', '')}"[:512] for item in news[:5]]
        titles = [item.get('title', '') for item in news[:5]]
        if not texts: return 0, "無新聞", []

        results = classifier(texts)
        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        total_score = 0; logs = []
        
        for i, res in enumerate(results):
            val = score_map[res['label']] * res['score']
            total_score += val
            icon = "🔥" if res['label']=="positive" else "❄️" if res['label']=="negative" else "⚪"
            logs.append(f"{icon} {res['label'][:3].upper()} {res['score']:.2f}: {titles[i]}")
            
        return total_score/len(texts), titles[0], logs
    except Exception as e: return 0, f"AI Error: {str(e)[:20]}", []

def predict_volatility(df):
    try:
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        if atr is None: return None, None
        return df['Close'].iloc[-1] + atr.iloc[-1], df['Close'].iloc[-1] - atr.iloc[-1]
    except: return None, None

def analyze_chips_volume(df, inst, short_pct):
    try:
        obv = ta.obv(df['Close'], df['Volume'])
        if obv is None or len(obv)<20: return "無量能數據"
        trend = "🔴 流入" if obv.iloc[-1] > ta.sma(obv, length=20).iloc[-1] else "🟢 渙散"
        msg = f"{trend}"
        if inst and inst > 0: msg += f" | 機構 {inst*100:.0f}%"
        if short_pct and short_pct > 0.2: msg += f" | ⚠️ 軋空警戒 ({short_pct*100:.1f}%)"
        return msg
    except: return "計算錯誤"

# ==========================================
# ★ 模組 D: 視覺化與回測 (含 200MA)
# ==========================================
def plot_interactive_chart(df, config, signals=None):
    if df is None or df.empty: return None
    COLOR_UP, COLOR_DOWN = '#089981', '#f23645'
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])

    # K線
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name='Price', increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN
    ), row=1, col=1)

    # ★ 200 EMA (牛熊分界線)
    try:
        ma200 = ta.ema(df['Close'], length=200)
        fig.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name='200 EMA (牛熊分界)', line=dict(color='#9c27b0', width=2)), row=1, col=1)
    except: pass

    # 策略指標
    if config['mode'] == "SUPERTREND":
        st_data = ta.supertrend(df['High'], df['Low'], df['Close'], length=config['period'], multiplier=config['multiplier'])
        if st_data is not None:
            fig.add_trace(go.Scatter(x=df.index, y=st_data[st_data.columns[0]], mode='lines', name='SuperTrend', line=dict(color='#ff9800')), row=1, col=1)
    elif config['mode'] == "MA_CROSS":
        fast = ta.sma(df['Close'], length=config['fast_ma'])
        slow = ta.sma(df['Close'], length=config['slow_ma'])
        fig.add_trace(go.Scatter(x=df.index, y=fast, name=f'MA {config["fast_ma"]}', line=dict(color='yellow', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=slow, name=f'MA {config["slow_ma"]}', line=dict(color='blue', width=1)), row=1, col=1)

    # 副圖
    if "RSI" in config['mode'] or config['mode'] == "FUSION" or config['mode'] == "BOLL_RSI":
        rsi = ta.rsi(df['Close'], length=config.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI', line=dict(color='#b39ddb')), row=2, col=1)
        fig.add_hline(y=config.get('entry_rsi', 30), line_dash="solid", line_color=COLOR_UP, row=2, col=1)
        fig.add_hline(y=config.get('exit_rsi', 70), line_dash="solid", line_color=COLOR_DOWN, row=2, col=1)
    elif config['mode'] == "KD":
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
        if stoch is not None:
            fig.add_trace(go.Scatter(x=df.index, y=stoch.iloc[:, 0], name='K', line=dict(color='#ffeb3b')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=stoch.iloc[:, 1], name='D', line=dict(color='#2962ff')), row=2, col=1)

    # 買賣訊號點
    if signals is not None:
        buy_pts = df.loc[signals == 1]; sell_pts = df.loc[signals == -1]
        if not buy_pts.empty: fig.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00e676'), name='Buy'), row=1, col=1)
        if not sell_pts.empty: fig.add_trace(go.Scatter(x=sell_pts.index, y=sell_pts['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff1744'), name='Sell'), row=1, col=1)

    # 版面設定
    adx_val = config.get('adx', 0); regime = config.get('regime', 'N/A')
    title_text = f"策略視圖 | 市場體制: {regime} (強度 ADX: {adx_val:.1f})"
    
    fig.update_layout(title=dict(text=title_text, font=dict(size=14, color='white')), height=500, margin=dict(t=50, b=0, l=10, r=10), paper_bgcolor='#131722', plot_bgcolor='#131722', font=dict(color='#d1d4dc'), showlegend=True, hovermode='x unified')
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    return fig

def quick_backtest(df, config):
    if df is None or len(df) < 50: return None, None
    bt_df = df.copy(); close = bt_df['Close']; signals = pd.Series(0, index=bt_df.index)
    
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
        regime_icon = "🦁" if config.get('regime') == "BULL_TREND" else "🐻" if config.get('regime') == "BEAR_TREND" else "🦀"
        regime_text = "多頭" if config.get('regime') == "BULL_TREND" else "空頭" if config.get('regime') == "BEAR_TREND" else "盤整"
        
        st.subheader(f"{row['Name']}")
        st.markdown(f"**市場狀態:** {regime_icon} {regime_text} (ADX:{config.get('adx',0):.0f})")
        
        if config.get('adaptive_msg'):
            st.info(f"🧬 AI 進化策略: {config['adaptive_msg']}")

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
        
        c1, c2 = st.columns(2)
        c1.markdown(f"**財報:** {row.get('Fund', '--')}\n\n**籌碼:** {row.get('Chip', '--')}")
        c2.markdown(f"**情緒:** {row.get('Sent', '--')}\n\n**預測:** {row.get('Pred', '--')}")

        if row.get("Raw_DF") is not None:
            with st.expander("📊 K線圖與驗證 (點擊展開)", expanded=False):
                signals, perf = quick_backtest(row["Raw_DF"], config)
                st.plotly_chart(plot_interactive_chart(row["Raw_DF"], config, signals), use_container_width=True)
                if perf: st.write(f"當前策略模擬績效: 報酬 {perf['Total_Return']:.1f}% | 勝率 {perf['Win_Rate']:.0f}%")
        
        st.divider()
        st.text(f"🛠 執行策略: {config['mode']} | 掛買: {row['Buy_At']} | 掛賣: {row['Sell_At']}")

# ==========================================
# 4. 主邏輯與策略庫
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
        
        # ---------------------------------------------------
        # ★★★ AI 進化區塊 (Evolution Block) ★★★
        # ---------------------------------------------------
        evolved_msg = ""
        # 只有當模式包含 RSI 時才啟用進化 (避免干擾 TSM/USD 邏輯)
        if "RSI" in base_config['mode'] or base_config['mode'] == "FUSION":
            best_params, best_perf = evolve_strategy(df_daily, symbol)
            
            if best_params:
                # 覆蓋原本的設定，這就是「進化」
                base_config['rsi_len'] = best_params['rsi_len']
                base_config['entry_rsi'] = best_params['entry']
                base_config['exit_rsi'] = best_params['exit']
                evolved_msg = f"{best_params['desc']} - {best_perf}"
        
        # ---------------------------------------------------
        # ★★★ 體制適應 (Regime Adaptation) ★★★
        # ---------------------------------------------------
        config = get_adaptive_config(df_daily, base_config)
        if evolved_msg: 
             # 將進化訊息與體制適應訊息合併
             config['adaptive_msg'] = f"{evolved_msg} ➔ {config.get('adaptive_msg', '')}"

        # ---------------------------------------------------
        # 計算訊號
        # ---------------------------------------------------
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

        if config['mode'] == "SUPERTREND":
            st_data = ta.supertrend(high, low, close, length=config['period'], multiplier=config['multiplier'])
            if st_data is not None:
                curr_dir, prev_dir, st_value = st_data.iloc[-1, 1], st_data.iloc[-2, 1], st_data.iloc[-1, 0]
                sell_at = f"${st_value:.2f}"
                if prev_dir == -1 and curr_dir == 1: signal, action_msg, signal_type = "🚀 BUY", "趨勢翻多 (Breakout)", "BUY"
                elif prev_dir == 1 and curr_dir == -1: signal, action_msg, signal_type = "📉 SELL", "趨勢翻空 (Breakdown)", "SELL"
                elif curr_dir == 1: signal, action_msg, signal_type = "✊ HOLD", f"趨勢多頭 (止損 {st_value:.2f})", "HOLD"
                else: signal, action_msg, signal_type = "☁️ EMPTY", f"趨勢空頭", "EMPTY"

        elif config['mode'] in ["RSI_RSI", "RSI_MA", "FUSION"]:
            rsi_len = config.get('rsi_len', 14)
            curr_rsi = ta.rsi(close, length=rsi_len).iloc[-1]
            b_price = find_price_for_rsi(df_daily, config['entry_rsi'], length=rsi_len)
            buy_at = f"${b_price:.2f}"
            
            if config['mode'] == "RSI_RSI" or config['mode'] == "FUSION": 
                s_val = find_price_for_rsi(df_daily, config['exit_rsi'], length=rsi_len)
                sell_at = f"${s_val:.2f}"
                if curr_rsi < config['entry_rsi']: 
                    signal, action_msg, signal_type = "🔥 BUY", f"RSI低檔 ({curr_rsi:.1f})", "BUY"
                elif curr_rsi > config['exit_rsi']: 
                    signal, action_msg, signal_type = "💰 SELL", f"RSI高檔 ({curr_rsi:.1f})", "SELL"
                else: action_msg = f"區間震盪 (RSI: {curr_rsi:.1f})"
            else: 
                s_val = ta.sma(close, length=config['exit_ma']).iloc[-1]
                sell_at = f"${s_val:.2f} (MA)"
                if curr_rsi < config['entry_rsi']: 
                    signal, action_msg, signal_type = "🔥 BUY", f"短線超賣", "BUY"
                elif curr_price > s_val: 
                    signal, action_msg, signal_type = "💰 SELL", f"觸及均線壓力", "SELL"

        elif config['mode'] == "KD":
            stoch = ta.stoch(high, low, close, k=9, d=3, smooth_k=3)
            curr_k = stoch.iloc[:, 0].iloc[-1]
            buy_at, sell_at = f"K<{config['entry_k']}", f"K>{config['exit_k']}"
            if curr_k < config['entry_k']: signal, action_msg, signal_type = "🚀 BUY", f"KD低檔交叉", "BUY"
            elif curr_k > config['exit_k']: signal, action_msg, signal_type = "💀 SELL", f"KD高檔鈍化", "SELL"
            else: action_msg = f"K值: {curr_k:.1f}"

        elif config['mode'] == "MA_CROSS":
             fast = ta.sma(close, length=config['fast_ma']); slow = ta.sma(close, length=config['slow_ma'])
             curr_fast, prev_fast = fast.iloc[-1], fast.iloc[-2]
             curr_slow, prev_slow = slow.iloc[-1], slow.iloc[-2]
             if prev_fast <= prev_slow and curr_fast > curr_slow: signal, action_msg, signal_type = "🔥 BUY", "黃金交叉", "BUY"
             elif prev_fast >= prev_slow and curr_fast < curr_slow: signal, action_msg, signal_type = "📉 SELL", "死亡交叉", "SELL"
             elif curr_fast > curr_slow: signal, action_msg, signal_type = "✊ HOLD", "多頭排列", "HOLD"
             else: signal, action_msg, signal_type = "☁️ EMPTY", "空頭排列", "EMPTY"

        # 整合財報與情緒
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
# 5. 策略清單與執行
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
    "BTC_W": { "symbol": "BTC-USD", "name": "BTC (波段)", "mode": "RSI_RSI", "entry_rsi": 44, "exit_rsi": 65, "rsi_len": 14, "ma_trend": 200 },
    "TSM": { "symbol": "TSM", "name": "TSM (趨勢)", "mode": "MA_CROSS", "fast_ma": 5, "slow_ma": 60 },
}

with st.sidebar:
    st.header("🇹🇼 台股雷達")
    try:
        with st.spinner('更新台股數據中...'):
            t = yf.Ticker("^TWII"); twii_now = t.fast_info['last_price']; twii_prev = t.fast_info['previous_close']
            st.metric("台股加權指數", f"{twii_now:,.0f}", f"{(twii_now - twii_prev) / twii_prev * 100:+.2f}%")
    except: st.error("連線異常")
    st.divider()
    st.info("🧬 AI 進化引擎已啟動：每次掃描皆會執行『步進最佳化』，為每檔股票尋找最佳參數。")

st.subheader("📋 核心持股清單 (AI Evolution + Regime)")
col1, col2 = st.columns(2)
placeholder_list = [col1.empty() if i % 2 == 0 else col2.empty() for i in range(len(strategies))]

for i, (key, config) in enumerate(strategies.items()):
    with placeholder_list[i].container(): st.text(f"🧬 AI 正在進化並分析 {config['name']} ...")
    row = analyze_ticker(config)
    placeholder_list[i].empty()
    display_stock_card(placeholder_list[i], row, get_adaptive_config(row.get('Raw_DF'), config))

st.success("✅ 掃描完成 | Strategies Evolved & Adapted")
