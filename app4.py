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
import json
import time

# ==========================================
# ★★★ 1. 強制編碼修復 ★★★
# ==========================================
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# ==========================================
# ★★★ 2. 套件安全匯入 ★★★
# ==========================================
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(
    page_title="2026 量化戰情室 (Ultimate v7.6)",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        h1, h2, h3, h4, h5, h6, span, div, p { color: #d1d4dc !important; font-family: 'Roboto', sans-serif; }
        div[data-testid="stMetric"] { background-color: #1c202a; border: 1px solid #2a2e39; border-radius: 8px; padding: 10px; }
        div[data-testid="stMetricLabel"] > div { color: #787b86 !important; }
        div[data-testid="stMetricValue"] > div { color: #d1d4dc !important; }
        section[data-testid="stSidebar"] { background-color: #161920; border-right: 1px solid #2a2e39; }
        .stButton > button { background-color: #2962ff; color: white; border: none; border-radius: 4px; font-weight: 600; }
        .stButton > button:hover { background-color: #1e4bd1; }
        .streamlit-expanderHeader { background-color: #1c202a !important; color: #d1d4dc !important; border: 1px solid #2a2e39; }
    </style>
""", unsafe_allow_html=True)

st.title("💎 量化交易 (Ultimate v7.6)")
st.caption("完全體：OBV+CMF 籌碼透視 | 垃圾新聞過濾 | 智慧緩存防爆 | 雙引擎 AI")

if st.button('🔄 強制刷新行情 (Clear Cache)'):
    st.cache_data.clear()
    if 'ai_cache' in st.session_state:
        del st.session_state['ai_cache']
    st.rerun()

if not HAS_GEMINI:
    st.warning("⚠️ 系統提示：google-generativeai 未安裝，無法使用 Gemini。")

# ==========================================
# ★★★ 策略清單 (保持不變) ★★★
# ==========================================
strategies = {
    "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (美元兌台幣匯率)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 25, "exit_k": 70 },
    "QQQ": { "symbol": "QQQ", "name": "QQQ (那斯達克100 ETF)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200 },
    "QLD": { "symbol": "QLD", "name": "QLD (那斯達克 2倍做多)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200 },
    "TQQQ": { "symbol": "TQQQ", "name": "TQQQ (那斯達克 3倍做多)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 85, "rsi_len": 2, "ma_trend": 200 },
    "SOXL_S": { "symbol": "SOXL", "name": "SOXL (費半 3倍做多 - 狙擊)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 10, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 100 },
    "SOXL_F": { "symbol": "SOXL", "name": "SOXL (費半 3倍做多 - 快攻)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 10, "exit_k": 75 },
    "EDZ": { "symbol": "EDZ", "name": "EDZ (新興市場 3倍做空 - 避險)", "category": "📊 指數/外匯", "mode": "BOLL_RSI", "entry_rsi": 9, "rsi_len": 2, "ma_trend": 20 },
    "BTC_W": { "symbol": "BTC-USD", "name": "BTC (比特幣 - 波段)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 44, "exit_rsi": 65, "rsi_len": 14, "ma_trend": 200 },
    "BTC_F": { "symbol": "BTC-USD", "name": "BTC (比特幣 - 閃電)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 50, "rsi_len": 2, "ma_trend": 100 },
    "NVDA": { "symbol": "NVDA", "name": "NVDA (AI 算力之王)", "category": "🤖 AI 硬體/晶片", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200, "vix_max": 32, "rvol_max": 2.5 },
    "TSM": { "symbol": "TSM", "name": "TSM (台積電 ADR - 晶圓代工)", "category": "🤖 AI 硬體/晶片", "mode": "MA_CROSS", "fast_ma": 5, "slow_ma": 60 },
    "AVGO": { "symbol": "AVGO", "name": "AVGO (博通 - AI 網通晶片)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 5, "entry_rsi": 55, "exit_rsi": 85, "ma_trend": 200 },
    "MRVL": { "symbol": "MRVL", "name": "MRVL (邁威爾 - ASIC 客製化晶片)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 20, "exit_rsi": 90, "ma_trend": 100 },
    "QCOM": { "symbol": "QCOM", "name": "QCOM (高通 - AI 手機/PC)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 8, "entry_rsi": 30, "exit_rsi": 70, "ma_trend": 100 },
    "GLW": { "symbol": "GLW", "name": "GLW (康寧 - 玻璃基板/光通訊)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
    "ONTO": { "symbol": "ONTO", "name": "ONTO (安圖 - CoWoS 檢測設備)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 50, "exit_rsi": 65, "ma_trend": 100 },
    "META": { "symbol": "META", "name": "META (臉書 - 廣告與元宇宙)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 40, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
    "GOOGL": { "symbol": "GOOGL", "name": "GOOGL (谷歌 - 搜尋與 Gemini)", "category": "💻 軟體/巨頭", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200, "vix_max": 32, "rvol_max": 2.5 },
    "ETN": { "symbol": "ETN", "name": "ETN (伊頓 - 電網與電力管理)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 40, "exit_rsi": 95, "ma_trend": 200 },
    "VRT": { "symbol": "VRT", "name": "VRT (維諦 - AI 伺服器液冷)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 35, "exit_rsi": 95, "ma_trend": 100 },
    "OKLO": { "symbol": "OKLO", "name": "OKLO (核能 - 微型反應堆)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 50, "exit_rsi": 95, "ma_trend": 0 },
    "SMR": { "symbol": "SMR", "name": "SMR (NuScale - 模組化核能)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 45, "exit_rsi": 90, "ma_trend": 0 },
    "KO": { "symbol": "KO", "name": "KO (可口可樂 - 消費必需品)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
    "JNJ": { "symbol": "JNJ", "name": "JNJ (嬌生 - 醫療與製藥)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 25, "exit_rsi": 90, "ma_trend": 200 },
    "PG": { "symbol": "PG", "name": "PG (寶僑 - 日用品龍頭)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 20, "exit_rsi": 80, "ma_trend": 0 },
    "BA": { "symbol": "BA", "name": "BA (波音 - 航太製造)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 15, "exit_rsi": 60, "ma_trend": 0 },
    "CHT": { "symbol": "2412.TW", "name": "中華電 (台灣電信龍頭)", "category": "🇹🇼 台股", "mode": "RSI_RSI", "rsi_len": 14, "entry_rsi": 45, "exit_rsi": 70, "ma_trend": 0 }
}

# ==========================================
# 1. 核心函數 (資料獲取 & 新聞過濾)
# ==========================================
def get_real_live_price(symbol):
    try:
        if symbol.endswith(".TW"):
             df_rt = yf.download(symbol, period="5d", interval="1m", progress=False)
        elif "-USD" in symbol or "=X" in symbol:
            df_rt = yf.download(symbol, period="1d", interval="1m", progress=False)
        else:
            df_rt = yf.download(symbol, period="5d", interval="1m", prepost=True, progress=False)
            
        if df_rt.empty: return None
        if isinstance(df_rt.columns, pd.MultiIndex): 
            df_rt.columns = df_rt.columns.get_level_values(0)
            
        return float(df_rt['Close'].iloc[-1])
    except: 
        try:
            return float(yf.Ticker(symbol).fast_info.get('last_price'))
        except:
            return None

def get_safe_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except: return None

def clean_text_for_llm(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^\w\s\u4e00-\u9fff.,:;%()\-]', '', text)

# ★★★ 智慧過濾新聞 (垃圾過濾器) ★★★
def get_news_content(symbol):
    try:
        if "=" in symbol or "^" in symbol: return []
        stock = yf.Ticker(symbol)
        news = stock.news
        if not news: return []
        clean_news = []
        
        # 黑名單：過濾廣告、機器人文章、標題黨
        BLACKLIST_SOURCES = ["Motley Fool", "Zacks", "InvestorPlace", "TheStreet", "Simply Wall St"]
        BAD_KEYWORDS = ["implied volatility", "put option", "call option", "zacks rank", "better buy", "forget", "prediction", "forecast", "10 stocks", "price target", "alert", "why is moving"]
        
        for n in news[:10]: # 掃描前 10 則
            title = n.get('title', n.get('content', {}).get('title', ''))
            publisher = n.get('publisher', 'Unknown')
            
            # 過濾 1: 來源
            if any(bad_src in publisher for bad_src in BLACKLIST_SOURCES): continue
            
            # 過濾 2: 關鍵字
            title_lower = title.lower()
            if any(bad_wd in title_lower for bad_wd in BAD_KEYWORDS): continue
            
            # 過濾 3: 長度
            if len(title) < 15: continue
            
            title = clean_text_for_llm(title)
            full_text = f"標題: {title}"
            clean_news.append(full_text)
            
            # 只要收集到 3 則優質新聞就夠了 (省 Token)
            if len(clean_news) >= 3: break
            
        return clean_news
    except: return []

# ==========================================
# 2. 基本面與 FinBERT
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
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
    except ImportError:
        return None
    except Exception as e:
        return None

def analyze_sentiment_finbert(symbol):
    if not HAS_TRANSFORMERS: return 0, "套件未安裝(跳過)", []
    try:
        if "=" in symbol or "^" in symbol: return 0, "非個股(跳過)", []
        stock = yf.Ticker(symbol)
        news_list = stock.news
        if not news_list: return 0, "無新聞", []
        
        classifier = load_finbert_model()
        if not classifier: return 0, "模型載入失敗", []
        
        texts = []
        raw_titles = [] 
        for n in news_list[:5]:
            t = n.get('title', '')
            if t: 
                clean_t = clean_text_for_llm(t)
                texts.append(clean_t)
                raw_titles.append(t)
        
        if not texts: return 0, "無新聞內容", []
        
        results = classifier(texts)
        total_score = 0
        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        debug_logs = []
        
        for i, res in enumerate(results):
            val = score_map[res['label']] * res['score']
            total_score += val
            icon = "🔥" if res['label'] == "positive" else "❄️" if res['label'] == "negative" else "⚪"
            title_preview = raw_titles[i][:30] + "..." if len(raw_titles[i]) > 30 else raw_titles[i]
            log_str = f"{icon} {res['label'].upper()} ({res['score']:.2f}): {title_preview}"
            debug_logs.append(log_str)
            
        return total_score/len(texts), texts[0], debug_logs
    except Exception as e:
        return 0, f"分析錯誤: {str(e)}", []

# ==========================================
# 3. AI 邏輯分析 (★ 二合一 One-Shot 核心 ★)
# ==========================================

# 通用重試器
def ai_retry_wrapper(func, *args):
    max_retries = 2
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(10) # 休息 10 秒再試
                    continue
                else:
                    return {"risk_decision": "PASS", "risk_reason": "429限速", "analysis_text": f"❌ 429 限速: {str(e)[:50]}"}
            else:
                return {"risk_decision": "PASS", "risk_reason": "AI錯誤", "analysis_text": f"❌ AI 錯誤: {str(e)[:50]}"}

# ★ Gemini 二合一核心 (呼叫一次，做兩件事)
def _analyze_gemini_unified_core(api_key, symbol, news_titles, tech_signal, rsi_val, model_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    if not news_titles:
        return {"risk_decision": "PASS", "risk_reason": "無新聞", "analysis_text": "⚠️ 無新聞可分析"}
        
    news_text = "\n".join(news_titles)
    
    # ★★★ 魔法 Prompt：單發射擊 JSON 格式 ★★★
    prompt = f"""
    Role: Professional Hedge Fund Manager & Risk Officer.
    Task: Analyze stock {symbol}.
    Technical Context: {tech_signal} (Current RSI: {rsi_val}).
    Recent News Headlines:
    {news_text}
    
    INSTRUCTIONS:
    1. First, check for "CATASTROPHIC RISKS" (Fraud, SEC Investigation, Bankruptcy, CEO Arrest). 
       - If ANY exists, risk_decision = "BLOCK".
       - If normal volatility/earnings miss, risk_decision = "PASS".
    2. Second, provide a concise trading analysis in Traditional Chinese (繁體中文).
       - Summarize the bull/bear logic based on news.
       - Give a Sentiment Score (-10 to +10).
       - Give a final Action Suggestion.

    OUTPUT FORMAT: JSON ONLY (No Markdown, No code blocks).
    {{
        "risk_decision": "BLOCK" or "PASS",
        "risk_reason": "Reason for decision (max 10 words, Chinese)",
        "analysis_text": "Detailed analysis in Traditional Chinese..."
    }}
    """
    
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    try:
        return json.loads(response.text)
    except:
        clean_text = response.text.replace("```json", "").replace("```", "")
        return json.loads(clean_text)

# 包裝後的呼叫函數
def analyze_stock_unified(api_provider, api_key, symbol, news_titles, tech_signal, rsi_val, model_name):
    if not news_titles:
        return "PASS", "無新聞", "⚪ 無新聞資料", False

    if api_provider == "Gemini (User Defined)" and api_key:
        res = ai_retry_wrapper(_analyze_gemini_unified_core, api_key, symbol, news_titles, tech_signal, rsi_val, model_name)
        
        decision = res.get("risk_decision", "PASS")
        reason = res.get("risk_reason", "AI Pass")
        text = res.get("analysis_text", "無分析內容")
        success = "❌" not in text
        return decision, reason, text, success
        
    elif api_provider == "Groq (Llama-3)" and api_key:
        return "PASS", "Groq未實作", "Groq 暫不支援二合一模式", False
        
    return "PASS", "未連線", "未設定 AI", False

def analyze_earnings_text(client, symbol, text):
    if not client: return "請先設定 Groq Key"
    short_text = text[:7000]
    prompt = f"分析 {symbol} 法說會重點：{short_text}..."
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

def analyze_earnings_audio(client, uploaded_file):
    try:
        st.info("👂 正在將語音轉為文字 (Whisper-v3)...")
        transcription = client.audio.transcriptions.create(
            file=(uploaded_file.name, uploaded_file.read()),
            model="whisper-large-v3",
            response_format="text"
        )
        return analyze_earnings_text(client, "Audio File", transcription), transcription
    except Exception as e: return f"語音分析失敗: {str(e)}", ""

# ==========================================
# 4. 技術指標與倉位計算 (含 v7.6 OBV/CMF 強化)
# ==========================================
def optimize_rsi_strategy(df, symbol):
    if df is None or df.empty: return None
    rsi_lengths = [6, 12, 14, 20]; entries = [20, 25, 30, 40]; exits = [60, 70, 75, 85]
    results = []
    close = df['Close'].values
    for l in rsi_lengths:
        rsi = ta.rsi(df['Close'], length=l)
        if rsi is None: continue
        rsi_val = rsi.values
        for ent in entries:
            for ext in exits:
                sig = np.zeros(len(close)); pos=0; entry=0; wins=0; trds=0; ret_tot=0
                sig[rsi_val < ent] = 1; sig[rsi_val > ext] = -1
                for i in range(len(close)):
                    if pos==0 and sig[i]==1: pos=1; entry=close[i]
                    elif pos==1 and sig[i]==-1:
                        pos=0; r=(close[i]-entry)/entry; ret_tot+=r; trds+=1
                        if r>0: wins+=1
                if trds>0:
                    results.append({"Length": l, "Buy": ent, "Sell": ext, "Return": ret_tot*100, "WinRate": wins/trds*100, "Trades": trds})
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

# ★★★ v7.6 強化版：OBV + CMF 綜合判讀 ★★★
def analyze_chips_volume(df, inst_percent, short_percent):
    try:
        if df is None or len(df) < 30: return "資料不足"
        
        # 1. OBV (長期趨勢 - 水庫水位)
        obv = ta.obv(df['Close'], df['Volume'])
        obv_ma = ta.sma(obv, length=20)
        if obv is None or obv_ma is None: return "OBV計算失敗"
        
        # 2. CMF (短期資金 - 水龍頭流速)
        cmf = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
        curr_cmf = cmf.iloc[-1] if cmf is not None else 0
        
        # 3. 綜合研判
        status = "⚪ 籌碼中性"
        details = []
        
        # OBV 趨勢判斷
        obv_trend = "↗️ OBV上升" if obv.iloc[-1] > obv_ma.iloc[-1] else "↘️ OBV下降"
        
        # 結合 CMF 做強弱判定
        if curr_cmf > 0.15 and obv.iloc[-1] > obv_ma.iloc[-1]:
            status = "🔥 籌碼雙多 (量滾量)"
            details.append("主力狂買")
        elif curr_cmf < -0.15 and obv.iloc[-1] < obv_ma.iloc[-1]:
            status = "❄️ 籌碼雙空 (人去樓空)"
            details.append("主力棄守")
        elif curr_cmf > 0.05 and obv.iloc[-1] < obv_ma.iloc[-1]:
            status = "❓ 內外分歧 (低檔吸籌?)"
            details.append("OBV低/CMF高")
        elif curr_cmf < -0.05 and obv.iloc[-1] > obv_ma.iloc[-1]:
            status = "⚠️ 高檔出貨 (拉高倒貨?)"
            details.append("OBV高/CMF低")
            
        if inst_percent > 0.1: details.append(f"法人{inst_percent*100:.0f}%") 
        if short_percent > 0.2: details.append(f"⚠️軋空警戒{short_percent*100:.1f}%")
        
        final_msg = f"{status} | {obv_trend} | CMF:{curr_cmf:.2f}"
        if details: final_msg += f" ({' '.join(details)})"
        return final_msg
    except Exception as e: return f"籌碼錯誤: {str(e)}"

def calculate_position_size(price, df, capital, risk_pct):
    try:
        if df is None or len(df) < 15: return "N/A"
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
        stop_loss_dist = 2 * atr
        risk_amount = capital * (risk_pct / 100)
        shares = risk_amount / stop_loss_dist
        total_cost = shares * price
        if total_cost > capital:
            shares = capital / price
            return f"{int(shares)}股 (滿倉)"
        return f"{int(shares)}股 (約${total_cost:.0f})"
    except: return "計算失敗"

# ==========================================
# 5. 主分析邏輯 (v7.4 智慧緩存版)
# ==========================================
def analyze_ticker(config, ai_provider, api_key_groq, api_key_gemini, gemini_model_name):
    symbol = config['symbol']
    
    # 1. 基礎數據 (每次都要抓最新的)
    df = get_safe_data(symbol)
    if df is None: return None

    lp = get_real_live_price(symbol)
    if lp is None: lp = df['Close'].iloc[-1]
    prev_c = df['Close'].iloc[-1]
    
    new_row = pd.DataFrame({'Close': [lp], 'High': [max(lp, df['High'].iloc[-1])], 'Low': [min(lp, df['Low'].iloc[-1])], 'Open': [lp], 'Volume': [0]}, index=[pd.Timestamp.now()])
    calc_df = pd.concat([df.copy(), new_row])
    c, h, l = calc_df['Close'], calc_df['High'], calc_df['Low']
    
    # 2. 技術指標計算 (本機運算)
    sig = "WAIT"; act = "觀望"; buy_at = "---"; sell_at = "---"; sig_type = "WAIT"; strategy_desc = ""
    
    if config['mode'] == "SUPERTREND":
        st_val = ta.supertrend(h, l, c, length=config['period'], multiplier=config['multiplier'])
        strategy_desc = f"SuperTrend (P={config['period']}, M={config['multiplier']})"
        if st_val is not None:
            dr = st_val.iloc[-1, 1]; p_dr = st_val.iloc[-2, 1]; s_line = st_val.iloc[-1, 0]
            sell_at = f"${s_line:.2f}"
            if p_dr == -1 and dr == 1: sig = "🚀 BUY"; act = "趨勢翻多"; sig_type="BUY"
            elif p_dr == 1 and dr == -1: sig = "📉 SELL"; act = "趨勢翻空"; sig_type="SELL"
            elif dr == 1: sig = "✊ HOLD"; act = f"多頭續抱 (損{s_line:.1f})"; sig_type="HOLD"
            else: sig = "☁️ EMPTY"; act = "空頭觀望"; sig_type="EMPTY"

    elif config['mode'] == "FUSION":
        rsi = ta.rsi(c, length=config['rsi_len']).iloc[-1]
        ma = ta.ema(c, length=config['ma_trend']).iloc[-1]
        buy_at = f"${find_price_for_rsi(df, config['entry_rsi'], config['rsi_len'])}"
        sell_at = f"${find_price_for_rsi(df, config['exit_rsi'], config['rsi_len'])}"
        strategy_desc = f"FUSION (RSI<{config['entry_rsi']} + EMA{config['ma_trend']})"
        if lp > ma and rsi < config['entry_rsi']: sig = "🔥 BUY"; act = "趨勢回檔超跌"; sig_type="BUY"
        elif rsi > config['exit_rsi']: sig = "💰 SELL"; act = "RSI過熱獲利"; sig_type="SELL"
        else: act = f"趨勢多頭 (RSI:{rsi:.1f})"

    elif config['mode'] in ["RSI_RSI", "RSI_MA"]:
        rsi = ta.rsi(c, length=config.get('rsi_len', 14)).iloc[-1]
        use_trend = config.get('ma_trend', 0) > 0
        is_trend_ok = True
        trend_msg = ""
        if use_trend:
            ma_val = ta.ema(c, length=config['ma_trend']).iloc[-1]
            if lp < ma_val: 
                is_trend_ok = False
                trend_msg = f"(逆勢: 破MA{config['ma_trend']})"
            else:
                trend_msg = f"(順勢: 上MA{config['ma_trend']})"

        buy_at = f"${find_price_for_rsi(df, config['entry_rsi'], config.get('rsi_len', 14))}"
        
        if config['mode'] == "RSI_RSI":
            strategy_desc = f"RSI區間 (L={config.get('rsi_len',14)}, Buy<{config['entry_rsi']}, Sell>{config['exit_rsi']})"
            sell_at = f"${find_price_for_rsi(df, config['exit_rsi'], config.get('rsi_len', 14))}"
            if rsi < config['entry_rsi']: 
                if is_trend_ok: sig = "🔥 BUY"; act = f"RSI低檔 ({rsi:.1f}) {trend_msg}"; sig_type="BUY"
                else: sig = "✋ WAIT"; act = f"RSI低但逆勢 {trend_msg} 不接刀"; sig_type="WAIT"
            elif rsi > config['exit_rsi']: sig = "💰 SELL"; act = f"RSI高檔 ({rsi:.1f})"; sig_type="SELL"
            else: act = f"區間震盪 (RSI:{rsi:.1f})"
        else:
            s_val = ta.sma(c, length=config['exit_ma']).iloc[-1]
            strategy_desc = f"RSI+MA (RSI<{config['entry_rsi']} 買, 破MA{config['exit_ma']} 賣)"
            sell_at = f"${s_val:.2f}"
            if rsi < config['entry_rsi']: 
                if is_trend_ok: sig = "🔥 BUY"; act = f"短線超賣 {trend_msg}"; sig_type="BUY"
                else: sig = "✋ WAIT"; act = f"超賣但逆勢 {trend_msg}"; sig_type="WAIT"
            elif lp > s_val: sig = "💰 SELL"; act = "觸及均線壓力"; sig_type="SELL"

    elif config['mode'] == "KD":
        k = ta.stoch(h, l, c, k=9, d=3).iloc[-1, 0]
        buy_at = f"K<{config['entry_k']}"; sell_at = f"K>{config['exit_k']}"
        strategy_desc = f"KD震盪 (K<{config['entry_k']} 買, K>{config['exit_k']} 賣)"
        if k < config['entry_k']: sig = "🚀 BUY"; act = f"KD低檔 ({k:.1f})"; sig_type="BUY"
        elif k > config['exit_k']: sig = "💀 SELL"; act = f"KD高檔 ({k:.1f})"; sig_type="SELL"
        else: act = f"盤整中 (K:{k:.1f})"

    elif config['mode'] == "MA_CROSS":
        f, s = ta.sma(c, config['fast_ma']), ta.sma(c, config['slow_ma'])
        curr_f, prev_f = f.iloc[-1], f.iloc[-2]; curr_s, prev_s = s.iloc[-1], s.iloc[-2]
        strategy_desc = f"均線交叉 (F:{config['fast_ma']}, S:{config['slow_ma']})"
        if prev_f <= prev_s and curr_f > curr_s: sig = "🔥 BUY"; act = "黃金交叉"; sig_type="BUY"
        elif prev_f >= prev_s and curr_f < curr_s: sig = "📉 SELL"; act = "死亡交叉"; sig_type="SELL"
        elif curr_f > curr_s: sig = "✊ HOLD"; act = "多頭排列"; sig_type="HOLD"
        else: sig = "☁️ EMPTY"; act = "空頭排列"; sig_type="EMPTY"
        
    elif config['mode'] == "BOLL_RSI":
        rsi = ta.rsi(c, length=config.get('rsi_len', 2)).iloc[-1]
        bb = ta.bbands(c, length=20, std=2)
        lower = bb.iloc[-1, 0]; mid = bb.iloc[-1, 1]; upper = bb.iloc[-1, 2]
        buy_at = f"${lower:.2f}"; sell_at = f"${mid:.2f}"
        strategy_desc = f"布林+RSI (破下軌 & RSI<{config['entry_rsi']})"
        if lp < lower and rsi < config['entry_rsi']: sig = "🚑 BUY"; act = "破底搶反彈"; sig_type="BUY"
        elif lp >= upper: sig = "💀 SELL"; act = "觸上軌快逃"; sig_type="SELL"
        elif lp >= mid: sig = "⚠️ HOLD"; act = "中軸震盪"; sig_type="HOLD"
    
    elif config['mode'] == "CHIPS":
        cmf = ta.cmf(h, l, c, calc_df['Volume'], length=20)
        curr_cmf = cmf.iloc[-1]
        strategy_desc = "主力籌碼分析 (CMF+MFI)"
        if curr_cmf > 0.15: sig="🔥 BUY"; act="主力強勢吃貨"; sig_type="BUY"
        elif curr_cmf < -0.15: sig="💀 SELL"; act="主力高檔出貨"; sig_type="SELL"
        else: sig="WAIT"; act="籌碼觀察中"; sig_type="WAIT"
    
    try:
        cmf_seq = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
        curr_cmf = cmf_seq.iloc[-1] if cmf_seq is not None else 0
        vwap = ta.vwma(df['Close'], df['Volume'], length=20).iloc[-1]
        
        if lp > vwap and curr_cmf > 0.05: act += " | 🚀量價齊揚"
        elif lp < vwap and curr_cmf > 0.05: act += " | 💎主力低接"
        elif lp > vwap and curr_cmf < -0.05: act += " | ⚠️高檔虛漲"
        elif lp < vwap and curr_cmf < -0.05: act += " | 🔻空頭確認"
    except: pass

    # 3. ★★★ 智慧 AI 緩存系統 ★★★
    cache_key = f"{symbol}_{ai_provider}_{gemini_model_name}"
    
    if 'ai_cache' not in st.session_state:
        st.session_state['ai_cache'] = {}
    
    ai_result = st.session_state['ai_cache'].get(cache_key)
    
    # 如果沒有緩存，或者用戶強制刷新 -> 呼叫 AI
    if not ai_result:
        news = get_news_content(symbol)
        fund = get_fundamentals(symbol)
        
        current_rsi = ta.rsi(c, length=14).iloc[-1] if len(c) > 14 else 50
        tech_ctx = f"Price: ${lp:.2f}. Signal: {sig}. Action: {act}."
        
        # 呼叫二合一 AI
        decision, reason, text, is_llm = analyze_stock_unified(
            ai_provider, api_key_gemini if "Gemini" in ai_provider else api_key_groq,
            symbol, news, tech_ctx, current_rsi, gemini_model_name
        )
        
        # 存入緩存
        ai_result = {
            "decision": decision, "reason": reason, "text": text, "is_llm": is_llm,
            "fund": fund, "news_count": len(news)
        }
        st.session_state['ai_cache'][cache_key] = ai_result
        
    # 從緩存讀取結果
    decision = ai_result['decision']
    reason = ai_result['reason']
    llm_res = ai_result['text']
    is_llm = ai_result['is_llm']
    fund = ai_result['fund']
    
    # 4. 整合最終訊號
    if decision == "BLOCK":
        sig = "⛔ DANGER"
        act = f"AI 攔截: {reason}"
        sig_type = "WAIT"
    else:
        if ai_provider != "不使用" and is_llm:
            act += f" (✅ AI 通過)"

    fund_msg = f"PE: {fund['pe']:.1f}" if fund and fund['pe'] else "N/A"
    
    # 其他計算
    p_high, p_low = predict_volatility(df)
    pred_msg = f"${p_low:.2f}~${p_high:.2f}" if p_high else ""
    chip_msg = analyze_chips_volume(df, fund['inst'] if fund else 0, fund['short'] if fund else 0)
    
    user_capital = st.session_state.get('user_capital', 10000)
    user_risk = st.session_state.get('user_risk', 1.0)
    pos_msg = calculate_position_size(lp, df, user_capital, user_risk)

    return {
        "Symbol": symbol, "Name": config['name'], "Price": lp, "Prev_Close": prev_c,
        "Signal": sig, "Action": act, "Type": sig_type, "Buy_At": buy_at, "Sell_At": sell_at,
        "Fund": fund_msg, "LLM_Analysis": llm_res, "Is_LLM": is_llm, 
        "Raw_DF": df, "Pred": pred_msg, "Chip": chip_msg, "Strat_Desc": strategy_desc,
        "Logs": [], "Position": pos_msg
    }

# ==========================================
# 6. 視覺化 (★ 雙軸籌碼圖 ★)
# ==========================================
def plot_chart(df, config, signals=None, show_signals=True):
    if df is None: return None
    
    # 建立 3 個子圖，第 3 個子圖啟用雙軸 (secondary_y=True)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        row_heights=[0.6, 0.2, 0.2], 
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]])
    
    # Row 1: K 線圖
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#089981', decreasing_line_color='#f23645'), row=1, col=1)
    if config.get('ma_trend', 0) > 0:
        ma_trend = ta.ema(df['Close'], length=config['ma_trend'])
        fig.add_trace(go.Scatter(x=df.index, y=ma_trend, name=f"EMA {config['ma_trend']}", line=dict(color='purple', width=2)), row=1, col=1)

    # Row 2: 策略指標
    if "RSI" in config['mode'] or config['mode'] == "FUSION" or config['mode'] == "BOLL_RSI":
        rsi = ta.rsi(df['Close'], length=config.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='#b39ddb', width=2)), row=2, col=1)
        fig.add_hline(y=config.get('entry_rsi', 30), line_dash="dash", line_color='#089981', row=2, col=1)
        fig.add_hline(y=config.get('exit_rsi', 70), line_dash="dash", line_color='#f23645', row=2, col=1)
    elif config['mode'] == "KD":
        k = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
        if k is not None:
            fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:, 0], name='K', line=dict(color='#ffeb3b', width=1.5)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:, 1], name='D', line=dict(color='#2962ff', width=1.5)), row=2, col=1)

    # ★ Row 3: 籌碼透視 (雙軸) ★
    cmf = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    if cmf is not None:
        colors = ['#089981' if v >= 0 else '#f23645' for v in cmf] 
        fig.add_trace(go.Bar(x=df.index, y=cmf, name='CMF (資金流)', marker_color=colors, opacity=0.6), row=3, col=1, secondary_y=False)
        fig.add_hline(y=0, line_color='gray', row=3, col=1, secondary_y=False)

    obv = ta.obv(df['Close'], df['Volume'])
    if obv is not None:
        fig.add_trace(go.Scatter(x=df.index, y=obv, name='OBV (累積量)', line=dict(color='#2962ff', width=2)), row=3, col=1, secondary_y=True)

    if show_signals and signals is not None:
        buy_pts = df.loc[signals == 1]; sell_pts = df.loc[signals == -1]
        if not buy_pts.empty: fig.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#089981', line=dict(width=1, color='black')), name='Buy'), row=1, col=1)
        if not sell_pts.empty: fig.add_trace(go.Scatter(x=sell_pts.index, y=sell_pts['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#f23645', line=dict(width=1, color='black')), name='Sell'), row=1, col=1)

    fig.update_layout(height=700, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='#131722', plot_bgcolor='#131722', font=dict(color='#d1d4dc', family="Roboto"), showlegend=True, 
                      xaxis=dict(showgrid=True, gridcolor='#2a2e39'), yaxis=dict(showgrid=True, gridcolor='#2a2e39'),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# ★★★ 修正回測邏輯 (支援所有策略) ★★★
def quick_backtest(df, config, fee=0.0005):
    if df is None or len(df) < 50: return None, None
    close = df['Close']; high = df['High']; low = df['Low']
    signals = pd.Series(0, index=df.index)
    
    try:
        if config['mode'] in ["RSI_RSI", "FUSION"]:
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            signals[rsi < config['entry_rsi']] = 1
            signals[rsi > config['exit_rsi']] = -1

        elif config['mode'] == "RSI_MA":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            ma_exit = ta.sma(close, length=config['exit_ma'])
            signals[rsi < config['entry_rsi']] = 1
            signals[close > ma_exit] = -1

        elif config['mode'] == "BOLL_RSI":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            bb = ta.bbands(close, length=20, std=2)
            lower = bb.iloc[:, 0]; upper = bb.iloc[:, 2]
            signals[(close < lower) & (rsi < config['entry_rsi'])] = 1
            signals[close > upper] = -1

        elif config['mode'] == "KD":
            k = ta.stoch(high, low, close, k=9, d=3).iloc[:, 0]
            signals[k < config['entry_k']] = 1
            signals[k > config['exit_k']] = -1

        elif config['mode'] == "SUPERTREND":
            st = ta.supertrend(high, low, close, length=config['period'], multiplier=config['multiplier'])
            dr = st.iloc[:, 1]
            signals[(dr == 1) & (dr.shift(1) == -1)] = 1
            signals[(dr == -1) & (dr.shift(1) == 1)] = -1

        elif config['mode'] == "MA_CROSS":
            f = ta.sma(close, config['fast_ma']); s = ta.sma(close, config['slow_ma'])
            signals[(f > s) & (f.shift(1) <= s.shift(1))] = 1; signals[(f < s) & (f.shift(1) >= s.shift(1))] = -1
            
        pos = 0; ent = 0; trd = 0; wins = 0; rets = []
        for i in range(len(df)):
            if pos == 0 and signals.iloc[i] == 1: 
                pos = 1; ent = close.iloc[i]
            elif pos == 1 and signals.iloc[i] == -1:
                pos = 0; raw_r = (close.iloc[i] - ent) / ent
                net_r = raw_r - (fee * 2)
                rets.append(net_r); trd += 1
                if net_r > 0: wins += 1
        return signals, {"Total_Return": sum(rets)*100, "Win_Rate": (wins/trd*100) if trd else 0, "Trades": trd}
    except: return None, None

def display_card(placeholder, row, config, unique_id, show_signals):
    with placeholder.container(border=True):
        st.subheader(f"{row['Name']}")
        c1, c2 = st.columns(2)
        c1.metric("昨日收盤", f"${row['Prev_Close']:,.2f}")
        c2.metric("即時價格", f"${row['Price']:,.2f}", f"{row['Price']-row['Prev_Close']:.2f}")
        
        sig_col = "green" if "BUY" in row['Signal'] else "red" if "SELL" in row['Signal'] else "gray"
        st.markdown(f"#### :{sig_col}[{row['Signal']}] - {row['Action']}")
        st.info(f"🛠️ **目前策略**: {row['Strat_Desc']}")
        
        st.warning(f"💰 **建議倉位 (Risk {st.session_state.get('user_risk', 1.0)}%)**: {row['Position']}")
        
        with st.expander("🎙️ AI 法說會工具箱 (手動版)", expanded=False):
            mode = st.radio("輸入模式", ["貼上逐字稿", "上傳錄音檔(mp3)"], horizontal=True, key=f"mode_{unique_id}")
            groq_client = st.session_state.get('groq_client_obj', None)
            
            if mode == "貼上逐字稿":
                txt_input = st.text_area("請貼上法說會內容...", height=150, key=f"txt_{unique_id}")
                if st.button("🧠 AI 分析文字", key=f"btn_txt_{unique_id}"):
                    if groq_client and txt_input:
                        with st.spinner("AI 正在研讀..."):
                            analysis = analyze_earnings_text(groq_client, row['Symbol'], txt_input)
                            st.markdown(analysis)
                    else: st.warning("請輸入內容並設定 Groq Key")
            else:
                aud_file = st.file_uploader("上傳錄音檔 (25MB內)", type=['mp3', 'wav', 'm4a'], key=f"aud_{unique_id}")
                if st.button("👂 AI 聽音辨位", key=f"btn_aud_{unique_id}"):
                    if groq_client and aud_file:
                        with st.spinner("AI 正在聆聽..."):
                            analysis, trans = analyze_earnings_audio(groq_client, aud_file)
                            st.markdown(analysis)
                            with st.expander("原始逐字稿"): st.text(trans[:1000]+"...")
                    else: st.warning("請上傳檔案並設定 Groq Key")

        if row['Is_LLM']:
            with st.expander("🧠 AI 觀點 (Gemini/Groq)", expanded=True):
                st.markdown(row['LLM_Analysis'])
        else:
            st.caption(f"FinBERT: {row['LLM_Analysis']}")
            if row.get('Logs'):
                with st.expander("📊 FinBERT 詳細情緒列表", expanded=False):
                    for log in row['Logs']:
                        st.text(log)

        if row['Raw_DF'] is not None:
            with st.expander("📊 K線與回測 (Pro Charts)", expanded=False):
                fee_rate = st.session_state.get('tx_fee', 0.0005)
                sig, perf = quick_backtest(row['Raw_DF'], config, fee_rate)
                st.plotly_chart(plot_chart(row['Raw_DF'], config, sig, show_signals), use_container_width=True)
                if perf: st.caption(f"模擬績效 (成本{fee_rate*100}%): 報酬 {perf['Total_Return']:.1f}% | 勝率 {perf['Win_Rate']:.0f}%")
        
        st.text(f"籌碼: {row['Chip']} | 波動: {row['Pred']}")

# ==========================================
# 8. 執行區 (UI 與 邏輯)
# ==========================================
with st.sidebar:
    st.header("⚙️ 設定")
    
    # 1. AI 模型選擇與 Key 輸入
    st.subheader("🤖 AI 模型選擇")
    ai_provider = st.selectbox("請選擇 AI 供應商", ["不使用", "Groq (Llama-3)", "Gemini (User Defined)"])
    
    groq_key = ""
    gemini_key = ""
    gemini_model_name = "models/gemini-2.0-flash" # Default
    
    if ai_provider == "Groq (Llama-3)":
        groq_key = st.text_input("Groq API Key", type="password")
        if groq_key: st.session_state['groq_client_obj'] = Groq(api_key=groq_key)
        
    elif ai_provider == "Gemini (User Defined)":
        gemini_key = st.text_input("Gemini API Key", type="password")
        gemini_model_name = st.text_input("Gemini Model Name", value="models/gemini-2.0-flash")
        st.caption("例如: models/gemini-2.0-flash 或 models/gemini-3-flash-preview")

    st.divider()
    st.header("💰 資金管理設定")
    capital_input = st.number_input("總操作資金 (USD)", min_value=1000, value=10000, step=1000)
    risk_input = st.number_input("單筆最大風險 (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    
    st.session_state['user_capital'] = capital_input
    st.session_state['user_risk'] = risk_input
    
    st.divider()
    
    # ---------------------------------------------
    # ★★★ 改用「點菜模式 (Selectbox)」 ★★★
    # ---------------------------------------------
    st.header("👆 選擇分析目標")
    
    market_filter = st.radio("市場區域：", ["全部", "美股", "台股"], horizontal=True)
    all_categories = sorted(list(set(s.get('category', '未分類') for s in strategies.values())))
    category_options = ["📂 全部產業"] + all_categories
    selected_category = st.selectbox("產業分類篩選：", category_options)

    # 1. 篩選出符合條件的股票清單
    filtered_strategies = {}
    for k, v in strategies.items():
        is_tw = ".TW" in v['symbol'] or "TWD" in v['symbol']
        if market_filter == "美股" and is_tw: continue
        if market_filter == "台股" and not is_tw: continue
        if selected_category != "📂 全部產業":
            if v.get('category') != selected_category: continue
        filtered_strategies[k] = v

    # 2. 製作選單 (顯示名稱而非代碼)
    option_map = {f"{v['symbol']} - {v['name']}": k for k, v in filtered_strategies.items()}
    selected_option = st.selectbox("請選擇要分析的股票：", list(option_map.keys()))
    
    # 3. 取得選中的 key
    target_key = option_map[selected_option]
    target_config = strategies[target_key]

    st.divider()
    st.header("🎛️ 顯示設定")
    show_signals = st.checkbox("顯示買賣訊號 (Buy/Sell)", value=True)
    tx_fee = st.number_input("單邊交易成本 (%)", min_value=0.0, max_value=5.0, value=0.05, step=0.01) / 100
    st.session_state['tx_fee'] = tx_fee

# ★★★ 側邊欄：日韓股早盤雷達 ★★★
st.sidebar.divider()
st.sidebar.header("🌏 亞股早盤雷達 (08:00)")

def get_market_status(symbol, name):
    try:
        data = yf.download(symbol, period="2d", interval="1d", progress=False)
        if len(data) >= 2:
            prev_close = float(data['Close'].iloc[-2])
            curr_price = get_real_live_price(symbol)
            if curr_price is None: curr_price = float(data['Close'].iloc[-1])
            change = curr_price - prev_close
            pct_change = (change / prev_close) * 100
            icon = "🔺" if change >= 0 else "🔻"
            return f"{name}", f"{curr_price:,.0f}", f"{icon} {pct_change:.2f}%"
        return name, "N/A", "N/A"
    except: return name, "連線失敗", "---"

m1, m2 = st.sidebar.columns(2)
with m1:
    n_name, n_price, n_chg = get_market_status("^N225", "🇯🇵 日經")
    st.metric(n_name, n_price, n_chg)
with m2:
    k_name, k_price, k_chg = get_market_status("^KS11", "🇰🇷 韓綜")
    st.metric(k_name, k_price, k_chg)

# ==========================================
# 9. 執行區 (改為單股分析)
# ==========================================
if target_key:
    st.subheader(f"📊 {target_config['name']} 深度分析")
    
    # 直接執行單股分析
    # 這裡會自動使用智慧緩存，不會重複呼叫 API
    with st.spinner(f"正在連線 {ai_provider} 分析 {target_config['symbol']} (已啟用智慧緩存)..."):
        row = analyze_ticker(target_config, ai_provider, groq_key, gemini_key, gemini_model_name)
        display_card(st.empty(), row, target_config, target_key, show_signals)
        
    # 如果有開啟參數優化，才跑這段
    if st.checkbox("🧪 執行 Grid Search 參數優化 (耗時)", value=False):
        if row['Raw_DF'] is not None:
            with st.expander(f"🧪 {target_config['symbol']} 最佳參數"):
                opt_res = optimize_rsi_strategy(row['Raw_DF'], target_config['symbol'])
                if opt_res is not None and not opt_res.empty:
                    best = opt_res.sort_values(by="Return", ascending=False).iloc[0]
                    st.write(f"最佳回報參數: RSI {int(best['Length'])} ({int(best['Buy'])}/{int(best['Sell'])}) -> 報酬 {best['Return']:.1f}%")

st.divider()
st.success("✅ 分析完成 (v7.6 終極完全體)")
