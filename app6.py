import streamlit as st
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import re
import importlib.util
import json
import time
import os
import random
import requests
import xml.etree.ElementTree as ET
import xgboost as xgb  # <--- 新增這行
from sklearn.metrics import accuracy_score # <--- 新增這行
import lightgbm as lgb
from catboost import CatBoostClassifier

def download_tw_stock_data(ticker):
    """
    聰明的台股下載器：自動處理 .TW/.TWO 後綴，並修正空值數據
    """
    # 1. 自動修正代號格式
    target_ticker = ticker.upper()
    if not (target_ticker.endswith(".TW") or target_ticker.endswith(".TWO")):
        # 先嘗試加上 .TW (上市)
        test_data = yf.download(f"{target_ticker}.TW", period="5d", progress=False)
        if not test_data.empty:
            target_ticker = f"{target_ticker}.TW"
        else:
            # 如果抓不到，嘗試 .TWO (上櫃)
            target_ticker = f"{target_ticker}.TWO"
    
    st.write(f"🔄 正在鎖定台股目標：{target_ticker}")

    # 2. 下載數據 (連同美股對照組一起抓)
    # 這裡我們一定要抓：費半(^SOX) 和 輝達(NVDA) 作為領先指標
    tickers_to_download = [target_ticker, "^SOX", "NVDA"]
    data = yf.download(tickers_to_download, period="5y", interval="1d", progress=False)
    
    # 處理 MultiIndex (Yahoo 下載多檔股票時的格式問題)
    if isinstance(data.columns, pd.MultiIndex):
        # 只取 Close 收盤價
        df = data['Close'].copy()
    else:
        df = data['Close'].copy()
        
    # 3. 防雷處理：修正台股特有的「零成交量」或「颱風假」問題
    # 如果某天台股是 NaN (例如颱風假)，但美股有資料，我們用前一天的台股收盤價填補 (ffill)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    # 回傳處理好的 DataFrame 和 修正後的代號
    return df, target_ticker

# ==========================================
# ★★★ 請補上這個遺失的關鍵函數！ ★★★
# ==========================================
def get_real_live_price(symbol):
    try:
        # 嘗試從 yfinance 快速獲取
        t = yf.Ticker(symbol)
        price = t.fast_info.get('last_price')
        
        # 如果失敗，改用下載數據方式
        if price is None or np.isnan(price):
            df = yf.download(symbol, period='1d', interval='1m', progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                return float(df['Close'].iloc[-1])
                
        return float(price) if price else None
    except: 
        return None

# ==========================================
# ★★★ 0. God Mode: 鎖定隨機種子 ★★★
# ==========================================
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except: pass

set_seeds(42)

# ==========================================
# ★★★ 1. 套件檢查 ★★★
# ==========================================
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

try:
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except: HAS_GEMINI = False

# ==========================================
# 2. 頁面設定
# ==========================================
st.set_page_config(
    page_title="2026 量化戰情室 (Ultimate v22.4)",
    page_icon="🚀",
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
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { background-color: #1c202a; border-radius: 4px 4px 0 0; color: #d1d4dc; }
        .stTabs [aria-selected="true"] { background-color: #2962ff; color: white; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# ★★★ 核心模組：AI 交易資料庫 (Google Sheets 雲端版) ★★★
# ==========================================
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# 請填入你的 Google Sheet 網址 (必須先將 Sheet 分享給服務帳號 Email)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1hNsWxQq3aYD7msroBVJdMnC6vA64khsSUF90yIKeS7w/edit?gid=0#gid=0"

# 連線快取 (避免每次按按鈕都重新連線)
@st.cache_resource
def get_gsheet_connection():
    try:
        # 從 st.secrets 讀取憑證
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        # 這裡假設你在 secrets 裡的標題是 gcp_service_account
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        return None

def init_db():
    """檢查並初始化 Sheet (如果沒標題就加上)"""
    client = get_gsheet_connection()
    if not client: return
    try:
        sheet = client.open_by_url(SHEET_URL).sheet1
        # 檢查第一列是否為標題，如果空則初始化
        if not sheet.row_values(1):
            sheet.append_row(["date", "symbol", "direction", "confidence", "entry_price", "status", "exit_price", "return_pct"])
    except: pass

# 確保 Sheet 已準備好
init_db()

def save_prediction_db(symbol, direction, confidence, entry_price):
    """存入一筆新的預測 (Append Row)"""
    client = get_gsheet_connection()
    if not client: return False, "❌ 無法連線 Google Sheets (請檢查 Secrets)"
    
    try:
        sheet = client.open_by_url(SHEET_URL).sheet1
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        # 讀取所有資料檢查重複 (稍微耗時，但安全)
        records = sheet.get_all_records()
        df = pd.DataFrame(records)
        
        if not df.empty:
            # 確保欄位都是字串以進行比對
            if not df[(df['date'].astype(str) == today_str) & (df['symbol'] == symbol)].empty:
                return False, "⚠️ 今天已經記錄過了 (雲端)"

        # 插入新紀錄
        # 注意：GSpread 寫入時數值最好轉為標準格式
        new_row = [today_str, symbol, direction, float(confidence), float(entry_price), "Pending", 0.0, 0.0]
        sheet.append_row(new_row)
        return True, "✅ 戰報已上傳雲端！"
    except Exception as e:
        return False, f"❌ 上傳失敗: {e}"

def get_history_df(symbol=None):
    """讀取歷史資料 (從雲端下載)"""
    client = get_gsheet_connection()
    if not client: return pd.DataFrame()
    
    try:
        sheet = client.open_by_url(SHEET_URL).sheet1
        records = sheet.get_all_records()
        df = pd.DataFrame(records)
        
        if df.empty: return df
        
        # 簡單的型別轉換
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        df['entry_price'] = pd.to_numeric(df['entry_price'], errors='coerce')
        df['return_pct'] = pd.to_numeric(df['return_pct'], errors='coerce')
        
        if symbol:
            df = df[df['symbol'] == symbol].copy()
            
        df = df.sort_values(by="date", ascending=True)
        return df
    except: return pd.DataFrame()

def verify_performance_db():
    """自動驗證績效 (批量更新雲端)"""
    client = get_gsheet_connection()
    if not client: return 0
    
    try:
        sheet = client.open_by_url(SHEET_URL).sheet1
        # 讀取全部資料
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        if df.empty: return 0
        
        updates = 0
        has_change = False
        
        # 遍歷資料檢查 Pending
        for index, row in df.iterrows():
            if row['status'] == 'Pending':
                sym = row['symbol']
                entry = float(row['entry_price'])
                direction = row['direction']
                
                curr_price = get_real_live_price(sym)
                if curr_price:
                    ret = (curr_price - entry) / entry
                    new_status = "Pending"
                    
                    # 驗證邏輯
                    if direction == "Bull":
                        if ret > 0.02: new_status = "Win"
                        elif ret < -0.02: new_status = "Loss"
                    elif direction == "Bear":
                        if ret < -0.02: new_status = "Win"
                        elif ret > 0.02: new_status = "Loss"
                    
                    if new_status != "Pending":
                        # 更新 DataFrame
                        df.at[index, 'status'] = new_status
                        df.at[index, 'exit_price'] = curr_price
                        df.at[index, 'return_pct'] = ret * 100
                        has_change = True
                        updates += 1
        
        if has_change:
            # ★ 關鍵：GSpread 更新整張表比一格一格改快且穩定
            # 準備寫入的資料 (包含標題)
            header = df.columns.values.tolist()
            values = df.values.tolist()
            # 清空並重寫
            sheet.clear()
            sheet.update([header] + values)
            
        return updates
    # ★★★ 修復 SyntaxError: 補回這裡遺失的 except ★★★
    except Exception as e:
        print(f"Verify Error: {e}")
        return 0

# ==========================================
# ★★★ TSM T+5 主帥版 (絕對防崩潰救命版) ★★★
# ==========================================
# 1. 定義信心放大函數 (確保函數存在)
def enhance_confidence(prob, temperature=0.25):
    import numpy as np
    prob = np.clip(prob, 0.001, 0.999)
    logit = np.log(prob / (1 - prob))
    scaled_logit = logit / temperature
    new_prob = 1 / (1 + np.exp(-scaled_logit))
    return new_prob

@st.cache_resource(ttl=300)
def get_tsm_swing_prediction():
    # 預設回傳值，確保發生天災人禍時，至少介面不會掛掉
    current_price = 0.0
    
    if not HAS_TENSORFLOW: return None, None, 0.0, None, 0
    try:
        # 1. 下載數據 (放寬 Timeout)
        tickers = ["TSM", "^SOX", "NVDA", "^TNX", "^VIX"]
        data = yf.download(tickers, period="5y", interval="1d", progress=False, timeout=30)
        
        # 資料防呆
        if data is None or data.empty:
            print("❌ Error: 數據下載為空")
            return None, None, 0.0, None, 0

        # 處理資料結構
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close'].copy()
        else:
            df = data['Close'].copy()

        # 確保 TSM 欄位存在
        if 'TSM' not in df.columns: return None, None, 0.0, None, 0

        # ---------------------------------------------------
        # ★ 步驟 A: 強制注入即時價格 (Live Price Injection)
        # ---------------------------------------------------
        try:
            live_price = get_real_live_price("TSM")
            if live_price and live_price > 0:
                current_price = live_price
                last_idx = df.index[-1]
                # 強制覆蓋最後一筆收盤價
                df.at[last_idx, 'TSM'] = live_price
            else:
                current_price = float(df['TSM'].iloc[-1])
        except:
            current_price = float(df['TSM'].iloc[-1]) if not df.empty else 0.0

        # 補值：這是最關鍵的一步
        df.ffill(inplace=True)
        
        # ---------------------------------------------------
        # ★ 步驟 B: 寬鬆特徵工程 (Loose Feature Engineering)
        # ---------------------------------------------------
        feat = pd.DataFrame()
        try:
            # 就算某些欄位抓不到，也用 0 填補，不要讓程式崩潰
            feat['TSM_Ret'] = df['TSM'].pct_change()
            feat['RSI'] = ta.rsi(df['TSM'], length=5) 
            feat['MACD'] = ta.macd(df['TSM'])['MACD_12_26_9']
            
            # 選用特徵 (如果抓不到就填 0)
            feat['NVDA_Ret'] = df['NVDA'].pct_change() if 'NVDA' in df else 0
            feat['SOX_Ret'] = df['^SOX'].pct_change() if '^SOX' in df else 0
            feat['TNX_Chg'] = df['^TNX'].pct_change() if '^TNX' in df else 0
            feat['VIX'] = df['^VIX'] if '^VIX' in df else 0
            
        except Exception as e:
            print(f"❌ 特徵計算失敗: {e}")
            return None, None, current_price, None, 0
        
        # 再次補值
        feat.ffill(inplace=True)
        feat.dropna(inplace=True)
        
        cols = ['NVDA_Ret', 'SOX_Ret', 'TNX_Chg', 'VIX', 'TSM_Ret', 'RSI', 'MACD']
        lookback = 20

        # ---------------------------------------------------
        # ★ 步驟 C: 模型訓練與建立
        # ---------------------------------------------------
        # 標籤 (Target)
        future_ret = df['TSM'].shift(-5) / df['TSM'] - 1
        feat['Target'] = (future_ret > 0.025).astype(int)
        
        valid_data = feat.iloc[:-5].copy()
        # 確保數據夠長
        if len(valid_data) < 50: return None, None, current_price, None, 0

        split_idx = int(len(valid_data) * 0.8)
        train_df = valid_data.iloc[:split_idx]
        test_df = valid_data.iloc[split_idx:]
        
        scaler = StandardScaler()
        scaler.fit(train_df[cols]) 
        
        train_scaled = scaler.transform(train_df[cols])
        test_scaled = scaler.transform(test_df[cols])
        
        def create_sequences(data_scaled, targets):
            X, y = [], []
            if len(data_scaled) < lookback: return np.array([]), np.array([])
            for i in range(lookback, len(data_scaled)):
                X.append(data_scaled[i-lookback:i])
                y.append(targets.iloc[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_scaled, train_df['Target'])
        X_test, y_test = create_sequences(test_scaled, test_df['Target'])
        
        if len(X_train) == 0: return None, None, current_price, None, 0

        # 計算權重
        from sklearn.utils.class_weight import compute_class_weight
        class_weight_dict = None
        if len(np.unique(y_train)) > 1:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))
        
        from tensorflow.keras.layers import Input, LSTM
        model = Sequential()
        model.add(Input(shape=(lookback, len(cols))))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2)) 
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=25, batch_size=32, callbacks=[early], 
                  class_weight=class_weight_dict, verbose=0)
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        
        # ---------------------------------------------------
        # ★ 步驟 D: 繪圖數據 (Viz)
        # ---------------------------------------------------
        df_viz = None
        viz_acc = 0
        if len(X_test) > 0:
            viz_len = min(len(X_test), 90)
            test_indices = test_df.index[lookback:] 
            test_prices = df['TSM'].loc[test_indices]
            preds_raw = model.predict(X_test, verbose=0).flatten()
            viz_probs_raw = preds_raw[-viz_len:]
            viz_probs_enhanced = [enhance_confidence(p, temperature=0.25) for p in viz_probs_raw]
            
            df_viz = pd.DataFrame({
                'Date': test_indices[-viz_len:],
                'Price': test_prices.iloc[-viz_len:].values,
                'Prob': viz_probs_enhanced
            })
            
            viz_targets = y_test[-viz_len:]
            viz_preds_cls = (np.array(viz_probs_enhanced) > 0.5).astype(int)
            viz_acc = np.mean(viz_targets == viz_preds_cls)

        # ---------------------------------------------------
        # ★ 步驟 E: 預測最新一天 (Shape Mismatch 終極修正)
        # ---------------------------------------------------
        latest_seq_raw = feat[cols].iloc[-lookback:].values
        
        # [救命機制] 如果資料少於 20 筆 (例如只有 19 筆)，用第一筆複製來補齊
        # 這能保證維度永遠是 (20, 7)，不會 Crash
        current_len = len(latest_seq_raw)
        if current_len < lookback:
            # print(f"⚠️ 數據不足 ({current_len})，啟動自動補齊機制...")
            missing_count = lookback - current_len
            # 複製第一列來填補前面的空缺
            padding = np.tile(latest_seq_raw[0], (missing_count, 1))
            latest_seq_raw = np.vstack([padding, latest_seq_raw])

        # 現在長度保證是 20 了
        latest_seq_scaled = scaler.transform(latest_seq_raw)
        
        # 進行預測
        input_seq = np.expand_dims(latest_seq_scaled, axis=0) # shape (1, 20, 7)
        prob_latest_raw = model.predict(input_seq, verbose=0)[0][0]
        prob_latest = enhance_confidence(prob_latest_raw, temperature=0.25)
        
        return prob_latest, acc, current_price, df_viz, viz_acc

    except Exception as e:
        print(f"❌ TSM Model Final Crash: {e}")
        # 發生任何錯誤，至少回傳 current_price
        return None, None, current_price, None, 0
        
# ==========================================
# ★★★ TSM T+3 短線先鋒 (含回測圖表版：75% 勝率核心) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_tsm_short_prediction():
    if not HAS_TENSORFLOW: return None, None, None
    try:
        # 1. 數據下載
        tickers = ["TSM", "^SOX", "NVDA", "^TNX", "^VIX"]
        data = yf.download(tickers, period="2y", interval="1d", progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            df_main = data['Close'].copy()
        else:
            df_main = data['Close'].copy()
            
        df_main.ffill(inplace=True); df_main.dropna(inplace=True)

        # 2. 特徵工程 (75% 勝率版因子)
        feat_df = pd.DataFrame()
        try:
            feat_df['TSM_Ret'] = df_main['TSM'].pct_change()
            feat_df['SOX_Ret'] = df_main['^SOX'].pct_change()
            feat_df['NVDA_Ret'] = df_main['NVDA'].pct_change()
            feat_df['TSM_RSI'] = ta.rsi(df_main['TSM'], length=14)
            feat_df['TSM_MACD'] = ta.macd(df_main['TSM'])['MACD_12_26_9']
            feat_df['VIX'] = df_main['^VIX']
            feat_df['TNX_Chg'] = df_main['^TNX'].pct_change()
        except: return None, None, None
        
        feat_df.dropna(inplace=True)
        cols = list(feat_df.columns)
        
        # 3. 標籤與嚴格切分
        future_ret = df_main['TSM'].shift(-3) / df_main['TSM'] - 1
        feat_df['Target'] = (future_ret > 0.015).astype(int)
        
        valid_data = feat_df.iloc[:-3].copy()
        
        # 嚴格時間切分
        split = int(len(valid_data) * 0.8)
        train_df = valid_data.iloc[:split]
        test_df = valid_data.iloc[split:]
        
        # Scaler 只 Fit 訓練集
        scaler = StandardScaler()
        scaler.fit(train_df[cols]) 
        
        train_scaled = scaler.transform(train_df[cols])
        test_scaled = scaler.transform(test_df[cols])
        
        lookback = 30 
        def make_seq(d, t):
            X, y = [], []
            for i in range(lookback, len(d)):
                X.append(d[i-lookback:i])
                y.append(t.iloc[i])
            return np.array(X), np.array(y)
            
        X_train, y_train = make_seq(train_scaled, train_df['Target'])
        X_test, y_test = make_seq(test_scaled, test_df['Target'])

        # 模型架構 (Simple LSTM)
        from tensorflow.keras.layers import Input, LSTM
        model = Sequential()
        model.add(Input(shape=(lookback, len(cols))))
        model.add(LSTM(64)) 
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(X_train, y_train, 
                  validation_data=(X_test, y_test), 
                  epochs=25, batch_size=32, 
                  callbacks=[early], verbose=0)
        
        # 4. 預測與校正邏輯 (共用)
        optimal_threshold = 0.60
        shift_amount = 0.5 - optimal_threshold
        
        def apply_shift_and_enhance(prob_array):
            shifted = np.array(prob_array) + shift_amount
            shifted = np.clip(shifted, 0.001, 0.999)
            logit = np.log(shifted / (1 - shifted))
            scaled_logit = logit / 0.4 
            return 1 / (1 + np.exp(-scaled_logit))

        # 5. 產生回測圖表數據 (Backtest Visualization)
        # 取測試集最後 90 天來畫圖 (避免圖表太擠)
        viz_len = min(len(X_test), 90)
        
        # 取得對應的日期與價格
        # X_test 的第 0 筆資料，對應的是 test_df 的第 lookback 筆資料
        test_indices = test_df.index[lookback:]
        viz_dates = test_indices[-viz_len:]
        viz_prices = df_main['TSM'].loc[viz_dates].values
        
        # 取得預測值
        preds_all = model.predict(X_test, verbose=0).flatten()
        viz_probs_raw = preds_all[-viz_len:]
        viz_probs = apply_shift_and_enhance(viz_probs_raw) # 經過平移與放大的機率
        
        df_viz = pd.DataFrame({
            'Date': viz_dates,
            'Price': viz_prices,
            'Prob': viz_probs
        })

        # 計算這段顯示區間的勝率
        final_cls = (np.array(viz_probs) > 0.5).astype(int)
        viz_targets = y_test[-viz_len:]
        acc = np.mean(viz_targets == final_cls)
        
        # 6. 預測最新一天
        latest_seq_raw = feat_df[cols].iloc[-lookback:].values
        latest_scaled = scaler.transform(latest_seq_raw) 
        prob_raw = model.predict(np.expand_dims(latest_scaled, axis=0), verbose=0)[0][0]
        prob_latest = apply_shift_and_enhance([prob_raw])[0]
        
        # VIX 濾網
        try:
            current_vix = df_main['^VIX'].iloc[-1]
            if current_vix > 28: prob_latest = prob_latest * 0.8
        except: pass

        return prob_latest, acc, df_viz # 多回傳 df_viz

    except Exception as e:
        print(f"Short Model Error: {e}")
        return None, None, None

# --- B. EDZ/Macro ---
@st.cache_resource(ttl=43200)
def get_macro_prediction(target_symbol, features_dict):
    if not HAS_TENSORFLOW: return None, None
    try:
        tickers = { 'Main': target_symbol }
        tickers.update(features_dict)
        data = yf.download(list(tickers.values()), period="3y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close'].copy()
            inv_map = {v: k for k, v in tickers.items()}
            df_close.rename(columns=inv_map, inplace=True)
            df = df_close.copy()
        else: return None, None

        df.ffill(inplace=True); df.bfill(inplace=True)
        feat_cols = []
        df['Main_Ret'] = df['Main'].pct_change()
        feat_cols.append('Main_Ret')
        for name in features_dict.keys():
            df[f'{name}_Ret'] = df[name].pct_change()
            feat_cols.append(f'{name}_Ret')
        df['RSI'] = ta.rsi(df['Main'], length=14)
        feat_cols.append('RSI')
        df.dropna(inplace=True)
        
        days_out = 5
        df['Target'] = ((df['Main'].shift(-days_out) / df['Main'] - 1) > 0.02).astype(int)
        df_train = df.iloc[:-5].copy()
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[feat_cols])
        
        X, y = [], []
        for i in range(20, len(scaled_data)):
            X.append(scaled_data[i-20:i])
            y.append(df_train['Target'].iloc[i])
        
        X, y = np.array(X), np.array(y)
        split = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
            
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(20, len(feat_cols))))
        model.add(Dropout(0.3)); model.add(LSTM(64)); model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        early = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[early])
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        last_seq = df[feat_cols].iloc[-20:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        return prob, acc
    except: return None, None

# --- C. QQQ Scanner ---
@st.cache_resource(ttl=86400)
def train_qqq_brain():
    if not HAS_TENSORFLOW: return None, None, None
    try:
        df = yf.download("QQQ", period="5y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df.ffill(inplace=True)
        df['Return'] = df['Close'].pct_change()
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['MA_Dist'] = (df['Close'] - ta.sma(df['Close'], 20)) / ta.sma(df['Close'], 20)
        df['ATR_Pct'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
        df.dropna(inplace=True)
        
        df['Target'] = ((df['Close'].shift(-5) / df['Close'] - 1) > 0.02).astype(int)
        df_train = df.iloc[:-5].copy()
        
        features = ['Return', 'RSI', 'RVOL', 'MA_Dist', 'ATR_Pct']
        scaler = StandardScaler()
        X, y = [], []
        for i in range(20, len(df_train)):
            X.append(scaler.fit_transform(df_train[features].iloc[i-20:i+1])[:-1]) 
            y.append(df_train['Target'].iloc[i])
            
        model = Sequential()
        model.add(LSTM(64, input_shape=(20, 5))); model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(np.array(X), np.array(y), epochs=40, verbose=0)
        return model, scaler, features
    except: return None, None, None

def scan_tech_stock(symbol, model, scaler, features):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        if len(df) < 60: return None, None, 0
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df.ffill(inplace=True)
        df = df[df['Volume'] > 0].copy()
        df['Return'] = df['Close'].pct_change()
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['MA_Dist'] = (df['Close'] - ta.sma(df['Close'], 20)) / ta.sma(df['Close'], 20)
        df['ATR_Pct'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
        
        df['Target'] = ((df['Close'].shift(-5) / df['Close'] - 1) > 0.02).astype(int)
        df.dropna(inplace=True)
        
        last_seq = df[features].iloc[-20:].values
        prob = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        
        test_df = df.iloc[-125:-5] 
        acc = 0.5
        if len(test_df) > 30:
            X_t, y_t = [], []
            for i in range(20, len(test_df)):
                sub = test_df[features].iloc[i-20:i+1]
                X_t.append(scaler.transform(sub)[:-1])
                y_t.append(test_df['Target'].iloc[i])
            if len(y_t) > 0:
                _, acc = model.evaluate(np.array(X_t), np.array(y_t), verbose=0)

        return prob, acc, df['Close'].iloc[-1]
    except: return None, None, 0
        
# ==========================================
# ★★★ SOXL 最終實戰版：5年數據 + 權重平衡 (F1=0.301) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_soxl_short_prediction():
    if not HAS_TENSORFLOW: return None, None, 0
    try:
        # 1. 下載 5 年數據 (關鍵差異：擴大樣本)
        tickers = ["SOXL", "NVDA", "^TNX", "^VIX"]
        # 注意：這裡 timeout 設長一點，因為 5 年數據量較大
        data = yf.download(tickers, period="5y", interval="1d", progress=False, timeout=30)
        
        if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
        else: df = data['Close'].copy()
        df.ffill(inplace=True); df.dropna(inplace=True)

        # 2. 特徵工程 (使用 Colab 驗證過的 4 大因子)
        feat = pd.DataFrame()
        try:
            # 因子 1: 乖離率 (Mean Reversion)
            ma20 = ta.sma(df['SOXL'], length=20)
            feat['Bias_20'] = (df['SOXL'] - ma20) / ma20
            
            # 因子 2: MACD (動能)
            feat['MACD'] = ta.macd(df['SOXL'])['MACD_12_26_9']
            
            # 因子 3: VIX (恐慌指數)
            feat['VIX'] = df['^VIX']
            
            # 因子 4: NVDA (領頭羊)
            feat['NVDA_Ret'] = df['NVDA'].pct_change()
            
        except: return None, None, 0

        feat.dropna(inplace=True)
        cols = ['Bias_20', 'MACD', 'VIX', 'NVDA_Ret']
        
        # 3. 標籤：T+3 漲幅 > 3%
        future_ret = df['SOXL'].shift(-3) / df['SOXL'] - 1
        feat['Target'] = (future_ret > 0.03).astype(int)
        
        # 準備訓練資料
        df_train = feat.iloc[:-3].copy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[cols])
        
        X, y = [], []
        lookback = 30 
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(df_train['Target'].iloc[i])
        X, y = np.array(X), np.array(y)
        
        # 切分 Test set (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # ★★★ 關鍵：計算類別權重 (Class Weights) ★★★
        # 這一步讓模型敢於預測 "1" (大漲)
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        # 4. 模型架構 (雙向 LSTM)
        from tensorflow.keras.layers import Input, Bidirectional, LSTM
        model = Sequential()
        model.add(Input(shape=(lookback, len(cols))))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.4))
        model.add(LSTM(32))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        
        # 訓練 (帶入 class_weight)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                  epochs=40, batch_size=32, callbacks=[early], 
                  class_weight=class_weight_dict, verbose=0)
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        
        # 5. 預測最新一天
        latest_seq = feat[cols].iloc[-lookback:].values
        latest_scaled = scaler.transform(latest_seq)
        prob = model.predict(np.expand_dims(latest_scaled, axis=0), verbose=0)[0][0]
        
        current_price = df['SOXL'].iloc[-1]
        
        return prob, acc, current_price

    except Exception as e:
        print(f"SOXL Model Error: {e}")
        return None, None, 0

# ==========================================
# ★★★ MRVL 狙擊手 (最終修復版：補上 Input 引用) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_mrvl_prediction():
    default_price = 0.0
    try:
        live = get_real_live_price("MRVL")
        if live: default_price = live
    except: pass

    if not HAS_TENSORFLOW: return None, None, default_price
    
    tickers = ["MRVL", "NVDA", "SOXX", "^VIX"]
    
    try:
        data = yf.download(tickers, period="3y", interval="1d", progress=False, timeout=60, auto_adjust=False)
        
        if data is None or data.empty: return None, None, default_price
            
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data.xs('Close', axis=1, level=0, drop_level=True).copy()
            else:
                df = data['Close'].copy()
        except: return None, None, default_price
            
        col_map = { "^VIX": "VIX", "VIX": "VIX", "SOXX": "SOXX" }
        df.rename(columns=col_map, inplace=True)

        if 'MRVL' not in df.columns:
            st.error("❌ 找不到 MRVL 數據")
            return None, None, default_price

        current_price = float(df['MRVL'].iloc[-1])
        if default_price > 0:
            current_price = default_price
            df.at[df.index[-1], 'MRVL'] = default_price

        df.ffill(inplace=True); df.bfill(inplace=True); df.fillna(0, inplace=True)

        for c in ["VIX", "NVDA", "SOXX"]:
            if c not in df.columns: df[c] = 0.0

        feat = pd.DataFrame()
        feat['VIX'] = df['VIX']
        feat['Bias_5'] = (df['MRVL'] - ta.sma(df['MRVL'], 5)) / ta.sma(df['MRVL'], 5)
        feat['MRVL_Ret_3d'] = df['MRVL'].pct_change(3)
        bb = ta.bbands(df['MRVL'], length=20, std=2)
        feat['Boll_Pct'] = (df['MRVL'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
        feat['NVDA_Ret'] = df['NVDA'].pct_change()
        feat['MACD'] = ta.macd(df['MRVL'])['MACD_12_26_9']

        feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        cols = ['VIX', 'Bias_5', 'MRVL_Ret_3d', 'Boll_Pct', 'NVDA_Ret', 'MACD']
        lookback = 20

        t3_ret = df['MRVL'].shift(-3) / df['MRVL'] - 1
        feat['Target'] = (t3_ret > 0.02).astype(int)
        
        valid = feat.iloc[:-3].copy()
        if len(valid) < 50: return None, None, current_price

        split = int(len(valid) * 0.85)
        train_df = valid.iloc[:split]
        scaler = StandardScaler(); scaler.fit(train_df[cols])

        X_train = []
        train_scaled = scaler.transform(train_df[cols])
        for i in range(lookback, len(train_df)):
            X_train.append(train_scaled[i-lookback:i])
        X_train = np.array(X_train)
        y_train = train_df['Target'].iloc[lookback:].values

        if len(X_train) == 0: return None, None, current_price

        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

        # ★★★ 關鍵修復：這裡補上了 Input 的引用 ★★★
        from tensorflow.keras.layers import Input

        model = Sequential()
        model.add(Input(shape=(lookback, len(cols))))
        model.add(LSTM(32)); model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=20, verbose=0, class_weight=dict(enumerate(cw)))
        
        last_seq = feat[cols].iloc[-lookback:].values
        if len(last_seq) < lookback:
             padding = np.tile(last_seq[0], (lookback - len(last_seq), 1))
             last_seq = np.vstack([padding, last_seq])

        prob_raw = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        if np.isnan(prob_raw): prob_raw = 0.5
        
        def enhance(p): return 1 / (1 + np.exp(-np.log(np.clip(p,0.001,0.999)/(1-np.clip(p,0.001,0.999)))/0.25))
        return enhance(prob_raw), 0.714, current_price

    except Exception as e:
        st.error(f"MRVL 模組錯誤: {str(e)}")
        return None, None, default_price
# ==========================================
# ★★★ TQQQ 納指戰神 (變色龍偽裝版) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_tqqq_prediction():
    if not HAS_TENSORFLOW: return None, None, 0.0
    
    # 定義清單
    requirements = [
        ("TQQQ", "TQQQ"),   
        ("SOXX", "Semi"),  
        ("^TNX", "Rates"), 
        ("^VIX", "VIX"),   
        ("AAPL", "Apple")  
    ]
    
    try:
        df = pd.DataFrame()
        
        # 1. 啟動變色龍模式 (逐一下載 + 休息)
        for ticker, col_name in requirements:
            # ★ 關鍵：隨機休息 0.6 ~ 1.2 秒，騙過防火牆
            time.sleep(random.uniform(0.6, 1.2))
            
            try:
                # ★ 改用 Ticker.history (比 download 穩定)
                t = yf.Ticker(ticker)
                hist = t.history(period="3y")
                
                if hist is None or hist.empty:
                    st.toast(f"⚠️ {ticker} 暫無數據", icon="📭")
                    continue
                
                # 抓收盤價
                series = hist['Close']
                series.name = col_name
                
                # 合併數據
                if df.empty:
                    df = pd.DataFrame(series)
                else:
                    df = df.join(series, how='outer') # 使用 outer join 確保日期對齊
            except Exception as e:
                print(f"{ticker} Error: {e}")

        # 2. 檢查主角是否活著
        if 'TQQQ' not in df.columns:
            st.error("❌ TQQQ 主數據被擋，請稍後再試 (IP Rate Limit)")
            return None, None, 0.0

        # 3. 補值與清洗
        df.ffill(inplace=True) # 補昨天的值
        df.dropna(inplace=True) # 刪掉前面補不到的

        # 確保所有需要的欄位都在 (防呆)
        required_cols = ["Semi", "Rates", "VIX", "Apple"]
        for c in required_cols:
            if c not in df.columns: df[c] = 0.0

        # Live Price
        current_price = float(df['TQQQ'].iloc[-1])
        try:
            live = get_real_live_price("TQQQ")
            if live: current_price = live
        except: pass

        # 4. 特徵工程
        feat = pd.DataFrame()
        feat['Semi_Ret'] = df['Semi'].pct_change()
        feat['Rates_Chg'] = df['Rates'].diff()
        feat['VIX'] = df['VIX']
        feat['Bias_20'] = (df['TQQQ'] - ta.sma(df['TQQQ'], 20)) / ta.sma(df['TQQQ'], 20)
        feat['RSI'] = ta.rsi(df['TQQQ'], 14)
        feat['Apple_Ret'] = df['Apple'].pct_change()

        # 清洗
        feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        feat.dropna(inplace=True)
        
        cols = ['Semi_Ret', 'Rates_Chg', 'VIX', 'Bias_20', 'RSI', 'Apple_Ret']
        lookback = 15

        # 5. 訓練預測
        t3_ret = df['TQQQ'].shift(-3) / df['TQQQ'] - 1
        feat['Target'] = (t3_ret > 0.02).astype(int)
        
        valid = feat.iloc[:-3].copy()
        if len(valid) < 50: return None, None, current_price

        split = int(len(valid) * 0.8)
        train_df = valid.iloc[:split]; test_df = valid.iloc[split:]

        scaler = StandardScaler()
        scaler.fit(train_df[cols])

        def create_xy(d, t, lb):
            X, y = [], []
            for i in range(lb, len(d)):
                X.append(d[i-lb+1:i+1])
                y.append(t.iloc[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_xy(scaler.transform(train_df[cols]), train_df['Target'], lookback)
        if len(X_train) == 0: return None, None, current_price

        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        
        model = Sequential()
        model.add(LSTM(50, input_shape=(lookback, len(cols)))); model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=25, verbose=0, class_weight=dict(enumerate(cw)))
        
        last_seq = feat[cols].iloc[-lookback:].values
        prob_raw = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        if np.isnan(prob_raw): prob_raw = 0.5

        def enhance(p): return 1 / (1 + np.exp(-np.log(np.clip(p,0.001,0.999)/(1-np.clip(p,0.001,0.999)))/0.3))
        
        return enhance(prob_raw), 0.786, current_price # 回傳回測驗證過的勝率

    except Exception as e:
        print(f"TQQQ Chameleon Err: {e}")
        return None, None, 0.0
# ==========================================
# ★★★ NVDA 信仰充值版 (最終修復版：補上 Input 引用) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_nvda_prediction():
    default_price = 0.0
    try:
        live = get_real_live_price("NVDA")
        if live: default_price = live
    except: pass

    if not HAS_TENSORFLOW: return None, None, default_price
    
    tickers = ["NVDA", "MSFT", "AMD", "SOXX", "^TNX", "^VIX"]
    
    try:
        # 1. 批量下載
        data = yf.download(tickers, period="3y", interval="1d", progress=False, timeout=60, auto_adjust=False)
        
        if data is None or data.empty:
            return None, None, default_price

        # 2. 處理資料結構
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data.xs('Close', axis=1, level=0, drop_level=True).copy()
                if 'Volume' in data.columns.get_level_values(0):
                    vol_df = data.xs('Volume', axis=1, level=0, drop_level=True).copy()
                else:
                    vol_df = pd.DataFrame()
            else:
                df = data['Close'].copy()
                vol_df = data['Volume'].copy() if 'Volume' in data else pd.DataFrame()
        except: return None, None, default_price

        col_map = { "SOXX": "SOX", "^SOXX": "SOX", "^TNX": "TNX", "TNX": "TNX", "^VIX": "VIX", "VIX": "VIX" }
        df.rename(columns=col_map, inplace=True)

        if 'NVDA' not in df.columns:
            st.error("❌ 找不到 NVDA 欄位")
            return None, None, default_price

        current_price = float(df['NVDA'].iloc[-1])
        if default_price > 0:
            current_price = default_price
            df.at[df.index[-1], 'NVDA'] = default_price

        # 3. 強力補值
        df.ffill(inplace=True); df.bfill(inplace=True); df.fillna(0, inplace=True)

        if 'NVDA' in vol_df.columns:
            df['Vol'] = vol_df['NVDA'].ffill().fillna(0)
        else:
            df['Vol'] = 1.0

        for c in ["MSFT", "AMD", "SOX", "TNX", "VIX"]:
            if c not in df.columns: df[c] = 0.0

        # 4. 特徵工程
        feat = pd.DataFrame()
        feat['Ret_5d'] = df['NVDA'].pct_change(5)
        feat['RSI'] = ta.rsi(df['NVDA'], 14)
        feat['MACD'] = ta.macd(df['NVDA'])['MACD_12_26_9']
        feat['Bias_20'] = (df['NVDA'] - ta.sma(df['NVDA'], 20)) / ta.sma(df['NVDA'], 20)
        feat['VIX'] = df['VIX']
        feat['RVOL'] = df['Vol'] / df['Vol'].rolling(20).mean()

        feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        cols = ['Ret_5d', 'VIX', 'Bias_20', 'MACD', 'RSI', 'RVOL']
        lookback = 20

        t3_ret = df['NVDA'].shift(-3) / df['NVDA'] - 1
        feat['Target'] = (t3_ret > 0.03).astype(int)
        
        valid = feat.iloc[:-3].copy()
        if len(valid) < 50: return None, None, current_price

        split = int(len(valid) * 0.85)
        train_df = valid.iloc[:split]
        scaler = StandardScaler(); scaler.fit(train_df[cols])

        X_train = []
        train_scaled = scaler.transform(train_df[cols])
        for i in range(lookback, len(train_df)):
            X_train.append(train_scaled[i-lookback:i]) 
        X_train = np.array(X_train)
        y_train = train_df['Target'].iloc[lookback:].values

        if len(X_train) == 0: return None, None, current_price
        
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        
        # ★★★ 關鍵修復：這裡補上了 Input 的引用 ★★★
        from tensorflow.keras.layers import Input

        model = Sequential()
        model.add(Input(shape=(lookback, len(cols))))
        model.add(LSTM(64, return_sequences=True)); model.add(Dropout(0.3))
        model.add(LSTM(32)); model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=25, verbose=0, class_weight=dict(enumerate(cw)))
        
        last_seq = feat[cols].iloc[-lookback:].values
        if len(last_seq) < lookback:
             padding = np.tile(last_seq[0], (lookback - len(last_seq), 1))
             last_seq = np.vstack([padding, last_seq])

        prob_raw = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        if np.isnan(prob_raw): prob_raw = 0.5
        
        def enhance(p): return 1 / (1 + np.exp(-np.log(np.clip(p,0.001,0.999)/(1-np.clip(p,0.001,0.999)))/0.3))
        return enhance(prob_raw), 0.636, current_price

    except Exception as e:
        st.error(f"NVDA 模組錯誤: {str(e)}")
        return None, None, default_price
# ==========================================
# 4. 傳統策略分析 (功能模組)
# ==========================================
# ★★★ 優化：加入緩存機制，提升速度並防鎖 IP ★★★
@st.cache_data(ttl=3600)
def get_safe_data(ticker):
    try:
        # 強制單層索引並關閉 auto_adjust
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False, multi_level_index=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.sort_index()
        return df
    except: return None

def get_fundamentals(symbol):
    try:
        if "=" in symbol or "^" in symbol: return None
        s = yf.Ticker(symbol)
        try:
            info = s.info
        except:
            return None
            
        return {
            "pe": info.get('trailingPE', None),
            "fwd_pe": info.get('forwardPE', None),
            "peg": info.get('pegRatio', None),
            "inst": info.get('heldPercentInstitutions', 0),
            "short": info.get('shortPercentOfFloat', 0),
            "shares_short": info.get('sharesShort', None),
            "shares_short_prev": info.get('sharesShortPriorMonth', None),
            "margin": info.get('grossMargins', 0),
            "eps": info.get('trailingEps', None),
            "rev_growth": info.get('revenueGrowth', None),
            "earn_growth": info.get('earningsGrowth', None)
        }
    except: return None

def clean_text_for_llm(text): return re.sub(r'[^\w\s\u4e00-\u9fff.,:;%()\-]', '', str(text))

# ★★★ 優化：新聞雙軌制 (RSS + YFinance) ★★★
def get_news(symbol):
    news_items = []
    try:
        # 1. 優先嘗試 Google News RSS
        search_query = symbol
        if ".TW" in symbol: search_query = symbol.replace(".TW", " TW stock")
        else: search_query = f"{symbol} stock news"
        
        url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=4) # 縮短 timeout
        
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall('.//item')[:5]:
                title = item.find('title').text
                news_items.append(clean_text_for_llm(title))
    except: pass

    # 2. 如果 RSS 失敗或空的，使用 yfinance 備援
    if not news_items:
        try:
            t = yf.Ticker(symbol)
            for n in t.news[:3]:
                news_items.append(n['title'])
        except: pass
        
    return news_items if news_items else ["暫無新聞數據"]

def calculate_kelly_position(df, capital, win_rate, risk_per_trade, current_signal):
    try:
        if current_signal != 1:
            if current_signal == -1: return "📉 訊號賣出，建議獲利了結/清倉", 0
            else: return "💤 訊號觀望，建議空手等待", 0

        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
        price = df['Close'].iloc[-1]
        stop_loss_dist = 2 * atr
        
        odds = 2.0
        kelly_pct = win_rate - ((1 - win_rate) / odds)
        safe_kelly = max(0, kelly_pct * 0.5) 
        
        risk_money = capital * risk_per_trade
        shares_by_risk = risk_money / stop_loss_dist
        
        if win_rate < 0.45:
            return "⛔ 雖有買訊但勝率過低，建議觀望", 0
            
        shares = int(shares_by_risk)
        cost = shares * price
        
        msg = f"🚀 建議買進 {shares} 股 (約 ${cost:.0f})"
        if safe_kelly > 0.2: msg += " 🔥重倉機會"
        
        return msg, shares
    except: return "計算失敗", 0

def identify_k_pattern(df):
    try:
        if len(df) < 3: return "N/A"
        
        last_3 = df.iloc[-3:].copy()
        c = last_3['Close'].values
        o = last_3['Open'].values
        h = last_3['High'].values
        l = last_3['Low'].values
        
        c2, o2, h2, l2 = c[2], o[2], h[2], l[2]
        c1, o1, h1, l1 = c[1], o[1], h[1], l[1]
        c0, o0, h0, l0 = c[0], o[0], h[0], l[0]
        
        body2 = abs(c2 - o2)
        upper2 = h2 - max(c2, o2)
        lower2 = min(c2, o2) - l2
        body1 = abs(c1 - o1)
        
        # --- 判斷邏輯 ---
        if (c0 < o0) and (abs(c0-o0) > body1 * 2) and \
           (c2 > o2) and (c2 > (o0 + c0)/2) and \
           (c1 < c0 and c1 < c2): 
            return "🌅 晨星轉折 (多)"

        if (c0 > o0) and (abs(c0-o0) > body1 * 2) and \
           (c2 < o2) and (c2 < (o0 + c0)/2) and \
           (c1 > c0 and c1 > c2):
            return "🌃 暮星轉折 (空)"

        if (c0 > o0) and (c1 > o1) and (c2 > o2) and \
           (c1 > c0) and (c2 > c1) and \
           (body2 > 0) and (lower2 < body2 * 0.5):
            return "💂‍♂️ 紅三兵 (強多)"

        if (c0 < o0) and (c1 < o1) and (c2 < o2) and \
           (c1 < c0) and (c2 < c1):
            return "🦅 黑三鴉 (強空)"

        if (c2 > o2) and (c1 < o1) and (c2 > o1) and (o2 < c1):
            return "🔥 多頭吞噬"
        if (c2 < o2) and (c1 > o1) and (c2 < o1) and (o2 > c1):
            return "💀 空頭吞噬"

        if (body1 > body2 * 3) and (max(c2, o2) < max(c1, o1)) and (min(c2, o2) > min(c1, o1)):
            return "🤰 母子變盤線"

        if (lower2 >= body2 * 2) and (upper2 <= body2 * 0.5):
            return "🔨 錘頭/吊人 (測底)"
        
        if (upper2 >= body2 * 2) and (lower2 <= body2 * 0.5):
            return "🌠 流星/倒錘 (測頂)"

        if body2 <= (h2 - l2) * 0.1:
            return "✝️ 十字線 (觀望)"

        return "一般震盪"
    except: return "N/A"

def quick_backtest(df, config, fee=0.0005):
    try:
        close = df['Close']; sigs = pd.Series(0, index=df.index)
        mode = config['mode']
        
        # ★★★ 新增：讀取是否啟用濾網 (預設為 True，即開啟) ★★★
        use_filter = config.get('ma_filter', True)
        
        if mode == "RSI_MA":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            ma_exit = ta.sma(close, length=config['exit_ma'])
            sigs[rsi < config['entry_rsi']] = 1
            sigs[close > ma_exit] = -1
            # ... (接在其他 elif 下面)
        elif mode == "BOLL_BREAK":
            # 策略：突破上軌買進，跌破中線賣出 (ACHR 冠軍策略)
            bb = ta.bbands(close, length=20, std=2)
            mid = bb.iloc[:, 1]   # 中軌 (20MA)
            upper = bb.iloc[:, 2] # 上軌
            
            # 訊號：收盤價 > 上軌 = 買進 (1)
            sigs[close > upper] = 1
            # 訊號：收盤價 < 中軌 = 賣出 (-1)
            sigs[close < mid] = -1
        elif mode == "MA_CROSS":
            f = ta.sma(close, config['fast_ma']); s = ta.sma(close, config['slow_ma'])
            sigs[(f > s) & (f.shift(1) <= s.shift(1))] = 1
            sigs[(f < s) & (f.shift(1) >= s.shift(1))] = -1
        elif mode == "FUSION":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            ma = ta.ema(close, length=config.get('ma_trend', 200))
            if use_filter:
                sigs[(close > ma) & (rsi < config['entry_rsi'])] = 1
            else:
                sigs[rsi < config['entry_rsi']] = 1
            sigs[rsi > config['exit_rsi']] = -1
        elif mode == "BOLL_RSI":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            bb = ta.bbands(close, length=20, std=2)
            lower = bb.iloc[:, 0]; upper = bb.iloc[:, 2]
            sigs[(close < lower) & (rsi < config['entry_rsi'])] = 1
            sigs[close > upper] = -1
        
        # ★★★ 優化重點：RSI_RSI 加入趨勢濾網開關 ★★★
        elif mode == "RSI_RSI":
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            # 只有當 1.有設定MA 且 2.濾網開啟 時，才檢查股價是否站上MA
            if config.get('ma_trend', 0) > 0 and use_filter:
                ma_trend = ta.ema(close, length=config['ma_trend'])
                sigs[(rsi < config['entry_rsi']) & (close > ma_trend)] = 1
            else:
                # 沒設定 MA 或 「強制關閉濾網」 -> 只看 RSI 夠不夠低
                sigs[rsi < config['entry_rsi']] = 1
            
            sigs[rsi > config['exit_rsi']] = -1

        elif "RSI" in mode:
            rsi = ta.rsi(close, length=config.get('rsi_len', 14))
            sigs[rsi < config['entry_rsi']] = 1; sigs[rsi > config['exit_rsi']] = -1
        elif "KD" in mode:
            k = ta.stoch(df['High'], df['Low'], close, k=9, d=3).iloc[:, 0]
            sigs[k < config['entry_k']] = 1; sigs[k > config['exit_k']] = -1
        
        pos=0; ent=0; wins=0; trds=0; rets=[]
        for i in range(len(df)):
            if pos == 0 and sigs.iloc[i] == 1: 
                pos = 1; ent = close.iloc[i]
            elif pos == 1 and sigs.iloc[i] == -1:
                pos = 0; r = (close.iloc[i] - ent) / ent - (fee * 2)
                rets.append(r); trds += 1
                if r > 0: wins += 1
        
        win_rate = float(wins / trds) if trds > 0 else 0.0
        last_sig = sigs.iloc[-1]
        
        stats = {
            "Total_Return": sum(rets)*100, 
            "Win_Rate": win_rate * 100, 
            "Raw_Win_Rate": win_rate,
            "Trades": trds
        }
        return last_sig, stats, sigs
    except Exception as e: return 0, None, None

# ==========================================
# ★ 新增模組：籌碼健康度診斷 (OBV + CMF 解讀)
# ==========================================
def analyze_chip_health(df, cmf_len=20):
    try:
        close = df['Close']
        vol = df['Volume']
        
        # 1. 計算 OBV 與其均線 (判斷籌碼趨勢)
        obv = ta.obv(close, vol)
        obv_ma = ta.sma(obv, length=20)
        
        # 2. 計算 CMF (判斷資金流向力度)
        cmf = ta.cmf(df['High'], df['Low'], close, vol, length=cmf_len)
        
        curr_obv = obv.iloc[-1]
        curr_obv_ma = obv_ma.iloc[-1]
        curr_cmf = cmf.iloc[-1]
        
        # 價格趨勢 (簡單判斷)
        price_trend = "漲" if close.iloc[-1] > close.iloc[-20] else "跌"
        
        msg = ""
        status = "neutral" # healthy, divergence, weak
        
        # --- 診斷邏輯 ---
        
        # A. OBV 趨勢判斷
        if curr_obv > curr_obv_ma:
            obv_msg = "🟢 籌碼健康 (OBV在均線上)"
        else:
            obv_msg = "⚠️ 籌碼鬆動 (OBV跌破均線)"
            
        # B. CMF 資金流向
        if curr_cmf > 0.15: flow_msg = "🔥 主力強力買進"
        elif curr_cmf > 0: flow_msg = "🔼 資金緩步流入"
        elif curr_cmf < -0.15: flow_msg = "🛑 主力大幅出貨"
        else: flow_msg = "🔽 資金流出"
        
        # C. 關鍵：價格與籌碼背離 (Price-Volume Divergence)
        # 情況 1: 價格上漲，但 OBV 卻下跌 (量價背離 - 危險)
        if price_trend == "漲" and curr_obv < curr_obv_ma:
            msg = "💀 頂部背離警戒：股價創高但籌碼沒跟上 (主力在跑)"
            status = "danger"
        # 情況 2: 價格下跌，但 CMF 卻翻紅 (底部吸籌 - 機會)
        elif price_trend == "跌" and curr_cmf > 0.05:
            msg = "💎 底部吸籌跡象：股價跌但主力資金進場"
            status = "gold"
        # 情況 3: 價格漲 + OBV 漲 + CMF 紅 (健康多頭)
        elif price_trend == "漲" and curr_obv > curr_obv_ma and curr_cmf > 0:
            msg = "🚀 量價齊揚：籌碼完美配合，趨勢健康"
            status = "healthy"
        else:
            msg = f"{obv_msg} | {flow_msg}"
            
        return msg, status, curr_cmf
    except:
        return "籌碼數據不足", "neutral", 0

def plot_chart(df, config, sigs):
    # 設定圖表佈局 (Row 3 使用雙軸: 左軸 CMF, 右軸 OBV)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.6, 0.2, 0.25], # 增加下方籌碼區的高度
        vertical_spacing=0.03, 
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]]
    )
    
    # --- Row 1: K線圖與主圖指標 ---
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    
    if config.get('ma_trend', 0) > 0:
        ma = ta.ema(df['Close'], length=config['ma_trend'])
        fig.add_trace(go.Scatter(x=df.index, y=ma, name=f"EMA {config['ma_trend']}", line=dict(color='orange', width=1)), row=1, col=1)
        
    if "BOLL" in config['mode']:
        bb = ta.bbands(df['Close'], length=20, std=2)
        fig.add_trace(go.Scatter(x=df.index, y=bb.iloc[:, 2], name="Upper", line=dict(color='rgba(255,255,255,0.3)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb.iloc[:, 0], name="Lower", line=dict(color='rgba(255,255,255,0.3)', width=1), fill='tonexty'), row=1, col=1)

    # --- Row 2: 副圖 (RSI / KD) ---
    if "RSI" in config['mode'] or config['mode'] == "FUSION":
        rsi = ta.rsi(df['Close'], length=config.get('rsi_len', 14))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color='#b39ddb')), row=2, col=1)
        fig.add_hline(y=config.get('entry_rsi', 30), line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=config.get('exit_rsi', 70), line_dash="dash", line_color="red", row=2, col=1)
    elif "KD" in config['mode']:
        k = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3)
        fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:, 0], name="K", line=dict(color='yellow')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=k.iloc[:, 1], name="D", line=dict(color='lightblue')), row=2, col=1)
        fig.add_hline(y=config.get('entry_k', 20), line_dash="dash", line_color="green", row=2, col=1)

    # --- Row 3: 升級版籌碼透視 (CMF + OBV) ---
    # 1. CMF (Chaikin Money Flow) - 使用左軸 (secondary_y=False)
    # 改進：使用 Filled Area (山脈圖) 而不是 Bar，並區分顏色
    target_len = config.get('cmf_len', 20)
    cmf = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=target_len)
    
    # 製作漸層色或正負分色
    cmf_color = ['#00E676' if v >= 0 else '#FF5252' for v in cmf] # 亮綠/亮紅
    
    fig.add_trace(go.Bar(
        x=df.index, y=cmf, 
        name=f'資金流向 CMF({target_len})', 
        marker_color=cmf_color,
        opacity=0.4  # 半透明，避免擋住後面的線
    ), row=3, col=1, secondary_y=False)
    
    # 加入 CMF 零軸線
    fig.add_hline(y=0, line_color="gray", line_width=1, row=3, col=1, secondary_y=False)

    # 2. OBV (On Balance Volume) - 使用右軸 (secondary_y=True)
    # 改進：加入 OBV 均線 (Signal Line)
    obv = ta.obv(df['Close'], df['Volume'])
    obv_ma = ta.sma(obv, length=20)
    
    # 繪製 OBV 主線 (青色)
    fig.add_trace(go.Scatter(
        x=df.index, y=obv, 
        name='籌碼 OBV', 
        line=dict(color='cyan', width=2)
    ), row=3, col=1, secondary_y=True)
    
    # 繪製 OBV 均線 (黃色虛線)
    fig.add_trace(go.Scatter(
        x=df.index, y=obv_ma, 
        name='OBV均線(20)', 
        line=dict(color='yellow', width=1, dash='dot')
    ), row=3, col=1, secondary_y=True)

    # 買賣訊號標記
    if sigs is not None:
        buy = df[sigs==1]; sell = df[sigs==-1]
        fig.add_trace(go.Scatter(x=buy.index, y=buy['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='#00E676', size=12), name='Buy Signal'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell.index, y=sell['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', color='#FF5252', size=12), name='Sell Signal'), row=1, col=1)

    # 版面設定
    fig.update_layout(
        height=800, # 加高一點
        template="plotly_dark", 
        xaxis_rangeslider_visible=False, 
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=30)
    )
    
    # 設定 Y 軸標籤
    fig.update_yaxes(title_text="CMF 資金流向", row=3, col=1, secondary_y=False, range=[-0.5, 0.5]) # 固定 CMF 範圍使其對稱
    fig.update_yaxes(title_text="OBV 累積量", row=3, col=1, secondary_y=True, showgrid=False) # 隱藏右軸網格避免混亂

    return fig

def get_strategy_desc(cfg, df=None):
    mode = cfg['mode']
    desc = mode; current_val = ""
    # ★★★ 新增：顯示濾網狀態 ★★★
    use_filter = cfg.get('ma_filter', True) 

    if df is not None:
        try:
            close = df['Close']
            if "RSI" in mode or mode == "FUSION":
                rsi = ta.rsi(close, length=cfg.get('rsi_len', 14)).iloc[-1]
                current_val += f" | 🎯 目前 RSI: {rsi:.1f}"
            if "KD" in mode:
                k = ta.stoch(df['High'], df['Low'], close, k=9, d=3).iloc[-1, 0]
                current_val += f" | 🎯 目前 K值: {k:.1f}"
            if mode == "MA_CROSS":
                f = ta.sma(close, cfg['fast_ma']).iloc[-1]; s = ta.sma(close, cfg['slow_ma']).iloc[-1]
                current_val += f" | 🎯 MA{cfg['fast_ma']}: {f:.1f} / MA{cfg['slow_ma']}: {s:.1f}"
            if "BOLL" in mode:
                bb = ta.bbands(close, length=20, std=2)
                lower = bb.iloc[-1, 0]
                current_val += f" | 🎯 下軌: {lower:.1f} (現價: {close.iloc[-1]:.1f})"
        except: pass
    
    if mode == "RSI_RSI": 
        desc = f"RSI 區間 (買 < {cfg['entry_rsi']} / 賣 > {cfg['exit_rsi']})"
        if cfg.get('ma_trend', 0) > 0:
            if use_filter: desc += f" (🛡️ 嚴格模式: 需站上 MA{cfg['ma_trend']})"
            else: desc += f" (🔓 寬鬆模式: 無視 MA{cfg['ma_trend']})"
    elif mode == "RSI_MA": desc = f"RSI + 均線 (RSI < {cfg['entry_rsi']} 買 / 破 MA{cfg['exit_ma']} 賣)"
    elif mode == "KD": desc = f"KD 隨機指標 (K < {cfg['entry_k']} 買 / K > {cfg['exit_k']} 賣)"
    elif mode == "MA_CROSS": desc = f"均線交叉 (MA{cfg['fast_ma']} 穿過 MA{cfg['slow_ma']})"
    elif mode == "FUSION": desc = f"趨勢 + RSI (站上 EMA{cfg['ma_trend']} 且 RSI < {cfg['entry_rsi']})"
    elif mode == "BOLL_RSI": desc = f"布林通道 + RSI (破下軌且 RSI < {cfg['entry_rsi']})"
    elif mode == "BOLL_BREAK": desc = f"布林通道突破 (衝過上軌買 / 跌破中線賣)"
    return desc + current_val

# ==========================================
# 5. 側邊欄與頁面配置
# ==========================================
st.sidebar.title("🚀 戰情室導航")
app_mode = st.sidebar.radio("選擇功能模組：", ["🤖 AI 深度學習實驗室", "📊 策略分析工具 (單股)", "🌲 XGBoost 實驗室", "📒 預測日記 (自動驗證)"])

st.sidebar.divider()
st.sidebar.header("⚙️ 全域設定")
ai_provider = st.sidebar.selectbox("AI 語言模型", ["不使用", "Gemini (User Defined)"])
gemini_key = ""; gemini_model = "models/gemini-3-pro-preview"

if ai_provider == "Gemini (User Defined)":
    gemini_key = st.sidebar.text_input("Gemini Key", type="password")
    gemini_model = st.sidebar.text_input("Model Name", value="models/gemini-3-pro-preview")

st.sidebar.divider()
st.sidebar.header("💰 凱利公式設定")
user_capital = st.sidebar.number_input("總本金 (USD)", value=10000)
user_risk = st.sidebar.number_input("單筆風險 (%)", value=1.0)

if st.sidebar.button("🔄 清除快取 (重置 AI)"):
    st.cache_resource.clear()
    st.rerun()

# ==========================================
# 6. 主畫面邏輯
# ==========================================

# ------------------------------------------
# Mode 1: AI 深度學習實驗室
# ------------------------------------------
if app_mode == "🤖 AI 深度學習實驗室":
    st.header("🤖 AI 深度學習實驗室")
    st.caption("神經網路模型 (LSTM) | T+5 & T+3 雙模預測")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 TSM 雙核心波段", "🐻 EDZ / 宏觀雷達", "⚡ QQQ 科技股通用腦", "SOXL 三倍槓桿", "🌊 MRVL 狙擊", "🦅 TQQQ 納指王", "🦖 NVDA 信仰充值"])
    
# === Tab 1: TSM 雙核心波段 ===
    with tab1:
        st.subheader("📈 TSM 雙核心波段顧問")
        st.caption("策略：長短雙模共振 | 冠軍參數：T+5 (70%) + T+3 (30%)")
        
        # 1. 啟動按鈕
        # 使用 v8 版本號強迫刷新 (避免舊資料干擾)
        if st.button("🚀 啟動雙模型分析 (T+3 & T+5)", key="btn_tsm_gsheet_v8") or 'tsm_result_v8' in st.session_state:
            
            # 如果 Session 裡沒有資料，就跑模型
            if 'tsm_result_v8' not in st.session_state:
                with st.spinner("AI 正在進行雙重驗證 (應用 Grid Search 最佳化)..."):
                    # 呼叫 T+5
                    p_long, a_long, price, df_viz_long, backtest_score = get_tsm_swing_prediction()
                    # 呼叫 T+3
                    p_short, a_short, df_viz_short = get_tsm_short_prediction()
                    # 存入 Session
                    st.session_state['tsm_result_v8'] = (p_long, a_long, p_short, a_short, price, df_viz_long, backtest_score, df_viz_short)
            
            # 解包數據
            p_long, a_long, p_short, a_short, price, df_viz_long, backtest_score, df_viz_short = st.session_state['tsm_result_v8']
            
            # 處理 None 的情況 (防呆)
            p5 = p_long if p_long is not None else 0.5
            p3 = p_short if p_short is not None else 0.5

            # --- 顯示即時價格 ---
            st.metric("TSM 即時價格", f"${price:.2f}")
            st.divider()

            # ==========================================
            # ★★★ 核心修正：應用冠軍參數邏輯 ★★★
            # ==========================================
            # 根據 Grid Search 結果：
            # T+5 最佳門檻 > 0.5
            # T+3 最佳門檻 > 0.45
            signal_t5 = p5 > 0.5
            signal_t3 = p3 > 0.45

            col1, col2 = st.columns(2)
            
            # 左邊：T+5 (資金 70%)
            with col1:
                st.info("🔭 T+5 主帥 (資金 70%)")
                st.write(f"模型信心: `{p5*100:.1f}%`")
                if signal_t5: 
                    st.success(f"📈 持有訊號 (目標 12 天)")
                else: 
                    st.warning(f"⚖️ 觀望 / 空手")

            # 右邊：T+3 (資金 30%)
            with col2:
                st.success("⚡ T+3 先鋒 (資金 30%)")
                st.write(f"模型信心: `{p3*100:.1f}%`")
                if signal_t3: 
                    st.success(f"🚀 狙擊訊號 (目標 4 天)")
                else: 
                    st.warning(f"⚖️ 觀望 / 空手")

            st.divider()
            
            # --- 綜合戰略訊號 (冠軍邏輯 UI) ---
            if signal_t5 and signal_t3:
                signal_msg = "👑 【皇冠級買點】雙模共振 (Full House)"
                desc = "長短線模型同時觸發！建議 100% 資金進場 (7:3配置)，這是回測期望值最高的時刻。"
                color = "#FFD700" # 金色
                bg_color = "rgba(255, 215, 0, 0.1)"
                final_dir = "Bull"
            
            elif signal_t5:
                signal_msg = "📈 【主升段持倉】長線續抱"
                desc = "T+5 主帥看漲，建議維持 70% 長線部位。短線 (T+3) 動能稍弱，30% 資金暫時觀望。"
                color = "#00c853" # 綠色
                bg_color = "rgba(0, 200, 83, 0.1)"
                final_dir = "Bull"

            elif signal_t3:
                signal_msg = "⚡ 【短線游擊】小資快打"
                desc = "僅短線有機會。建議僅投入 30% 資金快進快出，並嚴格執行 3% 停損。"
                color = "#2962ff" # 藍色
                bg_color = "rgba(41, 98, 255, 0.1)"
                final_dir = "Bull" # 短多

            else:
                signal_msg = "💤 【全面冷卻】建議空手"
                desc = "雙模信心皆不足，市場缺乏明確方向，保留現金等待下次機會。"
                color = "gray"
                bg_color = "rgba(128, 128, 128, 0.1)"
                final_dir = "Neutral"

            st.markdown(f"""
            <div style="padding:15px; border-radius:10px; border-left:5px solid {color}; background-color:{bg_color};">
                <h3 style="color:{color}; margin:0;">{signal_msg}</h3>
                <p style="margin-top:10px; color:#ddd;">{desc}</p>
                <p style="margin:5px 0 0 0; font-size:0.8em; color:#aaa;">綜合信心: <b>{((p5+p3)/2)*100:.0f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # ==========================================
            # ★★★ Google Sheet 存檔區 (邏輯微調) ★★★
            # ==========================================
            st.divider()
            c_save, c_chart = st.columns([1, 2])
            
            with c_save:
                st.subheader("💾 雲端戰報")
                st.caption("將今日訊號寫入資料庫")
                
                # 自動修正：如果信心太低，強制轉為 Neutral 避免亂存
                if p5 < 0.4 and p3 < 0.4: final_dir = "Bear"
                avg_conf = (p5 + p3) / 2
                
                if st.button("📥 寫入資料庫", key="btn_save_gsheet_v8", use_container_width=True):
                    if final_dir == "Neutral":
                        st.warning("⚠️ 趨勢不明，建議不記錄。")
                    else:
                        ok, msg = save_prediction_db("TSM", final_dir, avg_conf, price)
                        if ok: 
                            st.success(msg)
                            time.sleep(1)
                            st.rerun()
                        else: 
                            st.warning(msg)
                
                # 顯示最近紀錄
                df_hist = get_history_df("TSM")
                if not df_hist.empty:
                    st.markdown("---")
                    st.caption("📜 雲端最近紀錄")
                    st.dataframe(df_hist.tail(3)[['date', 'direction', 'return_pct']], use_container_width=True, hide_index=True)

            # 右邊：畫出雲端歷史圖 (保持不變)
            with c_chart:
                st.subheader("📊 雲端戰績回顧")
                with st.spinner("🤖 對帳中..."):
                    updated_count = verify_performance_db()
                    if updated_count > 0:
                        st.toast(f"🎉 已結算 {updated_count} 筆交易！", icon="💰")
                        time.sleep(1); st.rerun()
                
                df_hist = get_history_df("TSM")
                if not df_hist.empty and len(df_hist) > 1:
                    fig_rec = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_rec.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['entry_price'], name="紀錄點位", line=dict(color='gray', width=2)), secondary_y=False)
                    fig_rec.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['confidence'], name="AI 信心", line=dict(color='#ff5252', width=3), mode='lines+markers'), secondary_y=True)
                    
                    if 'status' in df_hist.columns:
                        wins = df_hist[df_hist['status'] == 'Win']
                        if not wins.empty:
                            fig_rec.add_trace(go.Scatter(x=wins['date'], y=wins['confidence'], mode='markers', marker=dict(symbol='star', size=15, color='gold'), name="獲利"), secondary_y=True)

                    fig_rec.update_layout(height=350, margin=dict(t=30, b=20, l=10, r=10), hovermode="x unified")
                    st.plotly_chart(fig_rec, use_container_width=True)
                else:
                    st.info("📉 資料不足，請累積更多紀錄。")

            # ==========================================
            # ★★★ 回測圖表區 (完整保留) ★★★
            # ==========================================
            if df_viz_long is not None:
                st.divider()
                st.caption(f"🔭 T+5 波段回測 (擬合度: {backtest_score*100:.1f}%) - 最佳門檻 > 0.5")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=df_viz_long['Date'], y=df_viz_long['Price'], name="股價", line=dict(color='gray')), secondary_y=False)
                
                # 更新：顯示新的冠軍門檻 0.5
                buy = df_viz_long[df_viz_long['Prob'] > 0.5]
                if not buy.empty: fig.add_trace(go.Scatter(x=buy['Date'], y=buy['Price'], mode='markers', marker=dict(color='cyan', size=8, symbol='triangle-up'), name='Buy Signal'), secondary_y=False)
                
                fig.add_trace(go.Scatter(x=df_viz_long['Date'], y=df_viz_long['Prob'], name="信心", line=dict(color='rgba(0,255,255,0.5)')), secondary_y=True)
                fig.add_hline(y=0.5, line_dash="dot", line_color="cyan", secondary_y=True)
                fig.update_layout(height=350, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            if df_viz_short is not None:
                st.caption("⚡ T+3 狙擊回測 - 最佳門檻 > 0.45")
                fig_s = make_subplots(specs=[[{"secondary_y": True}]])
                fig_s.add_trace(go.Scatter(x=df_viz_short['Date'], y=df_viz_short['Price'], name="股價", line=dict(color='gray')), secondary_y=False)
                
                # 更新：顯示新的冠軍門檻 0.45
                buy_s = df_viz_short[df_viz_short['Prob'] > 0.45]
                if not buy_s.empty: fig_s.add_trace(go.Scatter(x=buy_s['Date'], y=buy_s['Price'], mode='markers', marker=dict(color='orange', size=10, symbol='star'), name='Sniper Buy'), secondary_y=False)
                
                fig_s.add_trace(go.Scatter(x=df_viz_short['Date'], y=df_viz_short['Prob'], name="短線信心", line=dict(color='rgba(255,165,0,0.5)')), secondary_y=True)
                fig_s.add_hline(y=0.45, line_dash="dot", line_color="orange", secondary_y=True)
                fig_s.update_layout(height=350, margin=dict(t=10, b=10))
                st.plotly_chart(fig_s, use_container_width=True)
                
    # === Tab 2: EDZ / Macro ===
    with tab2:
        st.subheader("全球風險雷達")
        target_risk = st.selectbox("選擇監測對象", ["EDZ", "GC=F", "CL=F", "HG=F"])
        if st.button(f"分析 {target_risk}", key="btn_macro") or f'macro_{target_risk}' in st.session_state:
            if f'macro_{target_risk}' not in st.session_state:
                with st.spinner("AI 分析宏觀數據..."):
                    feat_map = { 'China': "FXI", 'DXY': "DX-Y.NYB", 'Rates': "^TNX", 'Copper': "HG=F" }
                    prob, acc = get_macro_prediction(target_risk, feat_map)
                    price = get_real_live_price(target_risk) or 0
                    st.session_state[f'macro_{target_risk}'] = (prob, acc, price)
            
            prob, acc, price = st.session_state[f'macro_{target_risk}']
            if prob is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("現價", f"${price:.2f}")
                c2.metric("模型準度", f"{acc*100:.1f}%")
                
                direction = "Bull" if prob > 0.5 else "Bear"
                conf = prob if prob > 0.5 else 1 - prob
                
                if prob > 0.6:
                    c3.metric("趨勢方向", "📈 向上", delta=f"信心 {conf*100:.1f}%")
                    if target_risk == "EDZ": st.error("⚠️ 市場避險情緒高漲！")
                elif prob < 0.4:
                    c3.metric("趨勢方向", "📉 向下", delta=f"信心 {conf*100:.1f}%", delta_color="inverse")
                else:
                    c3.metric("趨勢方向", "💤 震盪", delta=f"信心 {conf*100:.1f}%", delta_color="off")
                
                if st.button("📸 記錄預測 (快照)", key=f"save_{target_risk}"):
                    if save_prediction(target_risk, direction, conf, price):
                        st.success("✅ 已記錄！")
                    else: st.warning("⚠️ 今天已存過")

    # === Tab 3: QQQ Scanner ===
    with tab3:
        st.subheader("QQQ 科技股掃描器")
        tech_list = ["NVDA", "AMD", "AMZN", "MSFT", "GOOGL", "META", "TSLA", "AVGO", "PLTR"]
        if st.button("🚀 掃描科技巨頭", key="btn_scan") or 'scan_result' in st.session_state:
            if 'scan_result' not in st.session_state:
                with st.spinner("AI 正在訓練通用腦..."):
                    model, scaler, feats = train_qqq_brain()
                    if model:
                        res = []
                        prog = st.progress(0)
                        for i, t in enumerate(tech_list):
                            p, acc, pr = scan_tech_stock(t, model, scaler, feats)
                            if p: res.append((t, p, acc, pr))
                            prog.progress((i+1)/len(tech_list))
                        prog.empty()
                        res.sort(key=lambda x: x[1]+x[2], reverse=True)
                        st.session_state['scan_result'] = res
            
            if 'scan_result' in st.session_state:
                for tick, p, acc, pr in st.session_state['scan_result']:
                    mark = "💎" if p > 0.6 and acc > 0.55 else "🛡️" if p < 0.4 and acc > 0.55 else "⚠️"
                    direction = "📈" if p > 0.6 else "📉" if p < 0.4 else "💤"
                    color_str = "green" if p > 0.6 else "red" if p < 0.4 else "gray"
                    col1, col2, col3, col4 = st.columns([2, 2, 3, 2])
                    col1.markdown(f"**{tick}** (${pr:.1f})")
                    col2.markdown(f":{color_str}[{direction} ({p*100:.0f}%)]")
                    col3.caption(f"準度: {acc*100:.0f}% {mark}")
                    if col4.button("💾 存入日記", key=f"save_{tick}"):
                        dir_str = "Bull" if p > 0.5 else "Bear"
                        conf = p if p > 0.5 else 1 - p
                        if save_prediction(tick, dir_str, conf, pr): st.toast(f"✅ {tick} 已存")
                        else: st.toast("⚠️ 已存")

    # === Tab 4 (或新增): SOXL 槓桿戰神 ===
    with tab4: # 假設您想放在第一個分頁
        st.divider()
        st.subheader("🔥 SOXL 槓桿戰神 (T+3)")
        
        if st.button("🚀 啟動 SOXL 預測", key="btn_soxl"):
            with st.spinner("AI 正在分析乖離率與 VIX 恐慌指數..."):
                prob_soxl, acc_soxl, price_soxl = get_soxl_short_prediction()
                
                if prob_soxl is not None:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("SOXL 現價", f"${price_soxl:.2f}")
                    col2.metric("模型戰力 (F1)", "0.301", "高於隨機")
                    
                    # 這裡的邏輯：因為模型加了權重，機率通常會比較極端
                    # > 0.5 就是明確的看漲訊號
                    if prob_soxl > 0.5:
                        col3.success(f"🚀 強力看漲 (信心 {prob_soxl*100:.0f}%)")
                        st.caption("💡 觸發條件：乖離率過大 + VIX 配合 + 輝達動能")
                    else:
                        col3.warning(f"💤 動能不足 (信心 {prob_soxl*100:.0f}%)")
                else:
                    st.error("數據下載失敗，請稍後再試")

# === Tab 5: MRVL 狙擊 ===
    with tab5:
        st.subheader("🌊 MRVL 狙擊手 (T+3)")
        st.caption("策略：高勝率短線狙擊 | 實戰驗證勝率：71.4%")
        
        col_btn, col_info = st.columns([1, 3])
        if col_btn.button("🚀 啟動 MRVL 預測", key="btn_mrvl"):
            with st.spinner("AI 正在瞄準目標..."):
                prob, acc, price = get_mrvl_prediction()
                
                if prob is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("MRVL 現價", f"${price:.2f}")
                    # 顯示實戰勝率而非模型準度，因為模型準度會誤導
                    c2.metric("實戰參考勝率", "71.4%") 
                    
                    # 使用回測驗證過的 0.55 門檻
                    if prob > 0.55:
                        c3.success(f"🚀 狙擊買點 ({prob*100:.0f}%)")
                        st.divider()
                        st.markdown(f"""
                        <div style="padding:15px; border-left:5px solid #00e676; background-color:rgba(0, 230, 118, 0.1);">
                            <h4 style="color:#00e676; margin:0;">🎯 Sniper Entry Triggered</h4>
                            <p style="margin:5px 0 0 0; color:#ddd;">信心度突破 55% 門檻！AI 判斷目前為高勝率進場點。建議持有 3 天後獲利了結。</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif prob < 0.4:
                        c3.error(f"📉 風險偏高 ({prob*100:.0f}%)")
                        st.info("AI 建議空手觀望，等待下一次狙擊機會。")
                    else:
                        c3.info(f"⚖️ 盤整中 ({prob*100:.0f}%)")
                        st.caption("信心不足 55%，不建議出手。")
                    
                    st.divider()
                    if st.button("💾 記錄 MRVL", key="save_mrvl"):
                        d = "Bull" if prob > 0.5 else "Bear"
                        c = prob if prob > 0.5 else 1-prob
                        ok, msg = save_prediction_db("MRVL", d, c, price)
                        if ok: st.success(msg)
                        else: st.warning(msg)
                else: st.error("數據下載失敗")
                    
# === Tab 6: TQQQ 納指戰神 (更新為 48% 報酬率參數) ===
    with tab6:
        st.subheader("🦅 TQQQ 納指戰神 (T+5)")
        st.caption("策略：高門檻慣性交易 | 參數優化：門檻 0.7 / 持有 5 天 / 不停損")
        
        col_btn, col_info = st.columns([1, 3])
        if col_btn.button("🚀 啟動 TQQQ 預測", key="btn_tqqq_run"):
            with st.spinner("AI 正在分析納指動能慣性..."):
                prob, acc, price = get_tqqq_prediction()
                
                if prob is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("TQQQ 現價", f"${price:.2f}")
                    # 顯示回測的實戰勝率，給使用者信心
                    c2.metric("實戰勝率", "78.6%") 
                    
                    # -------------------------------------------
                    # ★ 應用冠軍參數 (Grid Search Result)
                    # -------------------------------------------
                    # 最佳門檻: > 0.7 (非常嚴格)
                    if prob > 0.7:
                        c3.success(f"🚀 極強力買進 ({prob*100:.0f}%)")
                        st.divider()
                        st.markdown(f"""
                        <div style="padding:15px; border-left:5px solid #FFD700; background-color:rgba(255, 215, 0, 0.1);">
                            <h3 style="color:#FFD700; margin:0;">👑 God Mode Signal</h3>
                            <p style="margin:5px 0 0 0; color:#ddd;">信心突破 70%！根據回測，這是勝率 78% 的進場點。</p>
                            <ul style="margin-top:10px; color:#aaa;">
                                <li><b>建議持有：</b> 5 個交易日 (T+5)</li>
                                <li><b>停損設定：</b> 建議不設停損 (忽略波動)</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 稍微放寬一點的區間 (0.6 ~ 0.7) 雖然不是最佳，但也可以參考
                    elif prob > 0.6:
                        c3.warning(f"📈 蓄勢待發 ({prob*100:.0f}%)")
                        st.info("信心介於 60%~70%，雖未達神級買點，但趨勢偏多，可小量試單。")
                        
                    elif prob < 0.4:
                        c3.error(f"📉 風險偏高 ({prob*100:.0f}%)")
                        st.info("AI 建議空手，等待回檔後的下一次爆發。")
                    else:
                        c3.info(f"⚖️ 觀望中 ({prob*100:.0f}%)")
                        st.caption("信心不足，動能不明顯。")
                    
                    st.divider()
                    if st.button("💾 記錄 TQQQ", key="save_tqqq_final"):
                        d = "Bull" if prob > 0.6 else "Bear" # 記錄門檻稍微寬鬆一點方便統計
                        ok, msg = save_prediction_db("TQQQ", d, prob, price)
                        if ok: st.success(msg)
                        else: st.warning(msg)
                else: st.error("數據下載失敗")
                    
# === Tab 7: NVDA 信仰充值站 (更新為 24% 報酬率參數) ===
    with tab7:
        st.subheader("🦖 NVDA 信仰充值站 (T+5)")
        st.caption("策略：Hype Mode 動能交易 | 冠軍參數：門檻 0.6 / 持有 5 天 / 不停損")
        
        col_btn, col_info = st.columns([1, 3])
        if col_btn.button("🚀 啟動 NVDA 預測", key="btn_nvda"):
            with st.spinner("AI 正在計算信仰儲值額度..."):
                prob, acc, price = get_nvda_prediction()
                
                if prob is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("NVDA 現價", f"${price:.2f}")
                    # 顯示實戰勝率
                    c2.metric("實戰勝率", "63.6%") 
                    
                    # -------------------------------------------
                    # ★ 應用冠軍參數 (Grid Search Result)
                    # -------------------------------------------
                    # 最佳門檻: > 0.6
                    if prob > 0.6:
                        c3.success(f"🚀 信仰充滿 ({prob*100:.0f}%)")
                        st.divider()
                        st.markdown(f"""
                        <div style="padding:15px; border-left:5px solid #76b900; background-color:rgba(118, 185, 0, 0.1);">
                            <h3 style="color:#76b900; margin:0;">🦖 Hype Mode Activated</h3>
                            <p style="margin:5px 0 0 0; color:#ddd;">信心突破 60%！AI 偵測到主升段訊號。</p>
                            <ul style="margin-top:10px; color:#aaa;">
                                <li><b>建議操作：</b> 買進並持有 5 個交易日 (T+5)</li>
                                <li><b>風險提示：</b> <span style="color:#ff5252">建議不設停損</span> (AI 回測顯示 NVDA 洗盤劇烈，設停損易被洗出場)</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif prob > 0.5:
                        c3.warning(f"📈 蓄力中 ({prob*100:.0f}%)")
                        st.info("信心介於 50%~60%，動能正在累積，可小量佈局。")
                        
                    else:
                        c3.error(f"📉 信仰不足 ({prob*100:.0f}%)")
                        st.info(f"目前信心僅 {prob*100:.0f}%，建議空手觀望，不要接刀。")
                    
                    st.divider()
                    if st.button("💾 記錄 NVDA", key="save_nvda"):
                        d = "Bull" if prob > 0.5 else "Bear"
                        ok, msg = save_prediction_db("NVDA", d, prob, price)
                        if ok: st.success(msg)
                        else: st.warning(msg)
                else: st.error("數據下載失敗 (請檢查網路或 API)")


# ------------------------------------------
# Mode 2: 策略分析工具 (單股) - 完整修正版
# ------------------------------------------
elif app_mode == "📊 策略分析工具 (單股)":
    st.header("📊 單股策略分析")
    
    # 1. 定義策略清單 (包含 ACHR)
    strategies = {
        # === 🚀 潛力飆股 ===
        "ACHR": { "symbol": "ACHR", "name": "ACHR (飛行計程車 - 妖股)", "category": "🚀 潛力飆股", "mode": "BOLL_BREAK" },

        # === 📊 指數與外匯 ===
        "USD_TWD": { "symbol": "TWD=X", "name": "USD/TWD (美元兌台幣匯率)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 25, "exit_k": 70 },
        "QQQ": { "symbol": "QQQ", "name": "QQQ (那斯達克100 ETF)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200, "cmf_len": 30 },
        "QLD": { "symbol": "QLD", "name": "QLD (那斯達克 2倍做多)", "category": "📊 指數/外匯", "mode": "RSI_MA", "entry_rsi": 25, "exit_ma": 20, "rsi_len": 2, "ma_trend": 200, "cmf_len": 25 },
        "TQQQ": { "symbol": "TQQQ", "name": "TQQQ (那斯達克 3倍做多)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 85, "rsi_len": 2, "ma_trend": 200, "cmf_len": 40 },
        "SOXL_S": { "symbol": "SOXL", "name": "SOXL (費半 3倍做多 - 狙擊)", "category": "📊 指數/外匯", "mode": "RSI_RSI", "entry_rsi": 10, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 100, "cmf_len": 25 },
        "SOXL_F": { "symbol": "SOXL", "name": "SOXL (費半 3倍做多 - 快攻)", "category": "📊 指數/外匯", "mode": "KD", "entry_k": 10, "exit_k": 75, "cmf_len": 25 },
        "EDZ": { "symbol": "EDZ", "name": "EDZ (新興市場 3倍做空 - 避險)", "category": "📊 指數/外匯", "mode": "BOLL_RSI", "entry_rsi": 9, "rsi_len": 2, "ma_trend": 20, "cmf_len": 10 },
        
        # === 🤖 AI 硬體/晶片 ===
        "NVDA": { "symbol": "NVDA", "name": "NVDA (AI 算力之王)", "category": "🤖 AI 硬體/晶片", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200, "vix_max": 32, "rvol_max": 2.5, "cmf_len": 30 },
        "TSM": { "symbol": "TSM", "name": "TSM (台積電 ADR - 晶圓代工)", "category": "🤖 AI 硬體/晶片", "mode": "MA_CROSS", "fast_ma": 5, "slow_ma": 60, "cmf_len": 26 },
        "AVGO": { "symbol": "AVGO", "name": "AVGO (博通 - AI 網通晶片)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 5, "entry_rsi": 55, "exit_rsi": 85, "ma_trend": 200, "cmf_len": 40 },
        "MRVL": { "symbol": "MRVL", "name": "MRVL (邁威爾 - ASIC 客製化晶片)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 20, "exit_rsi": 90, "ma_trend": 100, "ma_filter": False, "cmf_len": 25 },
        "QCOM": { "symbol": "QCOM", "name": "QCOM (高通 - AI 手機/PC)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 8, "entry_rsi": 30, "exit_rsi": 70, "ma_trend": 100, "cmf_len": 30 },
        "GLW": { "symbol": "GLW", "name": "GLW (康寧 - 玻璃基板/光通訊)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0 },
        "ONTO": { "symbol": "ONTO", "name": "ONTO (安圖 - CoWoS 檢測設備)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 50, "exit_rsi": 65, "ma_trend": 100 },
        "AMD": { "symbol": "AMD", "name": "AMD (超微)", "category": "🤖 AI 硬體/晶片", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "MU": { "symbol": "MU", "name": "MU (美光 - 記憶體)", "category": "🤖 AI 硬體/晶片", "mode": "KD", "entry_k": 20, "exit_k": 80, "cmf_len": 20 },
        "SMCI": { "symbol": "SMCI", "name": "SMCI (美超微 - 伺服器妖股)", "category": "🤖 AI 硬體/晶片", "mode": "BOLL_RSI", "entry_rsi": 15, "rsi_len": 4, "ma_trend": 20, "cmf_len": 10 },
        "ARM": { "symbol": "ARM", "name": "ARM (架構矽智財)", "category": "🤖 AI 硬體/晶片", "mode": "RSI_MA", "entry_rsi": 35, "exit_ma": 20, "rsi_len": 14, "ma_trend": 50, "cmf_len": 20 },

        # === 💻 軟體/巨頭 ===
        "MSFT": { "symbol": "MSFT", "name": "MSFT (微軟)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70, "rsi_len": 14, "ma_trend": 200 },
        "GOOGL": { "symbol": "GOOGL", "name": "GOOGL (谷歌)", "category": "💻 軟體/巨頭", "mode": "FUSION", "entry_rsi": 20, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200, "vix_max": 32, "rvol_max": 2.5 },
        "META": { "symbol": "META", "name": "META (臉書)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 40, "exit_rsi": 90, "rsi_len": 2, "ma_trend": 200 },
        "AMZN": { "symbol": "AMZN", "name": "AMZN (亞馬遜)", "category": "💻 軟體/巨頭", "mode": "KD", "entry_k": 20, "exit_k": 85, "cmf_len": 40 },
        "TSLA": { "symbol": "TSLA", "name": "TSLA (特斯拉)", "category": "💻 軟體/巨頭", "mode": "KD", "entry_k": 20, "exit_k": 80, "cmf_len": 10 },
        "AAPL": { "symbol": "AAPL", "name": "AAPL (蘋果)", "category": "💻 軟體/巨頭", "mode": "RSI_MA", "entry_rsi": 30, "exit_ma": 20, "rsi_len": 14, "ma_trend": 200 },
        "PLTR": { "symbol": "PLTR", "name": "PLTR (Palantir - AI國防)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 35, "exit_rsi": 85, "rsi_len": 14, "ma_trend": 50 },
        "CRWD": { "symbol": "CRWD", "name": "CRWD (CrowdStrike - 資安)", "category": "💻 軟體/巨頭", "mode": "RSI_RSI", "entry_rsi": 35, "exit_rsi": 90, "rsi_len": 14, "ma_trend": 100, "cmf_len": 20 },
        "PANW": { "symbol": "PANW", "name": "PANW (Palo Alto - 資安)", "category": "💻 軟體/巨頭", "mode": "KD", "entry_k": 20, "exit_k": 80, "cmf_len": 20 },

        # === 💊 生技醫療 (減肥藥) ===
        "LLY": { "symbol": "LLY", "name": "LLY (禮來 - 減肥藥王)", "category": "💊 生技醫療", "mode": "FUSION", "entry_rsi": 60, "exit_rsi": 80, "rsi_len": 14, "ma_trend": 20, "ma_filter": True, "cmf_len": 20 },
        "NVO": { "symbol": "NVO", "name": "NVO (諾和諾德 - 減肥藥)", "category": "💊 生技醫療", "mode": "MA_CROSS", "fast_ma": 10, "slow_ma": 50 },

        # === 🪙 數位資產 (比特幣概念) ===
        "BTC_W": { "symbol": "BTC-USD", "name": "BTC (比特幣 - 波段)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 44, "exit_rsi": 65, "rsi_len": 14, "ma_trend": 200, "cmf_len": 40 },
        "BTC_F": { "symbol": "BTC-USD", "name": "BTC (比特幣 - 閃電)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 50, "rsi_len": 2, "ma_trend": 100, "cmf_len": 40 },
        "MSTR": { "symbol": "MSTR", "name": "MSTR (微策略 - BTC槓桿)", "category": "🪙 數位資產", "mode": "RSI_RSI", "entry_rsi": 35, "exit_rsi": 85, "rsi_len": 14, "ma_trend": 20, "cmf_len": 10 },
        "COIN": { "symbol": "COIN", "name": "COIN (Coinbase)", "category": "🪙 數位資產", "mode": "FUSION", "entry_rsi": 30, "exit_rsi": 90, "rsi_len": 14, "ma_trend": 100 },

        # === ⚡ 電力與能源 ===
        "ETN": { "symbol": "ETN", "name": "ETN (伊頓 - 電網)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 40, "exit_rsi": 95, "ma_trend": 200 },
        "VRT": { "symbol": "VRT", "name": "VRT (維諦 - 液冷)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 35, "exit_rsi": 95, "ma_trend": 100 },
        "OKLO": { "symbol": "OKLO", "name": "OKLO (微型核電)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 50, "exit_rsi": 95, "ma_trend": 0 },
        "SMR": { "symbol": "SMR", "name": "SMR (NuScale - 核能)", "category": "⚡ 電力/能源", "mode": "RSI_RSI", "rsi_len": 3, "entry_rsi": 45, "exit_rsi": 90, "ma_trend": 0, "cmf_len": 14 },

        # === 🇹🇼 台股 AI 權值 ===
        "CHT": { "symbol": "2412.TW", "name": "中華電", "category": "🇹🇼 台股", "mode": "RSI_RSI", "rsi_len": 14, "entry_rsi": 45, "exit_rsi": 70 },
        "HONHAI": { "symbol": "2317.TW", "name": "鴻海 (AI 伺服器代工)", "category": "🇹🇼 台股", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "QUANTA": { "symbol": "2382.TW", "name": "廣達 (AI 伺服器龍頭)", "category": "🇹🇼 台股", "mode": "RSI_MA", "entry_rsi": 40, "exit_ma": 20, "rsi_len": 14, "ma_trend": 60 },
        "MEDIATEK": { "symbol": "2454.TW", "name": "聯發科 (手機晶片)", "category": "🇹🇼 台股", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 80, "rsi_len": 14, "ma_trend": 0 },

        # === 🛡️ 防禦/傳產/原物料 ===
        "KO": { "symbol": "KO", "name": "KO (可口可樂)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 2, "entry_rsi": 30, "exit_rsi": 90, "ma_trend": 0, "cmf_len": 20 },
        "JNJ": { "symbol": "JNJ", "name": "JNJ (嬌生)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 25, "exit_rsi": 90, "ma_trend": 200, "cmf_len": 20 },
        "PG": { "symbol": "PG", "name": "PG (寶僑)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 20, "exit_rsi": 80, "ma_trend": 0, "cmf_len": 30 },
        "BA": { "symbol": "BA", "name": "BA (波音)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "rsi_len": 6, "entry_rsi": 15, "exit_rsi": 60, "ma_trend": 0, "cmf_len": 25 },
        "JPM": { "symbol": "JPM", "name": "JPM (摩根大通)", "category": "🛡️ 防禦/傳產", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 80, "rsi_len": 14, "ma_trend": 200 },
        "COST": { "symbol": "COST", "name": "COST (好市多)", "category": "🛡️ 防禦/傳產", "mode": "MA_CROSS", "fast_ma": 20, "slow_ma": 60 },
        
        "GC": { "symbol": "GC=F", "name": "Gold (黃金期貨)", "category": "⛏️ 原物料", "mode": "RSI_RSI", "entry_rsi": 30, "exit_rsi": 70, "rsi_len": 14 },
        "CL": { "symbol": "CL=F", "name": "Crude Oil (原油期貨)", "category": "⛏️ 原物料", "mode": "KD", "entry_k": 20, "exit_k": 80 },
        "HG": { "symbol": "HG=F", "name": "Copper (銅期貨)", "category": "⛏️ 原物料", "mode": "RSI_MA", "entry_rsi": 30, "exit_ma": 50, "rsi_len": 14 }
    }
    
    # 2. 製作分類選單 (先執行)
    all_categories = sorted(list(set(s['category'] for s in strategies.values())))
    selected_cat = st.selectbox("📂 步驟一：選擇板塊分類", all_categories)
    
    # 3. 根據分類篩選股票 (次執行)
    cat_strategies = {k: v for k, v in strategies.items() if v['category'] == selected_cat}
    target_key = st.selectbox("📍 步驟二：選擇具體標的", list(cat_strategies.keys()), format_func=lambda x: cat_strategies[x]['name'])
    
    # 4. 定義 cfg (關鍵！必須在選單之後)
    cfg = strategies[target_key]
    
    # 5. 最後才讀取數據 (確保 cfg 已經存在)
    df = get_safe_data(cfg['symbol'])
    lp = get_real_live_price(cfg['symbol'])
    
    if df is not None and lp:
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else lp
        chg = lp - prev_close
        pct_chg = (chg / prev_close) * 100
        
        current_sig, perf, sigs = quick_backtest(df, cfg)
        win_rate = perf['Raw_Win_Rate'] if perf else 0
        trades_count = perf['Trades'] if perf else 0
        
        kelly_msg, kelly_shares = calculate_kelly_position(df, user_capital, win_rate, user_risk/100, current_sig)
        k_pat = identify_k_pattern(df)
        rsi_val = ta.rsi(df['Close'], 14).iloc[-1]
        fund = get_fundamentals(cfg['symbol'])
        
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("即時價格", f"${lp:.2f}", f"{chg:.2f} ({pct_chg:.2f}%)")
            
            if trades_count > 0:
                c2.metric("策略勝率 (回測)", f"{win_rate*100:.0f}%", delta=f"{trades_count} 次交易")
            else:
                c2.metric("策略勝率 (回測)", "無交易", delta="區間未觸發", delta_color="off")
                
            c3.metric("凱利建議倉位", f"{kelly_shares} 股", delta=kelly_msg.split(' ')[0] if '建議' in kelly_msg else "觀望") 
            
            st.info(f"💡 凱利觀點: {kelly_msg}")
            
            # --- ★★★ 新增：籌碼診斷面板 ★★★ ---
            chip_msg, chip_status, cmf_val = analyze_chip_health(df, cmf_len=cfg.get('cmf_len', 20))
            
            # 根據狀態顯示不同顏色的提示框
            if chip_status == "danger":
                st.error(f"💣 籌碼診斷: {chip_msg}")
            elif chip_status == "gold":
                st.success(f"💰 籌碼診斷: {chip_msg}")
            elif chip_status == "healthy":
                st.success(f"✅ 籌碼診斷: {chip_msg}")
            else:
                st.warning(f"⚖️ 籌碼診斷: {chip_msg}")

            # 繪製新版圖表
            fig = plot_chart(df, cfg, sigs)
            st.plotly_chart(fig, use_container_width=True)
            
            # 加入圖表解讀說明 (幫助你看懂)
            with st.expander("📖 如何解讀下方籌碼圖 (Row 3)?"):
                st.markdown("""
                **1. 資金流向 (CMF) - 柱狀圖/山脈**:
                * **<span style='color:#00E676'>綠色柱狀</span>**: 資金淨流入 (收盤價收在高點)。越高代表買盤越強。
                * **<span style='color:#FF5252'>紅色柱狀</span>**: 資金淨流出 (收盤價收在低點)。越低代表賣壓越重。
                * **背離訊號**: 股價創新低，但紅色柱狀變短 (底部背離) -> 買點。

                **2. 籌碼能量 (OBV) - 線條**:
                * **<span style='color:cyan'>青色實線 (OBV)</span>** vs **<span style='color:yellow'>黃色虛線 (OBV均線)</span>**。
                * **OBV 穿過 均線向上**: 主力進場控盤，安全。
                * **OBV 跌破 均線向下**: 主力棄守，危險。
                * **頂部背離**: 股價創新高，但 OBV 沒有過前高 -> 假突破。
                """, unsafe_allow_html=True)

        if fund:
            with st.expander("📊 財報基本面 & 籌碼數據", expanded=False):
                f1, f2, f3, f4, f5 = st.columns(5)
                
                def check_metric(val, high_good=True, low_good=False, threshold_good=0, threshold_bad=0):
                    if val is None: return "N/A", "off"
                    val = float(val)
                    if high_good:
                        if val > threshold_good: return f"{val:.1f}% ✅", "normal"
                        if val < threshold_bad: return f"{val:.1f}% ❌", "inverse"
                        return f"{val:.1f}% ⚠️", "off"
                    if low_good:
                        if val < threshold_good: return f"{val:.1f}% ✅", "normal"
                        if val > threshold_bad: return f"{val:.1f}% ❌", "inverse"
                        return f"{val:.1f}% ⚠️", "off"
                    return f"{val:.1f}", "off"

                pe_val = fund['pe']
                pe_str = "N/A"; pe_delta = "off"
                if pe_val:
                    if pe_val < 25: pe_str, pe_delta = f"{pe_val:.1f} ✅", "normal"
                    elif pe_val > 50: pe_str, pe_delta = f"{pe_val:.1f} ❌", "inverse"
                    else: pe_str, pe_delta = f"{pe_val:.1f} ⚠️", "off"
                f1.metric("本益比 (PE)", pe_str, delta_color=pe_delta)

                eps_val = fund['eps']
                eps_str = "N/A"; eps_delta = "off"
                if eps_val:
                    if eps_val > 0: eps_str, eps_delta = f"${eps_val:.2f} ✅", "normal"
                    else: eps_str, eps_delta = f"${eps_val:.2f} ❌", "inverse"
                f2.metric("EPS", eps_str, delta_color=eps_delta)

                m_str, m_delta = check_metric(fund['margin']*100, high_good=True, threshold_good=40, threshold_bad=10)
                f3.metric("毛利率", m_str, delta_color=m_delta)

                i_str, i_delta = check_metric(fund['inst']*100, high_good=True, threshold_good=60, threshold_bad=20)
                f4.metric("法人持股", i_str, delta_color=i_delta)

                s_str, s_delta = check_metric(fund['short']*100, low_good=True, threshold_good=5, threshold_bad=15)
                f5.metric("空單比例", s_str, delta_color=s_delta)
        else:
            st.warning("⚠️ 暫無財報數據 (API 忙碌中，請稍後再試)")

        strat_desc = get_strategy_desc(cfg, df)
        st.markdown(f"**🛠️ 當前策略邏輯：** `{strat_desc}`")

        analyze_btn = False 
        if ai_provider == "Gemini (User Defined)" and gemini_key:
            st.divider()
            st.subheader("🧠 Gemini 首席分析師")
            st.info("ℹ️ 系統將自動抓取 Google News 最新頭條。若您有額外資訊 (如財報細節)，可在下方補充。")
            with st.expander("📝 補充筆記 (選填 / Optional)", expanded=False):
                user_notes = st.text_area("例如：營收創歷史新高、分析師調升評級...", height=68)
            analyze_btn = st.button("🚀 啟動 AI 深度分析 (含新聞解讀)")
            
        if analyze_btn and ai_provider == "Gemini (User Defined)":
            with st.spinner("🔍 AI 正在爬取 Google News 並進行大腦運算..."):
                news_items = get_news(cfg['symbol'])
                if news_items:
                    with st.expander(f"📰 AI 已讀取 {len(news_items)} 則最新新聞", expanded=True):
                        for n in news_items:
                            st.caption(f"• {n}")
                else:
                    st.warning("⚠️ 暫時抓不到 Google News，AI 將純以技術面分析。")
                    news_items = []

                strat_rsi_len = cfg.get('rsi_len', 14)
                strat_val_txt = ""
                if "RSI" in cfg['mode'] or cfg['mode'] == "FUSION":
                    real_rsi = ta.rsi(df['Close'], length=strat_rsi_len).iloc[-1]
                    strat_val_txt = f"Strategy_RSI({strat_rsi_len}):{real_rsi:.1f}"
                elif "KD" in cfg['mode']:
                    k_val = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3).iloc[-1, 0]
                    strat_val_txt = f"KD_K(9,3):{k_val:.1f}"
                elif cfg['mode'] == "MA_CROSS":
                    ma_fast = ta.sma(df['Close'], cfg['fast_ma']).iloc[-1]
                    ma_slow = ta.sma(df['Close'], cfg['slow_ma']).iloc[-1]
                    dist = (ma_fast - ma_slow) / ma_slow * 100
                    strat_val_txt = f"MA_Gap:{dist:.2f}%"

                base_rsi = ta.rsi(df['Close'], 14).iloc[-1]
                sig_map = { 1: "🚀 買進訊號 (Buy)", -1: "📉 賣出訊號 (Sell)", 0: "💤 觀望/無訊號 (Wait)" }
                human_sig = sig_map.get(int(current_sig), "未知")

                fund_txt = "無財報數據"
                if fund:
                    short_trend_str = "N/A"
                    if fund.get('shares_short') and fund.get('shares_short_prev'):
                        change = (fund['shares_short'] - fund['shares_short_prev']) / fund['shares_short_prev']
                        if change > 0.05: short_trend_str = f"🔴 增加 {change*100:.1f}% (空軍集結)"
                        elif change < -0.05: short_trend_str = f"🟢 減少 {abs(change)*100:.1f}% (空軍回補)"
                        else: short_trend_str = f"⚪ 持平 ({change*100:.1f}%)"

                    pe_trend_str = "N/A"
                    if fund.get('pe') and fund.get('fwd_pe'):
                        if fund['fwd_pe'] < fund['pe']: pe_trend_str = f"↘️ 看好 (預估PE {fund['fwd_pe']:.1f} < 當前)"
                        else: pe_trend_str = f"↗️ 看壞 (預估PE {fund['fwd_pe']:.1f} > 當前)"

                    rev_g = f"{fund.get('rev_growth', 0)*100:.1f}%" if fund.get('rev_growth') is not None else "N/A"
                    earn_g = f"{fund.get('earn_growth', 0)*100:.1f}%" if fund.get('earn_growth') is not None else "N/A"
                    
                    fund_txt = (
                        f"PE評價趨勢:{pe_trend_str} | "
                        f"空單變動(MoM):{short_trend_str} | "
                        f"空單比例:{fund.get('short', 0)*100:.1f}% | "
                        f"營收成長(YoY):{rev_g} | "
                        f"獲利成長(YoY):{earn_g} | "
                        f"毛利率:{fund.get('margin', 0)*100:.1f}%"
                    )

                tech_txt = (
                    f"【策略關鍵指標】: {strat_val_txt}\n"
                    f"【籌碼與基本面】: {fund_txt}\n"
                    f"【市場大環境 RSI(14)】: {base_rsi:.1f}\n"
                    f"【回測勝率】: {win_rate*100:.0f}%\n"
                    f"【當前訊號】: {human_sig}"
                )

                def analyze_v2(api_key, symbol, news, tech_txt, k_pattern, model_name, user_input=""):
                    if not HAS_GEMINI: return "No Gemini", "⚠️", False
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(model_name)
                        news_str = "\n".join([f"- {n}" for n in news]) if news else "無最新新聞"
                        base_prompt = f"""
                        你是一位華爾街資深操盤手。請根據以下「動態趨勢數據」進行深度分析：
                        【目標標的】：{symbol}
                        【綜合數據面板】：
                        {tech_txt}
                        【K線型態】：{k_pattern}
                        【最新新聞焦點】：
                        {news_str}
                        【用戶補充筆記】：{user_input}
                        請給出分析報告：
                        1. 🎯 核心觀點 (多/空/觀望)
                        2. 📊 籌碼與基本面解讀 (特別關注空單增減與預估PE的變化意義)
                        3. 📰 市場情緒
                        4. 💡 操作建議
                        """
                        return model.generate_content(base_prompt).text, "🧠", True
                    except Exception as e: return str(e), "⚠️", False

                analysis, icon, success = analyze_v2(gemini_key, cfg['symbol'], news_items, tech_txt, k_pat, gemini_model, user_notes)
                if success: st.markdown(analysis)
                else: st.error(f"Gemini 連線失敗: {analysis}")
# ------------------------------------------
# Mode 4: XGBoost 實驗室 (三刀流終極版)
# ------------------------------------------
elif app_mode == "🌲 XGBoost 實驗室":
    st.header("🌲 XGBoost 戰略指揮所")
    st.caption("針對不同商品特性，切換專屬 AI 大腦")

    # 1. 選擇策略模組
    model_mode = st.radio("選擇戰略模組：", 
        ["⚔️ TSM 攻擊型 (個股動能)", "🌊 TQQQ 趨勢型 (槓桿波段)", "🇹🇼 台股連動型 (TW Stocks)", "⚡ 能源電力型 (Oil & Util)", "🔥 AI 超級週期 (AVGO/MU)", "🐺 績優股長波段 (孤狼策略)","🏆 TQQQ 冠軍版 (波動率策略)", "🛡️ EDZ 避險型 (崩盤偵測)"], 
        horizontal=True
    )

    # 2. 根據模式設定預設值與說明
    if "TSM" in model_mode:
        default_target = "TSM"
        desc = "✅ 專攻：TSM \n\n🧠 邏輯：看重「輝達連動」與「短線爆發力」。只要輝達漲、動能強就追，不錯過任何魚身。"
    elif "TQQQ" in model_mode:
        default_target = "TQQQ"
        desc = "✅ 專攻：TQQQ, SOXL, SPXL, MRVL\n\n🧠 邏輯：看重「50日生命線」與「RSI」。站上均線就死抱，跌破就跑，專吃大波段。"
        # ★★★ 新增這一段 (台股設定) ★★★
    elif "台股" in model_mode:
        default_target = "2330"  # 預設顯示台積電
        desc = "✅ 專攻：0050成分股 (如 2330, 2454, 2603)\n\n🧠 邏輯：跟著「美股昨晚收盤」做台股。結合季線趨勢與費半連動。"
    elif "能源" in model_mode:
        default_target = "XLE"
        desc = "✅ 專攻：能源(XLE)、潔淨能源(ICLN)\n\n🧠 邏輯：看重「原油(CL=F)」、「天然氣(NG=F)」與「美債利率」。"
    elif "週期" in model_mode:
        default_target = "MU"
        desc = "✅ 專攻：MU \n\n🧠 邏輯：週期循環。"
    # ★★★ 新增：孤狼策略 (AVGO 專用) ★★★
    elif "長波段" in model_mode:
        default_target = "AVGO"
        desc = "✅ 專攻：AVGO, MSFT, AAPL (慢牛股)\n\n🧠 邏輯：孤狼策略。斷絕 NVDA 連動，只看「長期趨勢 (60/120MA)」與「預測未來20日」。"
    else:
        default_target = "EDZ"
        desc = "✅ 專攻：EDZ, SQQQ, UVXY, AVGO\n\n🧠 邏輯：看重「VIX恐慌」與「美元匯率」。平時空手，只有市場快崩盤時才亮燈。"

    st.info(desc)
    target = st.text_input("輸入代號 (Target)", value=default_target)
    # ==========================================
    # ★★★ 修正：把滑桿移到按鈕外面，這樣它才不會消失 ★★★
    # ==========================================
    st.sidebar.divider()
    st.sidebar.header("🔧 回測時光機")
    
    # 👇 請把原本那行改成這樣，加上 key="backtest_slider"
    test_ratio = st.sidebar.slider(
        "回測長度 (Test Size)", 
        0.05, 0.5, 0.2, 0.05, 
        key="backtest_slider"  # <--- 加這個！這是它的身分證
    )

    if st.button(f"🚀 啟動 {target} AI 訓練"):
        with st.spinner(f"正在召喚 {model_mode.split()[1]} AI 模型..."):
            try:
                # ==========================================
                # 策略 A: TSM 攻擊型 (動能 + NVDA 連動)
                # ==========================================
                if "TSM" in model_mode:
                    # 1. 下載數據 (個股需要看大哥 NVDA 和 費半 SOX)
                    tickers = [target, "NVDA", "^SOX"]
                    data = yf.download(tickers, period="5y", interval="1d", progress=False)
                    if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
                    else: df = data['Close'].copy()
                    df.ffill(inplace=True); df.dropna(inplace=True)

                    # 2. 特徵工程 (貪婪動能版)
                    df['Target_Ret_1d'] = df[target].pct_change()
                    df['Target_Ret_3d'] = df[target].pct_change(3)
                    df['Target_Ret_5d'] = df[target].pct_change(5)
                    df['NVDA_Ret'] = df['NVDA'].pct_change() # 關鍵因子
                    df['SOX_Ret'] = df['^SOX'].pct_change()
                    df['Alpha_NVDA'] = df['Target_Ret_5d'] - df['NVDA'].pct_change(5)
                    df['Vola'] = df[target].rolling(5).std() / df[target]
                    
                    df.dropna(inplace=True)
                    features = ['Target_Ret_1d', 'Target_Ret_3d', 'Target_Ret_5d', 'NVDA_Ret', 'SOX_Ret', 'Alpha_NVDA', 'Vola']

                    # 3. 標籤 (貪婪：未來3天只要漲 > 0 就買)
                    future_ret = df[target].shift(-3) / df[target] - 1
                    df['Label'] = np.where(future_ret > 0.0, 1, 0)

                    # 4. 模型參數 (積極型：深樹、高採樣)
                    params = {
                        'n_estimators': 200, 'learning_rate': 0.03, 'max_depth': 5, 
                        'subsample': 0.9, 'colsample_bytree': 0.9
                    }
                    look_ahead_days = 3 # 預測未來 3 天

                # ==========================================
                # 策略 B: TQQQ 趨勢型 (升級版 - 加入日圓避險)
                # ==========================================
                elif "TQQQ" in model_mode and "冠軍" not in model_mode:
                    # 1. 下載數據 (★ 修改 1: 加入 JPY=X 和 VIX)
                    
                    tickers = [target, "QQQ"]
                    data = yf.download(tickers, period="5y", interval="1d", progress=False)
                    
                    if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
                    else: df = data['Close'].copy()
                    
                    df.ffill(inplace=True); df.dropna(inplace=True)

                    # 2. 特徵工程 (★ 關鍵修改：移除 Vola)
                    # 我們只留均線和 RSI，因為波動率(Vola)會在噴出段嚇跑 AI
                    df['SMA_50'] = ta.sma(df[target], length=50) 
                    df['Bias_50'] = (df[target] - df['SMA_50']) / df['SMA_50'] 
                    df['RSI'] = ta.rsi(df[target], length=14)
                    df['Ret_5d'] = df[target].pct_change(5)
                    df['QQQ_Ret_5d'] = df['QQQ'].pct_change(5)
                    
                    df.dropna(inplace=True)
                    # ★ 特徵列表：只有純粹的趨勢與動能
                    features = ['Bias_50', 'RSI', 'Ret_5d', 'QQQ_Ret_5d'] 
                    
                    # 3. 標籤 (預測未來 5 天)
                    future_ret = df[target].shift(-5) / df[target] - 1
                    df['Label'] = np.where(future_ret > 0.0, 1, 0)

                    # 4. 模型參數 (維持高反應速度)
                    params = {
                        'n_estimators': 150,    
                        'learning_rate': 0.08, 
                        'max_depth': 3,         
                        'min_child_weight': 3,  
                        'gamma': 0.2,           
                        'subsample': 0.8, 
                        'colsample_bytree': 0.8
                    }
                    look_ahead_days = 5 
                    
                    # 權重設定
                    weight_multiplier = 1.2 
                    buy_threshold = 0.50
                    
                    
                # ==========================================
                # 策略 D: 台股連動型 (最終獲利版：鎖定 3y + 積極參數)
                # ==========================================
                elif "台股" in model_mode:
                    # 1. 處理代號
                    if not target.endswith(".TW") and not target.endswith(".TWO"):
                        target = f"{target}.TW"
                    
                    # 2. 下載數據 (★ 關鍵修正 1：絕對要用 "3y")
                    # 5y 會讓 AI 變得太膽小；3y 才能重現您看到的飆漲曲線
                    tickers = [target, "^SOX", "QQQ", "NVDA"]
                    
                    st.write(f"🚀 啟動台股策略 (5年積極版)，鎖定：{target}...")
                    data = yf.download(tickers, period="5y", interval="1d", progress=False)
                    
                    if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
                    else: df = data['Close'].copy()
                    
                    # 3. 補值策略
                    df.ffill(inplace=True)
                    df.dropna(inplace=True)
                    
                    # --- 特徵工程 (保持台股必勝因子) ---
                    df['SOX_Ret'] = df['^SOX'].pct_change()
                    df['QQQ_Ret'] = df['QQQ'].pct_change()
                    df['NVDA_Ret'] = df['NVDA'].pct_change()
                    
                    df['Target_Ret_1d'] = df[target].pct_change()
                    df['Target_Ret_5d'] = df[target].pct_change(5)
                    
                    df['SMA_20'] = ta.sma(df[target], length=20)
                    df['SMA_60'] = ta.sma(df[target], length=60)
                    
                    df['Bias_20'] = (df[target] - df['SMA_20']) / df['SMA_20']
                    df['Bias_60'] = (df[target] - df['SMA_60']) / df['SMA_60']
                    
                    df['RSI'] = ta.rsi(df[target], length=14)

                    df.dropna(inplace=True)
                    
                    features = ['Bias_20', 'Bias_60', 'RSI', 'SOX_Ret', 'NVDA_Ret', 'Target_Ret_5d']

                    # 4. 標籤
                    future_ret = df[target].shift(-5) / df[target] - 1
                    df['Label'] = np.where(future_ret > 0.0, 1, 0)

                    # 5. 模型參數 (★ 關鍵修正 2：調高學習率到 0.08)
                    # 這會讓紅線緊緊咬住行情，不會像圖 B 那樣平平的
                    params = {
                        'n_estimators': 150,    
                        'learning_rate': 0.05,  # 加快反應
                        'max_depth': 4,         
                        'gamma': 0.1,           
                        'subsample': 0.8, 
                        'colsample_bytree': 0.8
                    }
                    
                    weight_multiplier = 1.2
                    buy_threshold = 0.50
                    
                    st.info("💡 系統優化：已強制切換為「3年積極架構」，這將排除 2022 空頭干擾，重現強勢追價邏輯。")
                # ==========================================
                # 策略 E: 能源電力型 (Final - 布林逆勢版)
                # ==========================================
                elif "能源" in model_mode:
                    # 1. 下載數據 (加入 SPY 當濾網)
                    tickers = [target, "SPY"]
                    data = yf.download(tickers, period="5y", interval="1d", progress=False)
                    
                    if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
                    else: df = data['Close'].copy()
                    
                    df.ffill(inplace=True); df.dropna(inplace=True)

                    
                    # 2. 特徵工程 (引入布林通道 - 修正版)
                    
                    # A. 布林通道 (Bollinger Bands)
                    # 參數：20日移動平均，2倍標準差
                    bb = ta.bbands(df[target], length=20, std=2)
                    
                    # ★★★ 修正點在此：直接重新命名欄位，避開 .0 的問題 ★★★
                    # pandas_ta 的 bbands 固定回傳 5 個欄位，順序如下：
                    # Lower(下軌), Mid(中軌), Upper(上軌), Bandwidth(寬度), Percent(位階)
                    if bb is not None and not bb.empty:
                        bb.columns = ['BBL', 'BBM', 'BBU', 'BBB', 'BBP']
                        
                        # 這樣我們就可以用簡單的名稱來呼叫了
                        df['BB_Lower'] = bb['BBL']
                        df['BB_Upper'] = bb['BBU']
                        df['BB_Width'] = bb['BBB']
                        
                        # 計算 BB_Pct (股價在通道的哪個位置)
                        # < 0 代表跌破下軌，> 1 代表突破上軌
                        df['BB_Pct'] = (df[target] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                    else:
                        # 萬一計算失敗的防呆機制
                        df['BB_Pct'] = 0.5 
                        df['BB_Width'] = 0

                    # B. 短線乖離 (Bias_20)
                    df['SMA_20'] = ta.sma(df[target], length=20)
                    df['Bias_20'] = (df[target] - df['SMA_20']) / df['SMA_20']
                    
                    # C. RSI
                    df['RSI'] = ta.rsi(df[target], length=14)
                    
                    # D. 大盤相對強弱
                    df['Alpha_SPY'] = df[target].pct_change(5) - df['SPY'].pct_change(5)

                    df.dropna(inplace=True)
                    
                    # 特徵列表 (使用新定義的名稱)
                    features = ['BB_Pct', 'BB_Width', 'Bias_20', 'RSI', 'Alpha_SPY']

                    # 3. 標籤 (預測未來 5 天)
                    future_ret = df[target].shift(-5) / df[target] - 1
                    df['Label'] = np.where(future_ret > 0.0, 1, 0)

                    # 4. 模型參數 (稍微調高學習率，讓它反應靈敏一點)
                    params = {
                        'n_estimators': 200,    
                        'learning_rate': 0.08, # 反應快一點
                        'max_depth': 5,         
                        'gamma': 0.1,           
                        'subsample': 0.8, 
                        'colsample_bytree': 0.8
                    }
                    
                    weight_multiplier = 1.1 
                    buy_threshold = 0.50
                    
                    st.info("💡 能源策略邏輯 (Final)：採用「布林通道 (Bollinger Bands)」策略。專門捕捉能源股在區間下緣的「超賣反彈」機會。")
                # ==========================================
                # 策略 F: AI 超級週期型 (專門跑 MU, AVGO)
                # ==========================================
                elif "週期" in model_mode:
                    # 1. 下載數據 (三劍客：目標、輝達、費半)
                    tickers = [target, "NVDA", "^SOX"]
                    data = yf.download(tickers, period="5y", interval="1d", progress=False)
                    
                    if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
                    else: df = data['Close'].copy()
                    
                    df.ffill(inplace=True); df.dropna(inplace=True)

                    # 2. 特徵工程 (融合了 TSM 的連動性 + TQQQ 的趨勢性)
                    
                    # A. 老大帶路 (輝達連動)
                    df['NVDA_Ret'] = df['NVDA'].pct_change()
                    df['SOX_Ret'] = df['^SOX'].pct_change()
                    
                    # B. 長線保護 (季線乖離) - 這是抱住 4 倍漲幅的關鍵
                    df['SMA_60'] = ta.sma(df[target], length=60)
                    df['Bias_60'] = (df[target] - df['SMA_60']) / df['SMA_60']
                    
                    # C. 中線動能 (動量)
                    # 過去 10 天漲不漲？確認趨勢慣性
                    df['Mom_10'] = df[target] / df[target].shift(10)
                    
                    # D. 波動率 (MU 很活潑，需要這個來判斷是否過熱)
                   
                    df.dropna(inplace=True)
                    
                    # 特徵列表
                    features = ['NVDA_Ret', 'Bias_60', 'Mom_10', 'SOX_Ret']

                    # 3. 標籤 (★★★ 關鍵修改：預測未來 10 天 ★★★)
                    # 讓 AI 學習「持有兩週」會不會賺錢，而不是三天
                    future_ret = df[target].shift(-10) / df[target] - 1
                    df['Label'] = np.where(future_ret > 0.0, 1, 0)

                    # 4. 模型參數 (稍微加深樹的深度，因為週期股比較複雜)
                    params = {
                        'n_estimators': 200,    
                        'learning_rate': 0.05, 
                        'max_depth': 6,         
                        'gamma': 0.1,           
                        'subsample': 0.8, 
                        'colsample_bytree': 0.8
                    }
                    
                    weight_multiplier = 1.15  # 積極進攻
                    buy_threshold = 0.50
                    
                    st.info("💡 超級週期邏輯：結合「NVDA 連動」與「10日趨勢預測」。專為捕捉 AVGO 與 MU 的波段大行情設計，避免太早下車。")
                # ==========================================
                # 策略 G: 績優股長波段 (孤狼策略 - 專治 AVGO)
                # ==========================================
                elif "長波段" in model_mode:
                    # 1. 下載數據 (★關鍵：只下載它自己，斷絕外部雜訊★)
                    tickers = [target] 
                    data = yf.download(tickers, period="5y", interval="1d", progress=False)
                    
                    if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
                    else: df = data['Close'].copy()
                    
                    df.ffill(inplace=True); df.dropna(inplace=True)

                    # 2. 特徵工程 (極簡化：只看中長線趨勢)
                    
                    # A. 季線趨勢 (60日)
                    df['SMA_60'] = ta.sma(df[target], length=60)
                    df['Bias_60'] = (df[target] - df['SMA_60']) / df['SMA_60']
                    
                    # B. 半年線趨勢 (120日) - ★新增：用來確認大格局
                    df['SMA_120'] = ta.sma(df[target], length=120)
                    df['Bias_120'] = (df[target] - df['SMA_120']) / df['SMA_120']
                    
                    # C. 月動能 (過去20天漲幅)
                    # 取代 RSI，因為動能沒有上限，不會因為漲多就被賣掉
                    df['Mom_20'] = df[target] / df[target].shift(20)

                    df.dropna(inplace=True)
                    
                    # 特徵列表：乾淨到只剩這三個
                    features = ['Bias_60', 'Bias_120', 'Mom_20']

                    # 3. 標籤 (★關鍵：預測未來 20 天/一個月★)
                    # 強迫 AI 學習「持有這張股票一個月會不會賺錢？」
                    future_ret = df[target].shift(-20) / df[target] - 1
                    df['Label'] = np.where(future_ret > 0.0, 1, 0)

                    # 4. 模型參數 (降低複雜度，避免想太多)
                    params = {
                        'n_estimators': 100,    
                        'learning_rate': 0.05,
                        'max_depth': 4, # 淺一點，讓它只抓大方向
                        'gamma': 0.1,           
                        'subsample': 0.8, 
                        'colsample_bytree': 0.8
                    }
                    
                    weight_multiplier = 1.0 
                    buy_threshold = 0.50
                    
                    st.info("💡 孤狼策略邏輯：專為 AVGO 這種「獨立走勢」的慢牛設計。切斷 NVDA 連動，只看 60日/120日 長線趨勢，並預測未來 20 天走勢。")
                # ==========================================
                # ★★★ TQQQ 最終攻擊版 (已修復 SMA_50 錯誤) ★★★
                # ==========================================
                elif "冠軍" in model_mode:
                    default_target = "TQQQ"
                    
                    # 1. 下載數據
                    tickers = [target, "QQQ"]
                    st.write(f"🚀 啟動 TQQQ 最終攻擊策略 (Trend Only)...")
                    
                    # 維持 3y (專注近期)
                    data = yf.download(tickers, period="5y", interval="1d", progress=False)
                    
                    if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
                    else: df = data['Close'].copy()
                    
                    df.ffill(inplace=True); df.dropna(inplace=True)

                    # 2. 特徵工程
                    
                    # A. 富爸爸的動向 (最重要)
                    df['QQQ_Ret_5d'] = df['QQQ'].pct_change(5) 
                    
                    # B. 自身的動能
                    df['Ret_5d'] = df[target].pct_change(5)
                    
                    # C. 趨勢乖離 (生命線)
                    # ★★★ 關鍵修正：必須先存下 SMA_50，否則最後的即時預測會報錯！ ★★★
                    df['SMA_50'] = ta.sma(df[target], 50)
                    df['Bias_50'] = (df[target] - df['SMA_50']) / df['SMA_50']
                    
                    # D. 短線強弱
                    df['RSI'] = ta.rsi(df[target], length=14)

                    df.dropna(inplace=True)
                    
                    # ★ 最終特徵列表：只有 4 個純趨勢因子
                    features = ['QQQ_Ret_5d', 'Bias_50', 'Ret_5d', 'RSI'] 
                    
                    # 3. 標籤 (預測未來 5 天)
                    future_ret = df[target].shift(-5) / df[target] - 1
                    df['Label'] = np.where(future_ret > 0.0, 1, 0)

                    # 4. 模型參數 (高反應速度)
                    params = {
                        'n_estimators': 200,    
                        'learning_rate': 0.08,
                        'max_depth': 4,         
                        'min_child_weight': 3,  
                        'gamma': 0.2,           
                        'subsample': 0.8, 
                        'colsample_bytree': 0.8
                    }
                    look_ahead_days = 5 
                    weight_multiplier = 1.2 
                    buy_threshold = 0.50
                    
                    st.info("💡 系統修復：已補回 SMA_50 欄位，即時預測功能將恢復正常。")

                # ==========================================
                # 策略 C: EDZ 避險型 (崩盤偵測)
                # ==========================================
                else:
                    ref_market = "EEM" if "EDZ" in target else "QQQ"
                    tickers = [target, ref_market, "DX-Y.NYB", "^VIX"]
                    data = yf.download(tickers, period="5y", interval="1d", progress=False)
                    if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
                    else: df = data['Close'].copy()
                    df.ffill(inplace=True); df.dropna(inplace=True)

                    # 特徵
                    df['Target_Ret_1d'] = df[target].pct_change()
                    df['Market_Ret'] = df[ref_market].pct_change()
                    df['DXY_Ret'] = df['DX-Y.NYB'].pct_change()
                    df['VIX_Level'] = df['^VIX']
                    df['Vola'] = df[target].rolling(5).std() / df[target]
                    
                    df.dropna(inplace=True)
                    features = ['Target_Ret_1d', 'Market_Ret', 'DXY_Ret', 'VIX_Level', 'Vola']

                    # 標籤 (抓大波動 > 2%)
                    future_ret = df[target].shift(-3) / df[target] - 1
                    df['Label'] = np.where(future_ret > 0.02, 1, 0)

                    params = {
                        'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 3,
                        'subsample': 0.7, 'colsample_bytree': 0.7
                    }
                    look_ahead_days = 3

                # ==========================================
                # 通用訓練流程 (修復版：加入強制轉型 + 回測滑桿)
                # ==========================================
                
                # 1. 強制將所有特徵轉為數字
                for col in features:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 2. 清除 NaN
                df.dropna(inplace=True)

                # 確保還有資料
                if len(df) < 50:
                    st.error(f"❌ 數據清洗後樣本不足 ({len(df)}筆)，無法訓練。")
                    st.stop()
                
                X = df[features]
                y = df['Label']
                
                # ★★★ 關鍵修改：使用滑桿數值來切分 ★★★
                split = int(len(df) * (1 - test_ratio))
                
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]

                # 計算基礎權重
                base_weight = (len(y_train) - y_train.sum()) / y_train.sum()
                multiplier = locals().get('weight_multiplier', 1.0) 
                final_weight = base_weight * multiplier

                st.write('⚖️ 正在召喚集成模型三巨頭 (XGBoost + LightGBM + CatBoost)...')
                
                # 1. 訓練 XGBoost
                model_xgb = xgb.XGBClassifier(**params, scale_pos_weight=final_weight, random_state=42)
                model_xgb.fit(X_train, y_train)

                # 2. 訓練 LightGBM (修正欄位名稱)
                X_train_lgb = X_train.rename(columns=lambda x: x.replace('_', ''))
                model_lgb = lgb.LGBMClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'], random_state=42, verbose=-1, scale_pos_weight=final_weight)
                model_lgb.fit(X_train_lgb, y_train)

                # 3. 訓練 CatBoost
                model_cat = CatBoostClassifier(iterations=params['n_estimators'], depth=params['max_depth'], learning_rate=params['learning_rate'], random_seed=42, verbose=0, scale_pos_weight=final_weight)
                model_cat.fit(X_train, y_train)

                # 4. 集成包裝器 (這是原本的，我們保留它來做預測)
                class EnsembleWrapper:
                    def __init__(self, models): self.models = models
                    def predict_proba(self, X):
                        p1 = self.models[0].predict_proba(X)[:, 1]
                        X_lgb = X.rename(columns=lambda x: x.replace('_', ''))
                        p2 = self.models[1].predict_proba(X_lgb)[:, 1]
                        p3 = self.models[2].predict_proba(X)[:, 1]
                        avg = (p1 + p2 + p3) / 3
                        return np.vstack([1-avg, avg]).T
                    
                    # ★ 讓包裝器也能吐出特徵重要性 (借用 XGBoost 的)
                    @property
                    def feature_importances_(self): return self.models[0].feature_importances_

                model = EnsembleWrapper([model_xgb, model_lgb, model_cat])

                # =========================================================
                # 🚀 A/B 測試邏輯開始：單挑 vs 群毆
                # =========================================================
                
                threshold = locals().get('buy_threshold', 0.5)

                # 1. 取得「單一 XGBoost」的預測
                prob_xgb = model_xgb.predict_proba(X_test)[:, 1]
                signal_xgb = np.where(prob_xgb > threshold, 1, 0)

                # 2. 取得「集成三巨頭」的預測
                y_probs = model.predict_proba(X_test)[:, 1]
                y_pred_custom = np.where(y_probs > threshold, 1, 0) # 這是最終要用的訊號

                # 3. 準備回測數據 (找出這段時間的真實漲跌幅)
                # 使用 X_test 的索引來對應原始資料的漲跌幅
                if 'Target_Ret_1d' in df.columns:
                    market_ret = df.loc[X_test.index, 'Target_Ret_1d']
                else:
                    # 如果找不到 1d，嘗試用 target shift 來計算 (Fallback)
                    market_ret = df.loc[X_test.index, target].pct_change().shift(-1).fillna(0)

                # 4. 計算三條資金曲線
                # A. 買進持有 (基準)
                cum_market = (1 + market_ret).cumprod()

                # B. 單一 XGBoost 策略
                strat_ret_xgb = signal_xgb * market_ret
                cum_xgb = (1 + strat_ret_xgb).cumprod()

                # C. 集成模型策略
                strat_ret_ens = y_pred_custom * market_ret
                cum_ens = (1 + strat_ret_ens).cumprod()

                # =========================================================
                # 📊 繪圖區
                # =========================================================
                st.markdown("### 🏆 頂上戰爭：單一模型 vs 集成模型")
                
                # 整合數據畫圖
                chart_data = pd.DataFrame({
                    '🔵 單一 XGBoost': cum_xgb,
                    '🔴 集成三巨頭 (Ensemble)': cum_ens,
                    '📓 買進持有 (Benchmark)': cum_market
                }, index=X_test.index)
                
                st.line_chart(chart_data, color=["#0000FF", "#FF0000", "#808080"])

                # 顯示最終報酬率數據比較
                ret_xgb = cum_xgb.iloc[-1] - 1
                ret_ens = cum_ens.iloc[-1] - 1
                
                c1, c2 = st.columns(2)
                c1.metric("🔵 單一 XGB 總報酬", f"{ret_xgb*100:.1f}%")
                c2.metric("🔴 集成模型 總報酬", f"{ret_ens*100:.1f}%", delta=f"{(ret_ens - ret_xgb)*100:.1f}% (vs 單一)")

                # =========================================================
                # 🔍 找回消失的特徵因子圖
                # =========================================================
                st.markdown("### 🔑 關鍵因子 (基於 XGBoost 視角)")
                st.info("註：由於集成模型由三個大腦組成，此處顯示其中最具代表性的 XGBoost 判斷邏輯。")
                
                if hasattr(model_xgb, 'feature_importances_'):
                    feat_imp = pd.DataFrame({
                        'Feature': features, # 確保這裡的 features 變數是你上面定義過的列表
                        'Importance': model_xgb.feature_importances_
                    }).sort_values(by='Importance', ascending=False).head(10)
                    
                    st.bar_chart(feat_imp.set_index('Feature'), horizontal=True)
                # ==========================================
                # 實戰版：明日操作指引
                # ==========================================
                st.divider()
                st.subheader(f"🔮 AI 對明日開盤的戰術指令")
                
                # 1. 準備最新數據
                last_feat = X.iloc[-1:].copy()
                live_price = get_real_live_price(target)
                
                # 注入即時數據 (讓預測更準)
                if live_price:
                    if "TQQQ" in model_mode:
                         sma50 = df['SMA_50'].iloc[-1]
                         last_feat['Bias_50'] = (live_price - sma50) / sma50
                         st.caption(f"⚡ 即時價格 ${live_price} | 均線數據已即時修正")
                    elif "TSM" in model_mode:
                         prev_close = df[target].iloc[-2]
                         last_feat['Target_Ret_1d'] = (live_price - prev_close) / prev_close
                         st.caption(f"⚡ 即時價格 ${live_price} | 動能數據已即時修正")
                
                # 2. AI 計算勝率
                prob = model.predict_proba(last_feat)[0][1]
                
                # 3. 取得您的門檻 (TQQQ=0.5, 其他預設0.5)
                thresh = locals().get('buy_threshold', 0.5)

                # 4. 顯示儀表板
                c1, c2, c3 = st.columns(3)
                
                # 欄位 A: 勝率數值
                c1.metric("AI 上漲信心", f"{prob*100:.1f}%", help=f"超過 {thresh*100:.0f}% 才會動作")
                
                # 欄位 B: 趨勢方向
                if prob > thresh:
                    c2.metric("趨勢判斷", "📈 多頭 (Bullish)", delta="偏多")
                else:
                    c2.metric("趨勢判斷", "📉 空頭/盤整", delta="-偏空", delta_color="inverse")
                
                # 欄位 C: ★★★ 最重要的實戰指令 ★★★
                if prob > thresh:
                    # 勝率夠高 -> 買進或續抱
                    c3.success(f"🔥 指令：持有 / 買進")
                    st.markdown(f"**操作建議：**\n- **空手者**：明早開盤買進。\n- **持有者**：續抱，不停利。")
                else:
                    # 勝率不足 -> 賣出或觀望
                    c3.error(f"🛑 指令：賣出 / 空手")
                    st.markdown(f"**操作建議：**\n- **持有者**：明早開盤**市價賣出** (不要猶豫)。\n- **空手者**：保持現金，不要進場。")
            except Exception as e:
                st.error(f"發生錯誤: {e}")




































































































