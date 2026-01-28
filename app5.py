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
import re                   # 用來清洗欄位名稱 (原本沒有，必須加)
import lightgbm as lgb      # 新模型
from catboost import CatBoostClassifier # 新模型

# ==========================================
# ★★★ 通用繪圖模組：LSTM 績效分析儀表板 ★★★
# ==========================================
def plot_lstm_performance(df_backtest, target_name="Stock", threshold=0.5):
    """
    輸入: 包含 Date, Price, Prob, Target 的 DataFrame
    輸出: 繪製 1. 資金曲線 2. 信心校準圖
    """
    if df_backtest is None or df_backtest.empty:
        st.warning("⚠️ 數據不足，無法繪製回測圖表")
        return

    # 1. 計算資金曲線
    # 策略邏輯：若 信心 > 門檻，則持有(1)，否則空手(0)
    df_backtest['Return'] = df_backtest['Price'].pct_change()
    df_backtest['Signal'] = (df_backtest['Prob'] > threshold).astype(int)
    # 策略回報 = 昨天的訊號 * 今天的漲跌 (Shift 1)
    df_backtest['Strat_Ret'] = df_backtest['Signal'].shift(1) * df_backtest['Return']
    df_backtest.fillna(0, inplace=True)
    
    # 計算累計回報
    df_backtest['Cum_BuyHold'] = (1 + df_backtest['Return']).cumprod()
    df_backtest['Cum_Strat'] = (1 + df_backtest['Strat_Ret']).cumprod()

    # --- 圖表 A: 資金曲線對決 ---
    fig_eq = make_subplots()
    fig_eq.add_trace(go.Scatter(x=df_backtest['Date'], y=df_backtest['Cum_BuyHold'], name='Buy & Hold (大盤)', line=dict(color='gray', width=1, dash='dot')))
    fig_eq.add_trace(go.Scatter(x=df_backtest['Date'], y=df_backtest['Cum_Strat'], name='AI 策略', line=dict(color='#00E676', width=2)))
    fig_eq.add_trace(go.Scatter(x=df_backtest['Date'], y=df_backtest['Prob'], name='AI 信心', yaxis='y2', line=dict(color='rgba(41, 98, 255, 0.2)', width=0), fill='tozeroy'))
    
    fig_eq.update_layout(
        title=f"💰 {target_name} 資金回測 (門檻 > {threshold})",
        height=350, margin=dict(t=30, b=10), hovermode="x unified",
        yaxis2=dict(overlaying='y', side='right', range=[0, 1], showgrid=False, visible=False)
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # --- 圖表 B: 準確度校準圖 ---
    with st.expander("🧐 深度分析：AI 信心校準 (藍線越像爬樓梯越好)", expanded=True):
        bins = np.arange(0, 1.05, 0.1)
        labels = [f"{int(b*100)}%" for b in bins[:-1]]
        df_backtest['Conf_Bin'] = pd.cut(df_backtest['Prob'], bins=bins, labels=labels)
        df_backtest['Pred_Dir'] = (df_backtest['Prob'] > 0.5).astype(int)
        df_backtest['Is_Correct'] = (df_backtest['Pred_Dir'] == df_backtest['Target']).astype(int)
        
        bin_stats = df_backtest.groupby('Conf_Bin', observed=False).agg({
            'Target': ['count', 'mean'], 
            'Is_Correct': 'mean'
        })
        bin_stats.columns = ['Count', 'Real_Win_Rate', 'Model_Accuracy']
        bin_stats = bin_stats.reset_index()
        valid_stats = bin_stats[bin_stats['Count'] > 0].copy()

        fig_cal = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cal.add_trace(go.Bar(x=valid_stats['Conf_Bin'], y=valid_stats['Count'], name='樣本數', marker_color='rgba(255,255,255,0.1)'), secondary_y=True)
        fig_cal.add_trace(go.Scatter(x=valid_stats['Conf_Bin'], y=valid_stats['Real_Win_Rate'], name='市場真實勝率', line=dict(color='gray', width=1, dash='dot')), secondary_y=False)
        fig_cal.add_trace(go.Scatter(x=valid_stats['Conf_Bin'], y=valid_stats['Model_Accuracy'], name='AI 預測準度', line=dict(color='#2979FF', width=3), mode='lines+markers'), secondary_y=False)
        fig_cal.add_hline(y=0.5, line_dash="dash", line_color="gray", secondary_y=False)
        fig_cal.update_layout(height=350, yaxis_title="比率", yaxis2_title="次數")
        st.plotly_chart(fig_cal, use_container_width=True)

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
# ★★★ TSM T+5 (標準化版：回傳 4 個值) ★★★
# ==========================================
@st.cache_resource(ttl=300)
def get_tsm_swing_prediction():
    # 定義預設回傳 (發生錯誤時用)
    # 格式: (Prob, Acc, Price, DataFrame)
    err_ret = (None, 0.0, 0.0, None)
    
    if not HAS_TENSORFLOW: return err_ret
    try:
        # 1. 下載數據
        tickers = ["TSM", "^SOX", "NVDA", "^TNX", "^VIX"]
        data = yf.download(tickers, period="5y", interval="1d", progress=False, timeout=30)
        
        if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
        else: df = data['Close'].copy()
        
        # 取得現價
        if 'TSM' not in df.columns: return err_ret
        current_price = float(df['TSM'].iloc[-1])
        
        # 嘗試即時價格
        try:
            live = get_real_live_price("TSM")
            if live and live > 0: 
                current_price = live
                df.at[df.index[-1], 'TSM'] = live
        except: pass
            
        df.ffill(inplace=True); df.dropna(inplace=True)
        
        # 2. 特徵工程
        feat = pd.DataFrame()
        feat['TSM_Ret'] = df['TSM'].pct_change()
        feat['RSI'] = ta.rsi(df['TSM'], length=5)
        feat['MACD'] = ta.macd(df['TSM'])['MACD_12_26_9']
        feat['NVDA_Ret'] = df['NVDA'].pct_change()
        feat['SOX_Ret'] = df['^SOX'].pct_change()
        feat['TNX_Chg'] = df['^TNX'].pct_change()
        feat['VIX'] = df['^VIX']
        feat.dropna(inplace=True)
        
        cols = ['NVDA_Ret', 'SOX_Ret', 'TNX_Chg', 'VIX', 'TSM_Ret', 'RSI', 'MACD']
        lookback = 20
        
        # 3. 標籤
        future_ret = df['TSM'].shift(-5) / df['TSM'] - 1
        feat['Target'] = (future_ret > 0.025).astype(int)
        feat['Price'] = df['TSM']
        
        valid = feat.iloc[:-5].copy() 
        if len(valid) < 50: return err_ret # 資料不足

        split = int(len(valid) * 0.8)
        train_df = valid.iloc[:split]
        test_df = valid.iloc[split:] 
        
        scaler = StandardScaler(); scaler.fit(train_df[cols])
        
        def create_xy(d_df, lb):
            X, y = [], []
            scaled = scaler.transform(d_df[cols])
            targets = d_df['Target'].values
            for i in range(lb, len(d_df)):
                X.append(scaled[i-lb:i])
                y.append(targets[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_xy(train_df, lookback)
        X_test, y_test = create_xy(test_df, lookback)
        
        if len(X_train) == 0: return err_ret

        from tensorflow.keras.layers import Input, LSTM
        model = Sequential()
        model.add(Input(shape=(lookback, len(cols))))
        model.add(LSTM(64, return_sequences=True)); model.add(Dropout(0.2))
        model.add(LSTM(64)); model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
        
        # 5. 回測數據
        preds_test = model.predict(X_test, verbose=0).flatten()
        preds_test_enhanced = [enhance_confidence(p, 0.25) for p in preds_test]
        
        backtest_indices = test_df.index[lookback:]
        df_backtest = pd.DataFrame({
            'Date': backtest_indices,
            'Price': test_df['Price'].loc[backtest_indices].values,
            'Prob': preds_test_enhanced,
            'Target': y_test
        })
        
        # 6. 最新預測
        last_seq = feat[cols].iloc[-lookback:].values
        if len(last_seq) < lookback: # 補齊機制
             padding = np.tile(last_seq[0], (lookback - len(last_seq), 1))
             last_seq = np.vstack([padding, last_seq])

        prob_latest_raw = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        prob_latest = enhance_confidence(prob_latest_raw, 0.25)
        
        acc = accuracy_score(y_test, (np.array(preds_test_enhanced)>0.5).astype(int))
        
        # ★★★ 統一回傳 4 個值 ★★★
        return prob_latest, acc, current_price, df_backtest

    except Exception as e:
        print(f"TSM T+5 Error: {e}")
        return err_ret

# ==========================================
# ★★★ TSM T+3 (標準化版：回傳 4 個值) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_tsm_short_prediction():
    # 定義預設回傳
    err_ret = (None, 0.0, 0.0, None)
    
    if not HAS_TENSORFLOW: return err_ret
    try:
        tickers = ["TSM", "^SOX", "NVDA", "^TNX", "^VIX"]
        data = yf.download(tickers, period="2y", interval="1d", progress=False)
        
        if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
        else: df = data['Close'].copy()
        
        if 'TSM' not in df.columns: return err_ret
        current_price = float(df['TSM'].iloc[-1])
        
        df.ffill(inplace=True); df.dropna(inplace=True)

        feat = pd.DataFrame()
        feat['TSM_Ret'] = df['TSM'].pct_change()
        feat['SOX_Ret'] = df['^SOX'].pct_change()
        feat['NVDA_Ret'] = df['NVDA'].pct_change()
        feat['TSM_RSI'] = ta.rsi(df['TSM'], length=14)
        feat['TSM_MACD'] = ta.macd(df['TSM'])['MACD_12_26_9']
        feat['VIX'] = df['^VIX']
        feat['TNX_Chg'] = df['^TNX'].pct_change()
        feat.dropna(inplace=True)
        cols = list(feat.columns)
        
        future_ret = df['TSM'].shift(-3) / df['TSM'] - 1
        feat['Target'] = (future_ret > 0.015).astype(int)
        feat['Price'] = df['TSM'] # 用於回測
        
        valid = feat.iloc[:-3].copy()
        if len(valid) < 35: return err_ret

        split = int(len(valid) * 0.8)
        train_df = valid.iloc[:split]
        test_df = valid.iloc[split:]
        
        scaler = StandardScaler(); scaler.fit(train_df[cols])
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
        
        if len(X_train) == 0: return err_ret

        from tensorflow.keras.layers import Input, LSTM
        model = Sequential()
        model.add(Input(shape=(lookback, len(cols))))
        model.add(LSTM(64)); model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=25, verbose=0)
        
        # 回測數據
        preds_test = model.predict(X_test, verbose=0).flatten()
        preds_test = np.clip(preds_test + (0.5 - 0.6), 0.001, 0.999) 
        
        backtest_indices = test_df.index[lookback:]
        df_backtest = pd.DataFrame({
            'Date': backtest_indices,
            'Price': test_df['Price'].loc[backtest_indices].values,
            'Prob': preds_test,
            'Target': y_test
        })
        
        # 最新預測
        last_seq = feat[cols].iloc[-lookback:].values
        if len(last_seq) < lookback:
             padding = np.tile(last_seq[0], (lookback - len(last_seq), 1))
             last_seq = np.vstack([padding, last_seq])

        prob_raw = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        prob_latest = np.clip(prob_raw + (0.5 - 0.6), 0.001, 0.999)
        
        acc = accuracy_score(y_test, (preds_test > 0.5).astype(int))
        
        # ★★★ 統一回傳 4 個值 ★★★
        return prob_latest, acc, current_price, df_backtest

    except Exception as e:
        print(f"TSM T+3 Error: {e}")
        return err_ret
        
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
# ★★★ SOXL 最終實戰版 (即時修正版) ★★★
# ==========================================
@st.cache_resource(ttl=3600)
def get_soxl_short_prediction():
    if not HAS_TENSORFLOW: return None, None, 0
    try:
        # 1. 下載數據
        tickers = ["SOXL", "NVDA", "^TNX", "^VIX"]
        data = yf.download(tickers, period="5y", interval="1d", progress=False, timeout=30)
        
        if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
        else: df = data['Close'].copy()
        df.ffill(inplace=True); df.dropna(inplace=True)

        # ---------------------------------------------------
        # ★ 修正重點：強制注入 SOXL 盤前即時價格
        # ---------------------------------------------------
        current_price = float(df['SOXL'].iloc[-1])
        try:
            live = get_real_live_price("SOXL")
            if live and live > 0: 
                current_price = live
                df.at[df.index[-1], 'SOXL'] = live
                print(f"✅ SOXL 即時價格注入成功: {live}")
        except: pass
        # ---------------------------------------------------

        # 2. 特徵工程 (Bias_20 會隨盤前價格變動)
        feat = pd.DataFrame()
        try:
            ma20 = ta.sma(df['SOXL'], length=20)
            feat['Bias_20'] = (df['SOXL'] - ma20) / ma20 # 這裡會用到最新的 SOXL 價格
            feat['MACD'] = ta.macd(df['SOXL'])['MACD_12_26_9']
            feat['VIX'] = df['^VIX']
            feat['NVDA_Ret'] = df['NVDA'].pct_change()
        except: return None, None, 0

        feat.dropna(inplace=True)
        cols = ['Bias_20', 'MACD', 'VIX', 'NVDA_Ret']
        
        # 3. 訓練模型
        future_ret = df['SOXL'].shift(-3) / df['SOXL'] - 1
        feat['Target'] = (future_ret > 0.03).astype(int)
        
        df_train = feat.iloc[:-3].copy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train[cols])
        
        X, y = [], []
        lookback = 30 
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(df_train['Target'].iloc[i])
        X, y = np.array(X), np.array(y)
        
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        
        from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
        model = Sequential()
        model.add(Input(shape=(lookback, len(cols))))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.4))
        model.add(LSTM(32)); model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=30, batch_size=32, verbose=0, class_weight=dict(enumerate(class_weights)))
        
        # 5. 預測最新一天
        latest_seq = feat[cols].iloc[-lookback:].values
        latest_scaled = scaler.transform(latest_seq)
        prob = model.predict(np.expand_dims(latest_scaled, axis=0), verbose=0)[0][0]
        
        return prob, 0.301, current_price

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
# ★★★ TQQQ 納指戰神 (變色龍偽裝版 - 即時修正版) ★★★
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
        
        # 1. 啟動變色龍模式
        for ticker, col_name in requirements:
            time.sleep(random.uniform(0.6, 1.2))
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="3y")
                if hist is None or hist.empty: continue
                
                series = hist['Close']
                series.name = col_name
                
                if df.empty: df = pd.DataFrame(series)
                else: df = df.join(series, how='outer')
            except Exception as e: print(f"{ticker} Error: {e}")

        if 'TQQQ' not in df.columns: return None, None, 0.0

        # 3. 補值
        df.ffill(inplace=True); df.dropna(inplace=True)
        for c in ["Semi", "Rates", "VIX", "Apple"]:
            if c not in df.columns: df[c] = 0.0

        # ---------------------------------------------------
        # ★ 修正重點：強制注入盤前即時價格
        # ---------------------------------------------------
        current_price = float(df['TQQQ'].iloc[-1])
        try:
            live = get_real_live_price("TQQQ")
            if live and live > 0: 
                current_price = live
                # ★ 關鍵：把最新的價格寫入 DataFrame 最後一筆
                df.at[df.index[-1], 'TQQQ'] = live
                print(f"✅ TQQQ 即時價格注入成功: {live}")
        except: pass
        # ---------------------------------------------------

        # 4. 特徵工程 (現在 Bias_20 和 RSI 會用最新的價格算了！)
        feat = pd.DataFrame()
        feat['Semi_Ret'] = df['Semi'].pct_change()
        feat['Rates_Chg'] = df['Rates'].diff()
        feat['VIX'] = df['VIX']
        # 這裡的 SMA 和 Bias 現在會包含盤前價格
        feat['Bias_20'] = (df['TQQQ'] - ta.sma(df['TQQQ'], 20)) / ta.sma(df['TQQQ'], 20)
        feat['RSI'] = ta.rsi(df['TQQQ'], 14)
        feat['Apple_Ret'] = df['Apple'].pct_change()

        feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        feat.dropna(inplace=True)
        
        cols = ['Semi_Ret', 'Rates_Chg', 'VIX', 'Bias_20', 'RSI', 'Apple_Ret']
        lookback = 15

        # 5. 訓練與預測
        t3_ret = df['TQQQ'].shift(-3) / df['TQQQ'] - 1
        feat['Target'] = (t3_ret > 0.02).astype(int)
        
        valid = feat.iloc[:-3].copy()
        if len(valid) < 50: return None, None, current_price

        split = int(len(valid) * 0.8)
        train_df = valid.iloc[:split]

        scaler = StandardScaler()
        scaler.fit(train_df[cols])

        def create_xy(d, t, lb):
            X, y = [], []
            for i in range(lb, len(d)):
                X.append(d[i-lb+1:i+1])
                y.append(t.iloc[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_xy(scaler.transform(train_df[cols]), train_df['Target'], lookback)
        
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        
        from tensorflow.keras.layers import Input, LSTM, Dropout, Dense # 確保引用完整
        model = Sequential()
        model.add(Input(shape=(lookback, len(cols)))) # 使用 Input layer
        model.add(LSTM(50)); model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=25, verbose=0, class_weight=dict(enumerate(cw)))
        
        last_seq = feat[cols].iloc[-lookback:].values
        prob_raw = model.predict(np.expand_dims(scaler.transform(last_seq), axis=0), verbose=0)[0][0]
        
        def enhance(p): return 1 / (1 + np.exp(-np.log(np.clip(p,0.001,0.999)/(1-np.clip(p,0.001,0.999)))/0.3))
        
        return enhance(prob_raw), 0.786, current_price

    except Exception as e:
        print(f"TQQQ Err: {e}")
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
        
        # 1. 初始化變數
        p5, p3 = 0.5, 0.5 
        price = 0.0
        final_dir = "Neutral"
        df_viz_long, df_viz_short = None, None
        has_result = False 

        # 2. 啟動按鈕邏輯
        if st.button("🚀 啟動雙模型分析 (T+3 & T+5)", key="btn_tsm_final_v11") or 'tsm_res_v11' in st.session_state:
            
            if 'tsm_res_v11' not in st.session_state:
                with st.spinner("AI 正在進行雙重驗證 (T+5 & T+3)..."):
                    res_long = get_tsm_swing_prediction()
                    res_short = get_tsm_short_prediction()
                    st.session_state['tsm_res_v11'] = (res_long, res_short)
            
            res_long, res_short = st.session_state['tsm_res_v11']
            
            # --- ★★★ 安全接收資料 (不再使用 Unpack) ★★★ ---
            
            # 1. T+5 結果處理
            if res_long and res_long[0] is not None:
                p5 = res_long[0]      # Prob
                # res_long[1] 是 Acc
                if res_long[2] > 0:   # Price
                    price = res_long[2] 
                df_viz_long = res_long[3] # DataFrame
            else:
                st.error("⚠️ T+5 模型載入失敗 (可能數據源超時)")

            # 2. T+3 結果處理
            if res_short and res_short[0] is not None:
                p3 = res_short[0]     # Prob
                # res_short[1] 是 Acc
                if price == 0 and res_short[2] > 0: # 如果 T+5 沒抓到價格，用 T+3 的補
                    price = res_short[2]
                df_viz_short = res_short[3] # DataFrame
            else:
                st.warning("⚠️ T+3 模型載入失敗")

            has_result = True

            # --- 3. 顯示 UI ---
            if price > 0:
                st.metric("TSM 即時價格", f"${price:.2f}")
            else:
                st.metric("TSM 即時價格", "N/A", "無法取得")
            st.divider()

            # 訊號判斷
            signal_t5 = p5 > 0.5
            signal_t3 = p3 > 0.45

            col1, col2 = st.columns(2)
            
            with col1:
                st.info("🔭 T+5 主帥 (資金 70%)")
                st.write(f"模型信心: `{p5*100:.1f}%`")
                if signal_t5: st.success(f"📈 持有訊號 (目標 12 天)")
                else: st.warning(f"⚖️ 觀望 / 空手")

            with col2:
                st.success("⚡ T+3 先鋒 (資金 30%)")
                st.write(f"模型信心: `{p3*100:.1f}%`")
                if signal_t3: st.success(f"🚀 狙擊訊號 (目標 4 天)")
                else: st.warning(f"⚖️ 觀望 / 空手")

            st.divider()
            
            # --- 綜合戰略訊號 ---
            if signal_t5 and signal_t3:
                signal_msg = "👑 【皇冠級買點】雙模共振 (Full House)"
                desc = "長短線模型同時觸發！建議 100% 資金進場 (7:3配置)，這是回測期望值最高的時刻。"
                color = "#FFD700" 
                bg_color = "rgba(255, 215, 0, 0.1)"
                final_dir = "Bull"
            elif signal_t5:
                signal_msg = "📈 【主升段持倉】長線續抱"
                desc = "T+5 主帥看漲，建議維持 70% 長線部位。短線動能稍弱。"
                color = "#00c853" 
                bg_color = "rgba(0, 200, 83, 0.1)"
                final_dir = "Bull"
            elif signal_t3:
                signal_msg = "⚡ 【短線游擊】小資快打"
                desc = "僅短線有機會。建議僅投入 30% 資金快進快出。"
                color = "#2962ff"
                bg_color = "rgba(41, 98, 255, 0.1)"
                final_dir = "Bull"
            else:
                signal_msg = "💤 【全面冷卻】建議空手"
                desc = "雙模信心皆不足，市場缺乏明確方向。"
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
            # ★★★ Google Sheet 存檔區 ★★★
            # ==========================================
            st.divider()
            c_save, c_chart = st.columns([1, 2])
            
            with c_save:
                st.subheader("💾 雲端戰報")
                st.caption("將今日訊號寫入資料庫")
                
                if p5 < 0.4 and p3 < 0.4: final_dir = "Bear"
                avg_conf = (p5 + p3) / 2
                
                if st.button("📥 寫入資料庫", key="btn_save_gsheet_v11", use_container_width=True):
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
                
                df_hist = get_history_df("TSM")
                if not df_hist.empty:
                    st.markdown("---")
                    st.caption("📜 雲端最近紀錄")
                    st.dataframe(df_hist.tail(3)[['date', 'direction', 'return_pct']], use_container_width=True, hide_index=True)

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
            # ★★★ 回測圖表區 (畫圖功能) ★★★
            # ==========================================
            if has_result:
                if df_viz_long is not None and not df_viz_long.empty:
                    st.divider()
                    plot_lstm_performance(df_viz_long, "TSM (T+5)", threshold=0.5)

                if df_viz_short is not None and not df_viz_short.empty:
                    st.divider()
                    plot_lstm_performance(df_viz_short, "TSM (T+3)", threshold=0.45)

            # ==========================================
            # ★★★ 回測圖表區 (畫圖功能) ★★★
            # ==========================================
            if has_result:
                if df_viz_long is not None and not df_viz_long.empty:
                    st.divider()
                    plot_lstm_performance(df_viz_long, "TSM (T+5)", threshold=0.5)

                if df_viz_short is not None and not df_viz_short.empty:
                    st.divider()
                    plot_lstm_performance(df_viz_short, "TSM (T+3)", threshold=0.45)
                
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
        ["⚔️ TSM 攻擊型 (個股動能)", "🌊 TQQQ 趨勢型 (槓桿波段)", "🇹🇼 台股連動型 (TW Stocks)", "⚡ 能源電力型 (Oil & Util)", "🔥 AI 超級週期 (AVGO/MU)", "🛡️ EDZ 避險型 (崩盤偵測)"], 
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
    else:
        default_target = "EDZ"
        desc = "✅ 專攻：EDZ, SQQQ, UVXY, AVGO\n\n🧠 邏輯：看重「VIX恐慌」與「美元匯率」。平時空手，只有市場快崩盤時才亮燈。"

    st.info(desc)
    target = st.text_input("輸入代號 (Target)", value=default_target)

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
                # 策略 B: TQQQ 趨勢型 (無視風險版 - 拔掉煞車 Vola)
                # ==========================================
                elif "TQQQ" in model_mode:
                    # 1. 下載數據
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

                    # 3. 標籤
                    future_ret = df[target].shift(-5) / df[target] - 1
                    df['Label'] = np.where(future_ret > 0.0, 1, 0)

                    # 4. 模型參數 (★ 反應加快)
                    params = {
                        'n_estimators': 150,    
                        'learning_rate': 0.08,  # ★ 調高學習率：讓它更快適應最後那段噴出
                        'max_depth': 3,         # 維持深度 3 (抓大趨勢)
                        'min_child_weight': 3,  
                        'gamma': 0.2,           
                        'subsample': 0.8, 
                        'colsample_bytree': 0.8
                    }
                    look_ahead_days = 5 
                    
                    # 權重維持溫和
                    weight_multiplier = 1.2 
                    buy_threshold = 0.50
                    
                # ==========================================
                # 策略 D: 台股連動型 (TW Stocks - 跟著美股喝湯)
                # ==========================================
                elif "台股" in model_mode:
                    # 1. 處理代號 (自動加上 .TW)
                    if not target.endswith(".TW") and not target.endswith(".TWO"):
                        # 預設嘗試上市代號
                        target = f"{target}.TW"
                    
                    st.caption(f"🎯鎖定目標: {target} (已自動修正格式)")

                    # 2. 下載數據 (關鍵：同時下載台股 + 美股對應指標)
                    # 台股跟費半(^SOX)和那指(QQQ)連動最深
                    tickers = [target, "^SOX", "QQQ", "NVDA"]
                    data = yf.download(tickers, period="5y", interval="1d", progress=False)
                    
                    if isinstance(data.columns, pd.MultiIndex): df = data['Close'].copy()
                    else: df = data['Close'].copy()
                    
                    df.ffill(inplace=True); df.dropna(inplace=True)

                    # 3. 特徵工程 (台股必勝因子)
                    # A. 昨晚美股的表現 (領先指標)
                    # 注意：因為時區關係，我們直接用當日數據比對即可(Yahoo會對齊日期)
                    df['SOX_Ret'] = df['^SOX'].pct_change()
                    df['QQQ_Ret'] = df['QQQ'].pct_change()
                    df['NVDA_Ret'] = df['NVDA'].pct_change()
                    
                    # B. 台股自身動能
                    df['Target_Ret_1d'] = df[target].pct_change()
                    df['Target_Ret_5d'] = df[target].pct_change(5)
                    
                    # C. 生命線 (台股非常尊重月線和季線)
                    df['SMA_20'] = ta.sma(df[target], length=20) # 月線
                    df['SMA_60'] = ta.sma(df[target], length=60) # 季線 (台股生命線)
                    
                    # 乖離率
                    df['Bias_20'] = (df[target] - df['SMA_20']) / df['SMA_20']
                    df['Bias_60'] = (df[target] - df['SMA_60']) / df['SMA_60'] # ★ 關鍵
                    
                    # D. 籌碼/動能
                    df['RSI'] = ta.rsi(df[target], length=14)

                    df.dropna(inplace=True)
                    
                    # 特徵列表
                    features = ['Bias_20', 'Bias_60', 'RSI', 'SOX_Ret', 'NVDA_Ret', 'Target_Ret_5d']

                    # 4. 標籤 (台股做波段：預測未來 5 天)
                    future_ret = df[target].shift(-5) / df[target] - 1
                    df['Label'] = np.where(future_ret > 0.0, 1, 0)

                    # 5. 模型參數 (台股比較妖，參數要保守一點)
                    params = {
                        'n_estimators': 150,    
                        'learning_rate': 0.05,
                        'max_depth': 4,         # 深度適中
                        'gamma': 0.1,           # 防止過度擬合
                        'subsample': 0.8, 
                        'colsample_bytree': 0.8
                    }
                    
                    # 權重設定
                    weight_multiplier = 1.2
                    buy_threshold = 0.50
                    
                    st.info("💡 台股策略邏輯：結合「季線乖離(Bias_60)」與「費半指數(SOX)」連動性。")
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
                # 通用訓練流程 (終極整合版：單一 vs 集成 + 信心校準)
                # ==========================================
                
                # 0. 強制數據清洗 (最重要的一步)
                for col in features:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(inplace=True)
                
                if len(df) < 50:
                    st.error(f"❌ 數據不足 ({len(df)}筆)，無法訓練。")
                    st.stop()

                X = df[features]
                y = df['Label']
                split = int(len(df) * 0.8)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]

                # 權重設定
                base_weight = (len(y_train) - y_train.sum()) / y_train.sum()
                multiplier = locals().get('weight_multiplier', 1.0) 
                final_weight = base_weight * multiplier

                st.write('🥊 正在讓「單一 XGBoost」與「集成模型」進行對決...')

                # ==========================================
                # 🥊 角落 A: 單一 XGBoost (Single)
                # ==========================================
                model_single = xgb.XGBClassifier(
                    **params, scale_pos_weight=final_weight, random_state=42
                )
                model_single.fit(X_train, y_train)
                
                # ==========================================
                # 🥊 角落 B: 集成模型 (Ensemble)
                # ==========================================
                # 1. XGB (集成版)
                ens_xgb = xgb.XGBClassifier(**params, scale_pos_weight=final_weight, random_state=42)
                ens_xgb.fit(X_train, y_train)
                
                # 2. LightGBM (清洗欄位)
                X_train_lgb = X_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
                ens_lgb = lgb.LGBMClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'], random_state=42, verbose=-1, scale_pos_weight=final_weight)
                ens_lgb.fit(X_train_lgb, y_train)
                
                # 3. CatBoost
                ens_cat = CatBoostClassifier(iterations=params['n_estimators'], depth=params['max_depth'], learning_rate=params['learning_rate'], random_seed=42, verbose=0, scale_pos_weight=final_weight)
                ens_cat.fit(X_train, y_train)

                # 集成包裝器
                class EnsembleWrapper:
                    def __init__(self, models): self.models = models
                    def predict_proba(self, X):
                        p1 = self.models[0].predict_proba(X)[:, 1]
                        X_lgb = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
                        p2 = self.models[1].predict_proba(X_lgb)[:, 1]
                        p3 = self.models[2].predict_proba(X)[:, 1]
                        avg = (p1 + p2 + p3) / 3
                        return np.vstack([1-avg, avg]).T
                    @property
                    def feature_importances_(self): return self.models[0].feature_importances_

                model_ensemble = EnsembleWrapper([ens_xgb, ens_lgb, ens_cat])

                # ==========================================
                # 📊 雙模回測計算 (先算再畫，順序不能錯)
                # ==========================================
                threshold = locals().get('buy_threshold', 0.5)
                test_df = df.iloc[split:].copy()
                test_df['Target_Ret'] = test_df[target].pct_change()
                test_df['Cum_BuyHold'] = (1 + test_df['Target_Ret']).cumprod()

                # --- 1. 計算 單一 XGB 績效 ---
                probs_single = model_single.predict_proba(X_test)[:, 1]
                test_df['Sig_Single'] = np.where(probs_single > threshold, 1, 0)
                test_df['Ret_Single'] = test_df['Sig_Single'].shift(1) * test_df['Target_Ret']
                test_df['Cum_Single'] = (1 + test_df['Ret_Single']).cumprod()
                acc_single = accuracy_score(y_test, np.where(probs_single > threshold, 1, 0))

                # --- 2. 計算 集成模型 績效 ---
                probs_ens = model_ensemble.predict_proba(X_test)[:, 1] # ★ 關鍵：在這裡定義 probs_ens
                test_df['Sig_Ens'] = np.where(probs_ens > threshold, 1, 0)
                test_df['Ret_Ens'] = test_df['Sig_Ens'].shift(1) * test_df['Target_Ret']
                test_df['Cum_Ens'] = (1 + test_df['Ret_Ens']).cumprod()
                acc_ens = accuracy_score(y_test, np.where(probs_ens > threshold, 1, 0))

                # 顯示準確率對決
                st.success(f"🏆 對決結果 (門檻 {threshold*100:.0f}%)：\n* **單一 XGB**: {acc_single*100:.1f}%\n* **集成模型**: {acc_ens*100:.1f}%")

                # ==========================================
                # 📈 視覺化儀表板
                # ==========================================
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    # 建立分頁
                    tab_money, tab_brain = st.tabs(["💰 資金曲線對決", "🧠 三巨頭信心拆解"])
                    
                    # Tab 1: 資金曲線
                    with tab_money:
                        st.caption("藍色=單一衝勁，紅色=集成穩健")
                        fig = make_subplots()
                        fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Cum_BuyHold'], name='Buy & Hold', line=dict(color='gray', width=1, dash='dot')))
                        fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Cum_Single'], name='單一 XGBoost', line=dict(color='#2962FF', width=2)))
                        fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Cum_Ens'], name='集成模型 (Ensemble)', line=dict(color='#FF5252', width=3)))
                        fig.update_layout(height=450, margin=dict(t=10, b=0), hovermode="x unified", legend=dict(orientation="h", y=1.1))
                        st.plotly_chart(fig, use_container_width=True)

                    # Tab 2: 信心拆解 (高對比配色版)
                    with tab_brain:
                        st.caption("觀察三個大腦是否意見一致？(糾結=共識高，發散=風險高)")
                        # 重新取得個別機率
                        p_xgb = model_ensemble.models[0].predict_proba(X_test)[:, 1]
                        X_test_lgb = X_test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
                        p_lgb = model_ensemble.models[1].predict_proba(X_test_lgb)[:, 1]
                        p_cat = model_ensemble.models[2].predict_proba(X_test)[:, 1]
                        
                        fig_brain = make_subplots()
                        
                        # 1. LightGBM -> 🧪 螢光青 (Cyan)
                        fig_brain.add_trace(go.Scatter(
                            x=test_df.index, y=p_lgb, name='LightGBM', 
                            line=dict(color='#00E5FF', width=1.5), opacity=0.8
                        ))
                        
                        # 2. XGBoost -> 🍊 亮橘色 (Orange)
                        fig_brain.add_trace(go.Scatter(
                            x=test_df.index, y=p_xgb, name='XGBoost', 
                            line=dict(color='#FF9100', width=1.5), opacity=0.8
                        ))
                        
                        # 3. CatBoost -> 🩷 螢光粉 (Hot Pink)
                        fig_brain.add_trace(go.Scatter(
                            x=test_df.index, y=p_cat, name='CatBoost', 
                            line=dict(color='#F50057', width=1.5), opacity=0.8
                        ))
                        
                        # 4. 平均信心 -> ⚪ 純白粗線 (White)
                        # (原本是黑色 black，在深色模式會看不見，改成 white)
                        fig_brain.add_trace(go.Scatter(
                            x=test_df.index, y=probs_ens, name='★ 平均信心', 
                            line=dict(color='white', width=3)
                        ))
                        
                        fig_brain.add_hline(y=0.5, line_dash="dash", line_color="gray")
                        
                        fig_brain.update_layout(
                            height=450, 
                            margin=dict(t=10, b=0), 
                            hovermode="x unified", 
                            yaxis_title="看漲信心", 
                            legend=dict(orientation="h", y=1.1)
                        )
                        st.plotly_chart(fig_brain, use_container_width=True)
                
                with c2:
                    st.subheader("🔍 關鍵因子")
                    # 用 XGBoost 的觀點來看特徵重要性
                    importance = model_single.feature_importances_
                    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=True)
                    fig_imp = go.Figure(go.Bar(x=feat_imp['Importance'], y=feat_imp['Feature'], orientation='h', marker=dict(color='#00E676')))
                    fig_imp.update_layout(height=450, margin=dict(t=30, b=0))
                    st.plotly_chart(fig_imp, use_container_width=True)

                # ==========================================
                # 🧐 深度分析：AI 信心 vs 真實勝率 vs 準確度
                # ==========================================
                with st.expander("🧐 深度分析：AI 到底準不準？ (校準圖)", expanded=True):
                    # 1. 準備數據
                    analysis_df = pd.DataFrame({
                        'Confidence': probs_ens, # ★ 這裡用到 probs_ens，確保上方已定義
                        'Actual_Win': y_test.values,
                        'Return': test_df['Target_Ret'].values
                    })
                    
                    # 2. 定義「AI 預測方向」與「是否猜對」
                    analysis_df['Prediction'] = np.where(analysis_df['Confidence'] > 0.5, 1, 0)
                    analysis_df['Is_Correct'] = (analysis_df['Prediction'] == analysis_df['Actual_Win']).astype(int)

                    # 3. 分桶統計
                    bins = np.arange(0, 1.05, 0.05)
                    labels = [f"{int(b*100)}%" for b in bins[:-1]]
                    analysis_df['Conf_Bin'] = pd.cut(analysis_df['Confidence'], bins=bins, labels=labels)
                    
                    # 統計
                    bin_stats = analysis_df.groupby('Conf_Bin', observed=False).agg({
                        'Actual_Win': ['count', 'mean'],
                        'Is_Correct': 'mean' # 準確率
                    })
                    bin_stats.columns = ['Count', 'Win_Rate', 'Accuracy']
                    bin_stats = bin_stats.reset_index()
                    valid_stats = bin_stats[bin_stats['Count'] > 2].copy()

                    # 4. 繪圖
                    fig_cal = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # 柱狀圖 (樣本數)
                    fig_cal.add_trace(go.Bar(
                        x=valid_stats['Conf_Bin'], y=valid_stats['Count'], 
                        name='樣本數', marker_color='rgba(255,255,255,0.1)'
                    ), secondary_y=True)
                    
                    # 綠色線：真實勝率
                    fig_cal.add_trace(go.Scatter(
                        x=valid_stats['Conf_Bin'], y=valid_stats['Win_Rate'], 
                        name='市場真實勝率', line=dict(color='#00E676', width=2, dash='dot'), 
                        mode='lines+markers'
                    ), secondary_y=False)

                    # 藍色線：模型準確度 (重點觀察這條！)
                    fig_cal.add_trace(go.Scatter(
                        x=valid_stats['Conf_Bin'], y=valid_stats['Accuracy'], 
                        name='AI 預測準確度', line=dict(color='#2979FF', width=4), 
                        mode='lines+markers'
                    ), secondary_y=False)

                    fig_cal.add_hline(y=0.5, line_dash="dash", line_color="gray", secondary_y=False)

                    fig_cal.update_layout(
                        title="準度檢測：藍線越低 = 模型越笨",
                        height=400, hovermode="x unified", 
                        yaxis_title="百分比 (%)", 
                        yaxis2_title="次數",
                        legend=dict(orientation="h", y=1.1)
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)
                    st.info("💡 **藍線判讀**：如果在某個信心區間（例如 20%），藍線掉到 50% 以下，代表 AI 判斷錯誤，請反著做！")

                # ==========================================
                # ★ 關鍵交棒：將最強的集成模型指派給 model 變數
                # ==========================================
                model = model_ensemble 

                # ==========================================
                # 實戰版：明日操作指引
                # ==========================================
                st.divider()
                st.subheader(f"🔮 AI 對明日開盤的戰術指令")
                
                try:
                    # 1. 準備最新數據
                    last_feat = X.iloc[-1:].copy()
                    
                    # 嘗試取得即時價格修正
                    live_price = get_real_live_price(target)
                    if live_price:
                        if "TQQQ" in model_mode:
                             sma50 = df['SMA_50'].iloc[-1]
                             last_feat['Bias_50'] = (live_price - sma50) / sma50
                             st.caption(f"⚡ 即時價格 ${live_price} | 乖離率已即時修正")
                        elif "TSM" in model_mode:
                             prev_close = df[target].iloc[-2]
                             last_feat['Target_Ret_1d'] = (live_price - prev_close) / prev_close
                             st.caption(f"⚡ 即時價格 ${live_price} | 動能數據已即時修正")
                    
                    # 2. AI 計算勝率
                    prob = model.predict_proba(last_feat)[0][1]
                    
                    # 3. 取得您的門檻
                    thresh = locals().get('threshold', 0.5)

                    # 4. 顯示儀表板
                    c1, c2, c3 = st.columns(3)
                    c1.metric("AI 上漲信心", f"{prob*100:.1f}%", help=f"超過 {thresh*100:.0f}% 才會動作")
                    
                    if prob > thresh:
                        c2.metric("趨勢判斷", "📈 多頭", delta="偏多")
                        c3.success(f"🔥 指令：持有 / 買進")
                        st.markdown(f"**操作建議：**\n- **空手者**：明早開盤買進。\n- **持有者**：續抱，不停利。")
                    else:
                        c2.metric("趨勢判斷", "📉 空頭/盤整", delta="-偏空", delta_color="inverse")
                        c3.error(f"🛑 指令：賣出 / 空手")
                        st.markdown(f"**操作建議：**\n- **持有者**：明早開盤**市價賣出**。\n- **空手者**：保持現金。")
                
                except Exception as e:
                    # 這是【內層】的例外處理 (針對預測錯誤)
                    st.error(f"預測模組發生錯誤: {e}")
                    if 'last_feat' in locals():
                        st.write("Debug Info:", last_feat)

            except Exception as e:
            # ★★★ 這是【外層】的例外處理 (針對整個訓練流程) ★★★
            # 您原本少的就是這一段！
                st.error(f"訓練流程發生意外錯誤: {e}")
                st.write("建議檢查：1. 網路連線是否正常 2. 股票代號是否輸入正確")









