import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from newsapi import NewsApiClient
from datetime import datetime, time
import pytz
from streamlit_autorefresh import st_autorefresh
import requests

# --- AUTO REFRESH (5 Minutes) ---
st_autorefresh(interval=5 * 60 * 1000, limit=None, key="auto_refresh")

TWELVE_DATA_KEY = '4c470e5fed854bbaa4b4c3a38a11499e'
NEWS_API_KEY = 'fd513a4322ea48e2b9be96cd2ad2e3c6'

st.set_page_config(page_title="Gold AI", page_icon="🪙", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #111827; color: white; }
    .card { background-color: #1F2937; padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #374151; }
    .title { color: #FFD700; font-size: 24px; font-weight: bold; }
    .subtitle { color: #9CA3AF; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🪙 Gold AI (Pro)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Live Signals & Analysis</div>', unsafe_allow_html=True)

# --- 1. MARKET SESSIONS ---
def get_market_sessions():
    now_utc = datetime.now(pytz.utc)
    utc_time = now_utc.time()
    
    sessions = {
        "🇦🇺 Sydney": (time(21, 0), time(6, 0)),
        "🇯🇵 Tokyo": (time(0, 0), time(9, 0)),
        "🇬🇧 London": (time(7, 0), time(16, 0)),
        "🇺🇸 New York": (time(12, 0), time(21, 0))
    }
    
    result = []
    for name, (start, end) in sessions.items():
        if start < end: 
            is_open = start <= utc_time <= end
        else: 
            is_open = utc_time >= start or utc_time <= end
            
        color = "#10B981" if is_open else "#EF4444"
        status = "OPEN 🟢" if is_open else "CLOSED 🔴"
        result.append(f'<div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span>{name}</span><span style="color: {color}; font-weight: bold;">{status}</span></div>')
    return "".join(result)

sessions_html = get_market_sessions()
st.markdown(f'<div class="card"><div class="title" style="font-size: 16px;">🌍 Market Sessions</div>{sessions_html}</div>', unsafe_allow_html=True)

# --- DATA FETCH ---
@st.cache_data(ttl=300) 
def fetch_twelvedata(symbol, interval, outputsize):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_DATA_KEY}"
    try:
        response = requests.get(url, timeout=15)
        data = response.json()
        if data.get("status") == "error" or "values" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except:
        return pd.DataFrame()

# --- AI SIGNAL ---
@st.cache_data(ttl=300) 
def get_pro_signal():
    try:
        gold_data = fetch_twelvedata("XAU/USD", "15min", 500)
        gold_1h = fetch_twelvedata("XAU/USD", "1h", 200)
        dxy = fetch_twelvedata("DXY", "15min", 500)

        if gold_data.empty or gold_1h.empty or dxy.empty:
            return "WAIT", 0, 0, 0, 0, 0, "N/A", "N/A", "Data Error / Market Closed"

        gold_data['SMA_20'] = gold_data['Close'].rolling(window=20).mean()
        gold_data['SMA_50'] = gold_data['Close'].rolling(window=50).mean()
        
        delta = gold_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        gold_data['RSI'] = 100 - (100 / (1 + rs))
        gold_data.dropna(inplace=True)

        if len(gold_data) < 50: 
            return "WAIT", 0, 0, 0, 0, 0, "N/A", "N/A", "Not enough data"

        gold_1h['SMA_50'] = gold_1h['Close'].rolling(window=50).mean()
        h1_trend = "UP" if gold_1h['Close'].iloc[-1] > gold_1h['SMA_50'].iloc[-1] else "DOWN"

        dxy['SMA_20'] = dxy['Close'].rolling(window=20).mean()
        dxy_trend = "UP" if dxy['Close'].iloc[-1] > dxy['SMA_20'].iloc[-1] else "DOWN"

        gold_data['Target'] = np.where(gold_data['Close'].shift(-1) > gold_data['Close'], 1, 0)
        gold_data.dropna(inplace=True)
        
        X = gold_data[['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50', 'RSI']]
        Y = gold_data['Target']
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, Y_train)
        
        latest = X.tail(1)
        ml_signal = model.predict(latest)[0]
        ml_prob = model.predict_proba(latest)[0][ml_signal] * 100

        direction = "WAIT"
        confidence = 0
        reason = "Market condition not clear."

        if ml_signal == 1 and h1_trend == "UP" and dxy_trend == "DOWN":
            direction = "BUY"
            confidence = ml_prob
            reason = "AI Bullish + 1H Trend Up + Dollar Weak"
        elif ml_signal == 0 and h1_trend == "DOWN" and dxy_trend == "UP":
            direction = "SELL"
            confidence = ml_prob
            reason = "AI Bearish + 1H Trend Down + Dollar Strong"
        else:
            reason = f"No Alignment (AI: {'BUY' if ml_signal==1 else 'SELL'}, 1H: {h1_trend}, DXY: {dxy_trend})"

        entry = round(float(latest['Close'].iloc[0]), 2)
        
        if direction == "BUY":
            sl = round(entry - 3.0, 2)
            tp1 = round(entry + 4.0, 2)
            tp2 = round(entry + 6.0, 2)
        elif direction == "SELL":
            sl = round(entry + 3.0, 2)
            tp1 = round(entry - 4.0, 2)
            tp2 = round(entry - 6.0, 2)
        else:
            sl, tp1, tp2 = 0, 0, 0

        return direction, confidence, entry, sl, tp1, tp2, h1_trend, dxy_trend, reason
    except Exception:
        return "WAIT", 0, 0, 0, 0, 0, "N/A", "N/A", "System Error"

# --- FETCH DATA ---
with st.spinner('Analysis...'):
    direction, confidence, entry, sl, tp1, tp2, h1_trend, dxy_trend, reason = get_pro_signal()

# --- LIVE CHART ---
live_chart_html = """
<div class="tradingview-widget-container" style="height:200px;width:100%">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
  {"symbol": "OANDA:XAUUSD", "width": "100%", "height": "100%", "locale": "en", "dateRange": "1D", "colorTheme": "dark", "isTransparent": false, "autosize": true}
  </script>
</div>
"""
components.html(live_chart_html, height=200)

# --- SIGNAL CARD ---
color = '#10B981' if direction == 'BUY' else '#EF4444' if direction == 'SELL' else '#FFD700'

st.markdown(f"""
<div class="card">
    <div style="display: flex; justify-content: space-between;"><span style="font-size: 18px; font-weight: bold; color: white;">⚡ AI Signal</span></div>
    <div style="margin-top: 15px; background-color: #111827; padding: 15px; border-radius: 10px; text-align: center;">
        <span style="color: #9CA3AF; font-size: 14px;">Direction</span><br>
        <span style="color: {color}; font-size: 26px; font-weight: bold;">{direction}</span>
        <div style="color: #9CA3AF; font-size: 12px; margin-top: 5px;">{reason}</div>
    </div>
    <div style="display: flex; justify-content: space-between; gap: 10px; margin-top: 10px;">
        <div style="background-color: #111827; padding: 8px; border-radius: 8px; text-align: center; flex: 1;">
            <div style="color: #9CA3AF; font-size: 12px;">1H Trend</div>
            <div style="color: {'#10B981' if h1_trend == 'UP' else '#EF4444'}; font-size: 14px; font-weight: bold;">{h1_trend}</div>
        </div>
        <div style="background-color: #111827; padding: 8px; border-radius: 8px; text-align: center; flex: 1;">
            <div style="color: #9CA3AF; font-size: 12px;">Dollar</div>
            <div style="color: {'#10B981' if dxy_trend == 'UP' else '#EF4444'}; font-size: 14px; font-weight: bold;">{dxy_trend}</div>
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; gap: 10px; margin-top: 10px;">
        <div style="background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; flex: 1;">
            <div style="color: #9CA3AF; font-size: 12px;">SL</div>
            <div style="color: #EF4444; font-size: 16px; font-weight: bold;">{sl if sl != 0 else '--'}</div>
        </div>
        <div style="background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; flex: 1;">
            <div style="color: #9CA3AF; font-size: 12px;">TP 1</div>
            <div style="color: #10B981; font-size: 16px; font-weight: bold;">{tp1 if tp1 != 0 else '--'}</div>
        </div>
        <div style="background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; flex: 1;">
            <div style="color: #9CA3AF; font-size: 12px;">TP 2</div>
            <div style="color: #10B981; font-size: 16px; font-weight: bold;">{tp2 if tp2 != 0 else '--'}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- NEWS ---
@st.cache_data(ttl=3600)
def get_news():
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_everything(q='"gold price" OR "XAUUSD"', language='en', sort_by='relevancy', page_size=10)
        filtered = []
        for art in articles['articles']:
            title = art['title']
            if title and any(word in title.lower() for word in ['gold', 'fed', 'dollar']):
                sentiment = "Neutral ⚖️"
                color = "#9CA3AF"
                if any(w in title.lower() for w in ['rise', 'surge', 'high', 'inflation']): 
                    sentiment = "Bullish 📈"
                    color = "#10B981"
                elif any(w in title.lower() for w in ['fall', 'drop', 'low', 'loss']): 
                    sentiment = "Bearish 📉"
                    color = "#EF4444"
                
                filtered.append({
                    "title": title, 
                    "url": art['url'], 
                    "sentiment": sentiment, 
                    "color": color
                })
            if len(filtered) >= 3: 
                break
        return filtered
    except: 
        return []

news_articles = get_news()
st.markdown('<div class="card"><div class="title" style="font-size: 18px;">📰 Market News</div><br>', unsafe_allow_html=True)
if news_articles:
    for article in news_articles:
        st.markdown(f"🔹 **[{article['title']}]({article['url']})**")
        st.markdown(f"<span style='color:{article['color']}; font-size:12px; font-weight:bold;'>{article['sentiment']}</span><br>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
