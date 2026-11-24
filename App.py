import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from newsapi import NewsApiClient
from datetime import datetime, time
import pytz

# --- APP DESIGN ---
st.set_page_config(page_title="Gold Pro AI", page_icon="🪙", layout="centered")
st.markdown("""
<style>
    .stApp { background-color: #111827; color: white; }
    .card { background-color: #1F2937; padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #374151; }
    .title { color: #FFD700; font-size: 24px; font-weight: bold; }
    .subtitle { color: #9CA3AF; font-size: 14px; }
    .signal-buy { color: #10B981; font-size: 22px; font-weight: bold; }
    .signal-sell { color: #EF4444; font-size: 22px; font-weight: bold; }
    .signal-wait { color: #FFD700; font-size: 22px; font-weight: bold; }
    .level-box { background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px;}
    .level-title { color: #9CA3AF; font-size: 12px; }
    .level-val { color: white; font-size: 16px; font-weight: bold; }
    .timing-box { background-color: #1F2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; margin-bottom: 15px; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🪙 Gold Pro AI (MTF)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Timeframe & Correlation Analysis</div>', unsafe_allow_html=True)

# --- MARKET TIMING ---
def get_market_timing():
    now_utc = datetime.now(pytz.utc)
    ny_time = now_utc.astimezone(pytz.timezone('America/New_York'))
    if time(8, 30) <= ny_time.time() <= time(17, 0):
        return "🟢 US Market OPEN", "#10B981"
    elif time(3, 0) <= ny_time.time() < time(8, 30):
        return "🟡 London Session", "#FFD700"
    else:
        return "🔴 Asian Session (Low Vol)", "#EF4444"

timing_text, timing_color = get_market_timing()
st.markdown(f'<div class="timing-box"><span style="color: #9CA3AF;">Market Timing:</span> <span style="color: {timing_color}; font-weight: bold;">{timing_text}</span></div>', unsafe_allow_html=True)

# --- PRO AI LOGIC ---
@st.cache_data(ttl=300) 
def get_pro_signal():
    # 1. Main Data (Gold 15M)
    gold_data = yf.download("GC=F", period="60d", interval="15m", progress=False)
    
    # 2. Higher Timeframe (Gold 1H) for Trend
    gold_1h = yf.download("GC=F", period="60d", interval="1h", progress=False)
    
    # 3. Dollar Index (DXY) for Correlation
    dxy = yf.download("DX-Y.NYB", period="60d", interval="15m", progress=False)

    if gold_data.empty or gold_1h.empty or dxy.empty:
        return "NO DATA", 0, 0, 0, 0, 0, 0, 0

    # Handle MultiIndex
    for df in [gold_data, gold_1h, dxy]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # --- Calculate Indicators for Gold 15M ---
    gold_data['SMA_20'] = gold_data['Close'].rolling(window=20).mean()
    gold_data['SMA_50'] = gold_data['Close'].rolling(window=50).mean()
    
    delta = gold_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    gold_data['RSI'] = 100 - (100 / (1 + rs))
    gold_data.dropna(inplace=True)

    # --- Calculate 1H Trend ---
    gold_1h['SMA_50'] = gold_1h['Close'].rolling(window=50).mean()
    h1_trend = "UP" if gold_1h['Close'].iloc[-1] > gold_1h['SMA_50'].iloc[-1] else "DOWN"

    # --- Calculate DXY Trend ---
    dxy['SMA_20'] = dxy['Close'].rolling(window=20).mean()
    dxy_trend = "UP" if dxy['Close'].iloc[-1] > dxy['SMA_20'].iloc[-1] else "DOWN"

    # --- AI Model Training ---
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

    # --- FINAL PRO LOGIC (MTF + Correlation) ---
    direction = "WAIT"
    confidence = 0
    reason = "Market condition not clear."

    # Logic: Jo AI Buy bole, H1 trend Up hoy, ane Dollar Down hoy to j Buy
    if ml_signal == 1 and h1_trend == "UP" and dxy_trend == "DOWN":
        direction = "BUY"
        confidence = ml_prob
        reason = "AI Bullish + 1H Trend Up + Dollar Weak"
    # Logic: Jo AI Sell bole, H1 trend Down hoy, ane Dollar Up hoy to j Sell
    elif ml_signal == 0 and h1_trend == "DOWN" and dxy_trend == "UP":
        direction = "SELL"
        confidence = ml_prob
        reason = "AI Bearish + 1H Trend Down + Dollar Strong"
    else:
        direction = "WAIT"
        confidence = 0
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

    return direction, confidence, entry, sl, tp1, tp2, h1_trend, dxy_trend

# --- NEWS LOGIC ---
NEWS_API_KEY = 'fd513a4322ea48e2b9be96cd2ad2e3c6'
@st.cache_data(ttl=3600)
def get_news():
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        query = '"gold price" OR "XAUUSD" OR "Federal Reserve" OR "US inflation"'
        articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=10)
        filtered = []
        for art in articles['articles']:
            title = art['title']
            if title and "[Removed]" not in title and any(word in title.lower() for word in ['gold', 'fed', 'dollar', 'inflation']):
                filtered.append(art)
            if len(filtered) >= 3:
                break
        return filtered
    except:
        return []

# --- APP UI ---
with st.spinner('Pro AI Analysis (MTF + DXY)...'):
    direction, confidence, entry, sl, tp1, tp2, h1_trend, dxy_trend = get_pro_signal()
    news_articles = get_news()

# Main Signal Card
if direction == "NO DATA":
    st.error("Data currently unavailable. Please try again.")
else:
    color = '#10B981' if direction == 'BUY' else '#EF4444' if direction == 'SELL' else '#FFD700'
    
    card_html = f"""
    <div style="background-color: #1F2937; padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #374151; font-family: sans-serif;">
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 18px; font-weight: bold; color: white;">⚡ Pro Signal (M15 + H1 + DXY)</span>
        </div>
        
        <div style="margin-top: 15px; text-align: center;">
            <span style="color: #9CA3AF; font-size: 14px;">Current Price</span><br>
            <span style="color: #FFD700; font-size: 28px; font-weight: bold;">${entry}</span>
        </div>
        
        <div style="margin-top: 15px; background-color: #111827; padding: 15px; border-radius: 10px; text-align: center;">
            <span style="color: #9CA3AF; font-size: 14px;">AI Direction</span><br>
            <span style="color: {color}; font-size: 26px; font-weight: bold;">{direction}</span>
            <div style="color: #9CA3AF; font-size: 12px; margin-top: 5px;">{reason}</div>
        </div>
        
        <div style="display: flex; justify-content: space-between; gap: 10px; margin-top: 10px;">
            <div style="background-color: #111827; padding: 8px; border-radius: 8px; text-align: center; flex: 1;">
                <div style="color: #9CA3AF; font-size: 12px;">1H Trend</div>
                <div style="color: {'#10B981' if h1_trend == 'UP' else '#EF4444'}; font-size: 14px; font-weight: bold;">{h1_trend}</div>
            </div>
            <div style="background-color: #111827; padding: 8px; border-radius: 8px; text-align: center; flex: 1;">
                <div style="color: #9CA3AF; font-size: 12px;">Dollar (DXY)</div>
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
    """
    components.html(card_html, height=360)

# News Card
st.markdown('<div class="card"><div class="title" style="font-size: 18px;">📰 Live Market News</div><br>', unsafe_allow_html=True)
if news_articles:
    for article in news_articles:
        st.markdown(f"🔹 [{article['title']}]({article['url']})")
else:
    st.write("News nathi mali.")
st.markdown('</div>', unsafe_allow_html=True)
