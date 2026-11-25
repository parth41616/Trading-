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

# --- TWELVE DATA API KEY ---
TWELVE_DATA_KEY = '4c470e5fed854bbaa4b4c3a38a11499e'

# --- APP DESIGN ---
st.set_page_config(page_title="Gold AI Ultimate", page_icon="🪙", layout="centered")
st.markdown("""
<style>
    .stApp { background-color: #111827; color: white; }
    .card { background-color: #1F2937; padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #374151; }
    .title { color: #FFD700; font-size: 24px; font-weight: bold; }
    .subtitle { color: #9CA3AF; font-size: 14px; }
    .timing-box { background-color: #1F2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; margin-bottom: 15px; text-align: center;}
    .session-row { display: flex; justify-content: space-between; margin-bottom: 8px; padding: 5px; border-bottom: 1px solid #374151; }
    .mtf-box { background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; flex: 1; margin: 0 2px;}
    .fib-box { background-color: #111827; padding: 5px; border-radius: 5px; text-align: center; margin: 2px; font-size: 13px;}
    .chat-box { background-color: #111827; padding: 15px; border-radius: 10px; margin-top: 10px; border: 1px solid #FFD700;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🪙 Gold AI (Ultimate Pro)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">All-in-One Trading Assistant</div>', unsafe_allow_html=True)

# --- 1. MARKET SESSIONS (Live Global Time) ---
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
        result.append(f'<div class="session-row"><span>{name}</span><span style="color: {color}; font-weight: bold;">{status}</span></div>')
    return "".join(result)

sessions_html = get_market_sessions()
st.markdown(f'<div class="timing-box"><div style="color: #FFD700; font-weight: bold; margin-bottom: 10px;">🌍 Global Forex Sessions (Live)</div>{sessions_html}</div>', unsafe_allow_html=True)

# --- DATA FETCH FUNCTION ---
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

# --- AI MAIN SIGNAL ---
@st.cache_data(ttl=300) 
def get_pro_signal():
    try:
        gold_data = fetch_twelvedata("XAU/USD", "15min", 500)
        gold_1h = fetch_twelvedata("XAU/USD", "1h", 200)
        dxy = fetch_twelvedata("DXY", "15min", 500)

        if gold_data.empty or gold_1h.empty or dxy.empty:
            return "WAIT", 0, 0, 0, 0, 0, "N/A", "N/A", "API Limit Reached / Market Closed"

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

        return direction, confidence, entry, sl, tp1, tp2, h1_trend, dxy_trend, reason
    except Exception as e:
        return "WAIT", 0, 0, 0, 0, 0, "N/A", "N/A", "System Error"

# --- 2. MTF TRENDS ---
@st.cache_data(ttl=300)
def get_mtf_trends():
    timeframes = {"5min": "5M", "15min": "15M", "1h": "1H", "4h": "4H"}
    trends = {}
    for tf, label in timeframes.items():
        df = fetch_twelvedata("XAU/USD", tf, 50)
        if not df.empty and len(df) >= 20:
            sma = df['Close'].rolling(window=20).mean()
            trends[label] = "BUY" if df['Close'].iloc[-1] > sma.iloc[-1] else "SELL"
        else:
            trends[label] = "N/A"
    return trends

# --- 3. SENTIMENT METER (RSI based) ---
@st.cache_data(ttl=300)
def get_sentiment():
    df = fetch_twelvedata("XAU/USD", "15min", 50)
    if df.empty: 
        return 50, 50
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    return round(current_rsi, 1), round(100 - current_rsi, 1)

# --- 4. VOLUME PROFILE ---
@st.cache_data(ttl=300)
def get_volume_profile():
    df = fetch_twelvedata("XAU/USD", "15min", 200)
    if df.empty: 
        return []
    
    price_min, price_max = df['Low'].min(), df['High'].max()
    bins = np.linspace(price_min, price_max, 11)
    
    df['Bin'] = pd.cut(df['Close'], bins=bins, labels=False)
    vol_profile = df.groupby('Bin')['Volume'].sum().reset_index()
    
    top_volumes = vol_profile.nlargest(2, 'Volume')
    result = []
    for idx, row in top_volumes.iterrows():
        bin_idx = int(row['Bin'])
        price_level = (bins[bin_idx] + bins[bin_idx+1]) / 2
        result.append(round(price_level, 2))
    
    result.sort()
    return result

# --- AI CHATBOT LOGIC ---
def get_ai_chatbot_response(query, direction, confidence, h1_trend, dxy_trend, entry):
    query = query.lower()
    if "buy" in query or "sell" in query or "trade" in query or "levu" in query:
        if direction == "BUY": 
            return f"AI keh chhe ke BUY karo. Entry: {entry}, Confidence: {confidence:.1f}%. 1H Trend UP chhe."
        elif direction == "SELL": 
            return f"AI keh chhe ke SELL karo. Entry: {entry}, Confidence: {confidence:.1f}%. 1H Trend DOWN chhe."
        else: 
            return f"Abhi WAIT karo. AI ane trend ma takkar chhe ({h1_trend} vs {dxy_trend}). Risk na lo."
    elif "news" in query or "khabar" in query:
        return "News section ma jao, tya latest Bullish/Bearish news dekhai jashe."
    elif "risk" in query or "lot" in query:
        return "Risk Calculator ma tamaru Balance ane Risk % nakho, e jate j lot size batavshe."
    else:
        return "Hu tamaro AI assistant chhu. Mane puchho: 'Trade levu?', 'News su chhe?', ke 'Lot size ketlo?'"

# --- APP UI START ---
with st.spinner('Data Analysis...'):
    direction, confidence, entry, sl, tp1, tp2, h1_trend, dxy_trend, reason = get_pro_signal()
    mtf_trends = get_mtf_trends()
    buy_sentiment, sell_sentiment = get_sentiment()
    vol_levels = get_volume_profile()

# LIVE PRICE WIDGET
live_chart_html = """
<div class="tradingview-widget-container" style="height:200px;width:100%">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
  {"symbol": "OANDA:XAUUSD", "width": "100%", "height": "100%", "locale": "en", "dateRange": "1D", "colorTheme": "dark", "isTransparent": false, "autosize": true}
  </script>
</div>
"""
components.html(live_chart_html, height=200)

# Main Signal Card
color = '#10B981' if direction == 'BUY' else '#EF4444' if direction == 'SELL' else '#FFD700'
card_html = f"""
<div style="background-color: #1F2937; padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #374151; font-family: sans-serif;">
    <div style="display: flex; justify-content: space-between;"><span style="font-size: 18px; font-weight: bold; color: white;">⚡ Live AI Signal</span></div>
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
components.html(card_html, height=320)

# --- MTF DASHBOARD ---
mtf_html = '<div style="display: flex; justify-content: space-between; margin-bottom: 15px;">'
for tf, trend in mtf_trends.items():
    color = "#10B981" if trend == "BUY" else "#EF4444" if trend == "SELL" else "#9CA3AF"
    mtf_html += f'<div class="mtf-box"><div style="color:#9CA3AF; font-size:12px;">{tf}</div><div style="color:{color}; font-weight:bold;">{trend}</div></div>'
mtf_html += '</div>'
st.markdown(f'<div class="card"><div class="title" style="font-size: 16px;">📊 Multi-Timeframe</div>{mtf_html}</div>', unsafe_allow_html=True)

# --- SENTIMENT METER ---
st.markdown(f"""
<div class="card">
    <div class="title" style="font-size: 16px;">⚖️ Market Sentiment (RSI)</div>
    <div style="margin-top: 15px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="color: #10B981; font-weight: bold;">Buyers: {buy_sentiment}%</span>
            <span style="color: #EF4444; font-weight: bold;">Sellers: {sell_sentiment}%</span>
        </div>
        <div style="background-color: #EF4444; border-radius: 10px; height: 15px; width: 100%; display: flex; overflow: hidden;">
            <div style="background-color: #10B981; height: 100%; width: {buy_sentiment}%;"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- VOLUME PROFILE ---
if vol_levels and len(vol_levels) >= 2:
    st.markdown(f"""
    <div class="card">
        <div class="title" style="font-size: 16px;">🔊 High Volume Zones (S/R)</div>
        <div style="margin-top: 10px; display: flex; justify-content: space-between;">
            <div style="background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; flex: 1; margin-right: 5px;">
                <div style="color: #9CA3AF; font-size: 12px;">Strong Support</div>
                <div style="color: #10B981; font-size: 16px; font-weight: bold;">${vol_levels[0]}</div>
            </div>
            <div style="background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; flex: 1; margin-left: 5px;">
                <div style="color: #9CA3AF; font-size: 12px;">Strong Resistance</div>
                <div style="color: #EF4444; font-size: 16px; font-weight: bold;">${vol_levels[-1]}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 5. FIBONACCI CARD ---
def get_fibonacci_levels(df):
    if df.empty or len(df) < 20: 
        return {}
    high, low = df['High'].tail(50).max(), df['Low'].tail(50).min()
    diff = high - low
    return {"0.0%": high, "23.6%": high-(diff*0.236), "38.2%": high-(diff*0.382), "50.0%": high-(diff*0.5), "61.8%": high-(diff*0.618), "100%": low}

gold_df_for_fib = fetch_twelvedata("XAU/USD", "15min", 100)
fib_levels = get_fibonacci_levels(gold_df_for_fib)
if fib_levels:
    st.markdown('<div class="card"><div class="title" style="font-size: 16px;">📐 Fibonacci Levels</div><div style="display: flex; flex-wrap: wrap; margin-top:10px;">', unsafe_allow_html=True)
    for level, price in fib_levels.items():
        st.markdown(f'<div class="fib-box" style="background-color: #111827;"><span style="color:#9CA3AF;">{level}:</span> <span style="color:#FFD700; font-weight:bold;">{round(price, 2)}</span></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# --- RISK CALCULATOR ---
st.markdown('<div class="card"><div class="title" style="font-size: 16px;">🧮 Risk Calculator</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1: balance = st.number_input("Balance ($)", value=1000, step=100)
with col2: risk_pct = st.number_input("Risk (%)", value=1.0, step=0.5)
with col3: sl_points = st.number_input("SL Points", value=30, step=10)
if balance > 0 and risk_pct > 0 and sl_points > 0:
    lot_size = (balance * (risk_pct / 100)) / (sl_points * 10)
    st.markdown(f'<div style="text-align: center; margin-top: 10px;"><span style="color:#9CA3AF;">Lot Size:</span> <span style="color:#10B981; font-size: 24px; font-weight: bold;">{round(lot_size, 2)} Lots</span></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- AI CHATBOT ---
st.markdown('<div class="card"><div class="title" style="font-size: 16px;">🤖 AI Chatbot Assistant</div>', unsafe_allow_html=True)
user_query = st.text_input("Tamaro sawal (ke 'Trade levu?', 'Lot size?')")
if user_query:
    bot_response = get_ai_chatbot_response(user_query, direction, confidence, h1_trend, dxy_trend, entry)
    st.markdown(f'<div class="chat-box"><span style="color:#FFD700;">AI:</span> {bot_response}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- ECONOMIC CALENDAR ---
@st.cache_data(ttl=3600)
def get_economic_calendar():
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        response = requests.get(url, timeout=10)
        data = response.json()
        filtered = [item for item in data if item['country'] == 'USD' and item['impact'] == 'High']
        return filtered[:3]
    except: 
        return []

st.markdown('<div class="card"><div class="title" style="font-size: 16px;">📅 High Impact US News</div><br>', unsafe_allow_html=True)
calendar_events = get_economic_calendar()
if calendar_events:
    for event in calendar_events:
        st.markdown(f"🚨 **{event['title']}**")
        st.markdown(f"<span style='color:#9CA3AF; font-size:12px;'>Time: {event['date']} | Forecast: {event.get('forecast', 'N/A')}</span><br>", unsafe_allow_html=True)
else: 
    st.write("Koi moti news nathi aaj.")
st.markdown('</div>', unsafe_allow_html=True)

# --- NEWS CARD ---
NEWS_API_KEY = 'fd513a4322ea48e2b9be96cd2ad2e3c6'
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
            if 
