import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from newsapi import NewsApiClient

# --- APP DESIGN ---
st.set_page_config(page_title="Gold AI Trade", page_icon="🪙", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #111827; color: white; }
    .card { background-color: #1F2937; padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #374151; }
    .title { color: #FFD700; font-size: 24px; font-weight: bold; }
    .subtitle { color: #9CA3AF; font-size: 14px; }
    .signal-buy { color: #10B981; font-size: 22px; font-weight: bold; }
    .signal-sell { color: #EF4444; font-size: 22px; font-weight: bold; }
    .level-box { background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px;}
    .level-title { color: #9CA3AF; font-size: 12px; }
    .level-val { color: white; font-size: 16px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🪙 Gold AI Trade</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by AI & News Analysis</div>', unsafe_allow_html=True)
st.write("---")

# --- AI LOGIC ---
@st.cache_data(ttl=3600)
def get_signal():
    data = yf.download("GC=F", period="2y", interval="1d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data.dropna(inplace=True)
    
    current_price = float(data['Close'].iloc[-1])
    
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data.dropna(inplace=True)
    X = data[['Open', 'High', 'Low', 'Close', 'SMA_50', 'SMA_200', 'RSI']]
    Y = data['Target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, Y_train)
    
    latest = X.tail(1)
    signal = model.predict(latest)
    prob = model.predict_proba(latest)
    
    direction = "BUY" if signal[0] == 1 else "SELL"
    confidence = prob[0][signal[0]] * 100
    
    entry = round(current_price, 2)
    if direction == "BUY":
        sl = round(entry - 10.0, 2)
        tp1 = round(entry + 15.0, 2)
        tp2 = round(entry + 25.0, 2)
    else:
        sl = round(entry + 10.0, 2)
        tp1 = round(entry - 15.0, 2)
        tp2 = round(entry - 25.0, 2)

    return direction, confidence, entry, sl, tp1, tp2

# --- NEWS LOGIC ---
NEWS_API_KEY = 'fd513a4322ea48e2b9be96cd2ad2e3c6'
@st.cache_data(ttl=3600)
def get_news():
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_everything(q='gold price OR XAUUSD', language='en', sort_by='publishedAt', page_size=5)
        return articles['articles']
    except Exception as e:
        return []

# --- APP UI ---
with st.spinner('AI Market analysis kari rahyu chhe...'):
    direction, confidence, entry, sl, tp1, tp2 = get_signal()
    news_articles = get_news()

# Main Signal Card (FIX: Using components.html instead of markdown)
card_html = f"""
<div style="background-color: #1F2937; padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #374151; font-family: sans-serif;">
    <div style="display: flex; justify-content: space-between;">
        <span style="font-size: 18px; font-weight: bold; color: white;">🇺🇸 XAU/USD (Gold)</span>
        <span style="color: #9CA3AF; font-size: 14px;">Just now</span>
    </div>
    <div style="margin-top: 10px;">
        <span style="color: #9CA3AF; font-size: 14px;">AI Confidence: <span style="color: #FFD700;">{confidence:.1f}%</span></span>
        <div style="background-color: #374151; border-radius: 5px; height: 8px; margin-top: 5px;">
            <div style="background-color: #FFD700; width: {confidence}%; height: 8px; border-radius: 5px;"></div>
        </div>
    </div>
    
    <div style="margin-top: 15px; background-color: #111827; padding: 15px; border-radius: 10px; text-align: center;">
        <span style="color: #9CA3AF; font-size: 14px;">AI Direction</span><br>
        <span style="color: {'#10B981' if direction == 'BUY' else '#EF4444'}; font-size: 22px; font-weight: bold;">{direction}</span>
    </div>
    
    <div style="display: flex; justify-content: space-between; gap: 10px;">
        <div style="background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px; flex: 1;">
            <div style="color: #9CA3AF; font-size: 12px;">ENTRY</div>
            <div style="color: white; font-size: 16px; font-weight: bold;">{entry}</div>
        </div>
        <div style="background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px; flex: 1;">
            <div style="color: #9CA3AF; font-size: 12px;">SL</div>
            <div style="color: #EF4444; font-size: 16px; font-weight: bold;">{sl}</div>
        </div>
        <div style="background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px; flex: 1;">
            <div style="color: #9CA3AF; font-size: 12px;">TP 1</div>
            <div style="color: #10B981; font-size: 16px; font-weight: bold;">{tp1}</div>
        </div>
        <div style="background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px; flex: 1;">
            <div style="color: #9CA3AF; font-size: 12px;">TP 2</div>
            <div style="color: #10B981; font-size: 16px; font-weight: bold;">{tp2}</div>
        </div>
    </div>
</div>
"""
components.html(card_html, height=320)

# News Card
st.markdown('<div class="card"><div class="title" style="font-size: 18px;">📰 Latest Gold News</div><br>', unsafe_allow_html=True)
if news_articles:
    for article in news_articles:
        st.markdown(f"🔹 [{article['title']}]({article['url']})")
else:
    st.write("News nathi mali, thodi vaar pachi try karo.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle" style="text-align:center;">*Aa AI suggestions chhe, financial advice nahi. Trade potana risk par lo.*</div>', unsafe_allow_html=True)
