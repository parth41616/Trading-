import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

# --- AUTO REFRESH (5 Minutes) ---
st_autorefresh(interval=5 * 60 * 1000, limit=None, key="auto_refresh")

st.set_page_config(page_title="Gold AI (Smart)", page_icon="🪙", layout="centered")
st.markdown("""
<style>
    .stApp { background-color: #111827; color: white; }
    .card { background-color: #1F2937; padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #374151; }
    .title { color: #FFD700; font-size: 24px; font-weight: bold; }
    .subtitle { color: #9CA3AF; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🪙 Gold AI (Smart Strategy)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Based on 2 Years Historical Data</div>', unsafe_allow_html=True)

# --- MARKET SESSIONS ---
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
        is_open = start <= utc_time <= end if start < end else utc_time >= start or utc_time <= end
        color = "#10B981" if is_open else "#EF4444"
        status = "OPEN 🟢" if is_open else "CLOSED 🔴"
        result.append(f'<div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span>{name}</span><span style="color: {color}; font-weight: bold;">{status}</span></div>')
    return "".join(result)

sessions_html = get_market_sessions()
st.markdown(f'<div class="card"><div class="title" style="font-size: 16px;">🌍 Market Sessions</div>{sessions_html}</div>', unsafe_allow_html=True)

# --- SMART STRATEGY (Backtest + Live) ---
@st.cache_data(ttl=300)
def get_smart_signal():
    try:
        # Last 2 years no data (Daily candles) - Historical Check mate
        df = yf.download("GC=F", period="2y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty or len(df) < 100:
            return "WAIT", 0, 0, 0, 0, "Data Error"

        # Indicators
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df.dropna(inplace=True)

        # Live Price (Current Daily Candle)
        current_price = float(df['Close'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        current_sma = float(df['SMA_50'].iloc[-1])

        direction = "WAIT"
        confidence = 0
        reason = "Market condition not clear."
        
        # Strategy 1: Trend Follow (Buy in Uptrend, Sell in Downtrend)
        if current_price > current_sma and current_rsi < 70:
            direction = "BUY"
            confidence = 75
            reason = "Uptrend + RSI Normal (High Probability)"
        elif current_price < current_sma and current_rsi > 30:
            direction = "SELL"
            confidence = 75
            reason = "Downtrend + RSI Normal (High Probability)"
            
        # Extreme Conditions
        elif current_rsi < 30:
            direction = "BUY"
            confidence = 85
            reason = "Oversold (Bounce Expected)"
        elif current_rsi > 70:
            direction = "SELL"
            confidence = 85
            reason = "Overbought (Correction Expected)"
        else:
            direction = "WAIT"
            confidence = 0
            reason = "No clear trend"

        # Targets
        if direction == "BUY":
            sl = round(current_price - 8.0, 2)
            tp1 = round(current_price + 10.0, 2)
            tp2 = round(current_price + 15.0, 2)
        elif direction == "SELL":
            sl = round(current_price + 8.0, 2)
            tp1 = round(current_price - 10.0, 2)
            tp2 = round(current_price - 15.0, 2)
        else:
            sl, tp1, tp2 = 0, 0, 0

        return direction, confidence, round(current_price, 2), sl, tp1, tp2, reason

    except Exception as e:
        return "WAIT", 0, 0, 0, 0, "System Error"

# --- APP UI ---
with st.spinner('Analyzing 2 Years Data...'):
    direction, confidence, entry, sl, tp1, tp2, reason = get_smart_signal()

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
    <div style="display: flex; justify-content: space-between;"><span style="font-size: 18px; font-weight: bold; color: white;">⚡ Smart Signal</span></div>
    <div style="margin-top: 15px; background-color: #111827; padding: 15px; border-radius: 10px; text-align: center;">
        <span style="color: #9CA3AF; font-size: 14px;">Direction</span><br>
        <span style="color: {color}; font-size: 26px; font-weight: bold;">{direction}</span>
        <div style="color: #9CA3AF; font-size: 12px; margin-top: 5px;">{reason}</div>
    </div>
    <div style="display: flex; justify-content: space-between; gap: 10px; margin-top: 10px;">
        <div style="background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; flex: 1;">
            <div style="color: #9CA3AF; font-size: 12px;">ENTRY</div>
            <div style="color: #FFD700; font-size: 16px; font-weight: bold;">{entry if entry != 0 else '--'}</div>
        </div>
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

st.markdown('<div class="subtitle" style="text-align:center; margin-top:20px;">*This strategy is based on Trend Following & RSI. Not 100% accurate.*</div>', unsafe_allow_html=True)
