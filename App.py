import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
from streamlit_autorefresh import st_autorefresh
import requests

# --- AUTO REFRESH (5 Minutes) ---
st_autorefresh(interval=5 * 60 * 1000, limit=None, key="auto_refresh")

TWELVE_DATA_KEY = '4c470e5fed854bbaa4b4c3a38a11499e'

st.set_page_config(page_title="Gold AI (Live Spot)", page_icon="🪙", layout="centered")
st.markdown("""
<style>
    .stApp { background-color: #111827; color: white; }
    .card { background-color: #1F2937; padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #374151; }
    .title { color: #FFD700; font-size: 24px; font-weight: bold; }
    .subtitle { color: #9CA3AF; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🪙 Gold AI (Spot Live)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Exact Spot Price with 15-Min Strategy</div>', unsafe_allow_html=True)

# --- DATA FETCH (Twelve Data - Spot XAU/USD) ---
def fetch_twelvedata(symbol, interval, outputsize):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_DATA_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("status") != "error" and "values" in data:
            df = pd.DataFrame(data["values"])
            df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float'})
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return df
    except:
        pass
    return pd.DataFrame()

# --- SMART STRATEGY (Live Spot) ---
def get_smart_signal():
    try:
        # Spot XAU/USD (Tamara MT5 sathe match thashe)
        df = fetch_twelvedata("XAU/USD", "15min", 500)
        
        if df.empty or len(df) < 50:
            return "WAIT", 0, 0, 0, 0, "Data Error", "N/A"

        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df.dropna(inplace=True)

        # Exact Live Price (Spot Price)
        live_price = round(float(df['Close'].iloc[-1]), 3)
        current_rsi = float(df['RSI'].iloc[-1])
        current_sma = float(df['SMA_50'].iloc[-1])

        direction = "WAIT"
        confidence = 0
        reason = "Market condition not clear."
        strategy_name = "No Strategy"

        # STRATEGY 1: Trend Follow
        if live_price > current_sma and current_rsi < 70:
            direction = "BUY"
            confidence = 70
            reason = "Price Above SMA 50 (Trend Up)"
            strategy_name = "Trend Follow (Up)"
        elif live_price < current_sma and current_rsi > 30:
            direction = "SELL"
            confidence = 70
            reason = "Price Below SMA 50 (Trend Down)"
            strategy_name = "Trend Follow (Down)"
            
        # STRATEGY 2: Reversal (RSI)
        elif current_rsi < 30:
            direction = "BUY"
            confidence = 80
            reason = "Oversold (RSI < 30)"
            strategy_name = "RSI Reversal (Buy)"
        elif current_rsi > 70:
            direction = "SELL"
            confidence = 80
            reason = "Overbought (RSI > 70)"
            strategy_name = "RSI Reversal (Sell)"
        
        else:
            direction = "WAIT"
            confidence = 0
            reason = "No clear trend"
            strategy_name = "No Setup"

        # Targets (Exact Live Price par thi)
        if direction == "BUY":
            sl = round(live_price - 8.0, 2)
            tp1 = round(live_price + 10.0, 2)
            tp2 = round(live_price + 15.0, 2)
        elif direction == "SELL":
            sl = round(live_price + 8.0, 2)
            tp1 = round(live_price - 10.0, 2)
            tp2 = round(live_price - 15.0, 2)
        else:
            sl, tp1, tp2 = 0, 0, 0

        return direction, confidence, live_price, sl, tp1, tp2, reason, strategy_name

    except Exception as e:
        return "WAIT", 0, 0, 0, 0, "System Error", "N/A"

# --- APP UI ---
with st.spinner('Fetching Spot Price & Strategy...'):
    direction, confidence, entry, sl, tp1, tp2, reason, strategy_name = get_smart_signal()

# --- LIVE CHART (Spot XAUUSD - Binance/TradingView source thi) ---
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
    <div style="display: flex; justify-content: space-between;"><span style="font-size: 18px; font-weight: bold; color: white;">⚡ Live Smart Signal</span></div>
    <div style="margin-top: 15px; background-color: #111827; padding: 15px; border-radius: 10px; text-align: center;">
        <span style="color: #9CA3AF; font-size: 14px;">Direction</span><br>
        <span style="color: {color}; font-size: 26px; font-weight: bold;">{direction}</span>
        <div style="color: #FFD700; font-size: 12px; margin-top: 5px;">Strategy: {strategy_name}</div>
        <div style="color: #9CA3AF; font-size: 12px; margin-top: 2px;">{reason}</div>
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
