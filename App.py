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

st.set_page_config(page_title="Gold AI (BoS Strategy)", page_icon="🪙", layout="centered")
st.markdown("""
<style>
    .stApp { background-color: #111827; color: white; }
    .card { background-color: #1F2937; padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #374151; }
    .title { color: #FFD700; font-size: 24px; font-weight: bold; }
    .subtitle { color: #9CA3AF; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🪙 Gold AI (BoS Strategy)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Break of Structure (Smart Money Concept)</div>', unsafe_allow_html=True)

# --- BOS STRATEGY (Break of Structure) ---
def get_bos_signal():
    try:
        # 15-Minute Data (Last 60 Days)
        df = yf.download("GC=F", period="60d", interval="15m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty or len(df) < 50:
            return "WAIT", 0, 0, 0, 0, "Data Error", "N/A"

        # Live Price
        current_price = round(float(df['Close'].iloc[-1]), 2)

        # 1. Last 20 Candles no High ane Low sodhiye
        recent_high = float(df['High'].iloc[-20:-1].max())
        recent_low = float(df['Low'].iloc[-20:-1].min())

        direction = "WAIT"
        confidence = 0
        reason = "Market condition not clear."
        strategy_name = "BoS Analysis"

        # 2. BoS Logic
        # Jo price last High ne break kare to BUY
        if current_price > recent_high:
            direction = "BUY"
            confidence = 85
            reason = f"Breakout Above {recent_high} (Bullish BoS)"
            strategy_name = "Break of Structure (Buy)"
        
        # Jo price last Low ne break kare to SELL
        elif current_price < recent_low:
            direction = "SELL"
            confidence = 85
            reason = f"Breakout Below {recent_low} (Bearish BoS)"
            strategy_name = "Break of Structure (Sell)"
        
        # 3. Targets (BoS mujab)
        if direction == "BUY":
            sl = round(recent_low, 2) # Last swing low ne SL
            tp1 = round(current_price + (current_price - sl), 2)
            tp2 = round(current_price + (current_price - sl) * 1.5, 2)
        elif direction == "SELL":
            sl = round(recent_high, 2) # Last swing high ne SL
            tp1 = round(current_price - (sl - current_price), 2)
            tp2 = round(current_price - (sl - current_price) * 1.5, 2)
        else:
            sl, tp1, tp2 = 0, 0, 0

        return direction, confidence, current_price, sl, tp1, tp2, reason, strategy_name

    except Exception as e:
        return "WAIT", 0, 0, 0, 0, "System Error", "N/A"

# --- APP UI ---
with st.spinner('Analyzing Break of Structure...'):
    direction, confidence, entry, sl, tp1, tp2, reason, strategy_name = get_bos_signal()

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
    <div style="display: flex; justify-content: space-between;"><span style="font-size: 18px; font-weight: bold; color: white;">⚡ Live BoS Signal</span></div>
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
