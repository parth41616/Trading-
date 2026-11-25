import os
from datetime import datetime, time

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

try:
    from newsapi import NewsApiClient
except ImportError:
    NewsApiClient = None

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Gold AI Ultimate", page_icon="🪙", layout="centered")
st_autorefresh(interval=60 * 1000, limit=None, key="auto_refresh")


def get_key(name):
    """Read key from .streamlit/secrets.toml or environment variable."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, "")


TWELVE_DATA_KEY = get_key("TWELVE_DATA_KEY")
NEWS_API_KEY = get_key("NEWS_API_KEY")

# ---------------- STYLE ----------------
st.markdown(
    """
<style>
    .stApp { background-color: #111827; color: white; }
    .card { background-color: #1F2937; padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #374151; }
    .title { color: #FFD700; font-size: 24px; font-weight: bold; }
    .subtitle { color: #9CA3AF; font-size: 14px; }
    .timing-box { background-color: #1F2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; margin-bottom: 15px; text-align: center;}
    .session-row { display: flex; justify-content: space-between; margin-bottom: 8px; padding: 5px; border-bottom: 1px solid #374151; }
    .mtf-box { background-color: #111827; padding: 10px; border-radius: 8px; text-align: center; flex: 1; margin: 0 2px;}
    .fib-box { background-color: #111827; padding: 5px 8px; border-radius: 5px; text-align: center; margin: 2px; font-size: 13px;}
    .chat-box { background-color: #111827; padding: 15px; border-radius: 10px; margin-top: 10px; border: 1px solid #FFD700;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🪙 Gold AI (Ultimate Pro)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">All-in-One Trading Assistant</div>', unsafe_allow_html=True)

if not TWELVE_DATA_KEY:
    st.error("TWELVE_DATA_KEY missing. Add it in .streamlit/secrets.toml or as an environment variable.")
    st.stop()


# ---------------- 1. MARKET SESSIONS ----------------
def get_market_sessions():
    utc_time = datetime.now(pytz.utc).time()
    sessions = {
        "🇦🇺 Sydney": (time(21, 0), time(6, 0)),
        "🇯🇵 Tokyo": (time(0, 0), time(9, 0)),
        "🇬🇧 London": (time(7, 0), time(16, 0)),
        "🇺🇸 New York": (time(12, 0), time(21, 0)),
    }
    rows = []
    for name, (start, end) in sessions.items():
        is_open = start <= utc_time <= end if start < end else (utc_time >= start or utc_time <= end)
        color = "#10B981" if is_open else "#EF4444"
        status = "OPEN 🟢" if is_open else "CLOSED 🔴"
        rows.append(
            f'<div class="session-row"><span>{name}</span>'
            f'<span style="color: {color}; font-weight: bold;">{status}</span></div>'
        )
    return "".join(rows)


st.markdown(
    f'<div class="timing-box"><div style="color: #FFD700; font-weight: bold; margin-bottom: 10px;">'
    f"🌍 Global Forex Sessions (Live)</div>{get_market_sessions()}</div>",
    unsafe_allow_html=True,
)


# ---------------- DATA FETCH ----------------
@st.cache_data(ttl=300)
def fetch_twelvedata(symbol, interval, outputsize):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_KEY,
    }
    try:
        r = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=15)
        data = r.json()
        if data.get("status") == "error" or "values" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        # Forex/commodity feeds often have NO volume column -> add it safely
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["open", "high", "low", "close", "volume"]]
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def add_indicators(df):
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    # scale-free features (raw prices don't generalise for tree models)
    df["Ret1"] = df["Close"].pct_change()
    df["Dist20"] = df["Close"] / df["SMA_20"] - 1
    df["Dist50"] = df["Close"] / df["SMA_50"] - 1
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]
    return df


FEATURES = ["Ret1", "Dist20", "Dist50", "RSI", "Range"]


def get_dollar_trend():
    """DXY if the plan supports it, otherwise inverse of EUR/USD as a proxy."""
    df = fetch_twelvedata("DXY", "15min", 100)
    inverted = False
    if df.empty:
        df = fetch_twelvedata("EUR/USD", "15min", 100)
        inverted = True
    if df.empty or len(df) < 20:
        return "N/A"
    up = df["Close"].iloc[-1] > df["Close"].rolling(20).mean().iloc[-1]
    if inverted:
        up = not up
    return "UP" if up else "DOWN"


# ---------------- AI MAIN SIGNAL ----------------
@st.cache_data(ttl=300)
def get_pro_signal():
    empty = dict(direction="WAIT", confidence=0, entry=0, sl=0, tp1=0, tp2=0,
                 h1="N/A", dxy="N/A", acc=0, reason="")
    try:
        gold = fetch_twelvedata("XAU/USD", "15min", 500)
        gold_1h = fetch_twelvedata("XAU/USD", "1h", 200)
        dxy_trend = get_dollar_trend()
        if gold.empty or gold_1h.empty:
            return {**empty, "reason": "API limit reached / market closed"}

        feat = add_indicators(gold)
        feat["Target"] = (feat["Close"].shift(-1) > feat["Close"]).astype(int)
        feat = feat.dropna(subset=FEATURES)
        if len(feat) < 100:
            return {**empty, "reason": "Not enough data"}

        latest = feat[FEATURES].tail(1)          # newest candle (no target yet)
        train_df = feat.iloc[:-1]                # last row has no real target

        split = int(len(train_df) * 0.8)         # chronological split (no shuffle)
        X_tr, y_tr = train_df[FEATURES].iloc[:split], train_df["Target"].iloc[:split]
        X_te, y_te = train_df[FEATURES].iloc[split:], train_df["Target"].iloc[split:]

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_tr, y_tr)
        acc = model.score(X_te, y_te) * 100

        ml_signal = int(model.predict(latest)[0])
        ml_prob = float(model.predict_proba(latest)[0][list(model.classes_).index(ml_signal)] * 100)

        sma50_1h = gold_1h["Close"].rolling(50).mean().iloc[-1]
        h1 = "UP" if gold_1h["Close"].iloc[-1] > sma50_1h else "DOWN"

        entry = round(float(gold["Close"].iloc[-1]), 2)
        direction, confidence, sl, tp1, tp2 = "WAIT", 0, 0, 0, 0
        reason = f"No alignment (AI: {'BUY' if ml_signal == 1 else 'SELL'}, 1H: {h1}, USD: {dxy_trend})"

        if ml_signal == 1 and h1 == "UP" and dxy_trend == "DOWN":
            direction, confidence = "BUY", ml_prob
            reason = "AI Bullish + 1H Trend Up + Dollar Weak"
            sl, tp1, tp2 = round(entry - 3, 2), round(entry + 4, 2), round(entry + 6, 2)
        elif ml_signal == 0 and h1 == "DOWN" and dxy_trend == "UP":
            direction, confidence = "SELL", ml_prob
            reason = "AI Bearish + 1H Trend Down + Dollar Strong"
            sl, tp1, tp2 = round(entry + 3, 2), round(entry - 4, 2), round(entry - 6, 2)

        return dict(direction=direction, confidence=confidence, entry=entry, sl=sl, tp1=tp1,
                    tp2=tp2, h1=h1, dxy=dxy_trend, acc=acc, reason=reason)
    except Exception as e:
        return {**empty, "reason": f"System error: {e}"}


# ---------------- 2. MTF TRENDS ----------------
@st.cache_data(ttl=300)
def get_mtf_trends():
    trends = {}
    for tf, label in {"5min": "5M", "15min": "15M", "1h": "1H", "4h": "4H"}.items():
        df = fetch_twelvedata("XAU/USD", tf, 60)
        if not df.empty and len(df) >= 20:
            sma = df["Close"].rolling(20).mean()
            trends[label] = "BUY" if df["Close"].iloc[-1] > sma.iloc[-1] else "SELL"
        else:
            trends[label] = "N/A"
    return trends


# ---------------- 3. SENTIMENT (RSI) ----------------
@st.cache_data(ttl=300)
def get_sentiment():
    df = fetch_twelvedata("XAU/USD", "15min", 500)
    if df.empty:
        return 50.0, 50.0
    rsi = add_indicators(df)["RSI"].iloc[-1]
    if pd.isna(rsi):
        return 50.0, 50.0
    return round(float(rsi), 1), round(100 - float(rsi), 1)


# ---------------- 4. VOLUME PROFILE ----------------
@st.cache_data(ttl=300)
def get_volume_profile():
    df = fetch_twelvedata("XAU/USD", "15min", 500).tail(200).copy()
    if df.empty:
        return []
    bins = np.linspace(df["Low"].min(), df["High"].max(), 11)
    df["Bin"] = pd.cut(df["Close"], bins=bins, labels=False, include_lowest=True)
    # Gold spot has no real volume -> use number of candles per price zone instead
    weight = "Volume" if df["Volume"].sum() > 0 else None
    prof = df.groupby("Bin")[weight].sum() if weight else df.groupby("Bin").size()
    top = prof.nlargest(2).index.tolist()
    levels = sorted(round((bins[int(b)] + bins[int(b) + 1]) / 2, 2) for b in top)
    return levels


# ---------------- 5. FIBONACCI ----------------
def get_fibonacci_levels(df):
    if df.empty or len(df) < 20:
        return {}
    high, low = df["High"].tail(50).max(), df["Low"].tail(50).min()
    diff = high - low
    return {
        "0.0%": high,
        "23.6%": high - diff * 0.236,
        "38.2%": high - diff * 0.382,
        "50.0%": high - diff * 0.5,
        "61.8%": high - diff * 0.618,
        "100%": low,
    }


# ---------------- CHATBOT ----------------
def get_ai_chatbot_response(query, sig):
    q = query.lower()
    if any(w in q for w in ["buy", "sell", "trade", "levu"]):
        if sig["direction"] in ("BUY", "SELL"):
            return (f"AI keh chhe ke {sig['direction']} karo. Entry: {sig['entry']}, "
                    f"Confidence: {sig['confidence']:.1f}%. 1H Trend {sig['h1']} chhe.")
        return f"Abhi WAIT karo. AI ane trend ma takkar chhe ({sig['h1']} vs {sig['dxy']}). Risk na lo."
    if "news" in q or "khabar" in q:
        return "News section ma jao, tya latest Bullish/Bearish news dekhai jashe."
    if "risk" in q or "lot" in q:
        return "Risk Calculator ma tamaru Balance ane Risk % nakho, e jate j lot size batavshe."
    return "Hu tamaro AI assistant chhu. Mane puchho: 'Trade levu?', 'News su chhe?', ke 'Lot size ketlo?'"


# ---------------- APP UI ----------------
with st.spinner("Data Analysis..."):
    sig = get_pro_signal()
    mtf_trends = get_mtf_trends()
    buy_sentiment, sell_sentiment = get_sentiment()
    vol_levels = get_volume_profile()

# Live price widget
components.html(
    """
<div class="tradingview-widget-container" style="height:200px;width:100%">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
  {"symbol": "OANDA:XAUUSD", "width": "100%", "height": "100%", "locale": "en", "dateRange": "1D", "colorTheme": "dark", "isTransparent": false, "autosize": true}
  </script>
</div>
""",
    height=200,
)

# Main signal card
direction = sig["direction"]
color = "#10B981" if direction == "BUY" else "#EF4444" if direction == "SELL" else "#FFD700"
acc_txt = f"Backtest accuracy: {sig['acc']:.0f}% (≈50% = coin flip)" if sig["acc"] else ""
conf_txt = f" • Confidence {sig['confidence']:.0f}%" if sig["confidence"] else ""
card_html = f"""
<div style="background-color:#1F2937;padding:20px;border-radius:15px;border:1px solid #374151;font-family:sans-serif;">
  <div style="font-size:18px;font-weight:bold;color:white;">⚡ Live AI Signal</div>
  <div style="margin-top:15px;background-color:#111827;padding:15px;border-radius:10px;text-align:center;">
    <span style="color:#9CA3AF;font-size:14px;">AI Direction</span><br>
    <span style="color:{color};font-size:26px;font-weight:bold;">{direction}</span>
    <span style="color:#9CA3AF;font-size:12px;">{conf_txt}</span>
    <div style="color:#9CA3AF;font-size:12px;margin-top:5px;">{sig['reason']}</div>
    <div style="color:#6B7280;font-size:11px;margin-top:3px;">{acc_txt}</div>
  </div>
  <div style="display:flex;gap:10px;margin-top:10px;">
    <div style="background-color:#111827;padding:8px;border-radius:8px;text-align:center;flex:1;">
      <div style="color:#9CA3AF;font-size:12px;">1H Trend</div>
      <div style="color:{'#10B981' if sig['h1']=='UP' else '#EF4444'};font-size:14px;font-weight:bold;">{sig['h1']}</div>
    </div>
    <div style="background-color:#111827;padding:8px;border-radius:8px;text-align:center;flex:1;">
      <div style="color:#9CA3AF;font-size:12px;">Dollar</div>
      <div style="color:{'#10B981' if sig['dxy']=='DOWN' else '#EF4444'};font-size:14px;font-weight:bold;">{sig['dxy']}</div>
    </div>
  </div>
  <div style="display:flex;gap:10px;margin-top:10px;">
    <div style="background-color:#111827;padding:10px;border-radius:8px;text-align:center;flex:1;">
      <div style="color:#9CA3AF;font-size:12px;">SL</div>
      <div style="color:#EF4444;font-size:16px;font-weight:bold;">{sig['sl'] or '--'}</div>
    </div>
    <div style="background-color:#111827;padding:10px;border-radius:8px;text-align:center;flex:1;">
      <div style="color:#9CA3AF;font-size:12px;">TP 1</div>
      <div style="color:#10B981;font-size:16px;font-weight:bold;">{sig['tp1'] or '--'}</div>
    </div>
    <div style="background-color:#111827;padding:10px;border-radius:8px;text-align:center;flex:1;">
      <div style="color:#9CA3AF;font-size:12px;">TP 2</div>
      <div style="color:#10B981;font-size:16px;font-weight:bold;">{sig['tp2'] or '--'}</div>
    </div>
  </div>
</div>
"""
components.html(card_html, height=340)

# MTF
mtf_html = '<div style="display:flex;justify-content:space-between;margin-top:10px;">'
for tf, trend in mtf_trends.items():
    c = "#10B981" if trend == "BUY" else "#EF4444" if trend == "SELL" else "#9CA3AF"
    mtf_html += (f'<div class="mtf-box"><div style="color:#9CA3AF;font-size:12px;">{tf}</div>'
                 f'<div style="color:{c};font-weight:bold;">{trend}</div></div>')
mtf_html += "</div>"
st.markdown(f'<div class="card"><div class="title" style="font-size:16px;">📊 Multi-Timeframe</div>{mtf_html}</div>',
            unsafe_allow_html=True)

# Sentiment
st.markdown(
    f"""
<div class="card">
  <div class="title" style="font-size:16px;">⚖️ Market Sentiment (RSI)</div>
  <div style="margin-top:15px;">
    <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
      <span style="color:#10B981;font-weight:bold;">Buyers: {buy_sentiment}%</span>
      <span style="color:#EF4444;font-weight:bold;">Sellers: {sell_sentiment}%</span>
    </div>
    <div style="background-color:#EF4444;border-radius:10px;height:15px;width:100%;display:flex;overflow:hidden;">
      <div style="background-color:#10B981;height:100%;width:{buy_sentiment}%;"></div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Volume zones
if len(vol_levels) >= 2:
    st.markdown(
        f"""
<div class="card">
  <div class="title" style="font-size:16px;">🔊 High Activity Zones (S/R)</div>
  <div style="margin-top:10px;display:flex;gap:10px;">
    <div style="background-color:#111827;padding:10px;border-radius:8px;text-align:center;flex:1;">
      <div style="color:#9CA3AF;font-size:12px;">Lower Zone</div>
      <div style="color:#10B981;font-size:16px;font-weight:bold;">${vol_levels[0]}</div>
    </div>
    <div style="background-color:#111827;padding:10px;border-radius:8px;text-align:center;flex:1;">
      <div style="color:#9CA3AF;font-size:12px;">Upper Zone</div>
      <div style="color:#EF4444;font-size:16px;font-weight:bold;">${vol_levels[-1]}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# Fibonacci (one single HTML block so the card wraps correctly)
fib_levels = get_fibonacci_levels(fetch_twelvedata("XAU/USD", "15min", 500))
if fib_levels:
    boxes = "".join(
        f'<div class="fib-box"><span style="color:#9CA3AF;">{lvl}:</span> '
        f'<span style="color:#FFD700;font-weight:bold;">{round(p, 2)}</span></div>'
        for lvl, p in fib_levels.items()
    )
    st.markdown(
        f'<div class="card"><div class="title" style="font-size:16px;">📐 Fibonacci Levels</div>'
        f'<div style="display:flex;flex-wrap:wrap;margin-top:10px;">{boxes}</div></div>',
        unsafe_allow_html=True,
    )

# Risk calculator
st.markdown('<div class="title" style="font-size:16px;">🧮 Risk Calculator</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    balance = st.number_input("Balance ($)", value=1000, step=100)
with c2:
    risk_pct = st.number_input("Risk (%)", value=1.0, step=0.5)
with c3:
    sl_points = st.number_input("SL Points", value=30, step=10)
if balance > 0 and risk_pct > 0 and sl_points > 0:
    # 1 point = $0.10 on gold, 1 lot = $10 per point
    lot_size = (balance * (risk_pct / 100)) / (sl_points * 10)
    st.markdown(
        f'<div style="text-align:center;margin:10px 0 20px;"><span style="color:#9CA3AF;">Lot Size:</span> '
        f'<span style="color:#10B981;font-size:24px;font-weight:bold;">{round(lot_size, 2)} Lots</span></div>',
        unsafe_allow_html=True,
    )

# Chatbot
st.markdown('<div class="title" style="font-size:16px;">🤖 AI Chatbot Assistant</div>', unsafe_allow_html=True)
user_query = st.text_input("Tamaro sawal (ke 'Trade levu?', 'Lot size?')")
if user_query:
    st.markdown(
        f'<div class="chat-box"><span style="color:#FFD700;">AI:</span> {get_ai_chatbot_response(user_query, sig)}</div>',
        unsafe_allow_html=True,
    )


# ---------------- ECONOMIC CALENDAR ----------------
@st.cache_data(ttl=3600)
def get_economic_calendar():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        data = r.json()
        if not isinstance(data, list):   # rate-limited response is a dict
            return []
        return [i for i in data if i.get("country") == "USD" and i.get("impact") == "High"][:3]
    except Exception:
        return []


st.markdown('<div class="card"><div class="title" style="font-size:16px;">📅 High Impact US News</div>',
            unsafe_allow_html=True)
events = get_economic_calendar()
if events:
    for ev in events:
        st.markdown(f"🚨 **{ev['title']}**")
        st.markdown(
            f"<span style='color:#9CA3AF;font-size:12px;'>Time: {ev['date']} | "
            f"Forecast: {ev.get('forecast') or 'N/A'}</span>",
            unsafe_allow_html=True,
        )
else:
    st.write("Koi moti news nathi.")
st.markdown("</div>", unsafe_allow_html=True)


# ---------------- NEWS ----------------
@st.cache_data(ttl=3600)
def get_news():
    if not NEWS_API_KEY or NewsApiClient is None:
        return []
    try:
        api = NewsApiClient(api_key=NEWS_API_KEY)
        res = api.get_everything(q='"gold price" OR "XAUUSD"', language="en",
                                 sort_by="relevancy", page_size=10)
        out = []
        for art in res.get("articles", []):
            title = art.get("title")
            if not title or not any(w in title.lower() for w in ["gold", "fed", "dollar"]):
