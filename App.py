import os
import html
import requests
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import streamlit.components.v1 as components

from datetime import datetime, time
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Gold AI Ultimate Pro",
    page_icon="🪙",
    layout="centered"
)

st_autorefresh(
    interval=60 * 1000,
    limit=None,
    key="auto_refresh"
)


def get_key(name):
    try:
        return st.secrets.get(name, os.environ.get(name, ""))
    except Exception:
        return os.environ.get(name, "")


TWELVE_DATA_KEY = get_key("TWELVE_DATA_KEY")
NEWS_API_KEY = get_key("NEWS_API_KEY")


# =====================================================
# DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #111827;
    color: white;
}

.card {
    background-color: #1F2937;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 18px;
    border: 1px solid #374151;
}

.title {
    color: #FFD700;
    font-size: 23px;
    font-weight: bold;
}

.subtitle {
    color: #9CA3AF;
    font-size: 14px;
}

.timing-box {
    background-color: #1F2937;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #374151;
    margin-bottom: 15px;
}

.session-row {
    display: flex;
    justify-content: space-between;
    padding: 7px;
    border-bottom: 1px solid #374151;
}

.mtf-box {
    background-color: #111827;
    padding: 10px 4px;
    border-radius: 8px;
    text-align: center;
    flex: 1;
    margin: 2px;
}

.fib-box {
    background-color: #111827;
    padding: 8px;
    border-radius: 6px;
    margin: 3px;
    font-size: 13px;
}

.chat-box {
    background-color: #111827;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
    border: 1px solid #FFD700;
}

.stButton button {
    background-color: #FFD700;
    color: #111827;
    font-weight: bold;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="title">🪙 Gold AI Ultimate Pro</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">All-in-One AI Gold Trading Assistant</div>',
    unsafe_allow_html=True
)


# =====================================================
# API STATUS
# =====================================================

if not TWELVE_DATA_KEY:
    st.error("Twelve Data API key missing. Configure TWELVE_DATA_KEY in Streamlit Secrets.")
    st.stop()


# =====================================================
# 1. GLOBAL FOREX SESSIONS
# =====================================================

def get_market_sessions():

    utc_time = datetime.now(pytz.utc).time()

    sessions = {
        "🇦🇺 Sydney": (time(21, 0), time(6, 0)),
        "🇯🇵 Tokyo": (time(0, 0), time(9, 0)),
        "🇬🇧 London": (time(7, 0), time(16, 0)),
        "🇺🇸 New York": (time(12, 0), time(21, 0))
    }

    result = []

    for name, (start, end) in sessions.items():

        if start < end:
            opened = start <= utc_time < end
        else:
            opened = utc_time >= start or utc_time < end

        color = "#10B981" if opened else "#EF4444"

        status = "OPEN 🟢" if opened else "CLOSED 🔴"

        result.append(
            f"""
            <div class="session-row">
                <span>{name}</span>
                <span style="color:{color};font-weight:bold">
                    {status}
                </span>
            </div>
            """
        )

    return "".join(result)


st.markdown(
    f"""
    <div class="timing-box">
        <div class="title" style="font-size:16px">
            🌍 Global Forex Sessions
        </div>
        {get_market_sessions()}
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 2. TWELVE DATA CONNECTION
# =====================================================

@st.cache_data(ttl=60)
def fetch_twelvedata(symbol, interval, outputsize):

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_KEY
    }

    try:

        response = requests.get(
            "https://api.twelvedata.com/time_series",
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if "values" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["values"])

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

        df.set_index("datetime", inplace=True)

        df.sort_index(inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:

            if col not in df.columns:
                df[col] = 0

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df[
            ["open", "high", "low", "close", "volume"]
        ]

        df.columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        return df.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

    except Exception:
        return pd.DataFrame()


# =====================================================
# 3. TECHNICAL INDICATORS
# =====================================================

def add_indicators(df):

    df = df.copy()

    df["SMA_20"] = df["Close"].rolling(20).mean()

    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()

    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    df["RSI"] = df["RSI"].fillna(50)

    df["Ret1"] = df["Close"].pct_change()

    df["Dist20"] = df["Close"] / df["SMA_20"] - 1

    df["Dist50"] = df["Close"] / df["SMA_50"] - 1

    df["Range"] = (
        df["High"] - df["Low"]
    ) / df["Close"].replace(0, np.nan)

    return df


FEATURES = [
    "Ret1",
    "Dist20",
    "Dist50",
    "RSI",
    "Range"
]


# =====================================================
# 4. DOLLAR INDEX TREND
# =====================================================

@st.cache_data(ttl=60)
def get_dollar_trend():

    df = fetch_twelvedata(
        "DXY",
        "15min",
        100
    )

    inverted = False

    if df.empty:

        df = fetch_twelvedata(
            "EUR/USD",
            "15min",
            100
        )

        inverted = True

    if df.empty or len(df) < 20:
        return "N/A"

    sma = df["Close"].rolling(20).mean().iloc[-1]

    up = df["Close"].iloc[-1] > sma

    if inverted:
        up = not up

    return "UP" if up else "DOWN"


# =====================================================
# 5. AI GOLD PREDICTION
# =====================================================

@st.cache_data(ttl=60)
def get_pro_signal():

    empty = {
        "direction": "WAIT",
        "confidence": 0,
        "entry": 0,
        "sl": 0,
        "tp1": 0,
        "tp2": 0,
        "h1": "N/A",
        "dxy": "N/A",
        "acc": 0,
        "reason": "Waiting for market data"
    }

    try:

        gold = fetch_twelvedata(
            "XAU/USD",
            "15min",
            500
        )

        gold_1h = fetch_twelvedata(
            "XAU/USD",
            "1h",
            200
        )

        dxy = get_dollar_trend()

        if gold.empty or gold_1h.empty:

            return {
                **empty,
                "reason": "Gold market data unavailable"
            }

        if len(gold_1h) < 50:
            return {
                **empty,
                "reason": "Insufficient 1H data"
            }

        feat = add_indicators(gold)

        feat["Target"] = (
            feat["Close"].shift(-1) > feat["Close"]
        ).astype(int)

        feat.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        feat.dropna(
            subset=FEATURES,
            inplace=True
        )

        if len(feat) < 100:

            return {
                **empty,
                "reason": "Not enough training data"
            }

        latest = feat[FEATURES].tail(1)

        train_df = feat.iloc[:-1].copy()

        split = int(len(train_df) * 0.8)

        X_train = train_df[FEATURES].iloc[:split]

        y_train = train_df["Target"].iloc[:split]

        X_test = train_df[FEATURES].iloc[split:]

        y_test = train_df["Target"].iloc[split:]

        if len(X_test) == 0 or y_train.nunique() < 2:

            return {
                **empty,
                "reason": "Insufficient model training classes"
            }

        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(
            X_train,
            y_train
        )

        accuracy = model.score(
            X_test,
            y_test
        ) * 100

        prediction = int(
            model.predict(latest)[0]
        )

        probabilities = model.predict_proba(latest)[0]

        confidence = float(
            probabilities[
                list(model.classes_).index(prediction)
            ] * 100
        )

        # 1H Trend

        sma50 = gold_1h["Close"].rolling(50).mean().iloc[-1]

        h1 = (
            "UP"
            if gold_1h["Close"].iloc[-1] > sma50
            else "DOWN"
        )

        entry = round(
            float(gold["Close"].iloc[-1]),
            2
        )

        direction = "WAIT"

        sl = tp1 = tp2 = 0

        reason = (
            f"No alignment: AI prediction "
            f"{'BUY' if prediction == 1 else 'SELL'}, "
            f"1H {h1}, USD {dxy}"
        )

        if confidence >= 55:

            if (
                prediction == 1
                and h1 == "UP"
                and dxy == "DOWN"
            ):

                direction = "BUY"

                sl = round(entry - 3, 2)

                tp1 = round(entry + 4, 2)

                tp2 = round(entry + 6, 2)

                reason = "Bullish AI + Strong 1H + Weak Dollar"

            elif (
                prediction == 0
                and h1 == "DOWN"
                and dxy == "UP"
            ):

                direction = "SELL"

                sl = round(entry + 3, 2)

                tp1 = round(entry - 4, 2)

                tp2 = round(entry - 6, 2)

                reason = "Bearish AI + Weak 1H + Strong Dollar"

        return {
            "direction": direction,
            "confidence": confidence if direction != "WAIT" else 0,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "h1": h1,
            "dxy": dxy,
            "acc": accuracy,
            "reason": reason
        }

    except Exception as e:

        return {
            **empty,
            "reason": f"AI Error: {str(e)}"
        }


# =====================================================
# 6. MULTI TIMEFRAME ANALYSIS
# =====================================================

@st.cache_data(ttl=60)
def get_mtf_trends():

    trends = {}

    timeframes = {
        "5min": "5M",
        "15min": "15M",
        "1h": "1H",
        "4h": "4H"
    }

    for tf, label in timeframes.items():

        df = fetch_twelvedata(
            "XAU/USD",
            tf,
            60
        )

        if not df.empty and len(df) >= 20:

            sma = df["Close"].rolling(20).mean()

            trends[label] = (
                "BUY"
                if df["Close"].iloc[-1] > sma.iloc[-1]
                else "SELL"
            )

        else:

            trends[label] = "N/A"

    return trends


# =====================================================
# 7. MARKET SENTIMENT
# =====================================================

@st.cache_data(ttl=60)
def get_sentiment():

    df = fetch_twelvedata(
        "XAU/USD",
        "15min",
        100
    )

    if df.empty:
        return 50.0, 50.0

    rsi = add_indicators(df)["RSI"].iloc[-1]

    if pd.isna(rsi):
        return 50.0, 50.0

    return (
        round(float(rsi), 1),
        round(100 - float(rsi), 1)
    )


# =====================================================
# 8. VOLUME / PRICE ACTIVITY PROFILE
# =====================================================

@st.cache_data(ttl=300)
def get_volume_profile():

    df = fetch_twelvedata(
        "XAU/USD",
        "15min",
        200
    )

    if df.empty:
        return []

    low = df["Low"].min()

    high = df["High"].max()

    if high <= low:
        return []

    bins = np.linspace(
        low,
        high,
        11
    )

    df["Bin"] = pd.cut(
        df["Close"],
        bins=bins,
        labels=False,
        include_lowest=True
    )

    if df["Volume"].sum() > 0:

        profile = df.groupby("Bin")["Volume"].sum()

    else:

        profile = df.groupby("Bin").size()

    profile = profile.dropna()

    top = profile.nlargest(2).index.tolist()

    levels = []

    for b in top:

        index = int(b)

        if 0 <= index < len(bins) - 1:

            price = (
                bins[index] + bins[index + 1]
            ) / 2

            levels.append(round(price, 2))

    return sorted(levels)


# =====================================================
# 9. FIBONACCI
# =====================================================

def get_fibonacci_levels(df):

    if df.empty or len(df) < 20:
        return {}

    high = df["High"].tail(50).max()

    low = df["Low"].tail(50).min()

    diff = high - low

    return {
        "0.0%": high,
        "23.6%": high - diff * 0.236,
        "38.2%": high - diff * 0.382,
        "50.0%": high - diff * 0.5,
        "61.8%": high - diff * 0.618,
        "100%": low
    }


# =====================================================
# 10. AI CHATBOT
# =====================================================

def get_ai_chatbot_response(query, sig):

    q = query.lower()

    if any(
        word in q
        for word in ["buy", "sell", "trade", "levu"]
    ):

        if sig["direction"] in ["BUY", "SELL"]:

            return (
                f"AI Signal: {sig['direction']} | "
                f"Entry: {sig['entry']} | "
                f"Confidence: {sig['confidence']:.1f}% | "
                f"1H Trend: {sig['h1']}"
            )

        return (
            f"WAIT recommended. "
            f"1H: {sig['h1']} | "
            f"Dollar: {sig['dxy']}. "
            f"Market alignment is not clear."
        )

    elif "news" in q or "khabar" in q:

        return "Latest Gold and Forex news is available in News section."

    elif "risk" in q or "lot" in q:

        return "Use Risk Calculator below to calculate estimated lot size."

    elif "rsi" in q:

        return "RSI shows market momentum. Above 70 may indicate overbought and below 30 may indicate oversold."

    elif "gold" in q or "price" in q:

        return f"Latest available Gold entry price: {sig['entry']}"

    else:

        return (
            "Hello! I am your Gold AI Assistant. "
            "Ask me: Trade levu? News su chhe? "
            "RSI su chhe? Lot size ketlo?"
        )


# =====================================================
# 11. ECONOMIC CALENDAR
# =====================================================

@st.cache_data(ttl=1800)
def get_economic_calendar():

    try:

        response = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            return []

        events = [
            item
            for item in data
            if item.get("country") == "USD"
            and item.get("impact") == "High"
        ]

        return events[:10]

    except Exception:
        return []


# =====================================================
# 12. GOLD NEWS
# =====================================================

@st.cache_data(ttl=1800)
def get_news():

    if not NEWS_API_KEY:
        return []

    try:

        params = {
            "q": '"gold price" OR XAUUSD OR Federal Reserve',
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 15,
            "apiKey": NEWS_API_KEY
        }

        response = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        if data.get("status") != "ok":
            return []

        articles = []

        for art in data.get("articles", []):

            title = art.get("title", "")

            if not title:
                continue

            title_lower = title.lower()

            sentiment = "Neutral ⚖️"

            color = "#9CA3AF"

            if any(
                word in title_lower
                for word in [
                    "surge", "rise", "rally",
                    "higher", "bullish", "gain"
                ]
            ):

                sentiment = "Bullish 📈"

                color = "#10B981"

            elif any(
                word in title_lower
                for word in [
                    "fall", "drop", "crash",
                    "lower", "bearish", "loss"
                ]
            ):

                sentiment = "Bearish 📉"

                color = "#EF4444"

            articles.append({
                "title": title,
                "url": art.get("url", ""),
                "source": art.get("source", {}).get("name", "News"),
                "date": art.get("publishedAt", ""),
                "sentiment": sentiment,
                "color": color
            })

        return articles

    except Exception:
        return []


# =====================================================
# 13. LOAD ALL DATA
# =====================================================

with st.spinner("🤖 AI Analyzing Gold Market..."):

    sig = get_pro_signal()

    mtf_trends = get_mtf_trends()

    buy_sentiment, sell_sentiment = get_sentiment()

    vol_levels = get_volume_profile()

    gold_df = fetch_twelvedata(
        "XAU/USD",
        "15min",
        100
    )

    fib_levels = get_fibonacci_levels(gold_df)


# =====================================================
# 14. LIVE GOLD PRICE CHART
# =====================================================

st.markdown(
    '<div class="title" style="font-size:17px">📈 Live Gold Chart</div>',
    unsafe_allow_html=True
)

components.html(
    """
    <div class="tradingview-widget-container" style="height:300px;width:100%">
        <div class="tradingview-widget-container__widget"></div>

        <script
        type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js"
        async>
        {
            "symbol": "OANDA:XAUUSD",
            "width": "100%",
            "height": "100%",
            "locale": "en",
            "dateRange": "1D",
            "colorTheme": "dark",
            "isTransparent": false,
            "autosize": true
        }
        </script>
    </div>
    """,
    height=300
)


# =====================================================
# 15. MAIN AI SIGNAL CARD
# =====================================================

direction = sig["direction"]

color = (
    "#10B981"
    if direction == "BUY"
    else "#EF4444"
    if direction == "SELL"
    else "#FFD700"
)

st.markdown(
    f"""
    <div class="card">

        <div class="title" style="font-size:19px">
            ⚡
