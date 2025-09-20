# -*- coding: utf-8 -*-
import streamlit as st
from binance import Client
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
import time
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

# ---------------- Env Loader ----------------
load_dotenv()
                  
# ---------------- Binance API ----------------
# Keyleri .env, secrets veya VPS environment variable üzerinden al
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET") 

# Fallback: environment variables if secrets not set
if not api_key or not api_secret:
    api_key = api_key or ""
    api_secret = api_secret or ""

# Safe Binance client setup
client = None
try:
    client = Client(api_key, api_secret)
    st.write("✅ Binance API connection successful")
except Exception as e:
    st.write("❌ Binance API connection failed:", e)
    st.stop()

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="Binance Live Tracker", layout="wide")
st.title("🚀 Binance Live Tracker + RSI & MACD + EMA + Volume + SL/TP + Trading Guide")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    coin = st.text_input("Enter Coin (e.g., BTCUSDT, ETHUSDT)", "BTCUSDT").upper()
    interval = st.selectbox("Time Interval", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    limit = st.slider("Number of Candles", min_value=50, max_value=1000, value=300, step=50)
    trend_mum = st.slider("Trend Channel Last X Candles", min_value=10, max_value=200, value=50)
    currency = st.selectbox("Show Price In", ["USDT", "USD", "TRY"], index=0)
    st.markdown("**Selected:**")
    st.write(f"Coin: {coin}, Interval: {interval}, Candles: {limit}, Trend Candles: {trend_mum}, Currency: {currency}")

# ---------------- Coin validation ----------------
if not re.match(r'^[A-Z0-9\-\_\.]{1,20}$', coin):
    st.error("Invalid coin symbol! Example: BTCUSDT, ETHUSDT")
    st.stop()

# ---------------- Helpers ----------------
@st.cache_data(ttl=60)
def load_data(symbol, interval, limit):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "Open time","Open","High","Low","Close","Volume","Close time",
        "Quote asset volume","Number of trades",
        "Taker buy base asset volume","Taker buy quote asset volume","Ignore"
    ])
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    return df

def compute_rsi(df, period=7):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(df, fast=6, slow=12, signal=3):
    exp1 = df["Close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

@st.cache_data(ttl=60)
def fetch_usd_to_try():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=USDTTRY"
        response = requests.get(url, timeout=5)
        data = response.json()
        print("Binance API cevabı:", data) #DEBUG
        return float(data["price"])
    except Exception as e:
        print("Error fetching USDT/TRY:", e)
        return None

@st.cache_data(ttl=60)
def fetch_usdt_to_usd():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=USDTUSD"
        response = requests.get(url, timeout=5).json()
        return float(response["price"])
    except:
        return 1.0
# ---------------- Session state ----------------
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = pd.DataFrame(columns=["Time", "Interval", "Price", "Signal"])
if 'usd_try' not in st.session_state:
    st.session_state.usd_try = fetch_usd_to_try()
    st.session_state.usd_try_time = time.time()
if "last_coin" not in st.session_state or st.session_state.last_coin != coin or st.session_state.last_interval != interval:
    st.session_state.signal_history = pd.DataFrame(columns=["Time", "Interval", "Price", "Signal"])
    st.session_state.last_coin = coin
    st.session_state.last_interval = interval

# ---------------- Placeholders ----------------
price_placeholder = st.empty()
price_tl_placeholder = st.empty()
graph_placeholder = st.empty()
table_placeholder = st.empty()
guide_placeholder = st.empty()

# ---------------- Main update function ----------------
def update_chart():
    df = load_data(coin, interval, limit)
    last_close = df["Close"].iloc[-1]
    last_high = df["High"].iloc[-1]
    last_low = df["Low"].iloc[-1]

    pivot = (last_high + last_low + last_close) / 3
    r1 = 2 * pivot - last_low
    r2 = pivot + (last_high - last_low)
    s1 = 2 * pivot - last_high
    s2 = pivot - (last_high - last_low)
    trend_high = df["High"].iloc[-trend_mum:].max()
    trend_low = df["Low"].iloc[-trend_mum:].min()

    # Indicators
    df["RSI"] = compute_rsi(df)
    last_rsi = df["RSI"].iloc[-1]
    rsi_signal = ""
    if pd.notna(last_rsi):
        if last_rsi < 30: rsi_signal = "BUY"
        elif last_rsi > 70: rsi_signal = "SELL"

    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df)
    last_macd = df["MACD"].iloc[-1]
    last_macd_signal = df["MACD_signal"].iloc[-1]
    macd_signal = ""
    if len(df) >= 2 and pd.notna(last_macd) and pd.notna(last_macd_signal):
        if last_macd > last_macd_signal and df["MACD"].iloc[-2] <= df["MACD_signal"].iloc[-2]:
            macd_signal = "BUY"
        elif last_macd < last_macd_signal and df["MACD"].iloc[-2] >= df["MACD_signal"].iloc[-2]:
            macd_signal = "SELL"

    # EMA
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # Final combined signal
    if rsi_signal == "BUY" and macd_signal == "BUY":
        final_signal = "BUY"
        signal_type = "🟢 BUY"
    elif rsi_signal == "SELL" and macd_signal == "SELL":
        final_signal = "SELL"
        signal_type = "🔴 SELL"
    else:
        final_signal = "NEUTRAL"
        signal_type = "⚪ NEUTRAL"

    # ---------------- Price widgets ----------------
    usdt_to_usd = fetch_usdt_to_usd()
    price_placeholder.metric(label=f"{coin} Price (USDT)", value=f"{last_close:.8f}")

    if currency == "USD":
        price_tl_placeholder.metric(label=f"{coin} Price (USD)", value=f"{last_close*usdt_to_usd:.8f}")
    elif currency == "TRY":
        if (time.time() - st.session_state.usd_try_time) > 300:
            st.session_state.usd_try = fetch_usd_to_try()
            st.session_state.usd_try_time = time.time()
        rate = st.session_state.usd_try
        if rate is not None:
            price_tl_placeholder.metric(label=f"{coin} Price (TRY)", value=f"{last_close*rate:.2f}")
        else:
            price_tl_placeholder.text("TRY conversion failed")
    else:
        price_tl_placeholder.text("Showing in USDT (default)")

    # Signal history
    st.session_state.signal_history = pd.concat(
        [st.session_state.signal_history, 
         pd.DataFrame([{"Time": df["Open time"].iloc[-1], "Interval": interval, "Price": last_close, "Signal": final_signal}])],
        ignore_index=True
    )

    # ---------------- Signal table ----------------
    def color_signal(val):
        if val=="BUY": return 'background-color: lightgreen'
        elif val=="SELL": return 'background-color: lightcoral'
        else: return 'background-color: gray'

    table_placeholder.dataframe(st.session_state.signal_history.tail(10).style.applymap(color_signal, subset=['Signal']))

     # CSV Export
    st.download_button("📥 Sinyal Geçmişini İndir",
                       st.session_state.signal_history.to_csv(index=False),
                       "signals.csv",
                       "text/csv")
    
    # ---------------- Candlestick + EMAs + Pivot + Volume ----------------
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Open time"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["Open time"], y=df["EMA50"], name="EMA50", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df["Open time"], y=df["EMA200"], name="EMA200", line=dict(width=1, dash='dash')))

    levels = [
        (pivot,"Pivot"), (r1,"R1"), (r2,"R2"), (s1,"S1"), (s2,"S2"), (trend_high,"Trend High"), (trend_low,"Trend Low")
    ]
    colors = {"Pivot":"yellow","R1":"orange","R2":"red","S1":"lightblue","S2":"blue","Trend High":"green","Trend Low":"green"}
    for level, name in levels:
        fig.add_hline(y=level, line_dash="dot", line_color=colors.get(name,"gray"), annotation_text=name)

    # Volume as secondary y-axis
    fig.add_trace(go.Bar(x=df["Open time"], y=df["Volume"], name="Volume", yaxis="y2", opacity=0.5))

    # Signal annotations
    last_points = st.session_state.signal_history.tail(10)
    for i, row in last_points.iterrows():
        if row["Signal"] != "NEUTRAL":
            x0 = row["Time"] - pd.Timedelta(seconds=30)
            x1 = row["Time"] + pd.Timedelta(seconds=30)
            color = "rgba(0,255,0,0.2)" if row["Signal"]=="BUY" else "rgba(255,0,0,0.2)"
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.3, line_width=0)
            symbol = "⬆️" if row["Signal"]=="BUY" else "⬇️"
            arrowcolor = "green" if row["Signal"]=="BUY" else "red"
            fig.add_annotation(x=row["Time"], y=row["Price"], text=symbol, showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor=arrowcolor, font=dict(size=14, color=arrowcolor))

    fig.update_layout(
        title=f"{coin} Chart ({interval})",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, position=1.0, title="Volume")
    )

    graph_placeholder.plotly_chart(fig, use_container_width=True, key=str(time.time()))

    # ---------------- SL / TP ----------------
    atr = (df['High'] - df['Low']).rolling(window=14).mean().iloc[-1]
    sl = last_close - atr * 1.0
    tp = last_close + atr * 1.5

    st.subheader("🛡️ Önerilen SL / 🎯 Önerilen TP (heuristic)")
    st.write(f"Stop-Loss önerisi: {sl:.6f}")
    st.write(f"Take-Profit önerisi: {tp:.6f}")

    # ---------------- Colorful Guide ----------------
    signal_color = {"BUY": "#4CAF50","SELL": "#F44336","NEUTRAL": "#03324F"}
    last_signal = signal_type.split()[-1]

    guide_html = f"""
    <div style="padding:10px; border-radius:8px; background-color:{signal_color[last_signal]}; color:white; font-weight:bold; text-align:center;">
        <div style="font-size:16px">Al/Sat Sinyali: {signal_type}</div>
        <div style="margin-top:5px; font-size:14px">RSI: {last_rsi:.2f} ({rsi_signal or 'NEUTRAL'})</div>
        <div style="font-size:14px">MACD: {last_macd:.6f} ({macd_signal or 'NEUTRAL'})</div>
        <div style="margin-top:5px; font-size:13px">
        SL önerisi: {sl:.6f} | TP önerisi: {tp:.6f}
        </div>
    </div>
    """
    guide_placeholder.markdown(guide_html, unsafe_allow_html=True)


    # ---------------- Göstergeler ----------------
    st.subheader("📊 Göstergeler")
    
    # 2 sütunlu layout
    col1, col2 = st.columns(2)
    
    # ---------------- Sol sütun ----------------
    with col1:
        st.write(f"RSI: {last_rsi:.2f} → " +
             ("Aşırı Satım (ALIM)" if last_rsi<30 else "Aşırı Alım (SATIM)" if last_rsi>70 else "NEUTRAL (Bekle)"))
        st.write(f"MACD: {last_macd:.6f} → {macd_signal or 'NEUTRAL'}")
        st.write(f"MACD Signal: {last_macd_signal:.6f}")
        st.write(f"MACD Histogram: {df['MACD_hist'].iloc[-1]:.6f}")
        st.write(f"SL: {sl:.6f} → Olası zarar sınırı")
        st.write(f"TP: {tp:.6f} → Hedeflenen kar")
        
    # ---------------- Sağ sütun ----------------
    with col2:
        st.write(f"Pivot: {pivot:.6f} → Fiyat dengede")
        st.write(f"R1: {r1:.6f}, R2: {r2:.6f} → Direnç seviyeleri")
        st.write(f"S1: {s1:.6f}, S2: {s2:.6f} → Destek seviyeleri")
    
    
    
    # ---------------- Educational Mini Guide ----------------
    st.subheader("📚 Trading Mini Rehberi")
    st.markdown("""
    **RSI (Relative Strength Index)**
    - 0-30 → Aşırı satım bölgesi → Potansiyel ALIM fırsatı
    - 70-100 → Aşırı alım bölgesi → Potansiyel SATIM fırsatı

    **MACD (Moving Average Convergence Divergence)**
    - MACD hattı sinyal hattını yukarı keserse (Histogram pozitif) → ALIM sinyali
    - MACD hattı sinyal hattını aşağı keserse (Histogram negatif) → SATIM sinyali
    - Aynı anda RSI ve MACD sinyali uyumlu ise hareket daha güçlüdür

    **SL / TP (Stop-Loss / Take-Profit)**
    - Stop-Loss: Olası zararı sınırlamak için belirlenen fiyat
    - Take-Profit: Hedeflenen kar seviyesi
    - Pivot, R1/R2, S1/S2 seviyeleri trend ve destek/direnç noktalarını gösterir
    """)

# ---------------- Auto-refresh ----------------
try:
    from streamlit import st_autorefresh
    st_autorefresh(interval=5000, key="auto_refresh")
except Exception:
    pass

# ---------------- Initial load ----------------
if client:
    update_chart()
else:
    st.stop()
