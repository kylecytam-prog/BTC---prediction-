import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import requests
from streamlit_autorefresh import st_autorefresh
import time

# Refresh every 2 seconds = no delay
st_autorefresh(interval=2000, key="fast_btc")

st.set_page_config(page_title="BTC Live Pro", page_icon="₿", layout="wide")

st.markdown("<h1 style='text-align:center'>₿ BTC Live Pro - REAL TIME</h1>", unsafe_allow_html=True)

# --- FAST LIVE PRICE FROM BINANCE (no delay) ---
@st.cache_data(ttl=1) # cache 1 second only
def get_live_price():
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=2)
        data = r.json()
        return {
            "price": float(data['lastPrice']),
            "high": float(data['highPrice']),
            "low": float(data['lowPrice']),
            "change": float(data['priceChangePercent']),
            "volume": float(data['volume'])
        }
    except:
        return None

live_data = get_live_price()

# --- HISTORY FOR CHART ---
@st.cache_data(ttl=300)
def load_history():
    df = yf.download("BTC-USD", period="1d", interval="1m", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    col = 'Date' if 'Date' in df.columns else df.columns[0]
    df['DateTime'] = pd.to_datetime(df[col])

    hist2y = yf.download("BTC-USD", period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(hist2y.columns, pd.MultiIndex):
        hist2y.columns = hist2y.columns.get_level_values(0)
    hist2y = hist2y.reset_index()
    col2 = 'Date' if 'Date' in hist2y.columns else hist2y.columns[0]
    hist2y['DateTime'] = pd.to_datetime(hist2y[col2])
    return df, hist2y

live_chart_df, hist_df = load_history()

if live_data:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("BTC REAL TIME", f"${live_data['price']:,.2f}", f"{live_data['change']:.2f}% 24h")
    c2.metric("24h High", f"${live_data['high']:,.2f}")
    c3.metric("24h Low", f"${live_data['low']:,.2f}")
    c4.metric("Status", "🔴 LIVE", f"{pd.Timestamp.now().strftime('%H:%M:%S')}")
else:
    st.warning("Binance API busy, using yfinance")
    live_data = {"price": float(live_chart_df['Close'].iloc[-1]), "change": 0, "high": 0, "low": 0}

# Keep live chart history for graph
st.subheader("Live Chart - Today (updates every 2 sec)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=live_chart_df['DateTime'], y=live_chart_df['Close'], name="Price", line=dict(color="#FF9900", width=2)))
fig.update_layout(template="plotly_dark", height=380)
st.plotly_chart(fig, use_container_width=True)

# Forecast
st.subheader("30-Day Forecast")
hist_df['Days'] = np.arange(len(hist_df))
model = LinearRegression()
model.fit(hist_df[['Days']], hist_df['Close'])
future = model.predict(np.arange(len(hist_df), len(hist_df)+30).reshape(-1,1))
future_dates = [hist_df['DateTime'].iloc[-1] + pd.Timedelta(days=i) for i in range(1,31)]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hist_df['DateTime'].tail(180), y=hist_df['Close'].tail(180), name="History", line=dict(color="#00C0FF")))
fig2.add_trace(go.Scatter(x=future_dates, y=future, name="Forecast", line=dict(color="#00FF88", dash="dash", width=3)))
fig2.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig2, use_container_width=True)
