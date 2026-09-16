import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

st.set_page_config(page_title="BTC Event Model V2", layout="wide")
st.title("BTC Price Generator - With Real Macro + ETF Data")

@st.cache_data
def load_btc():
    btc = yf.download("BTC-USD", start="2018-01-01", auto_adjust=True)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    return btc.reset_index()

@st.cache_data
def load_macro():
    # 1. Fed Funds Rate from FRED CSV (no API key needed)
    try:
        fed = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS")
        fed["DATE"] = pd.to_datetime(fed["DATE"])
        fed.rename(columns={"DATE":"Date","FEDFUNDS":"fed_rate"}, inplace=True)
    except:
        fed = pd.DataFrame({"Date":[pd.Timestamp.now()], "fed_rate":[5.0]})

    # 2. DXY Dollar Index and 10Y Yield as macro proxies
    dxy = yf.download("DX-Y.NYB", start="2018-01-01", auto_adjust=True)
    if isinstance(dxy.columns, pd.MultiIndex):
        dxy.columns = dxy.columns.get_level_values(0)
    dxy = dxy.reset_index()[["Date","Close"]].rename(columns={"Close":"dxy"})

    # 3. IBIT ETF as ETF flow proxy (net inflows move IBIT volume/price)
    ibit = yf.download("IBIT", start="2024-01-10", auto_adjust=True)
    if isinstance(ibit.columns, pd.MultiIndex):
        ibit.columns = ibit.columns.get_level_values(0)
    ibit = ibit.reset_index()[["Date","Close","Volume"]].rename(columns={"Close":"ibit_price","Volume":"etf_volume"})

    return fed, dxy, ibit

btc_raw = load_btc()
fed_df, dxy_df, ibit_df = load_macro()

# Merge all
df = btc_raw[["Date","Close"]].copy()
df.rename(columns={"Close":"price"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])

df = pd.merge_asof(df.sort_values("Date"), fed_df.sort_values("Date"), on="Date", direction="backward")
df = pd.merge_asof(df.sort_values("Date"), dxy_df.sort_values("Date"), on="Date", direction="backward")
df = pd.merge_asof(df.sort_values("Date"), ibit_df.sort_values("Date"), on="Date", direction="backward")

# Fill
df["fed_rate"] = df["fed_rate"].ffill().fillna(5.0)
df["dxy"] = df["dxy"].ffill()
df["etf_volume"] = df["etf_volume"].fillna(0)
df["ibit_price"] = df["ibit_price"].ffill().fillna(0)

# Events still matter
events = [
    ("2020-05-11", "Halving", 1.0),
    ("2022-11-11", "FTX", -1.0),
    ("2024-01-10", "ETF Approval", 1.0),
    ("2024-04-19", "Halving", 1.0),
]
events_df = pd.DataFrame(events, columns=["Date","event","impact"])
events_df["Date"] = pd.to_datetime(events_df["Date"])
df = pd.merge_asof(df.sort_values("Date"), events_df.sort_values("Date"), on="Date", direction="backward")
df["impact"] = df["impact"].fillna(0)
df["event"] = df["event"].fillna("none")

# Features
df["ma_30"] = df["price"].rolling(30).mean()
df["ma_90"] = df["price"].rolling(90).mean()
df["vol_30"] = df["price"].pct_change().rolling(30).std()
df["fed_change"] = df["fed_rate"].diff(30)
df["dxy_change"] = df["dxy"].pct_change(30)
df["etf_flow_proxy"] = df["etf_volume"].rolling(7).mean() / 1e7

last_halving = pd.to_datetime("2024-04-19")
df["days_since_halving"] = (df["Date"] - last_halving).dt.days
df.loc[df["Date"] < "2024-04-19", "days_since_halving"] = (df["Date"] - pd.to_datetime("2020-05-11")).dt.days

df["event_decay"] = df["impact"] * 0.95 ** df.groupby("event").cumcount()
df = df.dropna()

# Train
features = ["ma_30","ma_90","vol_30","fed_rate","fed_change","dxy","dxy_change","etf_flow_proxy","days_since_halving","event_decay"]
X = df[features]
y = df["price"].shift(-7)
X, y = X[:-7], y.dropna()

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

# UI
st.sidebar.header("Forecast Controls")
days_ahead = st.sidebar.slider("Days to generate", 7, 180, 30)
fed_scenario = st.sidebar.slider("Fed Rate Scenario (future)", 0.0, 7.0, float(df["fed_rate"].iloc[-1]))
etf_scenario = st.sidebar.slider("ETF Demand Multiplier", 0.5, 2.0, 1.0)

# Forecast
future = []
curr = X.iloc[-1:].copy()
for i in range(days_ahead):
    curr["days_since_halving"] += 1
    curr["event_decay"] *= 0.98
    curr["fed_rate"] = fed_scenario
    curr["etf_flow_proxy"] = curr["etf_flow_proxy"] * etf_scenario
    pred = model.predict(curr[features])[0]
    future.append(pred)
    curr["ma_30"] = curr["ma_30"]*0.97 + pred*0.03

# Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["price"], name="Real BTC"))
future_dates = pd.date_range(df["Date"].iloc[-1], periods=days_ahead+1, freq='D')[1:]
fig.add_trace(go.Scatter(x=future_dates, y=future, name="Generated with Macro+ETF", line=dict(dash='dash', width=3)))

for _, ev in events_df.iterrows():
    fig.add_vline(x=ev["Date"], line_dash="dot", annotation_text=ev["event"])

st.plotly_chart(fig, use_container_width=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Last BTC", f"${df['price'].iloc[-1]:,.0f}")
c2.metric("Fed Rate", f"{df['fed_rate'].iloc[-1]:.2f}%")
c3.metric("DXY", f"{df['dxy'].iloc[-1]:.2f}")
c4.metric(f"Forecast {days_ahead}d", f"${future[-1]:,.0f}")

# Feature importance
st.subheader("What drives the model?")
imp = pd.DataFrame({"factor": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
st.bar_chart(imp.set_index("factor"))

st.caption("Sources: BTC via Yahoo Finance, Fed Rate via FRED, DXY via Yahoo, IBIT volume as ETF flow proxy. For research only.")
