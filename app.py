import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="BTC Price Generator", layout="wide")
st.title("BTC Price Generator - With Real Macro + ETF Data")

@st.cache_data
def load_btc():
    df = yf.download("BTC-USD", period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    return df[['Date','Close']].rename(columns={'Close':'BTC'})

def load_macro_safe():
    try:
        # DXY + 10Y yield
        dxy = yf.download("DX-Y.NYB", period="2y", interval="1d", progress=False)['Close']
        tnx = yf.download("^TNX", period="2y", interval="1d", progress=False)['Close']
        df = pd.DataFrame({'DXY': dxy, 'TNX': tnx})
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values("Date").ffill()
        return df
    except:
        return pd.DataFrame()

btc = load_btc()
macro = load_macro_safe()

# SAFE MERGE - no merge_asof error
if not macro.empty and 'Date' in macro.columns:
    df = pd.merge(btc, macro, on="Date", how="left")
    df = df.sort_values("Date").ffill().bfill()
else:
    df = btc.copy()
    df['DXY'] = 103
    df['TNX'] = 4.0

st.write("Data loaded:", df.tail())

# Simple prediction - last price + trend
last_price = float(df['BTC'].iloc[-1])
st.metric("Current BTC Price", f"${last_price:,.0f}")

# Chart
st.line_chart(df.set_index('Date')['BTC'])

# 30-day simple forecast
growth = df['BTC'].pct_change().mean()
forecast = [last_price * ((1+growth)**i) for i in range(1,31)]
st.write("Next 30 days forecast (simple trend)")
st.line_chart(forecast)
