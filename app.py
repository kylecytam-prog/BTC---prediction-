import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=30 * 1000, key="btc_refresh")

st.set_page_config(page_title="BTC Live Pro", page_icon="₿", layout="wide")

st.markdown("""
<div style='text-align: center; padding: 10px;'>
<h1>₿ BTC Live Pro</h1>
<p style='color: #888;'>Real-time BTC + 30-Day Forecast</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30)
def load_btc_live():
    live = yf.download("BTC-USD", period="1d", interval="1m", progress=False, auto_adjust=True)
    if isinstance(live.columns, pd.MultiIndex):
        live.columns = live.columns.get_level_values(0)
    hist = yf.download("BTC-USD", period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    live = live.reset_index()
    hist = hist.reset_index()
    # Fix column name - yfinance sometimes uses Date, sometimes Datetime
    date_col_live = 'Date' if 'Date' in live.columns else live.columns[0]
    date_col_hist = 'Date' if 'Date' in hist.columns else hist.columns[0]
    live['DateTime'] = pd.to_datetime(live[date_col_live])
    hist['DateTime'] = pd.to_datetime(hist[date_col_hist])
    return live, hist

live_df, hist_df = load_btc_live()

current_price = float(live_df['Close'].iloc[-1])
open_24h = float(live_df['Close'].iloc[0])
change_24h = (current_price / open_24h - 1) * 100
high_24h = float(live_df['High'].max())
low_24h = float(live_df['Low'].min())

c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC Live", f"${current_price:,.2f}", f"{change_24h:.2f}% 24h")
c2.metric("24h High", f"${high_24h:,.2f}")
c3.metric("24h Low", f"${low_24h:,.2f}")
c4.metric("Status", "🟢 LIVE", "30s refresh")

st.subheader("Live Today (1-min)")
fig_live = go.Figure()
fig_live.add_trace(go.Scatter(x=live_df['DateTime'], y=live_df['Close'], mode='lines', name='Live', line=dict(color="#FF9900", width=2)))
fig_live.update_layout(template="plotly_dark", height=400, yaxis_title="USD")
st.plotly_chart(fig_live, use_container_width=True)

st.subheader("2-Year History + 30-Day Forecast")
hist_df['Days'] = np.arange(len(hist_df))
model = LinearRegression()
model.fit(hist_df[['Days']], hist_df['Close'])
future_days = np.arange(len(hist_df), len(hist_df)+30).reshape(-1,1)
forecast = model.predict(future_days)
last_date = hist_df['DateTime'].iloc[-1]
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1,31)]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hist_df['DateTime'].tail(180), y=hist_df['Close'].tail(180), name="Last 180 Days", line=dict(color="#00C0FF")))
fig2.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(color="#00FF88", width=3, dash="dash")))
fig2.update_layout(template="plotly_dark", height=450, yaxis_title="USD")
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(pd.DataFrame({"Date": future_dates, "Forecast": np.round(forecast,2)}), use_container_width=True)
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')} - auto refresh")
