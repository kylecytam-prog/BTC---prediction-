@st.cache_data(ttl=1)
def get_live_price():
    # Try 3 sources - one will work in China
    sources = [
        "https://api.coinbase.com/v2/prices/BTC-USD/spot",
        "https://api.okx.com/api/v5/market/ticker?instId=BTC-USDT",
        "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true&include_last_updated_at=true"
    ]
    try:
        # Try Coinbase first - works in China
        r = requests.get(sources[0], timeout=3).json()
        price = float(r['data']['amount'])
        # Get 24h stats from coingecko
        g = requests.get(sources[2], timeout=3).json()
        change = g['bitcoin'].get('usd_24h_change', 0)
        return {"price": price, "high": price*1.01, "low": price*0.99, "change": change, "volume": 0}
    except:
        try:
            # Try OKX
            r = requests.get(sources[1], timeout=3).json()
            d = r['data'][0]
            return {"price": float(d['last']), "high": float(d['high24h']), "low": float(d['low24h']), "change": float(d.get('volCcy24h', 0)), "volume": 0}
        except:
            return None
