"""S&P 500 Momentum Screener.

Scans every S&P 500 stock on 1-minute Alpaca bars and lists the ones matching
a bullish or a bearish intraday setup:

  Bullish: EMA9 > EMA21, MACD > signal, RSI in band, price > session VWAP,
           price inside the opening range, relative volume >= min, ADX > min
  Bearish: EMA9 < EMA21, MACD < signal, RSI in band, price < session VWAP,
           price inside the opening range, relative volume >= min, ADX > min

Indicator math mirrors the stock-analyzer app (Wilder's RSI/ADX, EMA 9/21,
MACD 12/26/9, 30-minute opening range, session-anchored VWAP).

Deploy: add this file to the stock-analyzer GitHub repo, then in Streamlit
Cloud create a new app from the same repo with main file `screener_app.py`
and copy the same Secrets (APCA_API_KEY_ID, APCA_API_SECRET_KEY, ALPACA_FEED).
"""
import numpy as np
import pandas as pd
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
SPX_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

st.set_page_config(page_title="S&P 500 Momentum Screener", layout="wide")


# ---------------------------------------------------------------- secrets
def _secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def alpaca_configured() -> bool:
    k = _secret("APCA_API_KEY_ID")
    s = _secret("APCA_API_SECRET_KEY")
    return bool(k and s and k not in ("", "Dummy", "PASTE_YOUR_KEY_HERE"))


def _headers() -> dict:
    return {"APCA-API-KEY-ID": _secret("APCA_API_KEY_ID"),
            "APCA-API-SECRET-KEY": _secret("APCA_API_SECRET_KEY"),
            "Accept": "application/json"}


def _feed() -> str:
    return _secret("ALPACA_FEED") or "iex"


# ---------------------------------------------------------------- universe
@st.cache_data(ttl=86400, show_spinner="Loading S&P 500 list…")
def get_spx():
    # StringIO: otherwise pd.read_html mistakes the HTML string for a URL
    # (lxml then throws FileNotFoundError trying to fetch it).
    # Keep raw Wikipedia symbols (BRK.B, BF.B): Alpaca needs the dots.
    # (The dash form is only for Yahoo Finance.)
    from io import StringIO
    html = requests.get(SPX_URL, headers=UA, timeout=30).text
    tbl = pd.read_html(StringIO(html))[0]
    syms = tbl["Symbol"].tolist()
    names = dict(zip(syms, tbl["Security"]))
    return syms, names


# ---------------------------------------------------------------- data
def _bars_to_df(bars) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"time": pd.Timestamp(b["t"]), "Open": b["o"], "High": b["h"],
          "Low": b["l"], "Close": b["c"], "Volume": b["v"]} for b in bars]
    ).set_index("time").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


class AlpacaAuthError(RuntimeError):
    """Alpaca answered 401/403 — the API keys are rejected."""


def _fetch_chunk(symbols, timeframe, start, end, limit):
    """One multi-symbol bars request. Raises requests.HTTPError on failure."""
    params = {"symbols": ",".join(symbols), "timeframe": timeframe,
              "start": start.isoformat(), "end": end.isoformat(),
              "limit": limit, "feed": _feed(), "adjustment": "raw", "sort": "desc"}
    r = requests.get("https://data.alpaca.markets/v2/stocks/bars",
                     headers=_headers(), params=params, timeout=60)
    r.raise_for_status()
    return {sym: _bars_to_df(bars)
            for sym, bars in (r.json().get("bars") or {}).items() if bars}


def _fetch_multi(tickers, timeframe, start, end, limit, chunk_size):
    """Threaded multi-symbol fetch -> (bars_by_ticker, [(ticker, http_code)]).

    A failed chunk is split and retried recursively, so one bad symbol can't
    sink ~200 tickers or red-screen the app. A 401/403 raises AlpacaAuthError
    (every chunk would fail the same way).
    """
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    out, failed = {}, []

    def one(chunk):
        try:
            return _fetch_chunk(chunk, timeframe, start, end, limit), []
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code in (401, 403):
                raise AlpacaAuthError(code)
            if len(chunk) > 1:
                mid = len(chunk) // 2
                r1, f1 = one(chunk[:mid])
                r2, f2 = one(chunk[mid:])
                r1.update(r2)
                return r1, f1 + f2
            return {}, [(chunk[0], code)]

    with ThreadPoolExecutor(max_workers=3) as ex:
        for d, f in ex.map(one, chunks):
            out.update(d)
            failed.extend(f)
    return out, failed


@st.cache_data(ttl=30, show_spinner=False)
def fetch_intraday(_tickers):
    """1-min bars, refreshed on every 30s cycle (TTL 30)."""
    end = datetime.now(ET)
    start = (end - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
    return _fetch_multi(_tickers, "1Min", start, end, 10000, 200)


@st.cache_data(ttl=3600, show_spinner="Pulling daily bars for volume baselines…")
def fetch_daily(_tickers):
    end = datetime.now(ET)
    start = end - timedelta(days=45)
    return _fetch_multi(_tickers, "1D", start, end, 100, 150)


# ---------------------------------------------------------------- indicators (vectorized Wilder's)
def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    ag = d.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False).mean()
    al = (-d.clip(upper=0.0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = ag / al.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def adx_vec(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    up, dn = high.diff(), -low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1.0 / n, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1.0 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False).mean(), pdi, mdi


# ---------------------------------------------------------------- screen
def screen(intraday, daily, names, window_end, relvol_min, adx_min, rsi_lo, rsi_hi):
    bull, bear = [], []
    for t, df in intraday.items():
        try:
            if df.empty:
                continue
            session_date = df.index.date.max()
            day = df[df.index.date == session_date]
            sess = day.between_time("09:30", window_end)
            if len(sess) < 29:  # ADX(14) needs 2n+1 bars
                continue
            o, h, l, c = sess["Open"], sess["High"], sess["Low"], sess["Close"]
            v = sess["Volume"].fillna(0.0)

            ema9 = c.ewm(span=9, adjust=False).mean().iloc[-1]
            ema21 = c.ewm(span=21, adjust=False).mean().iloc[-1]
            macd = (c.ewm(span=12, adjust=False).mean()
                    - c.ewm(span=26, adjust=False).mean())
            sig = macd.ewm(span=9, adjust=False).mean().iloc[-1]
            macd = macd.iloc[-1]
            rsi = wilder_rsi(c).iloc[-1]
            vwap = ((h + l + c) / 3.0 * v).cumsum().iloc[-1] / v.cumsum().iloc[-1]
            adx, _, _ = adx_vec(h, l, c)
            adx = adx.iloc[-1]
            price = float(c.iloc[-1])

            orw = sess.between_time("09:30", "10:00")
            orh, orl = float(orw["High"].max()), float(orw["Low"].min())

            d = daily.get(t)
            prev_close, pct_chg, avg_vol = np.nan, np.nan, np.nan
            if d is not None and len(d) >= 2:
                prev_close = float(d["Close"].iloc[-2]) if len(d) >= 2 else np.nan
                if prev_close:
                    pct_chg = 100 * (price / prev_close - 1)
                vols = d["Volume"].iloc[:-1].tail(20)
                if len(vols) >= 10:
                    avg_vol = float(vols.mean())
            relvol = (float(v.sum()) / (avg_vol * len(sess) / 390.0)
                      if avg_vol and avg_vol > 0 else np.nan)

            row = dict(Ticker=t, Name=names.get(t, ""), Price=round(price, 2),
                       Chg_Pct=round(pct_chg, 2), EMA9=round(ema9, 2),
                       EMA21=round(ema21, 2), MACD=round(macd, 3),
                       Signal=round(sig, 3), RSI=round(rsi, 1),
                       VWAP=round(vwap, 2), OR_Low=round(orl, 2),
                       OR_High=round(orh, 2), ADX=round(adx, 1),
                       RelVol=round(relvol, 2))

            in_or = orl <= price <= orh
            rsi_ok = rsi_lo < rsi < rsi_hi
            vol_ok = relvol >= relvol_min
            adx_ok = adx > adx_min
            if (ema9 > ema21 and macd > sig and rsi_ok and price > vwap
                    and in_or and vol_ok and adx_ok):
                bull.append(row)
            elif (ema9 < ema21 and macd < sig and rsi_ok and price < vwap
                    and in_or and vol_ok and adx_ok):
                bear.append(row)
        except Exception:
            continue
    b = pd.DataFrame(bull).sort_values("RelVol", ascending=False)
    br = pd.DataFrame(bear).sort_values("RelVol", ascending=False)
    return b, br, session_date


# ---------------------------------------------------------------- UI
st.title("S&P 500 Momentum Screener")
st.caption("1-minute Alpaca bars · same indicator math as the stock-analyzer "
           "(EMA 9/21, MACD 12/26/9, Wilder's RSI/ADX 14, 30-min opening range, session VWAP)")


def market_open_now() -> bool:
    now = datetime.now(ET)
    return now.weekday() < 5 and time(9, 30) <= now.time() <= time(16, 0)


with st.sidebar:
    st.header("Screen settings")
    st.selectbox("Evaluate as of",
                 ["10:00", "11:00", "12:00", "13:00", "Full session"],
                 index=1, key="window_label")
    st.session_state["window_end"] = ("16:00"
                                      if st.session_state["window_label"] == "Full session"
                                      else st.session_state["window_label"])
    st.number_input("Min relative volume", 0.5, 10.0, 1.5, 0.1, key="relvol_min")
    st.number_input("Min ADX", 10.0, 50.0, 25.0, 1.0, key="adx_min")
    st.slider("RSI band", 0, 100, (30, 70), key="rsi_band")
    st.divider()
    if st.button("↻ Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Feed: {_feed().upper()} · keys via Streamlit Secrets")
    if market_open_now():
        st.caption("🟢 Live: re-screens every 30 seconds during market hours.")
    else:
        st.caption("⚪ Market closed — showing the last session (no auto-refresh).")

if not alpaca_configured():
    st.warning("Add your Alpaca keys (APCA_API_KEY_ID, APCA_API_SECRET_KEY, ALPACA_FEED) "
               "under the app's Settings → Secrets, then reboot the app.")
    st.stop()

_cols = {"Ticker": st.column_config.TextColumn("Ticker"),
         "Name": st.column_config.TextColumn("Name"),
         "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
         "Chg_Pct": st.column_config.NumberColumn("Chg %", format="%.2f%%"),
         "EMA9": st.column_config.NumberColumn("EMA9", format="%.2f"),
         "EMA21": st.column_config.NumberColumn("EMA21", format="%.2f"),
         "MACD": st.column_config.NumberColumn("MACD", format="%.3f"),
         "Signal": st.column_config.NumberColumn("Signal", format="%.3f"),
         "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
         "VWAP": st.column_config.NumberColumn("VWAP", format="%.2f"),
         "OR_Low": st.column_config.NumberColumn("OR Low", format="%.2f"),
         "OR_High": st.column_config.NumberColumn("OR High", format="%.2f"),
         "ADX": st.column_config.NumberColumn("ADX", format="%.1f"),
         "RelVol": st.column_config.NumberColumn("RelVol", format="%.2fx")}


@st.fragment(run_every=30 if market_open_now() else None)
def screener_body():
    # The fragment re-runs every 30s in market hours; it reads the sidebar
    # settings from session state (never as fragment args) so control changes
    # always take effect.
    window_end = st.session_state.get("window_end", "11:00")
    relvol_min = st.session_state.get("relvol_min", 1.5)
    adx_min = st.session_state.get("adx_min", 25.0)
    rsi_lo, rsi_hi = st.session_state.get("rsi_band", (30, 70))

    try:
        tickers, names = get_spx()
        intra, intra_failed = fetch_intraday(tickers)
        daily, daily_failed = fetch_daily(tickers)
    except AlpacaAuthError as e:
        code = e.args[0]
        if code == 401:
            st.error("Alpaca rejected the API keys (HTTP 401). In **this** app's Settings → Secrets, "
                     "re-check APCA_API_KEY_ID and APCA_API_SECRET_KEY — secrets don't copy over to a "
                     "new app automatically, and a typo when pasting is the usual cause.")
        else:
            st.error(f"Alpaca refused the request (HTTP {code}) — the keys may lack market-data access.")
        st.stop()

    skipped = sorted({t for t, _ in intra_failed} | {t for t, _ in daily_failed})
    if skipped:
        st.warning(f"Skipped {len(skipped)} tickers Alpaca wouldn't serve: "
                   f"{', '.join(skipped[:10])}{' …' if len(skipped) > 10 else ''}")

    bull, bear, session_date = screen(intra, daily, names, window_end,
                                     relvol_min, adx_min, rsi_lo, rsi_hi)

    asof = (f"{session_date} as of {window_end} ET" if window_end != "16:00"
            else f"{session_date} full session")
    refreshed = datetime.now(ET).strftime("%H:%M:%S ET")
    st.info(f"**{len(intra)}** tickers scanned · session **{asof}** · "
            f"**{len(bull)}** bullish / **{len(bear)}** bearish matches · "
            f"updated {refreshed}", icon="📊")
    if window_end == "10:00":
        st.caption("⚠️ Early window: indicators run on ~30 bars, so EMA/MACD/ADX are on thin warmup.")

    tab_bull, tab_bear = st.tabs([f"🟢 Bullish ({len(bull)})", f"🔴 Bearish ({len(bear)})"])
    with tab_bull:
        if len(bull):
            st.dataframe(bull, column_config=_cols, use_container_width=True, hide_index=True)
            st.download_button("Download bullish CSV",
                               bull.to_csv(index=False).encode(),
                               f"bullish_{session_date}_{window_end.replace(':', '')}.csv",
                               "text/csv")
        else:
            st.write("No bullish matches — try loosening relative volume or ADX, or an earlier window.")
    with tab_bear:
        if len(bear):
            st.dataframe(bear, column_config=_cols, use_container_width=True, hide_index=True)
            st.download_button("Download bearish CSV",
                               bear.to_csv(index=False).encode(),
                               f"bearish_{session_date}_{window_end.replace(':', '')}.csv",
                               "text/csv")
        else:
            st.write("No bearish matches — try loosening relative volume or ADX, or an earlier window.")

    with st.expander("How the filters work"):
        st.markdown("""
**Bullish:** EMA9 > EMA21 · MACD > signal · RSI inside the band · price > session VWAP ·
price inside the 9:30–10:00 opening range · relative volume ≥ minimum · ADX > minimum.

**Bearish:** the mirror — EMA9 < EMA21 · MACD < signal · price < VWAP, everything else the same.

Relative volume = today's volume so far ÷ (20-day average daily volume, prorated for
elapsed session time). ADX measures trend *strength* (>25 = conviction); it says nothing
about direction. This is a setup screen, not a prediction — 1-minute directional calls
test around a coin flip; use these as candidates for your own read.
""")


screener_body()
