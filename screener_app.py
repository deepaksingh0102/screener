"""Momentum Screener — S&P 500 or every listed US stock.

Scans the chosen universe on 1-minute Alpaca bars and lists the ones matching
a bullish or a bearish intraday setup:

  Bullish: EMA9 > EMA21, MACD > signal, RSI in band, price > session VWAP,
           price inside the opening range, relative volume >= min, ADX > min
  Bearish: EMA9 < EMA21, MACD < signal, RSI in band, price < session VWAP,
           price inside the opening range, relative volume >= min, ADX > min

Indicator math mirrors the stock-analyzer app (Wilder's RSI/ADX, EMA 9/21,
MACD 12/26/9, 30-minute opening range, session-anchored VWAP).

"All listed stocks" runs a two-stage funnel: a snapshot-based volume
pre-filter (today's volume vs previous day's, >= 0.7x) narrows ~11k listings
to the day's active names, then the full 7-filter screen runs on those.
Final relative volume is always recomputed on the 20-day baseline.

Deploy: add this file to the stock-analyzer GitHub repo, then in Streamlit
Cloud create a new app from the same repo with main file `screener_app.py`
and copy the same Secrets (APCA_API_KEY_ID, APCA_API_SECRET_KEY, ALPACA_FEED).
"""
from time import sleep as _sleep

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

st.set_page_config(page_title="Momentum Screener", layout="wide")


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
# Timeframe labels -> trailing minutes the screen evaluates (None = whole
# session since market open). Mirrors the stock-analyzer's "Chart timeframe":
# candles stay 1-minute, the selector picks the window, and the indicators
# always carry the full session's warm-up (the window only sets which minutes
# the volume filter accumulates over).
TF_TRAILING = {"5 minutes": 5, "15 minutes": 15, "1 hour": 60,
               "Since market open": None}
NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"


@st.cache_data(ttl=86400, show_spinner="Loading market caps…")
def get_market_caps():
    """symbol -> market cap (USD) for US listings, from Nasdaq's public
    screener (no key needed). Nasdaq writes class shares as BF/A; Alpaca
    wants BF.B, hence the normalization. Rows with no cap are dropped."""
    r = requests.get(NASDAQ_SCREENER_URL,
                     params={"tableonly": "true", "limit": 25, "offset": 0,
                             "download": "true"},
                     headers={**UA, "Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    caps = {}
    for row in (r.json().get("data") or {}).get("rows") or []:
        sym = (row.get("symbol") or "").replace("/", ".")
        try:
            cap = float(row.get("marketCap") or 0)
        except (TypeError, ValueError):
            cap = 0.0
        if sym and cap > 0:
            caps[sym] = cap
    return caps


def _alpaca_listings():
    """Every tradable US equity on NYSE/NASDAQ/AMEX (unit/warrant-style
    symbols excluded)."""
    r = requests.get("https://api.alpaca.markets/v2/assets",
                     headers=_headers(),
                     params={"status": "active", "asset_class": "us_equity"},
                     timeout=60)
    if r.status_code in (401, 403):
        raise AlpacaAuthError(r.status_code)
    r.raise_for_status()
    syms, names = [], {}
    for a in r.json():
        sym = a.get("symbol", "")
        if (a.get("tradable") and a.get("exchange") in ("NYSE", "NASDAQ", "AMEX")
                and not any(x in sym for x in (".WS", ".U", "$", "+"))):
            syms.append(sym)
            names[sym] = a.get("name", "")
    return syms, names


def _spx500():
    # StringIO: otherwise pd.read_html mistakes the HTML string for a URL
    # (lxml then throws FileNotFoundError trying to fetch it).
    # Keep raw Wikipedia symbols (BRK.B, BF.B): Alpaca needs the dots.
    from io import StringIO
    html = requests.get(SPX_URL, headers=UA, timeout=30).text
    tbl = pd.read_html(StringIO(html))[0]
    syms = tbl["Symbol"].tolist()
    return syms, dict(zip(syms, tbl["Security"]))


@st.cache_data(ttl=86400, show_spinner="Loading stock universe…")
def get_universe(mode):
    """(tickers, names) for the chosen universe.

    "S&P 500": the Wikipedia list. "Top 1000 by market cap": Alpaca's
    listings ranked by Nasdaq screener caps (falls back to the S&P 500 with
    a notice if the cap feed fails). "All listed stocks": everything.
    """
    if mode == "S&P 500":
        return _spx500()
    tickers_all, names_all = _alpaca_listings()
    if mode == "Top 1000 by market cap":
        try:
            caps = get_market_caps()
            ranked = sorted(((caps[s], s) for s in tickers_all if s in caps),
                            reverse=True)
            top = [s for _, s in ranked[:1000]]
            if len(top) < 900:  # feed degraded; don't silently screen a stub
                raise ValueError("market-cap feed too thin")
            return top, {s: names_all[s] for s in top}
        except Exception:
            st.caption("⚠️ Market-cap feed was unreachable — screening the S&P 500 instead.")
            return _spx500()
    return tickers_all, names_all


@st.cache_data(ttl=600, show_spinner="Volume pre-filtering all listings…")
def prefilter_by_volume(_tickers, coarse_min=0.7):
    """Snapshot funnel for the all-listed universe: keep tickers whose volume
    so far today is >= coarse_min x yesterday's full-day volume. Coarse by
    design (yesterday is a noisy baseline) — the real screen recomputes
    relative volume on the 20-day average before applying the user's minimum.
    """
    def one(chunk):
        try:
            r = _get_with_backoff("https://data.alpaca.markets/v2/stocks/snapshots",
                                  _headers(), {"symbols": ",".join(chunk),
                                               "feed": _feed()}, timeout=60)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code in (401, 403):
                raise AlpacaAuthError(code)
            return []
        keep = []
        for sym, snap in (r.json() or {}).items():
            d = (snap or {}).get("dailyBar") or {}
            p = (snap or {}).get("prevDailyBar") or {}
            tv, pv = d.get("v") or 0, p.get("v") or 0
            if pv > 0 and tv / pv >= coarse_min:
                keep.append(sym)
        return keep

    chunks = [_tickers[i:i + 100] for i in range(0, len(_tickers), 100)]
    out = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for keep in ex.map(one, chunks):
            out.extend(keep)
    return out


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


def _get_with_backoff(url, headers, params, timeout=60, tries=3):
    """GET with exponential backoff on HTTP 429 (Alpaca caps data requests
    at 200/min; the all-listed universe comes close). Raises HTTPError after
    the final try."""
    delay = 2.0
    r = None
    for attempt in range(tries):
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 429 and attempt < tries - 1:
            _sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()  # pragma: no cover — loop always returns or raises
    return r


def _fetch_chunk(symbols, timeframe, start, end, limit):
    """One multi-symbol bars request, following next_page_token until done.

    Alpaca's `limit` is the TOTAL bars in one page across ALL symbols in the
    request (not per symbol), so a 200-ticker request for 3 days of 1-minute
    bars (~234k bars) returns only the first ~10k unless pages are followed —
    most tickers simply never appear in the response. Raises
    requests.HTTPError on failure.
    """
    out = {}
    page_token = None
    params = {"symbols": ",".join(symbols), "timeframe": timeframe,
              "start": start.isoformat(), "end": end.isoformat(),
              "limit": limit, "feed": _feed(), "adjustment": "raw", "sort": "asc"}
    while True:
        if page_token:
            params["page_token"] = page_token
        r = _get_with_backoff("https://data.alpaca.markets/v2/stocks/bars",
                              _headers(), params, timeout=60)
        js = r.json()
        for sym, bars in (js.get("bars") or {}).items():
            if bars:
                out.setdefault(sym, []).extend(bars)
        page_token = js.get("next_page_token")
        if not page_token:
            break
    return {sym: _bars_to_df(bars) for sym, bars in out.items()}


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
            if code == 429:
                # Already backed off inside _get_with_backoff; splitting would
                # only multiply requests against a hot rate limit.
                return {}, [(s, code) for s in chunk]
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
    """1-min bars, refreshed on every 30s cycle (TTL 30).

    Chunks of 25 symbols: one full session is 390 bars, so 25 x 390 = 9,750
    fits a single 10k-bar page (pagination is the safety net). Starts at
    today's open; if no session is found (weekend, pre-open, holiday), retries
    with a 5-day lookback so the last session still appears.
    """
    end = datetime.now(ET)
    today = end.replace(hour=0, minute=0, second=0, microsecond=0)
    out, failed = _fetch_multi(_tickers, "1Min", today, end, 10000, 25)
    if not out:
        out, failed = _fetch_multi(_tickers, "1Min",
                                   today - timedelta(days=5), end, 10000, 25)
    return out, failed


@st.cache_data(ttl=3600, show_spinner="Pulling daily bars for volume baselines…")
def fetch_daily(_tickers):
    end = datetime.now(ET)
    start = end - timedelta(days=45)
    return _fetch_multi(_tickers, "1D", start, end, 100, 80)


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
def screen(intraday, daily, names, timeframe, relvol_min, adx_min, rsi_lo, rsi_hi):
    bull, bear, session_date = [], [], None
    trail = TF_TRAILING[timeframe]
    for t, df in intraday.items():
        try:
            if df.empty:
                continue
            session_date = df.index.date.max()
            # Full regular session: indicators get the whole session's warm-up.
            day = df[df.index.date == session_date].between_time("09:30", "16:00")
            if len(day) < 29:  # ADX(14) needs 2n+1 bars
                continue
            o, h, l, c = day["Open"], day["High"], day["Low"], day["Close"]
            v = day["Volume"].fillna(0.0)

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

            orw = day.between_time("09:30", "10:00")
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
            # Relative volume over the trailing window vs the same slice of a
            # normal day (prorated 20-day average). This is what the timeframe
            # selector changes: "15 minutes" = hot in the last 15 minutes.
            win = day.tail(trail) if trail else day
            win_min = max((win.index[-1] - win.index[0]).total_seconds() / 60 + 1, 1)
            relvol = (float(win["Volume"].fillna(0.0).sum()) / (avg_vol * win_min / 390.0)
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
    b = pd.DataFrame(bull)
    if len(b):
        b = b.sort_values("RelVol", ascending=False)
    br = pd.DataFrame(bear)
    if len(br):
        br = br.sort_values("RelVol", ascending=False)
    return b, br, session_date


# ---------------------------------------------------------------- UI
st.title("Momentum Screener")
st.caption("1-minute Alpaca bars · same indicator math as the stock-analyzer "
           "(EMA 9/21, MACD 12/26/9, Wilder's RSI/ADX 14, 30-min opening range, session VWAP)")


def market_open_now() -> bool:
    now = datetime.now(ET)
    return now.weekday() < 5 and time(9, 30) <= now.time() <= time(16, 0)


with st.sidebar:
    st.header("Screen settings")
    st.selectbox("Universe",
                 ["S&P 500", "Top 1000 by market cap", "All listed stocks"],
                 index=1, key="universe",
                 help="Top 1000 = the 1,000 biggest US listings by market cap (Nasdaq screener, "
                      "refreshed daily). All listed stocks = ~11k tickers with a snapshot-based "
                      "volume pre-filter — a full scan takes ~1–2 minutes.")
    st.selectbox("Timeframe",
                 ["5 minutes", "15 minutes", "1 hour", "Since market open"],
                 index=3, key="timeframe",
                 help="Trailing window the screen evaluates. Like the stock-analyzer's "
                      "Chart timeframe: indicators always carry the full session's warm-up; "
                      "the window sets the minutes the relative-volume filter accumulates over.")
    st.number_input("Min relative volume", 0.5, 10.0, 1.5, 0.1, key="relvol_min")
    st.number_input("Min ADX", 10.0, 50.0, 25.0, 1.0, key="adx_min")
    st.slider("RSI band", 0, 100, (30, 70), key="rsi_band")
    st.divider()
    if st.button("↻ Refresh now", use_container_width=True):
        st.session_state["_force_scan"] = True
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Feed: {_feed().upper()} · keys via Streamlit Secrets")
    if market_open_now():
        st.caption("🟢 Live: re-screens during market hours "
                   "(~3 min on All listed, ~1 min on Top 1000, ~30s on S&P 500).")
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
    universe_mode = st.session_state.get("universe", "Top 1000 by market cap")
    timeframe = st.session_state.get("timeframe", "Since market open")
    relvol_min = st.session_state.get("relvol_min", 1.5)
    adx_min = st.session_state.get("adx_min", 25.0)
    rsi_lo, rsi_hi = st.session_state.get("rsi_band", (30, 70))
    big = universe_mode == "All listed stocks"
    mid = universe_mode == "Top 1000 by market cap"
    cadence = 180 if big else (60 if mid else 25)  # min seconds between full scans
    scan_key = (universe_mode, timeframe, relvol_min, adx_min, rsi_lo, rsi_hi)

    now = datetime.now(ET)
    force = st.session_state.pop("_force_scan", False)
    cached = st.session_state.get("_last_scan")
    last_ts = st.session_state.get("_last_scan_ts")
    if (cached and cached[0] == scan_key and not force and last_ts
            and (now - last_ts).total_seconds() < cadence):
        # Serve the previous scan; a full re-scan runs on the next cycle.
        bull, bear, session_date, asof, uni_label, n_scan = cached[1:]
        wait = int(cadence - (now - last_ts).total_seconds())
        st.caption(f"⏳ Next full scan in ~{wait}s — showing the last completed scan.")
    else:
        try:
            tickers, names = get_universe(universe_mode)
            if universe_mode == "All listed stocks":
                candidates = prefilter_by_volume(tickers)
                uni_note = f"{len(tickers):,} listed · {len(candidates):,} passed volume pre-filter"
            else:
                candidates, uni_note = tickers, f"{len(tickers):,} ({universe_mode})"
            intra, intra_failed = fetch_intraday(candidates)
            daily, daily_failed = fetch_daily(candidates)
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

        bull, bear, session_date = screen(intra, daily, names, timeframe,
                                         relvol_min, adx_min, rsi_lo, rsi_hi)
        if session_date is None:
            st.error("Alpaca returned no bars for any ticker — check the app logs, "
                     "or try Refresh now in a moment.")
            st.stop()

        asof = (f"{session_date} · last {timeframe}" if timeframe != "Since market open"
                else f"{session_date} · since market open")
        uni_label = uni_note
        n_scan = len(intra)
        st.session_state["_last_scan"] = (scan_key, bull, bear, session_date,
                                          asof, uni_label, n_scan)
        st.session_state["_last_scan_ts"] = datetime.now(ET)

    refreshed = datetime.now(ET).strftime("%H:%M:%S ET")
    st.info(f"**{n_scan}** tickers scanned · {uni_label} · session **{asof}** · "
            f"**{len(bull)}** bullish / **{len(bear)}** bearish matches · "
            f"updated {refreshed}", icon="📊")

    tf_slug = timeframe.replace(" ", "_")
    tab_bull, tab_bear = st.tabs([f"🟢 Bullish ({len(bull)})", f"🔴 Bearish ({len(bear)})"])
    with tab_bull:
        if len(bull):
            st.dataframe(bull, column_config=_cols, use_container_width=True, hide_index=True)
            st.download_button("Download bullish CSV",
                               bull.to_csv(index=False).encode(),
                               f"bullish_{session_date}_{tf_slug}.csv",
                               "text/csv")
        else:
            st.write("No bullish matches — try loosening relative volume or ADX, or a wider timeframe.")
    with tab_bear:
        if len(bear):
            st.dataframe(bear, column_config=_cols, use_container_width=True, hide_index=True)
            st.download_button("Download bearish CSV",
                               bear.to_csv(index=False).encode(),
                               f"bearish_{session_date}_{tf_slug}.csv",
                               "text/csv")
        else:
            st.write("No bearish matches — try loosening relative volume or ADX, or a wider timeframe.")

    with st.expander("How the filters work"):
        st.markdown("""
**Bullish:** EMA9 > EMA21 · MACD > signal · RSI inside the band · price > session VWAP ·
price inside the 9:30–10:00 opening range · relative volume ≥ minimum · ADX > minimum.

**Bearish:** the mirror — EMA9 < EMA21 · MACD < signal · price < VWAP, everything else the same.

Relative volume = the selected timeframe's volume ÷ (20-day average daily volume,
prorated for that many minutes). "15 minutes" = surging in the last 15 minutes; "Since market
open" = session-long volume buildup. On the "All listed stocks" universe a snapshot-based
pre-filter first keeps only names trading ≥ 0.7x yesterday's volume, so the full screen runs
on the day's active names; the final relative volume is always computed on the 20-day baseline.
ADX measures trend *strength* (>25 = conviction); it says nothing
about direction. This is a setup screen, not a prediction — 1-minute directional calls
test around a coin flip; use these as candidates for your own read.
""")


screener_body()
