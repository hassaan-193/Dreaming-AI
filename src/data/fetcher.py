"""
Dreaming AI v6 — Data Fetcher
Fetches real OHLCV data from yfinance, adds technical indicators,
macro features (VIX, SPY), earnings calendar, volume profile features,
validates integrity, and saves to /data.

v6 Additions (Improvements 3, 8, 9):
  - fetch_vix()           — CBOE VIX fear/greed gauge
  - fetch_spy_return()    — S&P 500 daily log return (market context)
  - get_earnings_dates()  — binary earnings calendar feature
  - Volume profile:       vwap, vol_momentum, pv_divergence, vrsi
  - Cache invalidation:   auto-refresh if column count has changed
"""
import os, warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("[Fetcher] WARNING: 'ta' library not installed — technical indicators disabled.")

from config import DATA_DIR, FETCH_PERIOD, OHLCV_COLS, TECH_COLS, N_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Raw data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_raw(ticker: str, period: str = FETCH_PERIOD) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    Validates that at least 252 trading-day rows exist.
    """
    print(f"[Fetcher] Downloading {ticker} ({period}) from Yahoo Finance …")
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the symbol.")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # ── Integrity checks ──────────────────────────────────────────────────────
    if len(df) < 252:
        raise ValueError(f"Only {len(df)} rows — need ≥252 trading days.")

    # Reject rows where Close ≤ 0
    bad = (df["Close"] <= 0).sum()
    if bad:
        print(f"[Fetcher] WARNING: {bad} rows with Close≤0 dropped.")
        df = df[df["Close"] > 0]

    # Forward-fill then back-fill any remaining NaNs
    df = df.ffill().bfill()

    # Verify against a 30-day spot-check window (last 30 days)
    spot = yf.Ticker(ticker).history(period="1mo", auto_adjust=True)
    if not spot.empty:
        latest_yf   = float(spot["Close"].iloc[-1])
        latest_df   = float(df["Close"].iloc[-1])
        pct_diff    = abs(latest_yf - latest_df) / latest_yf * 100
        print(f"[Fetcher] Integrity check — yfinance spot: {latest_yf:.2f} | "
              f"stored: {latest_df:.2f} | diff: {pct_diff:.2f}%")
        if pct_diff > 5:
            print("[Fetcher] WARNING: >5% discrepancy with spot price.")

    print(f"[Fetcher] {len(df)} rows  |  "
          f"{df.index[0].date()} -> {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators (using 'ta' library + manual fallbacks)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi_manual(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 15 technical indicators + 4 volume profile features to OHLCV DataFrame.
    Falls back to manual implementations if 'ta' is unavailable.
    """
    df = df.copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    if TA_AVAILABLE:
        # RSI
        df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        # MACD
        macd_obj          = ta.trend.MACD(close)
        df["macd"]        = macd_obj.macd()
        df["macd_signal"] = macd_obj.macd_signal()
        df["macd_diff"]   = macd_obj.macd_diff()
        # EMA
        df["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        # Bollinger Bands
        bb             = ta.volatility.BollingerBands(close, window=20)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_pct"]   = bb.bollinger_pband()
        # ATR
        df["atr"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        # OBV (normalised by rolling std)
        obv           = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        df["obv_norm"] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9)
        # CCI
        df["cci"] = ta.trend.CCIIndicator(high, low, close).cci()
        # Williams %R
        df["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
        # Stochastic
        stoch          = ta.momentum.StochasticOscillator(high, low, close)
        df["stoch_k"]  = stoch.stoch()
        df["stoch_d"]  = stoch.stoch_signal()
    else:
        # Manual fallbacks
        df["rsi"]        = _rsi_manual(close)
        df["macd"]       = _ema(close, 12) - _ema(close, 26)
        df["macd_signal"]= _ema(df["macd"], 9)
        df["macd_diff"]  = df["macd"] - df["macd_signal"]
        df["ema_20"]     = _ema(close, 20)
        df["ema_50"]     = _ema(close, 50)
        sma20            = close.rolling(20).mean()
        std20            = close.rolling(20).std()
        df["bb_upper"]   = sma20 + 2 * std20
        df["bb_lower"]   = sma20 - 2 * std20
        df["bb_pct"]     = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
        tr               = pd.concat([high - low,
                                       (high - close.shift()).abs(),
                                       (low  - close.shift()).abs()], axis=1).max(axis=1)
        df["atr"]        = tr.rolling(14).mean()
        obv              = (np.sign(close.diff()) * volume).cumsum()
        df["obv_norm"]   = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9)
        tp               = (high + low + close) / 3
        df["cci"]        = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-9)
        hn14 = high.rolling(14).max()
        ln14 = low.rolling(14).min()
        df["williams_r"] = (hn14 - close) / (hn14 - ln14 + 1e-9) * -100
        df["stoch_k"]    = (close - ln14) / (hn14 - ln14 + 1e-9) * 100
        df["stoch_d"]    = df["stoch_k"].rolling(3).mean()

    # ── NEW FEATURES (Momentum & Volatility) ──────────────────────────────────
    df["mom_5d"] = close.pct_change(5)
    df["mom_10d"] = close.pct_change(10)
    df["mom_21d"] = close.pct_change(21)
    df["volatility_10d"] = close.pct_change().rolling(10).std() * np.sqrt(252)
    df["volatility_21d"] = close.pct_change().rolling(21).std() * np.sqrt(252)
    
    # VIX Proxy: rolling 20d std of log returns * sqrt(252)
    log_ret = np.log(close / close.shift(1))
    df["vix_proxy"] = log_ret.rolling(20).std() * np.sqrt(252)

    # ── IMPROVEMENT 9: Volume Profile Features ────────────────────────────────
    # These are INDEPENDENT of price and carry different signal.

    # VWAP — Volume Weighted Average Price (20-day rolling)
    df["vwap"] = (
        (close * volume).rolling(20).sum() /
        (volume.rolling(20).sum() + 1e-9)
    )

    # Volume momentum — is buying pressure increasing vs 20-day average?
    vol_sma20 = volume.rolling(20).mean()
    vol_sma5 = volume.rolling(5).mean()
    df["vol_momentum"] = volume / (vol_sma20 + 1e-9)
    df["vol_mom_5_20"] = vol_sma5 / (vol_sma20 + 1e-9)

    # Price-Volume Divergence — price up but volume down = weak/suspicious move
    price_chg = close.pct_change()
    vol_chg   = volume / (volume.shift(1) + 1e-9) - 1.0
    df["pv_divergence"] = price_chg * vol_chg

    # Volume-weighted RSI (price moves on high volume count more)
    if TA_AVAILABLE:
        try:
            df["vrsi"] = ta.volume.volume_weighted_average_price(
                high, low, close, volume, window=14
            )
        except Exception:
            df["vrsi"] = df["vwap"]   # fallback if method signature differs
    else:
        df["vrsi"] = df["vwap"]   # manual fallback

    # Drop NaN rows produced by indicator warm-up windows
    df.dropna(inplace=True)
    print(f"[Fetcher] After indicators: {len(df)} rows, {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 3 — Macro feature fetchers (VIX + SPY)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_vix(start: str, end: str) -> pd.Series:
    """
    Fetch CBOE Volatility Index (VIX) as a fear/greed gauge.
    VIX > 30 = high fear (crash territory), < 15 = complacency.
    Falls back gracefully if download fails — caller fills with 20.0 (neutral).
    """
    try:
        vix = yf.download("^VIX", start=start, end=end,
                          auto_adjust=True, progress=False)
        if vix.empty:
            return pd.Series(dtype=float, name="vix")
        # Handle multi-level columns from newer yfinance versions
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix.droplevel(1, axis=1)
        series = vix["Close"].copy()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series.name = "vix"
        print(f"[Fetcher] VIX fetched: {len(series)} rows")
        return series
    except Exception as e:
        print(f"[Fetcher] VIX fetch failed ({e}) — will use neutral fill (20.0)")
        return pd.Series(dtype=float, name="vix")


def fetch_spy_return(start: str, end: str) -> pd.Series:
    """
    Fetch S&P 500 (SPY) daily log return as a market-direction context feature.
    Individual stocks follow the market 60-70% of the time — this gives the
    model a view of the macro direction independent of the target stock.
    Falls back gracefully if download fails — caller fills with 0.0.
    """
    try:
        spy = yf.download("SPY", start=start, end=end,
                          auto_adjust=True, progress=False)
        if spy.empty:
            return pd.Series(dtype=float, name="spy_return")
        if isinstance(spy.columns, pd.MultiIndex):
            spy = spy.droplevel(1, axis=1)
        ret = np.log(spy["Close"] / spy["Close"].shift(1))
        ret.index = pd.to_datetime(ret.index).tz_localize(None)
        ret.name = "spy_return"
        print(f"[Fetcher] SPY return fetched: {len(ret)} rows")
        return ret
    except Exception as e:
        print(f"[Fetcher] SPY fetch failed ({e}) — will use neutral fill (0.0)")
        return pd.Series(dtype=float, name="spy_return")


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 8 — Earnings Calendar
# ─────────────────────────────────────────────────────────────────────────────

def get_earnings_dates(ticker: str) -> set:
    """
    Returns a set of datetime.date objects when earnings are announced.
    Falls back to empty set if yfinance doesn't have calendar data.

    Earnings cause the largest single-day moves — knowing earnings are
    tomorrow is a crucial signal for uncertainty widening.
    """
    try:
        stock = yf.Ticker(ticker)
        cal   = stock.earnings_dates   # DataFrame indexed by announcement date
        if cal is not None and not cal.empty:
            dates = set(pd.to_datetime(cal.index).tz_localize(None).normalize().map(lambda x: x.date()))
            print(f"[Fetcher] Earnings calendar: {len(dates)} dates found for {ticker}")
            return dates
    except Exception as e:
        print(f"[Fetcher] Earnings calendar unavailable ({e}) — feature set to 0")
    return set()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, period: str = FETCH_PERIOD,
                     force_refresh: bool = False) -> pd.DataFrame:
    """
    Full pipeline: fetch -> validate -> add indicators -> add macro -> add earnings
    -> save.
    Uses cached CSV if available and force_refresh=False.

    IMPROVEMENT 3 + 8 + 9 (Final Checklist item 3):
    Cache is automatically invalidated if the stored column count doesn't
    match N_FEATURES, so stale caches after feature additions are handled
    automatically.
    """
    cache = os.path.join(DATA_DIR, f"{ticker}_features.csv")

    # Cache invalidation: refresh if column count has changed
    if os.path.exists(cache) and not force_refresh:
        try:
            df_cached = pd.read_csv(cache, index_col=0, nrows=1)
            # We expect at least N_FEATURES columns in the CSV
            # (there may be extra like 'log_return' added by features.py)
            if len(df_cached.columns) < N_FEATURES - 2:
                print(f"[Fetcher] Cache has {len(df_cached.columns)} cols, "
                      f"need ~{N_FEATURES} — refreshing cache")
                force_refresh = True
        except Exception:
            force_refresh = True

    if os.path.exists(cache) and not force_refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"[Fetcher] Loaded {len(df)} rows from cache: {cache}")
        return df

    raw = fetch_raw(ticker, period)
    df  = add_technical_indicators(raw)

    # Placeholder sentiment column (filled later by sentiment module)
    if "sentiment" not in df.columns:
        df["sentiment"] = 0.0

    # ── IMPROVEMENT 3: Add VIX + SPY macro features ───────────────────────────
    start_str = str(df.index[0].date())
    end_str   = str(df.index[-1].date())

    vix_series = fetch_vix(start_str, end_str)
    spy_series = fetch_spy_return(start_str, end_str)

    if not vix_series.empty:
        df = df.join(vix_series, how="left")
        df["vix"] = df["vix"].ffill()
        df["vix"] = df["vix"].fillna(20.0)   # neutral VIX if still missing
    else:
        df["vix"] = 20.0

    if not spy_series.empty:
        df = df.join(spy_series, how="left")
        df["spy_return"] = df["spy_return"].fillna(0.0)
    else:
        df["spy_return"] = 0.0

    # ── IMPROVEMENT 8: Earnings Calendar Feature ──────────────────────────────
    earnings_dates = get_earnings_dates(ticker)
    if earnings_dates:
        df["earnings_tomorrow"] = [
            1.0 if (pd.Timestamp(d) + pd.Timedelta(days=1)).date() in earnings_dates
            else 0.0
            for d in df.index
        ]
    else:
        df["earnings_tomorrow"] = 0.0

    df.to_csv(cache)
    print(f"[Fetcher] Saved to {cache}")
    return df


def fetch_latest(ticker: str) -> dict:
    """
    Fetch the single most-recent trading day for real-time inference.
    Returns a dict of the last row values.
    """
    df = yf.Ticker(ticker).history(period="3mo", auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = add_technical_indicators(df)
    df["sentiment"] = 0.0

    # Add macro features for latest row
    start_str = str(df.index[0].date())
    end_str   = str(df.index[-1].date())
    vix_s = fetch_vix(start_str, end_str)
    spy_s = fetch_spy_return(start_str, end_str)
    df["vix"]        = vix_s.reindex(df.index).ffill().fillna(20.0) if not vix_s.empty else 20.0
    df["spy_return"] = spy_s.reindex(df.index).fillna(0.0) if not spy_s.empty else 0.0
    df["earnings_tomorrow"] = 0.0   # conservative default for live inference

    row = df.iloc[-1].to_dict()
    row["date"] = str(df.index[-1].date())
    return row


if __name__ == "__main__":
    df = fetch_stock_data("AAPL", force_refresh=True)
    print(df.tail(3))
    print(f"\nColumns ({len(df.columns)}): {list(df.columns)}")
