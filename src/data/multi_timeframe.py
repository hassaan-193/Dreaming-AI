"""
Dreaming AI v3 — Multi-Timeframe Data Fetcher
Extends src/data/fetcher.py (v2) without modifying it.
Adds support for: 15m, 30m, 1h, 4h, 1d (existing), 1wk
"""
import os, warnings
import numpy as np
import pandas as pd
import yfinance as yf
warnings.filterwarnings("ignore")

# Import v2 fetcher — we extend it, not replace it
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.fetcher import add_technical_indicators, fetch_raw
from config import DATA_DIR, TIMEFRAMES


# ─────────────────────────────────────────────────────────────────────────────
# Core: fetch any timeframe
# ─────────────────────────────────────────────────────────────────────────────

def fetch_timeframe(ticker: str, interval: str, period: str,
                    force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV data at a specific interval from Yahoo Finance.
    Adds technical indicators (same 15 as v2 fetcher).

    Args:
        ticker:   Stock symbol (e.g., 'AAPL')
        interval: yfinance interval string ('15m','30m','1h','1d','1wk')
        period:   yfinance period string ('60d','730d','5y','10y')
        force_refresh: Bypass cache

    Returns:
        DataFrame with OHLCV + 15 technical indicators + DatetimeIndex
    """
    safe_interval = interval.replace("/", "_")
    cache = os.path.join(DATA_DIR, f"{ticker}_{safe_interval}_features.csv")

    if os.path.exists(cache) and not force_refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"[MTF] Loaded {ticker} {interval}: {len(df)} rows from cache")
        return df

    print(f"[MTF] Fetching {ticker} at {interval} ({period}) …")
    raw = yf.Ticker(ticker).history(period=period, interval=interval,
                                     auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No {interval} data for '{ticker}'. "
                         f"Note: intraday data is only available for the last 60 days.")

    raw = raw[["Open","High","Low","Close","Volume"]].copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    # Drop zero-price rows
    raw = raw[raw["Close"] > 0]
    raw = raw.ffill().bfill()

    # Reuse v2's indicator function
    df = add_technical_indicators(raw)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(cache)
    print(f"[MTF] {ticker} {interval}: {len(df)} bars saved -> {cache}")
    return df


def aggregate_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-hour bars into 4-hour bars.
    yfinance doesn't provide 4h natively so we build it from 1h.
    """
    df = df_1h.copy()
    # Create 4h buckets: 0-3, 4-7, 8-11, 12-15, 16-19, 20-23
    df["bucket"] = df.index.floor("4H")
    agg = df.groupby("bucket").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    })
    agg.index = pd.to_datetime(agg.index)
    agg = agg.dropna()
    return add_technical_indicators(agg)


def fetch_all_timeframes(ticker: str,
                         force_refresh: bool = False) -> dict:
    """
    Fetch data for all supported timeframes for a given ticker.
    Returns a dict: { '15m': df, '30m': df, '1h': df, '4h': df, '1d': df, '1wk': df }
    """
    result = {}
    for tf, cfg in TIMEFRAMES.items():
        try:
            if tf == "4h":
                # Build 4h from 1h
                df_1h = fetch_timeframe(ticker, "1h", cfg["period"], force_refresh)
                result[tf] = aggregate_to_4h(df_1h)
            else:
                result[tf] = fetch_timeframe(ticker, cfg["interval"],
                                              cfg["period"], force_refresh)
            print(f"[MTF] {tf}: {len(result[tf])} bars")
        except Exception as e:
            print(f"[MTF] WARNING: Could not fetch {tf} for {ticker}: {e}")
            result[tf] = None
    return result


def fetch_latest_bar(ticker: str, interval: str = "1d") -> dict:
    """
    Fetch the single most-recent bar for real-time inference.
    Returns dict of last-row values.
    """
    cfg = TIMEFRAMES.get(interval, TIMEFRAMES["1d"])
    actual_interval = "1h" if interval == "4h" else cfg["interval"]
    # Use a short recent window for real-time
    # Need enough rows for indicator warm-up (ATR needs 14, others need 20+)
    # Weekly: need at least 30 weeks. Intraday: 5d is enough for minute bars.
    if interval in ("15m", "30m"):
        period = "5d"
    elif interval in ("1h", "4h"):
        period = "30d"
    elif interval == "1wk":
        period = "2y"   # need 100+ weekly bars for indicators
    else:
        period = "90d"  # 1d default — need 60+ bars for ema_50 warmup
    raw = yf.Ticker(ticker).history(period=period,
                                     interval=actual_interval,
                                     auto_adjust=True)
    raw = raw[["Open","High","Low","Close","Volume"]].copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    if interval == "4h":
        raw = aggregate_to_4h(raw)
    else:
        raw = add_technical_indicators(raw)

    if raw.empty:
        return {}

    row = raw.iloc[-1].to_dict()
    row["date"]     = str(raw.index[-1])
    row["interval"] = interval
    return row


def build_window_for_inference(ticker: str, interval: str,
                                scaler, feature_cols: list,
                                window_size: int) -> np.ndarray:
    """
    Build a single inference window (1, window_size, n_features) for
    real-time prediction. Fetches fresh data, normalises with saved scaler.

    Returns:
        numpy array of shape (1, window_size, n_features)
    """
    cfg = TIMEFRAMES.get(interval, TIMEFRAMES["1d"])
    if interval == "4h":
        df_1h = fetch_timeframe(ticker, "1h", "730d", force_refresh=True)
        df    = aggregate_to_4h(df_1h)
    else:
        df = fetch_timeframe(ticker, cfg["interval"], cfg["period"],
                             force_refresh=True)

    # Add sentinel sentiment=0 if needed
    if "sentiment" not in df.columns:
        df["sentiment"] = 0.0

    available = [c for c in feature_cols if c in df.columns]
    if len(df) < window_size:
        raise ValueError(f"Only {len(df)} bars available for {ticker}/{interval}, "
                         f"need {window_size}")

    scaled = scaler.transform(df[available].values[-window_size:]).astype("float32")
    return scaled[np.newaxis]   # (1, window_size, n_features)


if __name__ == "__main__":
    data = fetch_all_timeframes("AAPL")
    for tf, df in data.items():
        if df is not None:
            print(f"  {tf}: {df.shape}")
