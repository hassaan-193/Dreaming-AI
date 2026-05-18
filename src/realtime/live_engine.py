"""
Dreaming AI v3 — Real-Time Prediction Engine
Converts static v2 predict-on-demand into a continuous live system.

Architecture:
  LivePredictionEngine:
    - Maintains a background thread that polls each subscribed ticker
    - Every REALTIME_POLL_SECONDS seconds: fetches latest bar -> runs inference
    - Stores predictions in an in-memory cache (LRU-like with TTL)
    - API endpoint /predict_live reads from this cache

The existing /predict endpoint from v2 is preserved as a one-shot endpoint.
This module adds a continuous, auto-refreshing layer on top.
"""
import os, sys, time, threading, datetime, traceback
from collections import deque, OrderedDict
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (REALTIME_POLL_SECONDS, REALTIME_CACHE_MAXSIZE,
                    TIMEFRAMES, MODELS_DIR, DEVICE, WINDOW_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction cache
# ─────────────────────────────────────────────────────────────────────────────

class PredictionCache:
    """
    Thread-safe LRU cache for live predictions.
    Key: (ticker, interval)
    Value: dict with prediction data + timestamp
    """
    def __init__(self, maxsize: int = REALTIME_CACHE_MAXSIZE):
        self._cache  = OrderedDict()
        self._lock   = threading.Lock()
        self.maxsize = maxsize

    def set(self, ticker: str, interval: str, data: dict):
        key = (ticker.upper(), interval)
        with self._lock:
            self._cache.pop(key, None)   # move to end
            self._cache[key] = {**data, "cached_at": str(datetime.datetime.now())}
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)   # evict oldest

    def get(self, ticker: str, interval: str) -> Optional[dict]:
        key = (ticker.upper(), interval)
        with self._lock:
            return self._cache.get(key, None)

    def all_keys(self) -> list:
        with self._lock:
            return list(self._cache.keys())


# Global cache instance
LIVE_CACHE = PredictionCache()


# ─────────────────────────────────────────────────────────────────────────────
# Single prediction runner
# ─────────────────────────────────────────────────────────────────────────────

def run_single_prediction(ticker: str, interval: str,
                           model_cache: dict) -> dict:
    """
    Run one full inference cycle for a given ticker + interval:
      1. Fetch latest bars
      2. Normalise with saved scaler
      3. Run DEBM v3 (or v2 fallback) inference
      4. Compute confidence band via Langevin perturbation
      5. Get real-time sentiment

    Args:
        ticker:      Stock symbol
        interval:    Timeframe string ('15m','1h','1d', etc.)
        model_cache: Dict loaded by _load_models() from v2 api/main.py

    Returns:
        Prediction dict ready for JSON serialisation
    """
    from src.data.multi_timeframe import build_window_for_inference
    from src.data.sentiment_v3 import get_enhanced_realtime_sentiment
    from config import FEATURE_COLS

    debm      = model_cache["debm"].to(DEVICE)
    n_features = model_cache["n_features"]
    close_idx  = model_cache["close_idx"]
    scaler     = model_cache["scaler"]

    if scaler is None:
        raise RuntimeError("No scaler found. Run /train first.")

    cfg         = TIMEFRAMES.get(interval, TIMEFRAMES["1d"])
    window_size = cfg["window"]

    # Build inference window
    x_input = build_window_for_inference(
        ticker, interval, scaler, FEATURE_COLS, window_size
    )
    x_tensor = __import__("torch").tensor(x_input, device=DEVICE)

    debm.eval()
    import torch
    with torch.no_grad():
        # Try v3 predict (timeframe-conditioned); fall back to v2
        try:
            pred_scaled = debm.predict(x_tensor, timeframe=interval).cpu().numpy().flatten()[0]
        except TypeError:
            pred_scaled = debm.predict(x_tensor).cpu().numpy().flatten()[0]

    # Confidence band: 20 latent perturbations
    with torch.no_grad():
        h_base = debm.encode(x_tensor)
    samples = []
    for _ in range(20):
        h_pert = h_base + torch.randn_like(h_base) * 0.05
        try:
            p = debm.predictor(h_pert).cpu().numpy().flatten()[0]
        except Exception:
            p = pred_scaled
        samples.append(p)
    std_scaled = float(np.std(samples))

    # Inverse-transform
    def inv(v):
        dummy = np.zeros((1, n_features), dtype=np.float32)
        dummy[0, close_idx] = v
        return float(scaler.inverse_transform(dummy)[0, close_idx])

    pred_usd = inv(pred_scaled)
    low_usd  = inv(pred_scaled - std_scaled)
    high_usd = inv(pred_scaled + std_scaled)

    # Latest known price (last row of fetched data)
    from src.data.multi_timeframe import fetch_latest_bar
    latest = fetch_latest_bar(ticker, interval)
    last_usd = inv(scaler.transform([[latest.get("Close", 0)] +
                                      [0]*(n_features-1)])[0, close_idx]) \
               if latest else pred_usd

    # Sentiment
    sent_info = get_enhanced_realtime_sentiment(ticker)

    return {
        "ticker":           ticker.upper(),
        "interval":         interval,
        "interval_label":   TIMEFRAMES.get(interval, {}).get("label", interval),
        "predicted_price":  round(pred_usd, 2),
        "confidence_band":  {"low": round(low_usd,2), "high": round(high_usd,2)},
        "last_known_price": round(last_usd, 2),
        "trend":            "UP" if pred_usd > last_usd else "DOWN",
        "trend_strength":   round(abs(pred_usd - last_usd) / (last_usd + 1e-9) * 100, 3),
        **sent_info,
        "model":            "DreamingAI-v3",
        "timestamp":        str(datetime.datetime.now()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Live prediction engine (background thread)
# ─────────────────────────────────────────────────────────────────────────────

class LivePredictionEngine:
    """
    Background polling engine.
    Maintains a set of (ticker, interval) subscriptions.
    Every REALTIME_POLL_SECONDS, refreshes all subscriptions and
    writes results to LIVE_CACHE.
    """
    def __init__(self, model_registry: dict,
                 poll_seconds: int = REALTIME_POLL_SECONDS):
        self._subscriptions = set()   # set of (ticker, interval) tuples
        self._registry      = model_registry
        self._poll_seconds  = poll_seconds
        self._thread        = None
        self._running       = False
        self._lock          = threading.Lock()
        self._history       = {}     # (ticker,interval) -> deque of last 100 preds

    def subscribe(self, ticker: str, interval: str = "1d"):
        with self._lock:
            self._subscriptions.add((ticker.upper(), interval))
            key = (ticker.upper(), interval)
            if key not in self._history:
                self._history[key] = deque(maxlen=100)
        print(f"[LiveEngine] Subscribed: {ticker}/{interval}")

    def unsubscribe(self, ticker: str, interval: str = "1d"):
        with self._lock:
            self._subscriptions.discard((ticker.upper(), interval))

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[LiveEngine] Started. Polling every {self._poll_seconds}s")

    def stop(self):
        self._running = False
        print("[LiveEngine] Stopped.")

    def _loop(self):
        while self._running:
            with self._lock:
                subs = list(self._subscriptions)
            for ticker, interval in subs:
                try:
                    ticker_upper = ticker.upper()
                    if ticker_upper not in self._registry:
                        continue
                    result = run_single_prediction(
                        ticker_upper, interval, self._registry[ticker_upper]
                    )
                    LIVE_CACHE.set(ticker_upper, interval, result)
                    key = (ticker_upper, interval)
                    self._history[key].append(result)
                    print(f"[LiveEngine] {ticker_upper}/{interval} -> "
                          f"${result['predicted_price']}  "
                          f"({result['trend']})")
                except Exception as e:
                    print(f"[LiveEngine] ERROR {ticker}/{interval}: {e}")
                    traceback.print_exc()
            time.sleep(self._poll_seconds)

    def get_history(self, ticker: str, interval: str) -> list:
        """Return list of last N predictions for a ticker/interval."""
        key = (ticker.upper(), interval)
        with self._lock:
            return list(self._history.get(key, []))

    def is_subscribed(self, ticker: str, interval: str) -> bool:
        return (ticker.upper(), interval) in self._subscriptions


# Global engine instance (initialised by API on startup)
_ENGINE: Optional[LivePredictionEngine] = None

def get_engine() -> Optional[LivePredictionEngine]:
    return _ENGINE

def init_engine(model_registry: dict) -> LivePredictionEngine:
    global _ENGINE
    _ENGINE = LivePredictionEngine(model_registry)
    _ENGINE.start()
    return _ENGINE
