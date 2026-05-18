"""
Dreaming AI v6 — Feature Engineering
Normalises all features, builds sliding-window sequences,
and returns train/val/test splits with proper temporal ordering.

v6 Additions (Improvement 1):
  - PREDICT_LOG_RETURN mode: target is log(Close_t / Close_{t-1})
    instead of raw scaled Close price.
  - log_return column is computed here BEFORE scaling (after macro join).
  - return_to_price() utility converts predicted log returns → USD prices.
  - target_col_idx is now determined dynamically based on config flag.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict

from config import (WINDOW_SIZE, TEST_SPLIT, VAL_SPLIT,
                    FEATURE_COLS, N_FEATURES, MODELS_DIR,
                    PREDICT_LOG_RETURN, LOG_RETURN_COL)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame,
               cols: list = FEATURE_COLS) -> RobustScaler:
    """Fit a RobustScaler on available columns (ignores missing ones)."""
    available = [c for c in cols if c in df.columns]
    scaler = RobustScaler()
    scaler.fit(df[available].values)
    return scaler


def scale(df: pd.DataFrame, scaler: RobustScaler,
          cols: list = FEATURE_COLS) -> np.ndarray:
    """Apply fitted scaler. Returns (N, n_features) float32 array."""
    available = [c for c in cols if c in df.columns]
    return scaler.transform(df[available].values).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(scaled: np.ndarray,
                    window: int = WINDOW_SIZE,
                    target_col_idx: int = 3   # 'Close' is index 3 in FEATURE_COLS
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding window: X[i] = scaled[i : i+window],  y[i] = scaled[i+window, target_idx]

    Returns:
        X: (N, window, n_features)  float32
        y: (N,)                     float32
    """
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window : i])
        y.append(scaled[i, target_col_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 1 — Log return utility
# ─────────────────────────────────────────────────────────────────────────────

def return_to_price(predicted_returns: np.ndarray,
                    last_known_price: float) -> np.ndarray:
    """
    Convert a sequence of predicted log returns back to absolute USD prices.

    Each predicted return r_t represents log(P_t / P_{t-1}).
    Starting from last_known_price, we compound forward:
        P_t = P_{t-1} * exp(r_t)

    Args:
        predicted_returns: (N,) array of log return predictions
        last_known_price:  last observed Close price in USD

    Returns:
        prices: (N,) array of reconstructed USD prices
    """
    prices = [last_known_price]
    for r in predicted_returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices[1:], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    ticker: str,
    window: int = WINDOW_SIZE,
) -> Dict[str, np.ndarray]:
    """
    Full feature engineering pipeline.

    Steps:
      1. Compute log_return if PREDICT_LOG_RETURN=True (before scaling).
      2. Select FEATURE_COLS (or available subset).
      3. Fit RobustScaler on train portion ONLY (no data leakage).
      4. Build sliding-window sequences.
      5. Split into train / val / test (preserving time order).
      6. Save scaler to /models.

    Returns a dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test,
        scaler, close_idx, target_col_idx, n_features
    """
    df = df.copy()

    # ── IMPROVEMENT 1: Compute log_return before scaling ─────────────────────
    # log_return must be the LAST computed feature to avoid NaN propagation
    # from the first-row shift.
    if PREDICT_LOG_RETURN:
        df[LOG_RETURN_COL] = np.log(df["Close"] / df["Close"].shift(1))
        df = df.dropna(subset=[LOG_RETURN_COL])
        print(f"[Features] log_return computed — {len(df)} rows after dropna")

    # Determine available features (some may not be in df if fetch failed)
    available = [c for c in FEATURE_COLS if c in df.columns]
    n_feat    = len(available)

    # Determine target column index
    if PREDICT_LOG_RETURN and LOG_RETURN_COL in available:
        target_col_idx = available.index(LOG_RETURN_COL)
        print(f"[Features] Target: {LOG_RETURN_COL} (directional mode) "
              f"at index {target_col_idx}")
    else:
        target_col_idx = available.index("Close") if "Close" in available else 3
        print(f"[Features] Target: Close price (legacy mode) "
              f"at index {target_col_idx}")

    close_idx = available.index("Close") if "Close" in available else 3

    print(f"[Features] N_FEATURES = {n_feat}  |  "
          f"Available cols: {len(available)}")

    n = len(df)
    test_n  = int(n * TEST_SPLIT)
    val_n   = int(n * VAL_SPLIT)
    train_n = n - test_n - val_n

    # Fit scaler on TRAIN rows only (prevent data leakage)
    train_df = df.iloc[:train_n]
    scaler   = fit_scaler(train_df, cols=available)
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[Features] Scaler saved -> {scaler_path}")

    # Scale entire dataset
    scaled = scale(df, scaler, cols=available)

    # Build sequences using the correct target column
    X, y = build_sequences(scaled, window=window,
                            target_col_idx=target_col_idx)

    # Adjust split indices to account for window warm-up
    total   = len(X)
    tr_end  = total - (test_n + val_n)
    val_end = total - test_n

    X_train, y_train = X[:tr_end],        y[:tr_end]
    X_val,   y_val   = X[tr_end:val_end], y[tr_end:val_end]
    X_test,  y_test  = X[val_end:],       y[val_end:]

    print(f"[Features] Splits — "
          f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}  "
          f"Features: {n_feat}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
        "close_idx":      close_idx,
        "target_col_idx": target_col_idx,
        "n_features":     n_feat,
    }


def inverse_close(y_scaled: np.ndarray, scaler: RobustScaler,
                  n_features: int, close_idx: int) -> np.ndarray:
    """
    Inverse-transform a 1-D array of scaled Close values back to USD.
    Pads with zeros for the other feature columns.

    NOTE: When PREDICT_LOG_RETURN=True, the model predicts log returns.
    Use return_to_price() to convert those back to USD prices instead.
    This function is kept for legacy/baseline compatibility.
    """
    dummy = np.zeros((len(y_scaled), n_features), dtype=np.float32)
    dummy[:, close_idx] = y_scaled
    return scaler.inverse_transform(dummy)[:, close_idx]


def inverse_target(y_scaled: np.ndarray, scaler: RobustScaler,
                   n_features: int, target_col_idx: int) -> np.ndarray:
    """
    Inverse-transform a 1-D array of scaled target values (log return or Close).
    Works for any target column index.
    """
    dummy = np.zeros((len(y_scaled), n_features), dtype=np.float32)
    dummy[:, target_col_idx] = y_scaled
    return scaler.inverse_transform(dummy)[:, target_col_idx]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.data.fetcher   import fetch_stock_data
    from src.data.sentiment import attach_sentiment
    df   = fetch_stock_data("AAPL")
    df   = attach_sentiment(df, "AAPL")
    data = prepare_data(df, "AAPL")
    print("X_train:", data["X_train"].shape)
    print("y_train:", data["y_train"].shape)
    print("Target col idx:", data["target_col_idx"])
