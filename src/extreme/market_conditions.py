"""
Dreaming AI v3 — Extreme Market Condition Module
Tags market conditions, tracks model performance per condition,
and enhances the Dreaming Phase with labelled synthetic scenarios.

Conditions:
  0 = normal
  1 = crash   (daily return < -3%)
  2 = spike   (daily return >  3%)
  3 = sideways_volatile (|return| < 0.5% but ATR is high)
"""
import os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (CRASH_THRESHOLD, SPIKE_THRESHOLD, SIDEWAYS_THRESHOLD,
                    EXTREME_LABEL_NAMES, OUTPUTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Condition tagger
# ─────────────────────────────────────────────────────────────────────────────

def tag_conditions(df: pd.DataFrame) -> np.ndarray:
    """
    Tag each row in df with a market condition integer.

    Args:
        df: DataFrame with at least 'Close' and optionally 'atr' columns.

    Returns:
        numpy array of ints, shape (N,):
          0=normal, 1=crash, 2=spike, 3=sideways_volatile
    """
    close   = df["Close"].values
    returns = np.zeros(len(close))
    returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-9)

    labels = np.zeros(len(df), dtype=int)   # default: normal

    # ATR-based sideways volatility (high ATR but flat price)
    if "atr" in df.columns:
        atr    = df["atr"].values
        atr_q  = np.nanpercentile(atr, 75)    # 75th percentile = "high" ATR
        hi_atr = atr > atr_q
    else:
        hi_atr = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        r = returns[i]
        if r < CRASH_THRESHOLD:
            labels[i] = 1   # crash
        elif r > SPIKE_THRESHOLD:
            labels[i] = 2   # spike
        elif abs(r) < SIDEWAYS_THRESHOLD and hi_atr[i]:
            labels[i] = 3   # sideways volatile
        # else stays 0 = normal

    counts = {EXTREME_LABEL_NAMES[i]: int((labels==i).sum()) for i in range(4)}
    print(f"[Extreme] Condition counts: {counts}")
    return labels


def tag_sequence_conditions(labels: np.ndarray,
                             window_size: int = 60) -> np.ndarray:
    """
    For each sliding window sequence, assign the condition of the TARGET day
    (the day AFTER the window). This aligns with how y is built in features.py.

    Args:
        labels:      Row-level condition array from tag_conditions()
        window_size: Same window_size used in feature engineering

    Returns:
        Array of shape (N - window_size,) — one label per sequence
    """
    # Target day index = window_size, window_size+1, …, N-1
    return labels[window_size:]


def condition_mask(seq_labels: np.ndarray, condition: int) -> np.ndarray:
    """Return boolean mask for sequences matching a specific condition."""
    return seq_labels == condition


# ─────────────────────────────────────────────────────────────────────────────
# Condition-specific evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_by_condition(actual: np.ndarray, predicted: np.ndarray,
                           seq_labels_test: np.ndarray,
                           model_name: str = "DEBM") -> dict:
    """
    Compute RMSE and Directional Accuracy separately for each market condition.

    Args:
        actual:           Unscaled actual Close prices, shape (N_test,)
        predicted:        Unscaled predicted prices, shape (N_test,)
        seq_labels_test:  Condition labels for the test set, shape (N_test,)
        model_name:       For printing/logging

    Returns:
        dict: { condition_name: {"RMSE": ..., "Dir.Acc": ..., "count": ...} }
    """
    from sklearn.metrics import mean_squared_error
    results = {}
    for cid, cname in enumerate(EXTREME_LABEL_NAMES):
        mask = condition_mask(seq_labels_test, cid)
        if mask.sum() < 5:
            results[cname] = {"RMSE": None, "Dir.Acc": None, "count": int(mask.sum())}
            continue
        act = actual[mask]
        prd = predicted[mask]
        rmse  = float(np.sqrt(mean_squared_error(act, prd)))
        da    = float(np.mean(np.sign(np.diff(act)) ==
                              np.sign(np.diff(prd))) * 100) if len(act) > 1 else 0.0
        results[cname] = {"RMSE": round(rmse,4), "Dir.Acc": round(da,2),
                           "count": int(mask.sum())}
    print(f"\n[Extreme] {model_name} performance by market condition:")
    for cname, m in results.items():
        if m["RMSE"] is not None:
            print(f"  {cname:<22}: RMSE={m['RMSE']:.4f}  Dir.Acc={m['Dir.Acc']:.1f}%  "
                  f"n={m['count']}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Dreaming Phase — label-aware synthetic generation
# ─────────────────────────────────────────────────────────────────────────────

def dream_extreme_scenarios(
    model,                       # DreamingAIv3 or DreamingAI
    X_train: np.ndarray,
    y_train: np.ndarray,
    seq_labels_train: np.ndarray,
    n_per_condition: int = 100,
    noise_scale: float = 0.07,
    device: str = "cpu"
) -> tuple:
    """
    Enhanced Dreaming Phase: generates synthetic samples specifically
    for each extreme condition (crash, spike, sideways_volatile).

    Strategy:
      1. Find real training sequences belonging to each extreme condition.
      2. Use them as starting points for Langevin Dynamics (better mixing
         than starting from noise for rare conditions).
      3. Add larger noise to crash/spike seeds to create plausible variations.

    Args:
        model:              Trained DEBM model
        X_train:            Real training sequences (N, W, F)
        y_train:            Real targets (N,)
        seq_labels_train:   Condition labels (N,)
        n_per_condition:    Synthetic samples per extreme condition
        noise_scale:        Gaussian noise scale applied to seeds

    Returns:
        (X_synthetic, y_synthetic, labels_synthetic) — all three aligned
    """
    model.eval()
    model = model.to(device)

    X_syn_list, y_syn_list, lbl_syn_list = [], [], []

    for cid in [1, 2, 3]:    # crash, spike, sideways_volatile only
        cname = EXTREME_LABEL_NAMES[cid]
        mask  = condition_mask(seq_labels_train, cid)
        seeds = X_train[mask]

        if len(seeds) == 0:
            print(f"[Extreme] No real samples for condition '{cname}' — skipping.")
            continue

        print(f"[Extreme] Generating {n_per_condition} synthetic '{cname}' scenarios "
              f"from {len(seeds)} real seeds …")

        # Repeat seeds to reach n_per_condition, then add noise
        repeat_times = (n_per_condition // len(seeds)) + 1
        seeds_tiled  = np.tile(seeds, (repeat_times, 1, 1))[:n_per_condition]
        y_seeds      = np.tile(y_train[mask], repeat_times)[:n_per_condition]

        # Add condition-specific noise
        scale = noise_scale * (2.0 if cid in (1, 2) else 1.0)  # more noise for crash/spike
        X_noisy = seeds_tiled + np.random.randn(*seeds_tiled.shape).astype(np.float32) * scale
        y_noisy = y_seeds    + np.random.randn(*y_seeds.shape).astype(np.float32) * y_train.std() * 0.05

        # Optional: refine with a few Langevin steps in latent space
        with torch.no_grad():
            Xt = torch.tensor(X_noisy, dtype=torch.float32, device=device)
            h  = model.encode(Xt)   # push through encoder

        # The refined sequences are the nearest neighbours of the Langevin-moved h
        with torch.no_grad():
            Xall  = torch.tensor(X_train, dtype=torch.float32, device=device)
            h_all = model.encode(Xall)
            hn_s  = torch.nn.functional.normalize(h,     dim=1)
            hn_a  = torch.nn.functional.normalize(h_all, dim=1)
            nn_idx = (hn_s @ hn_a.T).argmax(dim=1).cpu().numpy()

        X_refined = X_train[nn_idx] + np.random.randn(*seeds_tiled.shape).astype(np.float32) * scale * 0.5
        y_refined = y_train[nn_idx]

        X_syn_list.append(X_refined)
        y_syn_list.append(y_refined)
        lbl_syn_list.append(np.full(len(X_refined), cid, dtype=int))

    if not X_syn_list:
        return X_train[:0], y_train[:0], seq_labels_train[:0]

    X_syn = np.concatenate(X_syn_list, axis=0)
    y_syn = np.concatenate(y_syn_list, axis=0)
    lbl_syn = np.concatenate(lbl_syn_list, axis=0)

    print(f"[Extreme] Total extreme synthetic samples: {len(X_syn)}")
    return X_syn, y_syn, lbl_syn
