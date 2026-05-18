"""
Dreaming AI v3 — Extended Visualization Module
New charts added on top of v2 evaluate.py (not replacing it):
  1. Extreme condition performance comparison
  2. Confidence bands over time (TradingView-style)
  3. Multi-timeframe prediction summary
  4. Attention weight heatmap
  5. Live prediction sparkline
"""
import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import OUTPUTS_DIR, EXTREME_LABEL_NAMES, TIMEFRAMES

PALETTE = {
    "DEBM":    "#E63946",
    "DEBM_v3": "#FF6B6B",
    "LSTM":    "#3A86FF",
    "GAN":     "#06D6A0",
    "Actual":  "#CDD5E0",
    "normal":  "#06D6A0",
    "crash":   "#E63946",
    "spike":   "#FFB703",
    "sideways_volatile": "#8338EC",
    "band":    "rgba(230,57,70,0.15)",
}
BG    = "#080A0F"
CARD  = "#141820"
MUTED = "#5A6480"
TEXT  = "#CDD5E0"


def _ax_style(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#1E2433")
    ax.grid(alpha=0.12, color="#1E2433")
    return ax


def _savefig(fig, fname):
    path = os.path.join(OUTPUTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[VizV3] Saved -> {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Extreme condition performance comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_extreme_conditions(cond_results: dict, ticker: str):
    """
    Bar chart: RMSE per market condition (normal, crash, spike, sideways_volatile).
    Shows where the DEBM excels vs struggles.
    """
    conditions = [c for c in EXTREME_LABEL_NAMES if cond_results.get(c,{}).get("RMSE") is not None]
    rmse_vals  = [cond_results[c]["RMSE"] for c in conditions]
    da_vals    = [cond_results[c]["Dir.Acc"] for c in conditions]
    counts     = [cond_results[c]["count"] for c in conditions]
    colors     = [PALETTE.get(c, "#888") for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle(f"{ticker} — DEBM Performance by Market Condition",
                 color=TEXT, fontsize=14, fontweight="bold")

    # RMSE per condition
    ax = _ax_style(axes[0])
    bars = ax.bar(conditions, rmse_vals, color=colors, edgecolor=BG, alpha=0.85, width=0.6)
    for bar, v, n in zip(bars, rmse_vals, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(rmse_vals)*0.02,
                f"${v:.2f}\n(n={n})", ha="center", va="bottom",
                color=TEXT, fontsize=8)
    ax.set_title("RMSE by Condition (lower is better)", color=TEXT, fontsize=11)
    ax.set_ylabel("RMSE ($)", color=MUTED)
    ax.tick_params(axis="x", colors=TEXT)

    # Directional Accuracy per condition
    ax = _ax_style(axes[1])
    bars = ax.bar(conditions, da_vals, color=colors, edgecolor=BG, alpha=0.85, width=0.6)
    ax.axhline(50, color=MUTED, lw=1, ls="--", label="Random (50%)")
    for bar, v in zip(bars, da_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{v:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Directional Accuracy by Condition", color=TEXT, fontsize=11)
    ax.set_ylabel("Accuracy (%)", color=MUTED)
    ax.tick_params(axis="x", colors=TEXT)
    ax.legend(facecolor=CARD, labelcolor=MUTED)

    plt.tight_layout()
    _savefig(fig, f"{ticker}_extreme_conditions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confidence bands over time (TradingView-style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence_bands(actual: np.ndarray, predicted: np.ndarray,
                           ticker: str, std_scale: float = 0.015):
    """
    Plot predicted prices with shaded ±1σ confidence band.
    Highlights where prediction direction matches actual (green) vs mismatches (red).
    """
    n    = min(len(actual), len(predicted), 120)   # show last 120 bars
    act  = actual[-n:]
    pred = predicted[-n:]
    std  = np.abs(pred) * std_scale                 # ~1.5% band

    correct = np.sign(np.diff(np.append(act[0:1], act))) == \
              np.sign(np.diff(np.append(pred[0:1], pred)))

    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
    _ax_style(ax)
    x = np.arange(n)

    # Confidence band
    ax.fill_between(x, pred-std, pred+std, alpha=0.18,
                     color=PALETTE["DEBM"], label="±1σ confidence")

    # Actual price
    ax.plot(x, act, color=TEXT, lw=1.8, label="Actual", zorder=5)

    # Predicted — colour segments by correctness
    for i in range(1, n):
        col = "#06D6A0" if correct[i] else "#E63946"
        ax.plot([x[i-1], x[i]], [pred[i-1], pred[i]], color=col, lw=1.5, alpha=0.85)

    # Legend patches
    ax.plot([], [], color="#06D6A0", label="Correct direction", lw=2)
    ax.plot([], [], color="#E63946", label="Wrong direction",   lw=2)

    ax.set_title(f"{ticker} — DEBM Prediction with Confidence Band",
                 color=TEXT, fontsize=14, fontweight="bold")
    ax.set_xlabel("Test Day", color=MUTED)
    ax.set_ylabel("Price ($)", color=MUTED)
    ax.legend(facecolor=CARD, labelcolor=TEXT, framealpha=0.9)

    plt.tight_layout()
    _savefig(fig, f"{ticker}_confidence_bands.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-timeframe summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_timeframe_summary(ticker: str):
    """
    Visual summary showing which timeframes have data available
    and their approximate bar counts.
    """
    import pandas as pd
    from config import DATA_DIR

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    _ax_style(ax)

    tfs    = list(TIMEFRAMES.keys())
    labels = [TIMEFRAMES[t]["label"] for t in tfs]
    counts = []
    for tf in tfs:
        cache = os.path.join(DATA_DIR, f"{ticker}_{tf}_features.csv")
        if os.path.exists(cache):
            try:
                n = len(pd.read_csv(cache))
            except Exception:
                n = 0
        else:
            n = 0
        counts.append(n)

    colors = [PALETTE["DEBM"] if c > 0 else MUTED for c in counts]
    bars = ax.barh(labels, counts, color=colors, edgecolor=BG, alpha=0.85)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.01, bar.get_y()+bar.get_height()/2,
                str(c) if c > 0 else "N/A", va="center", color=TEXT, fontsize=9)

    ax.set_title(f"{ticker} — Available Data per Timeframe",
                 color=TEXT, fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Bars", color=MUTED)
    ax.tick_params(axis="y", colors=TEXT)

    plt.tight_layout()
    _savefig(fig, f"{ticker}_multi_timeframe.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Attention weight heatmap (if available)
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_weights(cross_weights: "torch.Tensor", ticker: str,
                            feature_names: list = None):
    """
    Heatmap of cross-attention weights (price attending to sentiment).
    cross_weights: (1, W, W) or (B, H, W_q, W_k) tensor from MultiheadAttention.
    """
    import torch
    if cross_weights is None:
        return

    w = cross_weights.detach().cpu().numpy()
    if w.ndim == 4:
        w = w[0].mean(0)   # average over heads, take first batch
    elif w.ndim == 3:
        w = w[0]

    # Limit to last 30 time steps for readability
    w = w[-30:, -30:] if w.shape[0] > 30 else w

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    _ax_style(ax)
    im = ax.imshow(w, cmap="magma", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_title(f"{ticker} — Price->Sentiment Cross-Attention",
                 color=TEXT, fontsize=12)
    ax.set_xlabel("Sentiment Time Steps", color=MUTED)
    ax.set_ylabel("Price Time Steps", color=MUTED)
    plt.tight_layout()
    _savefig(fig, f"{ticker}_attention_weights.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Live prediction sparkline (for dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def plot_live_sparkline(history: list, ticker: str, interval: str):
    """
    Tiny TradingView-style sparkline of recent live predictions.
    history: list of dicts from LivePredictionEngine.get_history()
    """
    if len(history) < 2:
        return

    preds  = [h["predicted_price"] for h in history]
    actual = [h.get("last_known_price", p) for h, p in zip(history, preds)]
    ts     = [h.get("timestamp","")[-8:] for h in history]   # HH:MM:SS

    fig, ax = plt.subplots(figsize=(12, 3), facecolor=BG)
    _ax_style(ax)

    x = np.arange(len(preds))
    ax.plot(x, actual, color=TEXT,           lw=1.5, label="Last Known")
    ax.plot(x, preds,  color=PALETTE["DEBM"], lw=1.5, ls="--", label="Predicted")

    # Final prediction label
    ax.annotate(f"${preds[-1]:.2f}",
                xy=(x[-1], preds[-1]),
                xytext=(x[-1]-2, preds[-1]+abs(max(preds)-min(preds))*0.1),
                color=PALETTE["DEBM"], fontsize=9)

    step = max(1, len(ts)//8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(ts[::step], rotation=30, ha="right", fontsize=7)
    ax.set_title(f"{ticker} {interval} — Live Predictions",
                 color=TEXT, fontsize=12, fontweight="bold")
    ax.set_ylabel("Price ($)", color=MUTED)
    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    _savefig(fig, f"{ticker}_{interval}_live_sparkline.png")
