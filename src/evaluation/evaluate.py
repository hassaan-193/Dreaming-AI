"""
Dreaming AI v6 — Evaluation Module
Computes RMSE, MAE, Directional Accuracy for DEBM vs LSTM vs GAN.
Generates publication-quality comparison plots.

v6 Additions (Improvements 6 + 10):
  - ensemble_predict()         — weighted blend of DEBM + LSTM predictions
  - compute_ensemble_metrics() — compare DEBM, LSTM, Ensemble side-by-side
  - walk_forward_evaluate()    — rolling-window walk-forward validation
    (honest, publishable accuracy estimate for FYP defence)
"""
import logging
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import torch

from config import OUTPUTS_DIR, DEVICE, CRASH_DAY_THRESHOLD, PREDICT_LOG_RETURN

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 7 — Crash Day Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_crash_days(actual: np.ndarray, predicted: np.ndarray,
                        threshold: float = CRASH_DAY_THRESHOLD) -> dict:
    """
    Evaluate model performance specifically on crash days.

    Crash days are defined as days where the actual return is below
    the threshold (default -2%).

    Args:
        actual:     (N,) array of actual prices or returns
        predicted:  (N,) array of predicted prices or returns
        threshold:  return threshold below which a day is a 'crash day'

    Returns:
        dict with keys: crash_dir_acc, crash_rmse, normal_dir_acc,
                        n_crash_days, n_total_days, confusion_matrix
    """
    # Compute daily returns from price sequences
    if PREDICT_LOG_RETURN:
        actual_ret = actual
        pred_ret   = predicted
    else:
        actual_ret = np.diff(actual) / (np.abs(actual[:-1]) + 1e-9)
        pred_ret   = np.diff(predicted) / (np.abs(predicted[:-1]) + 1e-9)

    crash_mask  = actual_ret < threshold
    normal_mask = ~crash_mask
    n_crash     = int(crash_mask.sum())

    result = {
        "n_crash_days": n_crash,
        "n_total_days": len(actual_ret),
        "crash_dir_acc":  None,
        "crash_rmse":     None,
        "normal_dir_acc": None,
        "confusion_matrix": None,
    }

    if n_crash > 0:
        result["crash_dir_acc"] = float(
            np.mean(np.sign(pred_ret[crash_mask]) == np.sign(actual_ret[crash_mask])) * 100
        )
        result["crash_rmse"] = float(
            np.sqrt(np.mean((actual_ret[crash_mask] - pred_ret[crash_mask]) ** 2))
        )
        # Confusion matrix: predicted down vs actual down on crash days
        y_true = (actual_ret[crash_mask] < 0).astype(int)
        y_pred = (pred_ret[crash_mask]   < 0).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0]).tolist()
        result["confusion_matrix"] = cm

    if normal_mask.sum() > 0:
        result["normal_dir_acc"] = float(
            np.mean(np.sign(pred_ret[normal_mask]) == np.sign(actual_ret[normal_mask])) * 100
        )

    return result


def save_crash_visualization(ticker: str,
                              actual: np.ndarray,
                              predicted: np.ndarray,
                              threshold: float = CRASH_DAY_THRESHOLD):
    """
    Save a crash analysis figure to outputs/{ticker}_crash_analysis.png.

    The figure contains:
      - Price chart with crash days highlighted in red
      - Bar chart: crash vs normal accuracy
      - Confusion matrix heatmap

    Args:
        ticker:    stock ticker label
        actual:    (N,) actual prices
        predicted: (N,) predicted prices
        threshold: crash return threshold
    """
    crash_info = evaluate_crash_days(actual, predicted, threshold)
    n_crash    = crash_info["n_crash_days"]

    if PREDICT_LOG_RETURN:
        actual_ret = actual
        pred_ret   = predicted
    else:
        actual_ret = np.diff(actual)   / (np.abs(actual[:-1])   + 1e-9)
        pred_ret   = np.diff(predicted) / (np.abs(predicted[:-1]) + 1e-9)
    crash_mask = actual_ret < threshold

    fig = plt.figure(figsize=(16, 12), facecolor="#0D0F14")
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#151820")
        ax.tick_params(colors="#8899AA")
        for sp in ax.spines.values():
            sp.set_color("#252A3A")

    # Price chart
    x = np.arange(len(actual))
    ax1.plot(x, actual,    color="#58A6FF", lw=1.8, label="Actual",    zorder=3)
    ax1.plot(x, predicted, color="#F85149", lw=1.2, ls="--",
             label="DEBM Pred", alpha=0.85, zorder=2)
    crash_idx = np.where(crash_mask)[0]
    for ci in crash_idx:
        ax1.axvspan(ci, ci + 1, color="#F85149", alpha=0.22, zorder=1)
    ax1.set_title(f"{ticker} — Crash Day Analysis  (red = crash days)",
                  color="#E8EAF0", fontsize=13)
    ax1.legend(facecolor="#1C2030", labelcolor="#E8EAF0")
    ax1.grid(alpha=0.12, color="#252A3A")
    ax1.set_ylabel("Price (USD)", color="#8899AA")

    # Accuracy bar chart
    categories = ["Normal Days", "Crash Days"]
    values = [
        crash_info.get("normal_dir_acc") or 0,
        crash_info.get("crash_dir_acc")  or 0,
    ]
    bars = ax2.bar(categories, values, color=["#3FB950", "#F85149"],
                   edgecolor="#0D0F14", alpha=0.85)
    ax2.axhline(50, color="#8899AA", lw=0.8, ls="--")
    ax2.set_ylim(0, 100)
    ax2.set_title("Directional Accuracy by Regime", color="#E8EAF0", fontsize=12)
    ax2.set_ylabel("Accuracy %", color="#8899AA")
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5, f"{v:.1f}%",
                 ha="center", color="#E8EAF0", fontweight="bold")

    # Confusion matrix
    if n_crash > 0 and crash_info["confusion_matrix"]:
        cm_arr = np.array(crash_info["confusion_matrix"])
        try:
            sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Reds",
                        xticklabels=["Pred Down", "Pred Up"],
                        yticklabels=["Act Down", "Act Up"],
                        ax=ax3, linewidths=0.5)
        except Exception:
            ax3.text(0.5, 0.5, "Confusion matrix\nunavailable",
                     ha="center", va="center", color="#8899AA",
                     transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, f"No crash days\n(threshold={threshold:.1%})",
                 ha="center", va="center", color="#8899AA",
                 transform=ax3.transAxes)
    ax3.set_title("Crash Day Confusion Matrix", color="#E8EAF0", fontsize=12)
    ax3.tick_params(colors="#8899AA")

    fig.suptitle(
        f"{ticker} — Crash Scenario Analysis  "
        f"(N={n_crash} crash days, threshold={threshold:.1%})",
        color="#E8EAF0", fontsize=14, y=1.01
    )
    path = os.path.join(OUTPUTS_DIR, f"{ticker}_crash_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"[Eval] Crash analysis saved -> {path}")
    return crash_info

PALETTE = {
    "DEBM":     "#E74C3C",
    "LSTM":     "#3498DB",
    "GAN":      "#2ECC71",
    "Ensemble": "#F39C12",
    "Actual":   "#2C3E50",
    "Synthetic":"#9B59B6",
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """RMSE, MAE, and Directional Accuracy."""
    rmse    = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae     = float(mean_absolute_error(actual, predicted))
    if PREDICT_LOG_RETURN:
        da = float(np.mean((actual > 0) == (predicted[:len(actual)] > 0)) * 100)
    else:
        da      = float(np.mean(
            np.sign(np.diff(actual)) == np.sign(np.diff(predicted[:len(actual)]))
        ) * 100)
    # Mean Absolute Percentage Error
    mape    = float(np.mean(np.abs((actual - predicted[:len(actual)]) /
                                   (np.abs(actual) + 1e-9))) * 100)
    return dict(RMSE=round(rmse,4), MAE=round(mae,4),
                DirectionalAcc=round(da,2), MAPE=round(mape,4))


def print_table(results: dict):
    hdr = f"{'Model':<10} {'RMSE':>10} {'MAE':>10} {'Dir.Acc%':>12} {'MAPE%':>10}"
    print("\n" + "="*58)
    print("  DREAMING AI — FINAL COMPARISON")
    print("="*58)
    print(hdr)
    print("-"*58)
    for m, v in results.items():
        print(f"  {m:<8} {v['RMSE']:>10.4f} {v['MAE']:>10.4f} "
              f"{v['DirectionalAcc']:>11.2f}% {v['MAPE']:>9.4f}%")
    print("="*58)
    # Highlight winner
    best = min(results, key=lambda m: results[m]["RMSE"])
    print(f"  🏆  Best RMSE: {best}  ({results[best]['RMSE']:.4f})")
    print("="*58 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 6 — Ensemble Predictions
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_predict(debm_preds: np.ndarray, lstm_preds: np.ndarray,
                     weights: tuple = (0.65, 0.35)) -> np.ndarray:
    """
    Weighted ensemble of DEBM and LSTM predictions.
    DEBM gets higher weight (0.65) as the primary model.
    Two models make different types of errors — averaging reduces total error.

    Args:
        debm_preds: array of DEBM predictions (scaled or USD)
        lstm_preds: array of LSTM predictions (same scale)
        weights:    (debm_weight, lstm_weight) — must sum to 1.0

    Returns:
        ensemble: weighted average predictions
    """
    w_debm, w_lstm = weights
    assert abs(w_debm + w_lstm - 1.0) < 1e-6, \
        f"Ensemble weights must sum to 1.0, got {w_debm + w_lstm}"
    return w_debm * debm_preds + w_lstm * lstm_preds


def compute_ensemble_metrics(actual: np.ndarray,
                              debm_preds: np.ndarray,
                              lstm_preds: np.ndarray,
                              weights: tuple = (0.65, 0.35)) -> dict:
    """
    Compute metrics for: DEBM alone, LSTM alone, and their weighted ensemble.

    Returns a comparison dict keyed by model name.
    """
    ensemble = ensemble_predict(debm_preds, lstm_preds, weights)

    results = {}
    for name, preds in [("DEBM", debm_preds), ("LSTM", lstm_preds),
                         ("Ensemble", ensemble)]:
        if PREDICT_LOG_RETURN:
            da = float(np.mean((actual > 0) == (preds > 0)) * 100)
        else:
            da = float(np.mean(np.sign(preds[1:] - preds[:-1]) ==
                               np.sign(actual[1:] - actual[:-1])) * 100)
        results[name] = {
            "rmse":    float(np.sqrt(np.mean((actual - preds) ** 2))),
            "mae":     float(np.mean(np.abs(actual - preds))),
            "dir_acc": da,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 10 — Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_evaluate(X: np.ndarray, y: np.ndarray,
                           model_builder_fn,
                           n_splits: int = 3,
                           epochs: int = 20) -> dict:
    """
    Rolling-window walk-forward validation.

    Unlike a single train/test split, this re-trains the model on an
    expanding window and tests on the next unseen block — matching how
    a model would actually be deployed in production.

    Split structure (n_splits=5, N total samples):
      Fold 1: train 0 → N//6,    test N//6  → 2N//6
      Fold 2: train 0 → 2N//6,   test 2N//6 → 3N//6
      …
      Fold 5: train 0 → 5N//6,   test 5N//6 → N

    Args:
        X:                  (N, window, n_features) sequences
        y:                  (N,) targets
        model_builder_fn:   callable() → returns an untrained DreamingAI instance
        n_splits:           number of rolling folds (default 5)
        epochs:             quick retrain epochs per fold (use fewer than full training)

    Returns:
        dict with per-fold metrics and mean across all folds
    """
    N          = len(X)
    split_size = N // (n_splits + 1)
    results    = {"folds": [], "mean_dir_acc": 0.0, "mean_rmse": 0.0}

    all_da   = []
    all_rmse = []

    print(f"\n[WalkForward] Running {n_splits}-fold walk-forward validation …")
    print(f"[WalkForward] Total samples={N}, fold_size≈{split_size}")

    for i in range(1, n_splits + 1):
        train_end = split_size * i
        test_end  = min(split_size * (i + 1), N)

        if test_end <= train_end or train_end < 10:
            print(f"  Fold {i}: skipped (insufficient data)")
            break

        X_tr = torch.tensor(X[:train_end], dtype=torch.float32).to(DEVICE)
        y_tr = torch.tensor(y[:train_end], dtype=torch.float32).to(DEVICE)
        X_te = torch.tensor(X[train_end:test_end], dtype=torch.float32).to(DEVICE)
        y_te = y[train_end:test_end]

        # Build and quick-train a fresh model instance
        model = model_builder_fn().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        # Neutral zero sentiment tensors — required by DreamingAI.forward(x, sentiment)
        s_tr = torch.zeros(X_tr.size(0), 1, dtype=torch.float32, device=DEVICE)
        s_te = torch.zeros(X_te.size(0), 1, dtype=torch.float32, device=DEVICE)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            _, pred, _ = model(X_tr, s_tr)   # Bug 1 fix: sentiment required
            loss = criterion(pred.squeeze(1), y_tr)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            _, preds_t, _ = model(X_te, s_te)  # Bug 1 fix: sentiment required
        preds = preds_t.squeeze(1).cpu().numpy()

        rmse = float(np.sqrt(np.mean((y_te - preds) ** 2)))
        # Directional accuracy on the test fold
        if len(preds) > 1:
            if PREDICT_LOG_RETURN:
                da = float(np.mean((preds > 0) == (y_te > 0)) * 100)
            else:
                da = float(np.mean(np.sign(preds[1:] - preds[:-1]) ==
                                   np.sign(y_te[1:] - y_te[:-1])) * 100)
        else:
            da = 50.0   # can't compute with 1 sample

        all_da.append(da)
        all_rmse.append(rmse)

        fold_result = {
            "fold":       i,
            "train_size": train_end,
            "test_size":  test_end - train_end,
            "rmse":       round(rmse, 4),
            "dir_acc":    round(da, 2),
        }
        results["folds"].append(fold_result)
        print(f"  Fold {i}: Dir.Acc={da:.1f}%  RMSE={rmse:.4f}  "
              f"(train={train_end}, test={test_end - train_end})")

        # BUG-9 fix: Clear memory to prevent OOM
        del model
        del optimizer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if all_da:
        results["mean_dir_acc"] = float(round(np.mean(all_da), 2))
        results["mean_rmse"]    = float(round(np.mean(all_rmse), 4))
    print(f"\n[WalkForward] Mean Dir.Acc = {results['mean_dir_acc']:.1f}%  "
          f"Mean RMSE = {results['mean_rmse']:.4f}")
    print("[WalkForward] This is your honest, publishable accuracy estimate.\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def infer_debm(model, X_test, device=DEVICE) -> np.ndarray:
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        out = model.predict(
            torch.tensor(X_test, dtype=torch.float32, device=device)
        )
    return out.cpu().numpy().flatten()


def infer_lstm(model, X_test, device=DEVICE) -> np.ndarray:
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        out = model(torch.tensor(X_test, dtype=torch.float32, device=device))
    return out.cpu().numpy().flatten()


def infer_gan(model, X_test, device=DEVICE) -> np.ndarray:
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        out = model.predict(
            torch.tensor(X_test, dtype=torch.float32, device=device)
        )
    return out.cpu().numpy().flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig, fname: str):
    path = os.path.join(OUTPUTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[Eval] Saved -> {path}")
    plt.close(fig)


def plot_predictions(actual, preds: dict, ticker: str):
    """Multi-panel prediction vs actual + residuals."""
    fig = plt.figure(figsize=(16, 10), facecolor="#0D0F14")
    gs  = gridspec.GridSpec(2, 1, hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in [ax1, ax2]:
        ax.set_facecolor("#151820")
        ax.tick_params(colors="#8899AA")
        for sp in ax.spines.values(): sp.set_color("#252A3A")

    ax1.plot(actual, label="Actual", color=PALETTE["Actual"], lw=2, zorder=5)
    for name, p in preds.items():
        clr = PALETTE.get(name, "#AAAAAA")
        ax1.plot(p[:len(actual)], label=name, color=clr,
                 lw=1.5, alpha=0.85, ls="--")
    ax1.set_title(f"{ticker} — Predicted vs Actual Close Price",
                  color="#E8EAF0", fontsize=13, pad=10)
    ax1.set_ylabel("Normalised Price", color="#8899AA")
    ax1.legend(facecolor="#1C2030", labelcolor="#E8EAF0", framealpha=0.9)
    ax1.grid(alpha=0.15, color="#252A3A")

    for name, p in preds.items():
        res = actual - p[:len(actual)]
        clr = PALETTE.get(name, "#AAAAAA")
        ax2.plot(res, label=f"{name} residual", color=clr, lw=1.2, alpha=0.8)
    ax2.axhline(0, color="#8899AA", lw=0.8)
    ax2.set_title("Residuals (Actual − Predicted)", color="#E8EAF0", fontsize=13, pad=10)
    ax2.set_ylabel("Error", color="#8899AA")
    ax2.set_xlabel("Test Time Step", color="#8899AA")
    ax2.legend(facecolor="#1C2030", labelcolor="#E8EAF0", framealpha=0.9)
    ax2.grid(alpha=0.15, color="#252A3A")

    _savefig(fig, f"{ticker}_predictions.png")


def plot_metrics_bar(results: dict, ticker: str):
    """Side-by-side bar chart for all metrics."""
    metrics = ["RMSE", "MAE", "MAPE"]
    models  = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0D0F14")
    ax.set_facecolor("#151820")
    ax.tick_params(colors="#8899AA"); ax.yaxis.label.set_color("#8899AA")
    for sp in ax.spines.values(): sp.set_color("#252A3A")

    for i, m in enumerate(models):
        vals = [results[m][k] for k in metrics]
        clr  = PALETTE.get(m, "#AAAAAA")
        bars = ax.bar(x + i * width, vals, width, label=m,
                      color=clr, alpha=0.85, edgecolor="#0D0F14")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.015,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=8, color="#E8EAF0")

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, color="#8899AA")
    ax.set_title(f"{ticker} — Model Performance Comparison (lower is better)",
                 color="#E8EAF0", fontsize=13)
    ax.legend(facecolor="#1C2030", labelcolor="#E8EAF0")
    ax.grid(axis="y", alpha=0.15, color="#252A3A")
    _savefig(fig, f"{ticker}_metrics_bar.png")


def plot_directional_accuracy(results: dict, ticker: str):
    """Bar chart for Directional Accuracy (higher is better)."""
    models = list(results.keys())
    vals   = [results[m]["DirectionalAcc"] for m in models]
    colors = [PALETTE.get(m, "#AAAAAA") for m in models]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0D0F14")
    ax.set_facecolor("#151820")
    ax.tick_params(colors="#8899AA")
    for sp in ax.spines.values(): sp.set_color("#252A3A")

    bars = ax.bar(models, vals, color=colors, edgecolor="#0D0F14", alpha=0.85)
    ax.axhline(50, color="#8899AA", lw=0.8, ls="--", label="Random baseline (50%)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5, f"{v:.1f}%",
                ha="center", va="bottom", color="#E8EAF0", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_title(f"{ticker} — Directional Accuracy (higher is better)",
                 color="#E8EAF0", fontsize=13)
    ax.set_ylabel("Accuracy %", color="#8899AA")
    ax.legend(facecolor="#1C2030", labelcolor="#E8EAF0")
    ax.grid(axis="y", alpha=0.15, color="#252A3A")
    _savefig(fig, f"{ticker}_directional_acc.png")


def plot_energy_landscape(debm_model, X_test, ticker: str, device=DEVICE):
    """PCA projection of test sequences coloured by energy."""
    debm_model.eval()
    debm_model = debm_model.to(device)
    Xt = torch.tensor(X_test, dtype=torch.float32, device=device)
    St = torch.zeros(len(Xt), 1, device=device)

    with torch.no_grad():
        h      = debm_model.encode(Xt)
        energy = debm_model.energy_fn(h, St).cpu().numpy().flatten()
    h_np = h.cpu().numpy()

    pca  = PCA(n_components=2)
    h_2d = pca.fit_transform(h_np)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0D0F14")
    for ax in axes:
        ax.set_facecolor("#151820")
        ax.tick_params(colors="#8899AA")
        for sp in ax.spines.values(): sp.set_color("#252A3A")

    # Scatter: PCA coloured by energy
    sc = axes[0].scatter(h_2d[:,0], h_2d[:,1], c=energy,
                         cmap="RdYlGn_r", s=15, alpha=0.7)
    cb = fig.colorbar(sc, ax=axes[0])
    cb.set_label("Energy E(h)", color="#8899AA")
    cb.ax.yaxis.set_tick_params(color="#8899AA")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8899AA")
    axes[0].set_title("Energy Landscape (PCA projection)\nGreen=Low E / Realistic  "
                       "Red=High E / Extreme", color="#E8EAF0", fontsize=11)
    axes[0].set_xlabel("PC1", color="#8899AA")
    axes[0].set_ylabel("PC2", color="#8899AA")

    # Histogram of energies
    axes[1].hist(energy, bins=40, color=PALETTE["DEBM"], alpha=0.75, edgecolor="#0D0F14")
    axes[1].set_title("Distribution of Energy Scores", color="#E8EAF0", fontsize=11)
    axes[1].set_xlabel("Energy E(h, s)", color="#8899AA")
    axes[1].set_ylabel("Count", color="#8899AA")
    axes[1].grid(alpha=0.15, color="#252A3A")

    fig.suptitle(f"{ticker} — DEBM Energy Landscape", color="#E8EAF0",
                 fontsize=14, y=1.02)
    _savefig(fig, f"{ticker}_energy_landscape.png")


def plot_loss_curves(histories: dict, ticker: str):
    """Training + validation loss curves for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor="#0D0F14")
    titles = ["DEBM", "LSTM", "GAN"]

    for ax, name in zip(axes, titles):
        ax.set_facecolor("#151820")
        ax.tick_params(colors="#8899AA")
        for sp in ax.spines.values(): sp.set_color("#252A3A")

        if name not in histories:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", color="#8899AA")
            continue
        h = histories[name]
        tr = h.get("train", h.get("g_loss", []))
        vl = h.get("val", [])
        clr = PALETTE.get(name, "#AAAAAA")
        ax.plot(tr, label="Train", color=clr, lw=1.8)
        if vl:
            ax.plot(vl, label="Val", color=clr, lw=1.8, ls="--", alpha=0.7)
        ax.set_title(f"{name} Loss", color="#E8EAF0", fontsize=12)
        ax.set_xlabel("Epoch", color="#8899AA")
        ax.set_ylabel("Loss", color="#8899AA")
        ax.legend(facecolor="#1C2030", labelcolor="#E8EAF0")
        ax.grid(alpha=0.15, color="#252A3A")

    fig.suptitle(f"{ticker} — Training Curves", color="#E8EAF0", fontsize=14)
    _savefig(fig, f"{ticker}_loss_curves.png")


def full_evaluation(debm_model, lstm_model, gan_model,
                    X_test, y_test, scaler,
                    n_features, target_col_idx, ticker,
                    histories=None, device=DEVICE):
    """
    Run end-to-end evaluation:
      1. Infer predictions from all three models.
      2. Inverse-transform to USD prices.
      3. Compute metrics (incl. ensemble).
      4. Compute crash day accuracy (Objective 7).
      5. Generate all plots including crash analysis PNG.
    Returns results dict.
    """
    from src.data.features import inverse_target

    # Infer (scaled)
    dp = infer_debm(debm_model, X_test, device)
    lp = infer_lstm(lstm_model, X_test, device)
    gp = infer_gan (gan_model,  X_test, device)

    # IMPROVEMENT 6: Ensemble prediction
    ep_ = ensemble_predict(dp, lp, weights=(0.65, 0.35))

    # Inverse-transform
    actual      = inverse_target(y_test,     scaler, n_features, target_col_idx)
    debm_usd    = inverse_target(dp,         scaler, n_features, target_col_idx)
    lstm_usd    = inverse_target(lp,         scaler, n_features, target_col_idx)
    gan_usd     = inverse_target(gp,         scaler, n_features, target_col_idx)
    ensemble_usd = inverse_target(ep_,       scaler, n_features, target_col_idx)

    results = {
        "DEBM":     compute_metrics(actual, debm_usd),
        "LSTM":     compute_metrics(actual, lstm_usd),
        "GAN":      compute_metrics(actual, gan_usd),
        "Ensemble": compute_metrics(actual, ensemble_usd),
    }
    print_table(results)

    # OBJECTIVE 7: Crash day evaluation
    crash_info = save_crash_visualization(ticker, actual, debm_usd)
    results["crash_analysis"] = crash_info
    logger.info(
        f"[EVAL] Crash Day Accuracy: "
        f"{crash_info.get('crash_dir_acc', 'N/A')}%  "
        f"(N={crash_info['n_crash_days']} crash days)"
    )

    # Save raw predictions
    np.save(os.path.join(OUTPUTS_DIR, f"{ticker}_actual.npy"),        actual)
    np.save(os.path.join(OUTPUTS_DIR, f"{ticker}_debm_pred.npy"),     debm_usd)
    np.save(os.path.join(OUTPUTS_DIR, f"{ticker}_lstm_pred.npy"),     lstm_usd)
    np.save(os.path.join(OUTPUTS_DIR, f"{ticker}_gan_pred.npy"),      gan_usd)
    np.save(os.path.join(OUTPUTS_DIR, f"{ticker}_ensemble_pred.npy"), ensemble_usd)

    # Plots
    plot_predictions(actual, {"DEBM": debm_usd, "LSTM": lstm_usd,
                               "GAN": gan_usd, "Ensemble": ensemble_usd}, ticker)
    plot_metrics_bar(results, ticker)
    plot_directional_accuracy(results, ticker)
    plot_energy_landscape(debm_model, X_test, ticker, device)
    if histories:
        plot_loss_curves(histories, ticker)

    return results
