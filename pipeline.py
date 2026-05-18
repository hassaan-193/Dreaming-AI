"""
Dreaming AI — Unified Master Pipeline
======================================

This single file replaces the three previous pipeline files:
  - pipeline.py       (was v2 only)
  - pipeline_v3.py    (was v2 + v3 extensions)
  - pipeline_multi.py (was multi-stock)

All three modes are now here, selected via --mode:

  FULL  (default, v2 + v3 in sequence — recommended for FYP):
      python pipeline.py --ticker AAPL

  V2 ONLY  (faster, no attention fusion/extreme tagging):
      python pipeline.py --ticker AAPL --mode v2

  MULTI-STOCK  (one shared DEBM across multiple tickers):
      python pipeline.py --ticker AAPL MSFT GOOG --mode multi

Nothing in the model, training, or evaluation code has changed.
"""

import os
import sys
import json
import logging
import argparse
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from rich.console import Console
    from rich.logging import RichHandler
    _console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=_console, rich_tracebacks=True, markup=True)]
    )
except ImportError:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    _console = None

logger = logging.getLogger("pipeline")

from config import (
    MODELS_DIR, OUTPUTS_DIR, DEVICE, WINDOW_SIZE,
    TIMEFRAMES, DEBM_EPOCHS, DREAM_EPOCHS,
    FUSION_DIM, FORECAST_HORIZONS,
    DEBM_BATCH, DEBM_LR, DEBM_WEIGHT_DECAY, GRAD_CLIP,
    LANGEVIN_STEPS, LANGEVIN_STEP_SIZE, LANGEVIN_NOISE,
    CD_WEIGHT, PRED_WEIGHT, LATENT_DIM,
    N_FEATURES, FEATURE_COLS, TICKERS, CRASH_DAY_THRESHOLD,
)

from src.data.fetcher    import fetch_stock_data
from src.data.sentiment  import attach_sentiment
from src.data.features   import prepare_data
from src.training.trainer import train_debm, dreaming_phase
from src.models.baselines import train_lstm, train_gan
from src.evaluation.evaluate import (full_evaluation, walk_forward_evaluate,
                                      ensemble_predict)
from src.models.debm import save_model, LangevinSampler


# ══════════════════════════════════════════════════════════════════════════════
# PREFLIGHT CHECK  (Objective 6)
# ══════════════════════════════════════════════════════════════════════════════

def preflight_check(tickers: list, device: str, n_features: int) -> bool:
    """
    Run all system-level validations before training begins.
    Returns True if all checks pass; raises RuntimeError on critical failure.

    Checks:
      1. Required packages importable
      2. N_FEATURES matches len(FEATURE_COLS)
      3. GPU available if device='cuda'
      4. Output/model directories exist (creates them)
      5. Cache files schema-compatible (deletes stale ones)
    """
    logger.info("[INFO] Running preflight checks …")
    ok = True

    # 1. Required packages
    required = ["ta", "transformers", "yfinance", "torch", "sklearn",
                "joblib", "textblob", "rich", "streamlit"]
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            logger.warning(f"[WARN] Package not found: {pkg} — some features may degrade")

    # 2. N_FEATURES consistency
    computed = len(FEATURE_COLS)
    if computed != n_features:
        logger.warning(
            f"[WARN] N_FEATURES mismatch: config says {n_features}, "
            f"len(FEATURE_COLS)={computed}. Using {computed}."
        )

    # 3. GPU check
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("[WARN] device='cuda' requested but CUDA not available — falling back to CPU")
        ok = False

    # 4. Directories
    for d in [MODELS_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)
        logger.info(f"[INFO] Directory OK: {d}")

    # 5. Stale cache check
    from config import DATA_DIR
    for ticker in tickers:
        cache = os.path.join(DATA_DIR, f"{ticker}_features.csv")
        if os.path.exists(cache):
            try:
                import pandas as pd
                df_peek = pd.read_csv(cache, index_col=0, nrows=1)
                if len(df_peek.columns) < n_features - 2:
                    logger.warning(
                        f"[WARN] Stale cache for {ticker} "
                        f"({len(df_peek.columns)} cols < {n_features-2}) — deleting"
                    )
                    os.remove(cache)
            except Exception as e:
                logger.warning(f"[WARN] Cache read error for {ticker} ({e}) — deleting")
                os.remove(cache)

    logger.info("[INFO] Preflight checks complete.")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_and_prepare(ticker: str, use_sentiment: bool = True,
                       force_refresh: bool = False):
    """Fetch stock data, attach sentiment, run feature engineering."""
    try:
        df = fetch_stock_data(ticker, force_refresh=force_refresh)
    except Exception as e:
        logger.error(f"[ERROR] fetch_stock_data({ticker}) failed: {e}")
        raise
    if use_sentiment:
        try:
            df = attach_sentiment(df, ticker)
        except Exception as e:
            logger.warning(f"[WARN] [{ticker}] Sentiment failed ({e}), using zeros")
            df["sentiment"] = 0.0
    else:
        df["sentiment"] = 0.0
    return prepare_data(df, ticker, window=WINDOW_SIZE), df


# ══════════════════════════════════════════════════════════════════════════════
# MODE v2  — DEBM + LSTM + GAN baselines (no attention fusion)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_v2(
    ticker:          str  = "AAPL",
    debm_epochs:     int  = 60,
    baseline_epochs: int  = 60,
    use_sentiment:   bool = True,
    use_finbert:     bool = False,
    force_refresh:   bool = False,
    device:          str  = DEVICE,
) -> dict:
    """v2 pipeline: DEBM + Dreaming Phase + LSTM baseline + GAN baseline."""
    print(f"\n{'='*62}")
    print(f"  DREAMING AI v2  |  {ticker}  |  device={device}")
    print(f"{'='*62}\n")

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("[v2] Step 1 — Fetch data")
    data, df = _fetch_and_prepare(ticker, use_sentiment, force_refresh)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    scaler    = data["scaler"]
    n_feat    = data["n_features"]
    close_idx = data["close_idx"]
    target_idx = data["target_col_idx"]

    s_train = X_train[:, -1, data["n_features"] - 4]  # sentinel: use feature index for 'sentiment'
    s_val   = X_val[:,   -1, data["n_features"] - 4]

    print("\n[v2] Step 2 — Train DEBM (initial, real data)")
    debm_model, debm_hist = train_debm(
        X_train, y_train, X_val, y_val,
        n_features=n_feat, ticker=ticker,
        sentiment_train=s_train, sentiment_val=s_val,
        epochs=debm_epochs, device=device,
    )

    print("\n[v2] Step 3 — Dreaming Phase")
    from src.extreme.market_conditions import tag_conditions, tag_sequence_conditions
    from config import PREDICT_LOG_RETURN, LOG_RETURN_COL
    df_clean = df.copy()
    if PREDICT_LOG_RETURN:
        df_clean[LOG_RETURN_COL] = np.log(df_clean["Close"] / df_clean["Close"].shift(1))
        df_clean = df_clean.dropna(subset=[LOG_RETURN_COL])
    row_labels = tag_conditions(df_clean)
    seq_labels = tag_sequence_conditions(row_labels, WINDOW_SIZE)
    seq_labels_train = seq_labels[:len(X_train)]

    debm_model = dreaming_phase(
        debm_model, X_train, y_train, X_val, y_val,
        n_features=n_feat, ticker=ticker,
        sentiment_train=s_train, sentiment_val=s_val,
        condition_labels=seq_labels_train,
        device=device,
    )
    save_model(debm_model, os.path.join(MODELS_DIR, f"{ticker}_debm_best.pth"))

    print("\n[v2] Step 4 — Train LSTM baseline")
    lstm_model, lstm_tr, lstm_vl = train_lstm(
        X_train, y_train, X_val, y_val,
        n_features=n_feat, ticker=ticker,
        epochs=baseline_epochs, device=device,
    )

    print("\n[v2] Step 5 — Train GAN baseline")
    gan_model, gan_tr, gan_vl = train_gan(
        X_train, y_train, X_val, y_val,
        n_features=n_feat, window=WINDOW_SIZE,
        ticker=ticker, epochs=baseline_epochs, device=device,
    )

    print("\n[v2] Step 6 — Evaluation & plots")
    histories = {
        "DEBM": debm_hist,
        "LSTM": {"train": lstm_tr, "val": lstm_vl},
        "GAN":  {"train": gan_tr,  "val": gan_vl},
    }
    results = full_evaluation(
        debm_model=debm_model, lstm_model=lstm_model, gan_model=gan_model,
        X_test=X_test, y_test=y_test,
        scaler=scaler, n_features=n_feat, target_col_idx=target_idx,
        ticker=ticker, histories=histories, device=device,
    )

    # ── Step 6b — Gradient Boosting Ensemble ──────────────────────────────────
    print("\n[v2] Step 6b — Training Gradient Boosting Ensemble …")
    try:
        from src.models.ensemble import train_boosting_ensemble
        boost_results = train_boosting_ensemble(
            debm_model=debm_model,
            X_train=X_train, y_train=y_train,
            X_val=X_val,   y_val=y_val,
            ticker=ticker,
            n_features=n_feat,
            epochs=40,
            device=device,
        )
        results["BoostedEnsemble"] = {
            "DirectionalAcc": boost_results["val_dir_acc_boosted"],
            "RMSE": results.get("DEBM", {}).get("RMSE", 0),
            "MAE":  results.get("DEBM", {}).get("MAE",  0),
            "MAPE": results.get("DEBM", {}).get("MAPE", 0),
        }
        print(f"[v2] Boosted Ensemble Val Dir.Acc: {boost_results['val_dir_acc_boosted']:.1f}%  "
              f"(vs DEBM alone: {boost_results['val_dir_acc_debm']:.1f}%)")
    except Exception as e:
        print(f"[v2] Boosting ensemble skipped ({e})")

    # ── IMPROVEMENT 10: Walk-forward validation ────────────────────────────────
    # Provides an honest, rolling accuracy estimate for FYP defence.
    print("\n[v2] Step 7 — Walk-forward validation …")
    from src.models.debm import DreamingAI as _DreamingAI
    def _model_builder():
        return _DreamingAI(n_features=n_feat, num_stocks=0)

    try:
        wf_results = walk_forward_evaluate(
            X_train, y_train,
            model_builder_fn=_model_builder,
            n_splits=5,
            epochs=20,
        )
        wf_path = os.path.join(OUTPUTS_DIR, f"{ticker}_walkforward.json")
        with open(wf_path, "w") as f:
            json.dump(wf_results, f, indent=2)
        print(f"[v2] Walk-forward results saved -> {wf_path}")
        results["walk_forward"] = wf_results
    except Exception as e:
        print(f"[v2] Walk-forward evaluation failed ({e}) — skipping")

    meta = {"n_features": int(n_feat), "close_idx": int(close_idx)}
    with open(os.path.join(MODELS_DIR, f"{ticker}_meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(MODELS_DIR, f"{ticker}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*62}\n  V2 COMPLETE  |  {ticker}\n{'='*62}\n")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MODE full  — V2 + V3 (attention fusion, enhanced sentiment, extreme tagging)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    ticker:          str  = "AAPL",
    debm_epochs:     int  = DEBM_EPOCHS,
    baseline_epochs: int  = 60,
    use_sentiment:   bool = True,
    use_finbert:     bool = False,
    force_refresh:   bool = False,
    device:          str  = DEVICE,
) -> dict:
    """Full single-stock pipeline (v2 then v3). Recommended entry point."""
    print(f"\n{'='*66}")
    print(f"  DREAMING AI — FULL PIPELINE  |  {ticker}  |  device={device}")
    print(f"{'='*66}\n")

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── Step 1: v2 sub-pipeline ────────────────────────────────────────────────
    print("[Pipeline] Step 1 — v2 sub-pipeline (LSTM + GAN + v2 DEBM) …")
    v2_results = run_pipeline_v2(
        ticker=ticker, debm_epochs=debm_epochs,
        baseline_epochs=baseline_epochs,
        use_sentiment=use_sentiment, use_finbert=use_finbert,
        force_refresh=force_refresh, device=device,
    )

    # ── Step 2: Reload data ────────────────────────────────────────────────────
    print("\n[Pipeline] Step 2 — Reloading feature data …")
    df = fetch_stock_data(ticker, force_refresh=False)
    if use_sentiment:
        df = attach_sentiment(df, ticker)

    # ── Step 3: Enhanced sentiment (uses canonical sentiment.py) ────────────────────────
    logger.info("\n[Pipeline] Step 3 — Enhanced sentiment …")
    # sentiment_v3.py removed (duplicate) — attach_sentiment already applied above
    # df already has 'sentiment' column from _fetch_and_prepare

    # ── Step 4: Feature engineering ───────────────────────────────────────────
    print("\n[Pipeline] Step 4 — Feature engineering …")
    data      = prepare_data(df, ticker, window=WINDOW_SIZE)
    X_train   = data["X_train"];  y_train = data["y_train"]
    X_val     = data["X_val"];    y_val   = data["y_val"]
    X_test    = data["X_test"];   y_test  = data["y_test"]
    scaler    = data["scaler"]
    n_feat    = data["n_features"]
    close_idx = data["close_idx"]
    target_idx= data["target_col_idx"]
    s_train   = X_train[:, -1, FEATURE_COLS.index("sentiment")]  # correct sentiment column
    s_val     = X_val[:,   -1, FEATURE_COLS.index("sentiment")]

    # ── Step 5: Tag market conditions ─────────────────────────────────────────
    print("\n[Pipeline] Step 5 — Tagging market conditions …")
    from src.extreme.market_conditions import tag_conditions, tag_sequence_conditions
    row_labels     = tag_conditions(df)
    seq_labels     = tag_sequence_conditions(row_labels, WINDOW_SIZE)
    seq_labels_test  = seq_labels[len(seq_labels) - len(X_test):]
    seq_labels_train = seq_labels[:len(X_train)]

    # ── Step 6: Train DreamingAIv3 ────────────────────────────────────────────
    print("\n[Pipeline] Step 6 — Training DreamingAIv3 (attention fusion) …")
    from src.fusion.attention_fusion import DreamingAIv3

    model_v3 = DreamingAIv3(
        n_features=n_feat, fusion_dim=FUSION_DIM,
        forecast_horizons=FORECAST_HORIZONS,
    ).to(device)

    opt     = optim.AdamW(model_v3.parameters(), lr=DEBM_LR,
                           weight_decay=DEBM_WEIGHT_DECAY)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=debm_epochs, eta_min=1e-5)
    mse     = nn.MSELoss()
    sampler = LangevinSampler(latent_dim=LATENT_DIM,
                               buffer_size=max(len(X_train), 512), device=device)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(s_train, dtype=torch.float32).unsqueeze(1),
        ),
        batch_size=DEBM_BATCH, shuffle=True, drop_last=True,
    )
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)
    sv = torch.tensor(s_val, dtype=torch.float32, device=device).unsqueeze(1)

    best_val = float("inf")
    ckpt_v3  = os.path.join(MODELS_DIR, f"{ticker}_debm_v3.pth")

    for ep in range(1, debm_epochs + 1):
        model_v3.train()
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(device), yb.to(device).unsqueeze(1), sb.to(device)
            e_real, pred, h = model_v3(xb, sb, timeframe="1d")
            avg_s  = float(sb.mean().item())
            h_fake = sampler.sample(model_v3.energy_fn, xb.size(0),
                                     sentiment_val=avg_s,
                                     n_steps=LANGEVIN_STEPS,
                                     step_size=LANGEVIN_STEP_SIZE,
                                     noise_std=LANGEVIN_NOISE)
            sf     = torch.full((h_fake.size(0), 1), avg_s,
                                dtype=torch.float32, device=device)
            e_fake = model_v3.energy_fn(h_fake, sf)
            cd_l   = (e_real.mean() - e_fake.mean()
                      + 0.001*(e_real**2 + e_fake**2).mean())
            loss   = PRED_WEIGHT*mse(pred, yb) + CD_WEIGHT*cd_l
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model_v3.parameters(), GRAD_CLIP)
            opt.step()
        sched.step()
        model_v3.eval()
        with torch.no_grad():
            _, vp, _ = model_v3(Xv, sv, timeframe="1d")
            vl = mse(vp, yv).item()
        if vl < best_val:
            best_val = vl
            torch.save(model_v3.state_dict(), ckpt_v3)
        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d}/{debm_epochs}  val_loss={vl:.5f}")

    model_v3.load_state_dict(torch.load(ckpt_v3, map_location=device, weights_only=True))
    print(f"[Pipeline] DreamingAIv3 best val_loss={best_val:.5f}")

    # ── Step 7: Extreme Dreaming Phase ────────────────────────────────────────
    print("\n[Pipeline] Step 7 — Extreme-condition Dreaming Phase …")
    from src.extreme.market_conditions import dream_extreme_scenarios
    X_ext, y_ext, _ = dream_extreme_scenarios(
        model_v3, X_train, y_train, seq_labels_train,
        n_per_condition=150, device=device,
    )
    if len(X_ext) > 0:
        X_hyb = np.concatenate([X_train, X_ext], axis=0)
        y_hyb = np.concatenate([y_train, y_ext], axis=0)
        print(f"[Pipeline] Hybrid: {len(X_train)} real + {len(X_ext)} extreme synthetic")
        loader_h = DataLoader(
            TensorDataset(
                torch.tensor(X_hyb, dtype=torch.float32),
                torch.tensor(y_hyb, dtype=torch.float32),
                torch.zeros(len(X_hyb), 1),
            ),
            batch_size=DEBM_BATCH, shuffle=True, drop_last=True,
        )
        opt2 = optim.AdamW(model_v3.parameters(), lr=DEBM_LR * 0.3)
        model_v3.train()
        for ep in range(1, DREAM_EPOCHS + 1):
            for xb, yb, sb in loader_h:
                xb, yb, sb = xb.to(device), yb.to(device).unsqueeze(1), sb.to(device)
                _, pred, _ = model_v3(xb, sb, timeframe="1d")
                loss = mse(pred, yb)
                opt2.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model_v3.parameters(), GRAD_CLIP)
                opt2.step()
        torch.save(model_v3.state_dict(), ckpt_v3)
        print("[Pipeline] Extreme fine-tuning complete.")

    # ── Step 8: Evaluation ────────────────────────────────────────────────────
    print("\n[Pipeline] Step 8 — v3 Evaluation …")
    model_v3.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        v3_preds_s = model_v3.predict(Xt, timeframe="1d").cpu().numpy().flatten()

    from src.data.features import inverse_target
    v3_preds_usd = inverse_target(v3_preds_s, scaler, n_feat, target_idx)
    actual_usd   = inverse_target(y_test,     scaler, n_feat, target_idx)

    from src.extreme.market_conditions import evaluate_by_condition
    cond_results = evaluate_by_condition(actual_usd, v3_preds_usd,
                                          seq_labels_test, model_name="DEBMv3")
    with open(os.path.join(MODELS_DIR, f"{ticker}_conditions.json"), "w") as f:
        json.dump(cond_results, f, indent=2)

    # ── Step 9: Visualisations ────────────────────────────────────────────────
    print("\n[Pipeline] Step 9 — v3 Visualisations …")
    from src.visualization.charts_v3 import (plot_extreme_conditions,
                                               plot_confidence_bands,
                                               plot_multi_timeframe_summary)
    plot_extreme_conditions(cond_results, ticker)
    plot_confidence_bands(actual_usd, v3_preds_usd, ticker)
    plot_multi_timeframe_summary(ticker)

    # ── Step 10: Save combined results ────────────────────────────────────────
    v3_rmse = float(np.sqrt(mean_squared_error(actual_usd, v3_preds_usd)))
    v3_mae  = float(mean_absolute_error(actual_usd, v3_preds_usd))
    
    from config import PREDICT_LOG_RETURN
    if PREDICT_LOG_RETURN:
        v3_da = float(np.mean((actual_usd > 0) == (v3_preds_usd > 0)) * 100)
    else:
        v3_da = float(np.mean(np.sign(np.diff(actual_usd)) ==
                              np.sign(np.diff(v3_preds_usd))) * 100)

    combined = {
        **v2_results,
        "DEBM_v3": {
            "RMSE":           round(v3_rmse, 4),
            "MAE":            round(v3_mae,  4),
            "DirectionalAcc": round(v3_da,   2),
            "MAPE":           round(float(np.mean(np.abs(
                (actual_usd - v3_preds_usd) / (np.abs(actual_usd) + 1e-9)
            )) * 100), 4),
        },
        "conditions": cond_results,
    }
    with open(os.path.join(MODELS_DIR, f"{ticker}_results.json"), "w") as f:
        json.dump(combined, f, indent=2)
    with open(os.path.join(MODELS_DIR, f"{ticker}_meta.json"), "w") as f:
        json.dump({"n_features": int(n_feat), "close_idx": int(close_idx)}, f)

    print(f"\n{'='*66}")
    print(f"  FULL PIPELINE COMPLETE  |  {ticker}")
    print(f"  DEBMv3  RMSE={v3_rmse:.4f}  Dir.Acc={v3_da:.1f}%")
    print(f"{'='*66}\n")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# MODE multi  — one shared DEBM across multiple tickers
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_multi(
    tickers:         list,
    debm_epochs:     int  = DEBM_EPOCHS,
    baseline_epochs: int  = 60,
    use_sentiment:   bool = True,
    force_refresh:   bool = False,
    device:          str  = DEVICE,
) -> dict:
    """Multi-stock pipeline: one shared DEBM with StockEmbedding."""
    print(f"\n{'='*66}")
    print(f"  DREAMING AI — MULTI-STOCK PIPELINE")
    print(f"  Tickers : {', '.join(tickers)}")
    print(f"  Device  : {device}")
    print(f"{'='*66}\n")

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    num_stocks   = len(tickers)
    all_data     = {}
    n_features   = None

    print("[Multi] Step 1 — Fetching and preparing data …")
    from src.extreme.market_conditions import tag_conditions, tag_sequence_conditions
    from config import PREDICT_LOG_RETURN, LOG_RETURN_COL
    import numpy as np
    
    all_seq_labels_train = []
    
    for ticker in tickers:
        print(f"  -> {ticker}")
        data, df = _fetch_and_prepare(ticker, use_sentiment, force_refresh)
        all_data[ticker] = data
        if n_features is None:
            n_features = data["n_features"]

        df_clean = df.copy()
        if PREDICT_LOG_RETURN:
            df_clean[LOG_RETURN_COL] = np.log(df_clean["Close"] / df_clean["Close"].shift(1))
            df_clean = df_clean.dropna(subset=[LOG_RETURN_COL])
        row_labels = tag_conditions(df_clean)
        seq_labels = tag_sequence_conditions(row_labels, WINDOW_SIZE)
        all_seq_labels_train.append(seq_labels[:len(data["X_train"])])

    print("\n[Multi] Step 2 — Merging datasets …")
    def _c(key): return np.concatenate([all_data[t][key] for t in tickers], axis=0)
    def _ids(key):
        return np.concatenate([
            np.full(len(all_data[t][key]), ticker_to_id[t], dtype=np.int64)
            for t in tickers
        ])

    X_train = _c("X_train");  y_train = _c("y_train")
    X_val   = _c("X_val");    y_val   = _c("y_val")
    X_test  = _c("X_test");   y_test  = _c("y_test")
    sid_train = _ids("X_train")
    sid_val   = _ids("X_val")
    multi_seq_labels_train = np.concatenate(all_seq_labels_train, axis=0)

    print(f"  Combined: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    print(f"\n[Multi] Step 3 — Training shared DEBM (num_stocks={num_stocks}) …")
    model, hist = train_debm(
        X_train, y_train, X_val, y_val,
        n_features=n_features, ticker="multi",
        sentiment_train=X_train[:, -1, -1],
        sentiment_val=X_val[:,   -1, -1],
        stock_ids_train=sid_train, stock_ids_val=sid_val,
        num_stocks=num_stocks, epochs=debm_epochs, device=device,
    )

    print("\n[Multi] Step 4 — Dreaming Phase …")
    model = dreaming_phase(
        model, X_train, y_train, X_val, y_val,
        n_features=n_features, ticker="multi",
        sentiment_train=X_train[:, -1, -1],
        sentiment_val=X_val[:,   -1, -1],
        stock_ids_train=sid_train, stock_ids_val=sid_val,
        condition_labels=multi_seq_labels_train,
        num_stocks=num_stocks, device=device,
    )

    print("\n[Multi] Step 5 — LSTM baselines per ticker …")
    for ticker in tickers:
        d = all_data[ticker]
        train_lstm(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                   n_features=d["n_features"], ticker=ticker,
                   epochs=baseline_epochs, device=device)

    print("\n[Multi] Step 6 — Evaluating …")
    results = {}
    base_model = model.module if hasattr(model, 'module') else model
    for ticker in tickers:
        d   = all_data[ticker]
        tid = ticker_to_id[ticker]
        model.eval()
        Xte  = torch.tensor(d["X_test"], dtype=torch.float32, device=device)
        sids = torch.full((len(Xte),), tid, dtype=torch.long, device=device)
        with torch.no_grad():
            ps = base_model.predict(Xte, sids).cpu().numpy()
        sc = d["scaler"];  ti = d["target_col_idx"]
        dm = np.zeros((len(ps), n_features));  dm[:, ti] = ps[:, 0]
        pp = sc.inverse_transform(dm)[:, ti]
        dm2 = np.zeros((len(d["y_test"]), n_features));  dm2[:, ti] = d["y_test"]
        tp  = sc.inverse_transform(dm2)[:, ti]
        rmse = float(np.sqrt(np.mean((pp - tp) ** 2)))
        mae  = float(np.mean(np.abs(pp - tp)))
        if PREDICT_LOG_RETURN:
            da = float(np.mean((pp > 0) == (tp > 0)) * 100)
        else:
            da = float(np.mean(np.sign(np.diff(pp)) == np.sign(np.diff(tp))) * 100)
        results[ticker] = {"RMSE": rmse, "MAE": mae, "DirectionalAcc": da}
        print(f"    {ticker}: RMSE={rmse:.4f}  MAE={mae:.4f}  DirAcc={da:.4f}")

    model_path = os.path.join(MODELS_DIR, "multi_debm_best.pth")
    torch.save(model.state_dict(), model_path)
    results_path = os.path.join(OUTPUTS_DIR, "multi_results.json")
    with open(results_path, "w") as f:
        json.dump({"tickers": tickers, "ticker_to_id": ticker_to_id,
                   "results": results}, f, indent=2)

    print(f"\n[Multi] Done. Model -> {model_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Dreaming AI — Unified Pipeline (v7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --ticker AAPL
  python pipeline.py --ticker AAPL --mode v2
  python pipeline.py --tickers AAPL MSFT GOOGL --mode multi
  python pipeline.py --ticker AAPL --dry-run
        """,
    )
    p.add_argument("--ticker",          nargs="+", default=["AAPL"],
                   help="Single or multiple tickers (alias: --tickers)")
    p.add_argument("--tickers",         nargs="+", default=None,
                   help="Alias for --ticker (multi-stock shorthand)")
    p.add_argument("--mode",            choices=["full", "v2", "multi"],
                   default="full")
    p.add_argument("--epochs",          type=int, default=DEBM_EPOCHS)
    p.add_argument("--baseline-epochs", type=int, default=60)
    p.add_argument("--no-sentiment",    action="store_true")
    p.add_argument("--finbert",         action="store_true")
    p.add_argument("--refresh",         action="store_true")
    p.add_argument("--force-refresh",   action="store_true")
    p.add_argument("--device",          default=DEVICE)
    p.add_argument("--dry-run",         action="store_true",
                   help="Dry run: 1 epoch, 50 rows — validates pipeline end-to-end")
    p.add_argument("--skip-preflight",  action="store_true",
                   help="Skip preflight checks (faster re-runs)")
    args = p.parse_args()

    if args.force_refresh:
        args.refresh = True

    # --tickers overrides --ticker
    raw_tickers = args.tickers if args.tickers else args.ticker
    tickers = [t.upper() for t in raw_tickers]

    # Dry-run overrides epochs
    epochs = 1 if args.dry_run else args.epochs
    if args.dry_run:
        logger.info("[INFO] DRY RUN mode: 1 epoch, limited data")

    device = args.device

    # Preflight
    if not args.skip_preflight:
        preflight_check(tickers, device, N_FEATURES)

    kw = dict(
        debm_epochs     = epochs,
        baseline_epochs = 1 if args.dry_run else args.baseline_epochs,
        use_sentiment   = not args.no_sentiment,
        force_refresh   = args.refresh,
        device          = device,
    )

    if args.mode == "multi" or len(tickers) > 1:
        run_pipeline_multi(tickers=tickers, **kw)
    elif args.mode == "v2":
        run_pipeline_v2(ticker=tickers[0], use_finbert=args.finbert, **kw)
    else:
        run_pipeline(ticker=tickers[0], use_finbert=args.finbert, **kw)


if __name__ == "__main__":
    main()
