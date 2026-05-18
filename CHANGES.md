# Dreaming AI v6 — CHANGES.md
## Master Accuracy Upgrade Log

**Date:** 2026-05-03  
**Session:** All 10 accuracy improvements applied in one session.  
**Baseline:** Dreaming AI v5/v6 with 22 features (OHLCV + 15 tech + sentiment)  
**After:**    29 features (OHLCV + 15 tech + 4 volume + sentiment + log_return + VIX + SPY + earnings_tomorrow)

---

## Files Modified

### `config.py` — Full Rewrite
- **Improvement 1:** Added `PREDICT_LOG_RETURN = True`, `LOG_RETURN_COL = "log_return"`
- **Improvement 2:** Added `DREAM_EXTREME_RATIO = 0.60`, `DREAM_NORMAL_RATIO = 0.40`
- **Improvement 3:** Added `MACRO_COLS = ["vix", "spy_return"]`
- **Improvement 4:** Changed `DIR_LOSS_WEIGHT = 0.5` (was 0.3)
- **Improvement 5:** Changed `FETCH_PERIOD = "10y"` (was "5y"); weekly period → "15y"
- **Improvement 8:** `"earnings_tomorrow"` added to `FEATURE_COLS`
- **Improvement 9:** Added `VOLUME_COLS = ["vwap", "vol_momentum", "pv_divergence", "vrsi"]`
- **Final Checklist:** `N_FEATURES = len(FEATURE_COLS)` — now computed dynamically (result: **29**)

### `src/data/fetcher.py` — Full Rewrite
- **Improvement 3:** Added `fetch_vix(start, end)` — downloads ^VIX, falls back to fill=20.0
- **Improvement 3:** Added `fetch_spy_return(start, end)` — downloads SPY log return, falls back to fill=0.0
- **Improvement 8:** Added `get_earnings_dates(ticker)` — binary earnings calendar via yfinance.Ticker.earnings_dates
- **Improvement 9:** Added 4 volume profile features inside `add_technical_indicators()`:
  - `vwap` — 20-day volume-weighted average price
  - `vol_momentum` — volume / 20-day SMA(volume)
  - `pv_divergence` — price_change × volume_change (divergence signal)
  - `vrsi` — volume-weighted RSI (ta library) or VWAP fallback
- **Improvement 3+8:** `fetch_stock_data()` now joins VIX, SPY, and earnings_tomorrow after indicators
- **Final Checklist:** Auto cache invalidation — if CSV column count < N_FEATURES-2, forces refresh
- `fetch_latest()` updated to also compute macro features for live inference

### `src/data/features.py` — Full Rewrite
- **Improvement 1:** `prepare_data()` computes `log_return = log(Close / Close.shift(1))` before scaling
- **Improvement 1:** `target_col_idx` is now dynamically selected:
  - `PREDICT_LOG_RETURN=True` → target = `log_return` column
  - `PREDICT_LOG_RETURN=False` → target = `Close` column (legacy)
- **Improvement 1:** Added `return_to_price(predicted_returns, last_known_price)` utility
- Added `inverse_target(y_scaled, scaler, n_features, target_col_idx)` generic version
- `prepare_data()` returns `target_col_idx` in the output dict (was only `close_idx`)
- Prints `[Features] N_FEATURES = X` on each run for easy verification

### `src/training/trainer.py` — Full Rewrite
- **Improvement 2:** Dreaming Phase now uses 60/40 split:
  - `n_standard = int(DREAM_N_SYNTHETIC * 0.40) = 200` normal Langevin samples
  - `n_per_condition = int(DREAM_N_SYNTHETIC * 0.60) // 2 = 150` crash + 150 spike
  - Prints `[Dream] Budget: 200 normal + 300 rare (60.0% rare)`
- **Improvement 4:** `_directional_loss()` completely replaced with weighted BCE:
  - Works in both price-prediction and log-return modes
  - Large moves (crashes/spikes) get higher weight → model focuses on extreme days
  - Uses `binary_cross_entropy_with_logits` with detached move-magnitude weights
- **Final Checklist:** DataLoader now uses `pin_memory=True` and `num_workers=4` on CUDA
- `train_debm()` prints training target mode in header line

### `src/data/sentiment.py` — Full Rewrite  
- **Improvement 7:** Complete FinBERT scorer added:
  - `_FINBERT_MODEL`, `_FINBERT_TOKENIZER`, `_FINBERT_LOADED` module-level state
  - `_load_finbert()` — lazy loader with full exception handling; ~440MB first download
  - `score_headline_textblob(headline)` — named TextBlob wrapper
  - `score_headline_finbert(headline)` — single headline FinBERT; TextBlob fallback
  - `score_headlines_finbert(headlines)` — batch FinBERT; mean score
  - Returns `positive_prob - negative_prob ∈ [-1, +1]`
  - GPU auto-detection via `config.DEVICE`
- `build_sentiment_series()` upgraded to call FinBERT when `use_finbert=True`
- All existing public functions (`attach_sentiment`, `get_realtime_sentiment`) preserved exactly

### `src/evaluation/evaluate.py` — Full Rewrite
- **Improvement 6:** Added `ensemble_predict(debm_preds, lstm_preds, weights=(0.65, 0.35))`
- **Improvement 6:** Added `compute_ensemble_metrics(actual, debm_preds, lstm_preds, weights)`
- **Improvement 10:** Added `walk_forward_evaluate(X, y, model_builder_fn, n_splits=5, epochs=20)`:
  - Rolling expanding-window train/test, 5 folds
  - Prints per-fold Dir.Acc + RMSE
  - Returns dict with `folds`, `mean_dir_acc`, `mean_rmse`
  - Described as the "honest, publishable accuracy estimate" for FYP defence
- `PALETTE` dict extended with `"Ensemble": "#F39C12"` (orange)
- `full_evaluation()` now includes Ensemble as 4th model in all metrics and plots
- Ensemble predictions saved to `outputs/{ticker}_ensemble_pred.npy`

### `src/api/main_v3.py` — Targeted Edit
- **Improvement 6:** `/predict` endpoint now blends DEBM + LSTM via `ensemble_predict(0.65, 0.35)`
- Returns `"ensemble_prediction"` and `"ensemble_trend"` in JSON response
- Ensemble is only computed if LSTM model is loaded (graceful skip if not)

### `pipeline.py` — Targeted Edits
- **Improvement 10:** `run_pipeline_v2()` now calls `walk_forward_evaluate()` after `full_evaluation()`
- Walk-forward results saved to `outputs/{ticker}_walkforward.json`
- Added `--force-refresh` CLI flag as alias for `--refresh`
- Imports updated: `walk_forward_evaluate` and `ensemble_predict` imported from evaluate

### `requirements.txt` — Minor Update
- **Improvement 7:** `transformers>=4.28.0` → `transformers>=4.30.0`

---

## Feature Column Summary (Final)

| Column Group | Columns | Count |
|---|---|---|
| OHLCV | Open, High, Low, Close, Volume | 5 |
| Tech | rsi, macd, macd_signal, macd_diff, ema_20, ema_50, bb_upper, bb_lower, bb_pct, atr, obv_norm, cci, williams_r, stoch_k, stoch_d | 15 |
| Volume Profile | vwap, vol_momentum, pv_divergence, vrsi | 4 |
| Sentiment | sentiment | 1 |
| Log Return | log_return | 1 |
| Macro | vix, spy_return | 2 |
| Earnings | earnings_tomorrow | 1 |
| **TOTAL** | | **29** |

---

## Key Design Decisions

1. **Log return as target** — More stationary than raw price; directional accuracy is directly measurable as `sign(predicted_return) == sign(actual_return)`
2. **Graceful fallbacks everywhere** — VIX/SPY failures fill with neutral values (20.0, 0.0); FinBERT falls back to TextBlob; earnings calendar falls back to 0.0
3. **60/40 extreme bias** — The Dreaming Phase now dedicates 60% of its synthetic budget to crash/spike scenarios, addressing the class imbalance in rare market events
4. **Dynamic N_FEATURES** — Never hardcoded again; computed from `len(FEATURE_COLS)` so adding a new feature is a one-line change in FEATURE_COLS
5. **Walk-forward as primary metric** — FYP defence should cite walk-forward Dir.Acc as the primary honest estimate; single test-split RMSE is supplementary

---

## How to Run After Upgrades

```bash
# Full pipeline with force-refresh (clears stale feature cache)
python pipeline.py --ticker AAPL --force-refresh

# Expected terminal output milestones:
# [Config] N_FEATURES    : 29  (FEATURE_COLS count)
# [Features] N_FEATURES = 29  |  Available cols: 29
# [Features] Target: log_return (directional mode) at index 20
# [Dream] Budget: 200 normal + 300 rare (60.0% rare)
# [WalkForward] Running 5-fold walk-forward validation …
# [WalkForward] Mean Dir.Acc = XX.X%   Mean RMSE = X.XXXX
# [v2] Walk-forward results saved → outputs/AAPL_walkforward.json
```

---

## Expected Performance Gains (Estimated)

| Metric | Before | After (estimated) |
|---|---|---|
| Directional Accuracy (normal days) | ~58-62% | ~66-70% |
| Directional Accuracy (crash days) | ~45-55% | ~60-68% |
| RMSE | baseline | -10 to -20% |
| Walk-forward mean Dir.Acc | N/A | 62-68% honest estimate |

---

# Dreaming AI v7 — Production-Grade Overhaul
**Date:** 2026-05-14 | **Objectives:** 7

## Files Changed

| File | Change |
|---|---|
| `src/training/trainer.py` | AMP GradScaler, GPU memory monitor, label smoothing, early stopping on val dir acc |
| `src/models/baselines.py` | AMP + pin_memory/num_workers on CUDA |
| `pipeline.py` | preflight_check(), --dry-run, --tickers, rich logging, try/except on external calls |
| `src/evaluation/evaluate.py` | evaluate_crash_days(), save_crash_visualization(), crash PNG auto-save |
| `src/api/main_v3.py` | v7.0.0, _ok() envelope, clean schema for all endpoints |
| `app.py` | NEW — Streamlit 4-tab dashboard (Train/Evaluate/Predict/Crash Scenarios) |
| `requirements.txt` | Added streamlit>=1.28.0, rich>=13.0.0, torch>=2.0.0 |

## Files Deleted

| File | Reason |
|---|---|
| `src/data/sentiment_v3.py` | Duplicate FinBERT code — merged into sentiment.py |

## How to Run (v7)
```bash
pip install -r requirements.txt
python pipeline.py --ticker AAPL --dry-run   # validate
python pipeline.py --ticker AAPL             # full run
streamlit run app.py                          # dashboard
uvicorn src.api.main_v3:app --port 8000       # API
```
