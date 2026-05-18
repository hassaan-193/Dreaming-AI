"""
Dreaming AI v8 — Role-Based Streamlit Dashboard
Roles: System Admin | ML Engineer | Financial Analyst
"""
import os
import sys
import json
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st

# ── MUST be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="Dreaming AI v8",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import project config safely ──────────────────────────────────────────────
try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from config import (
        DEVICE, TICKERS, TIMEFRAMES, MODELS_DIR, OUTPUTS_DIR,
        WINDOW_SIZE, FEATURE_COLS, CRASH_DAY_THRESHOLD
    )
except Exception as cfg_err:
    st.error(f"❌ Failed to load config.py: {cfg_err}")
    st.stop()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d1117; }
section[data-testid="stSidebar"] { background: #161b22; }
.metric-card {
    background: linear-gradient(135deg, #1c2130 0%, #0d1117 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
}
.metric-card .label { color: #8b949e; font-size:0.82rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }
.metric-card .value { color: #e6edf3; font-size:2rem; font-weight:700; margin-top:0.2rem; }
.metric-card .delta { font-size:0.85rem; margin-top:0.1rem; }
.delta-up   { color: #3fb950; }
.delta-down { color: #f85149; }
.crash-badge { background: rgba(248,81,73,0.15); border:1px solid #f85149; border-radius:8px; padding:0.8rem 1.2rem; color:#f85149; font-weight:600; }
.stTabs [data-baseweb="tab-list"] { gap:1rem; background:#161b22; border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:0.5rem 1.2rem; color:#8b949e; font-weight:600; }
.stTabs [aria-selected="true"] { background:#1f6feb !important; color:#e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Dreaming AI v8")
    st.markdown("---")
    role = st.selectbox(
        "👤 Select User Role",
        ["System Admin", "ML Engineer", "Financial Analyst"],
        help="Role-based access control"
    )
    st.markdown("---")
    if TORCH_OK and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.markdown(f"**Device:** 🟢 `{device_name}`")
            st.markdown(f"**VRAM:** `{vram:.1f} GB`")
        except Exception:
            st.markdown(f"**Device:** 🔵 `{DEVICE}`")
    else:
        st.markdown(f"**Device:** 🔵 `{DEVICE}`")

    st.markdown("---")
    st.markdown("**Available Tickers**")
    st.code(", ".join(TICKERS[:7]), language=None)


# ══════════════════════════════════════════════════════════════════════════════
# TAB FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_deploy_tab():
    st.markdown("## ⚙️ Manage Users and Deploy")
    st.info("System Admin Panel — Real-time deployment status and system controls.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🖥️ System Health")
        if TORCH_OK and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                pct = allocated / total if total > 0 else 0
                st.progress(min(pct, 1.0))
                st.write(f"GPU VRAM: `{allocated:.2f} GB / {total:.1f} GB`")
                st.write(f"CUDA: `{torch.version.cuda}`")
            except Exception as e:
                st.warning(f"GPU stats unavailable: {e}")
        else:
            st.warning("No GPU detected — Running on CPU")

        st.markdown("---")
        st.markdown("**Python:** `{}`".format(sys.version.split()[0]))
        st.markdown("**CWD:** `{}`".format(os.path.dirname(os.path.abspath(__file__))))

    with c2:
        st.markdown("### 🔑 Environment")
        env_keys = {
            "NEWS_API_KEY":    os.environ.get("NEWS_API_KEY",    "❌ Not set"),
            "FRED_API_KEY":    os.environ.get("FRED_API_KEY",    "⚠️ Optional"),
            "ALPHAVANTAGE_KEY": os.environ.get("ALPHAVANTAGE_KEY", "⚠️ Optional"),
        }
        for k, v in env_keys.items():
            masked = "✅ Loaded" if (v and "Not set" not in v and "Optional" not in v) else v
            st.markdown(f"**{k}:** `{masked}`")

        st.markdown("---")
        if st.button("🗑️ Purge Data Cache", use_container_width=True):
            import glob
            removed = 0
            for f in glob.glob(os.path.join(MODELS_DIR, "..", "data", "*.csv")):
                try:
                    os.remove(f)
                    removed += 1
                except Exception:
                    pass
            st.success(f"Purged {removed} cached data file(s).")

        if st.button("📋 Show Saved Models", use_container_width=True):
            models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")] if os.path.exists(MODELS_DIR) else []
            if models:
                st.write("\n".join(f"• `{m}`" for m in sorted(models)))
            else:
                st.info("No trained models found yet.")


def render_train_tab():
    st.markdown("## 🏋️ Train DEBM Model")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Configuration")
        selected_tickers = st.multiselect("Tickers", TICKERS, default=["AAPL"])
        time_interval = st.selectbox("Time Interval", list(TIMEFRAMES.keys()), index=4)
        train_mode = st.selectbox("Training Mode", ["full", "v2", "multi"],
                                  help="full=DEBM+v3, v2=DEBM only, multi=shared model")
        epochs = st.slider("Epochs", 1, 200, 60)
        use_sentiment = st.toggle("Sentiment Analysis", value=True)
        dry_run = st.toggle("Dry Run (1 epoch, fast test)", value=False)
        st.markdown("---")
        run_btn = st.button("🚀 Start Training", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📡 Live Training Status")

        # Show current GPU state
        if TORCH_OK and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                st.markdown(f"**GPU Memory:** `{allocated:.2f} GB / {total:.1f} GB`")
                st.progress(min(allocated / total, 1.0))
            except Exception:
                pass
        else:
            st.info("GPU not available — training will use CPU.")

        log_placeholder = st.empty()

        if run_btn:
            if not selected_tickers:
                st.error("Please select at least one ticker.")
            else:
                mode = "multi" if (len(selected_tickers) > 1 or train_mode == "multi") else train_mode
                cmd_parts = [
                    sys.executable, "pipeline.py",
                    "--ticker", *selected_tickers,
                    "--mode", mode,
                    "--epochs", str(1 if dry_run else epochs),
                    "--skip-preflight",
                ]
                if not use_sentiment:
                    cmd_parts.append("--no-sentiment")

                log_placeholder.info(f"▶ Running: `{' '.join(cmd_parts)}`")
                with st.spinner("Training in progress… This may take several minutes."):
                    env_vars = os.environ.copy()
                    env_vars["PYTHONIOENCODING"] = "utf-8"
                    result = subprocess.run(
                        cmd_parts,
                        capture_output=True, text=True, encoding="utf-8",
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        env=env_vars,
                    )

                if result.returncode == 0:
                    st.success("✅ Training complete!")
                    log_placeholder.code(result.stdout[-4000:], language=None)
                else:
                    st.error("❌ Training failed!")
                    log_placeholder.code(
                        (result.stderr or result.stdout)[-4000:], language=None
                    )


def render_eval_tab():
    st.markdown("## 📊 Compare Model Results")
    eval_ticker = st.selectbox("Select Ticker", TICKERS, key="eval_ticker")
    results_path = os.path.join(MODELS_DIR, f"{eval_ticker}_results.json")

    if not os.path.exists(results_path):
        st.warning(f"No results for **{eval_ticker}**. Run training first.")
        return

    with open(results_path) as f:
        res = json.load(f)

    # Model metric cards
    st.markdown("### 🏆 Model Performance")
    _NON_MODEL = {"crash_analysis", "walk_forward", "conditions"}
    model_keys = [k for k in ["DEBM", "LSTM", "GAN", "Ensemble", "DEBM_v3"]
                  if k in res and isinstance(res[k], dict) and "DirectionalAcc" in res[k]]

    if model_keys:
        cols = st.columns(len(model_keys))
        for col, mk in zip(cols, model_keys):
            m = res[mk]
            da = m.get("DirectionalAcc", 0)
            color = "#3fb950" if da >= 65 else "#f85149"
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">{mk}</div>
                  <div class="value" style="color:{color};">{da:.1f}%</div>
                  <div class="delta">Dir. Accuracy (target ≥65%)</div>
                  <div style="color:#8b949e;font-size:0.8rem;margin-top:0.4rem;">
                    RMSE: {m.get('RMSE', 0):.4f} &nbsp;|&nbsp; MAE: {m.get('MAE', 0):.4f}
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # Walk-forward results
    wf_path = os.path.join(OUTPUTS_DIR, f"{eval_ticker}_walkforward.json")
    if os.path.exists(wf_path):
        with open(wf_path) as f:
            wf = json.load(f)
        st.markdown("### 📈 Walk-Forward Validation")
        wc1, wc2 = st.columns(2)
        with wc1:
            st.metric("Mean Directional Accuracy", f"{wf.get('mean_dir_acc', 0):.1f}%")
        with wc2:
            st.metric("Mean RMSE", f"{wf.get('mean_rmse', 0):.4f}")

    # Charts
    st.markdown("### 📉 Evaluation Charts")
    chart_keys = ["predictions", "directional_acc", "metrics_bar", "loss_curves", "energy_landscape"]
    chart_cols = st.columns(2)
    shown = 0
    for img_key in chart_keys:
        img_path = os.path.join(OUTPUTS_DIR, f"{eval_ticker}_{img_key}.png")
        if os.path.exists(img_path):
            with chart_cols[shown % 2]:
                st.image(img_path, caption=img_key.replace("_", " ").title(), use_column_width=True)
            shown += 1
    if shown == 0:
        st.info("No chart images found. Run the full pipeline to generate them.")


def render_predict_tab():
    st.markdown('## Prediction Pipeline')

    if not TORCH_OK:
        st.error('PyTorch is not installed. Cannot run inference.')
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        pred_ticker = st.selectbox('Ticker', TICKERS, key='pred_ticker')
    with c2:
        pred_interval = st.selectbox(
            'Time Interval', list(TIMEFRAMES.keys()), index=4, key='pred_interval'
        )
    with c3:
        model_mode = st.selectbox('Model Mode', ['Single Stock', 'Multi Stock'], key='pred_mode')

    pred_btn = st.button('Get Prediction', type='primary', key='pred_btn')

    if pred_btn:
        if model_mode == 'Multi Stock':
            debm_path    = os.path.join(MODELS_DIR, 'multi_debm_best.pth')
            boost_ticker = 'multi'
        else:
            debm_path    = os.path.join(MODELS_DIR, pred_ticker + '_debm_best.pth')
            boost_ticker = pred_ticker

        scaler_path = os.path.join(MODELS_DIR, pred_ticker + '_scaler.pkl')

        if not os.path.exists(debm_path):
            st.error('No model found. Run training first.')
            return
        if not os.path.exists(scaler_path):
            st.error('Scaler not found for ' + pred_ticker)
            return

        with st.spinner('Running boosted inference...'):
            try:
                import joblib
                from src.data.fetcher import fetch_stock_data
                from src.models.debm import DreamingAI
                from src.models.ensemble import boosted_predict

                scaler = joblib.load(scaler_path)

                meta_path = os.path.join(MODELS_DIR, pred_ticker + '_meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path) as _mf:
                        _m = json.load(_mf)
                    n_feat    = int(_m.get('n_features', len(FEATURE_COLS)))
                    close_idx = int(_m.get('close_idx', 3))
                else:
                    n_feat    = len(FEATURE_COLS)
                    close_idx = 3

                if model_mode == 'Multi Stock':
                    debm = DreamingAI(n_features=n_feat, num_stocks=len(TICKERS))
                else:
                    debm = DreamingAI(n_features=n_feat)
                debm.load_state_dict(
                    torch.load(debm_path, map_location='cpu', weights_only=True)
                )
                debm.eval()

                df     = fetch_stock_data(pred_ticker, force_refresh=True)
                avail  = [c for c in FEATURE_COLS if c in df.columns]
                scaled = scaler.transform(df[avail].values).astype('float32')

                if len(scaled) < WINDOW_SIZE:
                    st.error('Not enough data: need ' + str(WINDOW_SIZE) + ' rows, got ' + str(len(scaled)))
                    return

                x_input = torch.tensor(scaled[-WINDOW_SIZE:][np.newaxis])
                ps = boosted_predict(debm, x_input, ticker=boost_ticker, device='cpu')

                def inv_transform(v):
                    d = np.zeros((1, n_feat), dtype='float32')
                    d[0, close_idx] = v
                    return float(scaler.inverse_transform(d)[0, close_idx])

                pred_usd = inv_transform(ps)
                last_usd = float(df['Close'].iloc[-1])
                pct_chg  = (pred_usd - last_usd) / last_usd * 100
                sign     = '+' if pred_usd > last_usd else ''

                res_ckpt   = os.path.join(MODELS_DIR, boost_ticker + '_residual_lstm.pth')
                meta_ckpt2 = os.path.join(MODELS_DIR, boost_ticker + '_meta_learner.pth')
                is_boosted = os.path.exists(res_ckpt) and os.path.exists(meta_ckpt2)
                badge = 'Boosted Ensemble (ResidualLSTM + MetaLearner)' if is_boosted else 'DEBM'

                st.metric(
                    label=badge + ' -- ' + pred_ticker + ' Next Close (' + pred_interval + ')',
                    value='$' + str(round(pred_usd, 2)),
                    delta=sign + str(round(pct_chg, 2)) + '% vs $' + str(round(last_usd, 2))
                )

                if not is_boosted:
                    st.info('Tip: Run the full training pipeline to enable Boosted Ensemble for higher accuracy.')

            except Exception as e:
                st.error('Prediction failed: ' + str(e))
                import traceback
                st.code(traceback.format_exc())

def render_live_feed_tab():
    st.markdown("## 📡 Subscribe Live Feed")
    st.info("Polls real-time price data from Yahoo Finance for the selected ticker.")

    feed_ticker = st.selectbox("Subscribe to Ticker", TICKERS, key="feed_ticker")

    if st.button("▶ Start Live Feed", type="primary"):
        try:
            import yfinance as yf
            placeholder = st.empty()
            history_prices = []

            for i in range(10):
                data = yf.Ticker(feed_ticker).fast_info
                price = float(data.get("last_price", 0) or data.get("previousClose", 0))
                if price == 0:
                    # Fallback: use history
                    hist = yf.Ticker(feed_ticker).history(period="1d", interval="1m")
                    price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0

                history_prices.append(price)
                delta = price - history_prices[-2] if len(history_prices) > 1 else 0.0

                with placeholder.container():
                    col_m, col_c = st.columns(2)
                    with col_m:
                        st.metric(
                            label=f"📈 {feed_ticker} Live Price",
                            value=f"${price:.2f}",
                            delta=f"{delta:+.2f}"
                        )
                    with col_c:
                        if len(history_prices) > 1:
                            import pandas as pd
                            st.line_chart(pd.Series(history_prices, name="Price"))
                time.sleep(2)

            st.success("Live feed session ended.")
        except Exception as e:
            st.error(f"Live feed error: {e}")


def render_landscape_tab():
    st.markdown("## 🌌 View Energy Landscape")
    st.info("Visualizes the latent energy space of the DEBM — Green = Normal, Red = Extreme/Crash.")

    land_ticker = st.selectbox("Ticker", TICKERS, key="land_ticker")

    # Try energy landscape first, fall back to predictions chart
    for img_key in ["energy_landscape", "predictions", "directional_acc"]:
        img_path = os.path.join(OUTPUTS_DIR, f"{land_ticker}_{img_key}.png")
        if os.path.exists(img_path):
            st.image(img_path, caption=img_key.replace("_", " ").title(), use_column_width=True)

    # Show conditions breakdown if available
    cond_path = os.path.join(MODELS_DIR, f"{land_ticker}_conditions.json")
    if os.path.exists(cond_path):
        with open(cond_path) as f:
            conds = json.load(f)
        st.markdown("### 📋 Market Condition Breakdown")
        for cond, metrics in conds.items():
            if isinstance(metrics, dict):
                st.markdown(f"**{cond.upper()}** — Dir.Acc: `{metrics.get('dir_acc', 'N/A')}`")

    if not any(
        os.path.exists(os.path.join(OUTPUTS_DIR, f"{land_ticker}_{k}.png"))
        for k in ["energy_landscape", "predictions"]
    ):
        st.warning(f"No charts found for **{land_ticker}**. Run the full pipeline first.")


def render_crash_tab():
    st.markdown("## 💥 Crash Scenario Analysis")

    crash_ticker = st.selectbox("Ticker", TICKERS, key="crash_ticker2")
    actual_path = os.path.join(OUTPUTS_DIR, f"{crash_ticker}_actual.npy")
    debm_path_np = os.path.join(OUTPUTS_DIR, f"{crash_ticker}_debm_pred.npy")
    crash_png = os.path.join(OUTPUTS_DIR, f"{crash_ticker}_crash_analysis.png")

    if not (os.path.exists(actual_path) and os.path.exists(debm_path_np)):
        st.warning(f"No prediction data for **{crash_ticker}**. Run training first.")
        return

    actual = np.load(actual_path)
    pred = np.load(debm_path_np)

    actual_ret = np.diff(actual) / (np.abs(actual[:-1]) + 1e-9)
    pred_ret = np.diff(pred) / (np.abs(pred[:-1]) + 1e-9)
    crash_mask = actual_ret < CRASH_DAY_THRESHOLD
    normal_mask = ~crash_mask
    n_crash = int(crash_mask.sum())
    n_total = len(actual_ret)

    overall_da = float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)) * 100)

    col1, col2, col3 = st.columns(3)
    with col1:
        if n_crash > 0:
            crash_da = float(
                np.mean(np.sign(pred_ret[crash_mask]) == np.sign(actual_ret[crash_mask])) * 100
            )
            st.markdown(f"""
            <div class="crash-badge">
              💥 Crash Day Accuracy<br>
              <span style="font-size:1.8rem;">{crash_da:.1f}%</span><br>
              <span style="font-size:0.8rem;">N={n_crash} crash days</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No crash days in test set.")

    with col2:
        if normal_mask.sum() > 0:
            normal_da = float(
                np.mean(np.sign(pred_ret[normal_mask]) == np.sign(actual_ret[normal_mask])) * 100
            )
            st.markdown(f"""
            <div style="background:rgba(63,185,80,0.15);border:1px solid #3fb950;
                        border-radius:8px;padding:0.8rem 1.2rem;color:#3fb950;font-weight:600;">
              📈 Normal Day Accuracy<br>
              <span style="font-size:1.8rem;">{normal_da:.1f}%</span><br>
              <span style="font-size:0.8rem;">N={int(normal_mask.sum())} normal days</span>
            </div>""", unsafe_allow_html=True)

    with col3:
        st.metric("Overall Directional Accuracy", f"{overall_da:.1f}%")
        st.metric("Total Test Days", f"{n_total}")

    # Pre-saved crash PNG if available
    if os.path.exists(crash_png):
        st.markdown("### 📊 Full Crash Analysis Chart")
        st.image(crash_png, use_column_width=True)
    else:
        # Render a quick inline chart
        st.markdown("### 📊 Actual vs Predicted")
        import pandas as pd
        chart_data = pd.DataFrame({"Actual": actual, "Predicted": pred})
        st.line_chart(chart_data)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTING — Role-Based Tab Rendering
# ══════════════════════════════════════════════════════════════════════════════

if role == "System Admin":
    tabs = st.tabs(["⚙️ Manage Users & Deploy"])
    with tabs[0]:
        render_deploy_tab()

elif role == "ML Engineer":
    tabs = st.tabs([
        "🏋️ Train DEBM Model",
        "📊 Compare Model Results",
        "🔮 Run Prediction",
    ])
    with tabs[0]: render_train_tab()
    with tabs[1]: render_eval_tab()
    with tabs[2]: render_predict_tab()

elif role == "Financial Analyst":
    tabs = st.tabs([
        "🔮 Run Prediction",
        "📡 Subscribe Live Feed",
        "🌌 Energy Landscape",
        "📊 Compare Model Results",
        "💥 Crash Scenarios",
    ])
    with tabs[0]: render_predict_tab()
    with tabs[1]: render_live_feed_tab()
    with tabs[2]: render_landscape_tab()
    with tabs[3]: render_eval_tab()
    with tabs[4]: render_crash_tab()
