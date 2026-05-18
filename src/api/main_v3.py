"""
Dreaming AI v3 — Extended FastAPI
Imports and mounts all v2 routes, then ADDS new v3 endpoints.
v2 endpoints are preserved exactly.

New v3 endpoints:
  GET  /predict_live       — real-time cached prediction
  POST /subscribe          — subscribe ticker to live engine
  GET  /history/{ticker}   — last N predictions
  GET  /timeframes         — list supported timeframes
  POST /predict_multi      — predict all horizons + all timeframes
  GET  /conditions/{ticker}— extreme market condition breakdown
"""
import os, sys, json, base64, traceback
from datetime import datetime
from typing import Optional

# ── path fix ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
for _p in [_ROOT, _SRC]:
    if _p not in sys.path: sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import torch

from config import (MODELS_DIR, OUTPUTS_DIR, DEVICE, WINDOW_SIZE,
                    N_FEATURES, TIMEFRAMES)

# ── Build app ─────────────────────────────────────────────────────────────────
_MODEL_VERSION = "7.0.0"

app = FastAPI(
    title="Dreaming AI v7",
    description="DEBM + Multi-Timeframe + Live Predictions + Attention Fusion (v7 Production)",
    version=_MODEL_VERSION,
    docs_url="/docs",
)

def _ok(ticker: str, data: dict) -> dict:
    """Standardised success envelope for all endpoints."""
    return {
        "status": "ok",
        "ticker": ticker,
        "data": data,
        "meta": {"device": DEVICE, "model_version": _MODEL_VERSION},
    }
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

_TEMPLATES_DIR = os.path.join(_ROOT, "templates")
_INDEX_HTML    = os.path.join(_TEMPLATES_DIR, "index.html")

# ── In-memory state ──────────────────────────────────────────────────────────
_model_cache: dict  = {}
_train_status: dict = {}

# ── Schemas ───────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    ticker: str = "AAPL"
    force_refresh: bool = False

class PredictRequest(BaseModel):
    ticker: str = "AAPL"
    use_sentiment: bool = True
    interval: str = "1d"

class SubscribeRequest(BaseModel):
    ticker: str = "AAPL"
    interval: str = "1d"


# ── Model loader ──────────────────────────────────────────────────────────────
def _load_models(ticker: str) -> dict:
    if ticker in _model_cache:
        return _model_cache[ticker]

    # Try v3 model first, fall back to v2
    debm_v3_path = os.path.join(MODELS_DIR, f"{ticker}_debm_v3.pth")
    debm_v2_path = os.path.join(MODELS_DIR, f"{ticker}_debm_best.pth")
    meta_path    = os.path.join(MODELS_DIR, f"{ticker}_meta.json")

    if not os.path.exists(debm_v2_path) and not os.path.exists(debm_v3_path):
        raise FileNotFoundError(f"No model for '{ticker}'. Call POST /train first.")

    meta      = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    n_feat    = int(meta.get("n_features", N_FEATURES))
    close_idx = int(meta.get("close_idx", 3))

    if os.path.exists(debm_v3_path):
        from src.fusion.attention_fusion import DreamingAIv3
        debm = DreamingAIv3(n_features=n_feat)
        debm.load_state_dict(torch.load(debm_v3_path, map_location="cpu", weights_only=True))
        model_ver = "v3"
    else:
        from src.models.debm import DreamingAI
        debm = DreamingAI(n_features=n_feat)
        debm.load_state_dict(torch.load(debm_v2_path, map_location="cpu", weights_only=True))
        model_ver = "v2"
    debm.eval()

    lstm_path = os.path.join(MODELS_DIR, f"{ticker}_lstm_best.pth")
    lstm = None
    if os.path.exists(lstm_path):
        from src.models.baselines import LSTMModel
        lstm = LSTMModel(n_features=n_feat)
        lstm.load_state_dict(torch.load(lstm_path, map_location="cpu", weights_only=True))
        lstm.eval()

    import joblib
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    _model_cache[ticker] = {
        "debm": debm, "lstm": lstm,
        "n_features": n_feat, "close_idx": close_idx,
        "scaler": scaler, "model_ver": model_ver,
    }
    print(f"[API] Loaded {model_ver} model for {ticker}")
    return _model_cache[ticker]


# ── Background training ───────────────────────────────────────────────────────
def _train_bg(ticker: str, force_refresh: bool):
    try:
        _train_status[ticker] = "running"
        from pipeline import run_pipeline
        run_pipeline(ticker=ticker, force_refresh=force_refresh)
        _train_status[ticker] = "done"
        _model_cache.pop(ticker, None)
    except Exception as e:
        _train_status[ticker] = f"error: {e}"
        traceback.print_exc()


# ── v2 routes (preserved exactly) ────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    if not os.path.exists(_INDEX_HTML):
        return HTMLResponse("<h2>templates/index.html not found</h2>"
                            "<p>API running. See <a href='/docs'>/docs</a></p>")
    return HTMLResponse(open(_INDEX_HTML, encoding="utf-8").read())

@app.get("/health")
def health():
    """Health check with device and version info."""
    return _ok("system", {
        "version": _MODEL_VERSION,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": str(datetime.now()),
    })

@app.post("/train")
def train(req: TrainRequest, bg: BackgroundTasks):
    """Start background training for a ticker."""
    t = req.ticker.upper().strip()
    if _train_status.get(t) == "running":
        return _ok(t, {"train_status": "already_running"})
    bg.add_task(_train_bg, t, req.force_refresh)
    return _ok(t, {"train_status": "started",
                   "message": "Poll GET /status/{ticker} for progress"})

@app.get("/status/{ticker}")
def status(ticker: str):
    """Poll training status for a ticker."""
    t = ticker.upper()
    return _ok(t, {"train_status": _train_status.get(t, "not_started")})

@app.post("/predict")
def predict(req: PredictRequest):
    ticker = req.ticker.upper().strip()
    try:
        cache = _load_models(ticker)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    debm      = cache["debm"].to(DEVICE)
    lstm      = cache["lstm"]   # may be None if LSTM not trained yet
    n_feat    = cache["n_features"]
    close_idx = cache["close_idx"]
    scaler    = cache["scaler"]

    from src.data.fetcher   import fetch_stock_data
    from src.data.sentiment import attach_sentiment
    from src.data.features  import FEATURE_COLS

    df = fetch_stock_data(ticker, force_refresh=True)
    if req.use_sentiment:
        df = attach_sentiment(df, ticker)

    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(df) < WINDOW_SIZE:
        raise HTTPException(400, f"Not enough data ({len(df)} rows).")

    scaled  = scaler.transform(df[available].values).astype("float32")
    window  = scaled[-WINDOW_SIZE:]
    x_input = torch.tensor(window[np.newaxis], device=DEVICE)

    debm.eval()
    with torch.no_grad():
        try:
            ps = debm.predict(x_input, timeframe=req.interval).cpu().numpy().flatten()[0]
        except TypeError:
            ps = debm.predict(x_input).cpu().numpy().flatten()[0]

    with torch.no_grad():
        h_base = debm.encode(x_input)
        samps = [debm.predictor(h_base + torch.randn_like(h_base)*0.05
                                ).detach().cpu().numpy().flatten()[0] for _ in range(20)]
    std_s = float(np.std(samps))

    def inv(v):
        d = np.zeros((1, n_feat), dtype="float32")
        d[0, close_idx] = v
        return float(scaler.inverse_transform(d)[0, close_idx])

    pred_usd = inv(ps)
    last_usd = float(df["Close"].iloc[-1])

    # IMPROVEMENT 6: Ensemble prediction (DEBM + LSTM blend)
    ensemble_usd = None
    if lstm is not None:
        try:
            lstm = lstm.to(DEVICE)
            lstm.eval()
            with torch.no_grad():
                ls = lstm(x_input).cpu().numpy().flatten()[0]
            lstm_usd = inv(ls)
            from src.evaluation.evaluate import ensemble_predict
            ens_scaled = ensemble_predict(
                np.array([ps]), np.array([ls]), weights=(0.65, 0.35)
            )[0]
            ensemble_usd = round(inv(ens_scaled), 2)
        except Exception as _e:
            print(f"[API] Ensemble blend failed ({_e}) — skipping")

    from src.data.sentiment import get_realtime_sentiment
    sent = {"sentiment_scalar": round(get_realtime_sentiment(ticker), 4)} if req.use_sentiment else {}

    response = {
        "ticker":             ticker,
        "interval":           req.interval,
        "predicted_price":    round(pred_usd, 2),
        "confidence_band":    {"low": round(inv(ps - std_s), 2),
                               "high": round(inv(ps + std_s), 2)},
        "last_known_price":   round(last_usd, 2),
        "trend":              "UP" if pred_usd > last_usd else "DOWN",
        "model":              f"DreamingAI-{cache['model_ver']}",
        "timestamp":          str(datetime.now()),
        **sent,
    }
    if ensemble_usd is not None:
        response["ensemble_prediction"] = ensemble_usd
        response["ensemble_trend"]      = "UP" if ensemble_usd > last_usd else "DOWN"

    return response

@app.get("/results/{ticker}")
def results(ticker: str):
    """Return evaluation metrics and base64 plot images."""
    t = ticker.upper()
    rpath = os.path.join(MODELS_DIR, f"{t}_results.json")
    if not os.path.exists(rpath):
        raise HTTPException(404, f"No results for '{t}'. Run /train first.")
    metrics = json.load(open(rpath))
    plots = {}
    for k in ["predictions", "metrics_bar", "directional_acc",
              "energy_landscape", "loss_curves", "extreme_conditions",
              "multi_timeframe", "confidence_bands", "crash_analysis"]:
        fp = os.path.join(OUTPUTS_DIR, f"{t}_{k}.png")
        if os.path.exists(fp):
            plots[k] = base64.b64encode(open(fp, "rb").read()).decode()
    return JSONResponse(_ok(t, {"metrics": metrics, "plots": plots}))


# ── v3 NEW endpoints ──────────────────────────────────────────────────────────

@app.get("/timeframes")
def timeframes():
    """List all supported prediction timeframes."""
    return {"timeframes": [
        {"id":k, "label":v["label"], "interval":v["interval"], "window":v["window"]}
        for k,v in TIMEFRAMES.items()
    ]}


@app.get("/predict_live")
def predict_live(
    ticker:   str = Query(..., description="Stock ticker e.g. AAPL"),
    interval: str = Query("1d", description="Timeframe: 15m,30m,1h,4h,1d,1wk")
):
    """
    Return the latest cached real-time prediction.
    If not cached yet, runs inference immediately (blocking).
    Subscribe with POST /subscribe for continuous background updates.
    """
    ticker = ticker.upper()
    from src.realtime.live_engine import LIVE_CACHE

    cached = LIVE_CACHE.get(ticker, interval)
    if cached:
        return {**cached, "source": "cache"}

    # Not cached yet — run immediately
    try:
        cache = _load_models(ticker)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    from src.realtime.live_engine import run_single_prediction
    try:
        result = run_single_prediction(ticker, interval, cache)
        LIVE_CACHE.set(ticker, interval, result)
        return {**result, "source": "fresh"}
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@app.post("/subscribe")
def subscribe(req: SubscribeRequest):
    """
    Subscribe a ticker+interval to the live prediction engine.
    Predictions will refresh every REALTIME_POLL_SECONDS seconds.
    """
    ticker = req.ticker.upper()
    from src.realtime.live_engine import get_engine, init_engine
    engine = get_engine()
    if engine is None:
        try:
            cache = _load_models(ticker)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        engine = init_engine({ticker: cache})

    engine.subscribe(ticker, req.interval)
    return {"status":"subscribed","ticker":ticker,"interval":req.interval,
            "poll_seconds":30}


@app.get("/history/{ticker}")
def history(ticker: str,
            interval: str = Query("1d"),
            n: int = Query(50, ge=1, le=100)):
    """Return the last N live predictions for a ticker/interval."""
    from src.realtime.live_engine import get_engine
    engine = get_engine()
    if engine is None:
        return {"ticker":ticker,"interval":interval,"history":[]}
    hist = engine.get_history(ticker.upper(), interval)[-n:]
    return {"ticker":ticker.upper(),"interval":interval,
            "count":len(hist),"history":hist}


@app.post("/predict_multi")
def predict_multi(req: PredictRequest):
    """
    Multi-horizon + multi-timeframe prediction.
    Returns predictions for 1-day, 3-day, 5-day horizons.
    """
    ticker = req.ticker.upper()
    try:
        cache = _load_models(ticker)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    debm      = cache["debm"].to(DEVICE)
    n_feat    = cache["n_features"]
    close_idx = cache["close_idx"]
    scaler    = cache["scaler"]

    from src.data.fetcher  import fetch_stock_data
    from src.data.features import FEATURE_COLS

    df        = fetch_stock_data(ticker, force_refresh=True)
    available = [c for c in FEATURE_COLS if c in df.columns]
    scaled    = scaler.transform(df[available].values).astype("float32")
    window    = scaled[-WINDOW_SIZE:]
    x_input   = torch.tensor(window[np.newaxis], device=DEVICE)

    def inv(v):
        d = np.zeros((1,n_feat),dtype="float32"); d[0,close_idx]=v
        return float(scaler.inverse_transform(d)[0,close_idx])

    debm.eval()
    results = {}
    with torch.no_grad():
        try:
            preds = debm.predict_multihorizon(x_input, timeframe=req.interval)
            for hname, pt in preds.items():
                results[hname] = round(inv(pt.cpu().numpy().flatten()[0]), 2)
        except (AttributeError, TypeError):
            # v2 model fallback — only 1-step
            ps = debm.predict(x_input).cpu().numpy().flatten()[0]
            results["h1"] = round(inv(ps), 2)

    return {
        "ticker":     ticker,
        "interval":   req.interval,
        "horizons":   results,
        "last_price": round(float(df["Close"].iloc[-1]), 2),
        "timestamp":  str(datetime.now()),
    }


@app.get("/conditions/{ticker}")
def conditions(ticker: str):
    """Return extreme market condition breakdown for latest test results."""
    t = ticker.upper()
    cpath = os.path.join(MODELS_DIR, f"{t}_conditions.json")
    if not os.path.exists(cpath):
        raise HTTPException(404, f"No condition data for '{t}'. Run /train first.")
    return JSONResponse(_ok(t, json.load(open(cpath))))


# ── Startup: init live engine ─────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Log startup info including device and version."""
    print(f"[API] Dreaming AI v{_MODEL_VERSION} started.")
    print(f"[API] Device: {DEVICE}  CUDA: {torch.cuda.is_available()}")
    print("[API] Live engine will initialise on first /subscribe call.")
    print(f"[API] Docs: http://localhost:8000/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main_v3:app", host="0.0.0.0", port=8000, reload=False)
