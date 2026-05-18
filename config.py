"""
Dreaming AI v7 — Central Configuration
All hyperparameters, paths, and toggles in one place.

v7 Production-Grade Overhaul:
  - TICKERS: default multi-stock list for combined training
  - AMP_ENABLED: automatic mixed precision (CUDA only)
  - GPU startup check: prints device name, VRAM, CUDA version
  - All v6 accuracy improvements preserved
"""
import os
import logging

logger = logging.getLogger(__name__)

# ── Load .env file automatically ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        logger.info(f"[Config] Loaded .env from {_env_path}")
    else:
        logger.info("[Config] No .env file found — using system environment variables.")
except ImportError:
    logger.warning("[Config] python-dotenv not installed. Run: pip install python-dotenv")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
CACHE_DIR   = DATA_DIR  # ticker feature CSVs live in DATA_DIR

for _d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Device — OBJECTIVE 1: Full GPU Utilization ────────────────────────────────
import torch

def _gpu_startup_check() -> str:
    """
    Detect GPU, print device name / VRAM / CUDA version at launch.
    Returns the device string 'cuda' or 'cpu'.
    """
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_ver = torch.version.cuda or "unknown"
        print(f"[GPU] Device     : {dev_name}")
        print(f"[GPU] VRAM       : {vram_gb:.1f} GB")
        print(f"[GPU] CUDA       : {cuda_ver}")
        print(f"[GPU] PyTorch    : {torch.__version__}")
        return "cuda"
    else:
        print("[GPU] CUDA not available — running on CPU")
        return "cpu"

DEVICE = _gpu_startup_check()

# Automatic Mixed Precision — enabled only on CUDA
AMP_ENABLED = (DEVICE == "cuda")

# ── Multi-Stock Tickers — OBJECTIVE 2 ────────────────────────────────────────
# Default list for combined multi-stock training.
# Override via --tickers CLI arg in pipeline.py.
TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META"]

# ── Data ──────────────────────────────────────────────────────────────────────
FETCH_PERIOD   = "10y"         # yfinance history window
WINDOW_SIZE    = 60            # look-back window (trading days)
FORECAST_STEPS = 1             # how many days ahead to predict
TEST_SPLIT     = 0.15
VAL_SPLIT      = 0.10

# ── Log-Return Target ─────────────────────────────────────────────────────────
PREDICT_LOG_RETURN = True
LOG_RETURN_COL     = "log_return"

# ── Feature Column Definitions ────────────────────────────────────────────────
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

TECH_COLS = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "ema_20", "ema_50",
    "bb_upper", "bb_lower", "bb_pct",
    "atr", "obv_norm", "cci", "williams_r", "stoch_k", "stoch_d",
]

VOLUME_COLS = ["vwap", "vol_momentum", "pv_divergence", "vrsi"]

MACRO_COLS = ["vix", "spy_return"]

FEATURE_COLS = (
    OHLCV_COLS
    + TECH_COLS
    + VOLUME_COLS
    + ["sentiment", "log_return"]
    + MACRO_COLS
    + ["earnings_tomorrow"]
)

N_FEATURES = len(FEATURE_COLS)   # auto-computed — currently 29

# ---- Architecture gives exactly ~300M parameters (verified) ---------------
# LSTM_HIDDEN=2192, LSTM_LAYERS=3, LATENT_DIM=512, ENERGY_HIDDEN=[6144...]
# Encoder: 271.9M  +  EnergyFn: 27.8M  +  Predictor: 0.18M  = 299.9M total
# ---------------------------------------------------------------------------
LSTM_HIDDEN    = 2192    # tuned for exactly 300M params
LSTM_LAYERS    = 3
LSTM_DROPOUT   = 0.4
ENERGY_HIDDEN  = [6144, 3072, 1536, 512, 256, 128]
LATENT_DIM     = 512

DEBM_EPOCHS        = 100
DEBM_BATCH         = 64       # Reduced to prevent OOM with 250M parameters
DEBM_LR            = 1.0e-4   # Slightly lower LR for stability with higher penalty
DEBM_WEIGHT_DECAY  = 1e-4
CD_WEIGHT          = 0.05     # Increased for better generative learning of rare events
PRED_WEIGHT        = 1.0
GRAD_CLIP          = 1.0       # raised to 1.0 per Objective 5 spec

DIR_LOSS_WEIGHT    = 0.5       # Raised to 0.5 to force the model to focus heavily on direction (key accuracy driver)
LABEL_SMOOTHING    = 0.05      # Reduced to 0.05 for sharper directional classification
EARLY_STOP_PATIENCE = 30       # Increased to 30 so it survives warmup

# ── Learning Rate Scheduler ───────────────────────────────────────────────────
# OBJECTIVE 5: CosineAnnealingLR
COSINE_T_MAX   = DEBM_EPOCHS   # period of cosine cycle
COSINE_ETA_MIN = 1e-6          # minimum LR

# Langevin sampling
LANGEVIN_STEPS      = 100
LANGEVIN_STEP_SIZE  = 0.003
LANGEVIN_NOISE      = 0.001
LANGEVIN_N_CHAINS   = 10

# Dream Phase
DREAM_CYCLES      = 3
DREAM_N_SYNTHETIC = 500
DREAM_EPOCHS      = 30

# 60/40 extreme bias in Dreaming Phase
DREAM_EXTREME_RATIO = 0.60
DREAM_NORMAL_RATIO  = 0.40

# ── Baselines ─────────────────────────────────────────────────────────────────
LSTM_EPOCHS     = 60
LSTM_BATCH      = 64
LSTM_LR         = 3e-4

GAN_EPOCHS      = 60
GAN_BATCH       = 64
GAN_LR_G        = 2e-4
GAN_LR_D        = 2e-4
GAN_NOISE_DIM   = 128

# ── External API Keys ─────────────────────────────────────────────────────────
NEWS_API_KEY      = os.getenv("NEWS_API_KEY", "")
FRED_API_KEY      = os.getenv("FRED_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

def _key_status(name, val):
    return f"{'[SET]' if val else '[MISSING - fallback active]'}"

print(f"[Config] NEWS_API_KEY  : {_key_status('NEWS_API_KEY',  NEWS_API_KEY)}")
print(f"[Config] FRED_API_KEY  : {_key_status('FRED_API_KEY',  FRED_API_KEY)}")
print(f"[Config] N_FEATURES    : {N_FEATURES}  (FEATURE_COLS count)")
print(f"[Config] AMP_ENABLED   : {AMP_ENABLED}")

# ── Sentiment ─────────────────────────────────────────────────────────────────
SENTIMENT_DECAY = 0.85

# ── v3 Extensions ─────────────────────────────────────────────────────────────
TIMEFRAMES = {
    "15m":  {"interval": "15m",  "period": "60d",  "window": 64,  "label": "15 Minutes"},
    "30m":  {"interval": "30m",  "period": "60d",  "window": 48,  "label": "30 Minutes"},
    "1h":   {"interval": "1h",   "period": "730d", "window": 48,  "label": "1 Hour"},
    "4h":   {"interval": "1h",   "period": "730d", "window": 32,  "label": "4 Hours"},
    "1d":   {"interval": "1d",   "period": "10y",  "window": 60,  "label": "1 Day"},
    "1wk":  {"interval": "1wk",  "period": "15y",  "window": 52,  "label": "1 Week"},
}
DEFAULT_TIMEFRAME = "1d"

REALTIME_POLL_SECONDS   = 60
REALTIME_CACHE_MAXSIZE  = 100
LIVE_PREDICTION_HORIZON = 5

SENTIMENT_EMBED_DIM   = 64
FINBERT_FULL_DIM      = 768
USE_SENTIMENT_EMBED   = True
NEWS_INTENSITY_WINDOW = 7

FUSION_HEADS     = 8
FUSION_DIM       = 512
PRICE_FEAT_DIM   = 17
SENTIMENT_FEAT_DIM = 5

# ── Crash / Extreme Market Thresholds ─────────────────────────────────────────
CRASH_THRESHOLD      = -0.03
SPIKE_THRESHOLD      =  0.03
SIDEWAYS_THRESHOLD   =  0.005
EXTREME_LABEL_NAMES  = ["normal", "crash", "spike", "sideways_volatile"]

# OBJECTIVE 7: threshold for crash-day evaluation
CRASH_DAY_THRESHOLD = -0.02   # daily return < -2% = crash day

# ── Multi-Head Prediction ─────────────────────────────────────────────────────
FORECAST_HORIZONS = [1, 3, 5]
