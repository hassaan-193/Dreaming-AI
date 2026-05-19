"""
Dreaming AI v6 — Sentiment Module
Loads NEWS_API_KEY from .env file automatically.

v6 Additions (Improvement 7):
  - Full FinBERT scorer with lazy loading and TextBlob fallback
  - score_headline_textblob()    — named TextBlob wrapper
  - score_headline_finbert()     — single headline via FinBERT
  - score_headlines_finbert()    — batch via FinBERT
  - Existing build_sentiment_series() + attach_sentiment() upgraded to
    use FinBERT when available (transparent fallback chain)
"""
import os, sys, time, warnings, datetime
import numpy as np
import pandas as pd
import requests
from textblob import TextBlob
warnings.filterwarnings("ignore")

# ── Load .env automatically ───────────────────────────────────────────────────
def _load_dotenv():
    """Search up the directory tree for a .env file and load it."""
    search_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),                          # src/data/
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),         # src/
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # project root
    ]
    for d in search_dirs:
        for fname in [".env", ".env.example"]:
            path = os.path.join(d, fname)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and val and key not in os.environ:
                        os.environ[key] = val
            return path
    return None

_env_file = _load_dotenv()

# Also try python-dotenv if installed
try:
    from dotenv import load_dotenv as _ld
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _ef in [os.path.join(_root, ".env"), os.path.join(_root, ".env.example")]:
        if os.path.exists(_ef):
            _ld(_ef, override=False)
            break
except ImportError:
    pass

# Read key after all loading attempts
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "").strip()

# Reject placeholder values
if NEWS_API_KEY.startswith("your_") or NEWS_API_KEY == "":
    NEWS_API_KEY = ""
    if _env_file:
        print(f"[Sentiment] WARNING: .env found ({_env_file}) but NEWS_API_KEY is "
              f"a placeholder — using synthetic sentiment.")
    else:
        print("[Sentiment] No .env file — using synthetic sentiment.")
else:
    print(f"[Sentiment] SUCCESS: NEWS_API_KEY loaded from {_env_file or 'environment'}")

SENTIMENT_DECAY = 0.85


# ── IMPROVEMENT 7: Full FinBERT with lazy loading + TextBlob fallback ─────────

_FINBERT_MODEL     = None
_FINBERT_TOKENIZER = None
_FINBERT_LOADED    = False   # True once load attempt has been made (success or fail)


def _load_finbert():
    """
    Lazy-load FinBERT (ProsusAI/finbert) on first call.
    Falls back to TextBlob gracefully if:
      - transformers is not installed
      - model download fails (no internet)
      - any other error occurs

    Returns True if FinBERT is available, False if TextBlob fallback is active.
    """
    global _FINBERT_MODEL, _FINBERT_TOKENIZER, _FINBERT_LOADED
    if _FINBERT_LOADED:
        return _FINBERT_MODEL is not None

    try:
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        print("[Sentiment] Loading FinBERT (ProsusAI/finbert) — "
              "first run downloads ~440MB …")
        _FINBERT_TOKENIZER = BertTokenizer.from_pretrained("ProsusAI/finbert")
        _FINBERT_MODEL     = BertForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )
        _FINBERT_MODEL.eval()

        # Move to GPU if available
        from config import DEVICE
        _FINBERT_MODEL = _FINBERT_MODEL.to(DEVICE)
        print(f"[Sentiment] FinBERT loaded on {DEVICE}")
        _FINBERT_LOADED = True
        return True

    except Exception as e:
        print(f"[Sentiment] FinBERT unavailable ({e}) — falling back to TextBlob")
        _FINBERT_LOADED = True
        return False


def score_headline_textblob(headline: str) -> float:
    """Score a single headline with TextBlob. Returns float in [-1, +1]."""
    return TextBlob(str(headline)).sentiment.polarity


def score_headline_finbert(headline: str) -> float:
    """
    Score a single headline with FinBERT.
    Returns float in [-1, +1]: positive_prob - negative_prob.

    FinBERT output classes: [negative=0, neutral=1, positive=2]
    Falls back to TextBlob if FinBERT is unavailable.
    """
    if not _load_finbert() or _FINBERT_MODEL is None:
        return score_headline_textblob(headline)

    try:
        import torch
        from config import DEVICE

        inputs = _FINBERT_TOKENIZER(
            headline, return_tensors="pt",
            truncation=True, max_length=512, padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = _FINBERT_MODEL(**inputs).logits

        probs = torch.softmax(logits, dim=1)[0]
        # positive_prob - negative_prob -> range [-1, +1]
        score = float(probs[2].item() - probs[0].item())
        return score

    except Exception as e:
        return score_headline_textblob(headline)


def score_headlines_finbert(headlines: list) -> float:
    """
    Score a list of headlines with FinBERT and return the mean score.
    Empty list returns 0.0 (neutral).
    Falls back to TextBlob per headline if FinBERT is unavailable.
    """
    if not headlines:
        return 0.0
    scores = [score_headline_finbert(h) for h in headlines]
    return float(sum(scores) / len(scores))


# ── Legacy scoring wrapper (kept for backward compatibility) ──────────────────

def _score_tb(text):
    """TextBlob scorer — legacy alias."""
    return score_headline_textblob(str(text))


def _score_finbert(texts):
    """FinBERT scorer for a list — legacy alias."""
    pipe_ok = _load_finbert()
    if not pipe_ok or _FINBERT_MODEL is None:
        return [_score_tb(t) for t in texts]
    return [score_headline_finbert(t) for t in texts]


def score_headlines(headlines, use_finbert: bool = False) -> float:
    """
    Score a list of headlines.
    use_finbert=True -> attempts FinBERT (falls back to TextBlob automatically).
    use_finbert=False -> TextBlob only.
    Returns mean sentiment score in [-1, +1].
    """
    if not headlines:
        return 0.0
    if use_finbert:
        return score_headlines_finbert(headlines)
    return float(np.mean([_score_tb(h) for h in headlines]))


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def _fetch_newsapi(ticker, from_date, to_date, page_size=20):
    if not NEWS_API_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": f"{ticker} stock", "from": from_date, "to": to_date,
              "language": "en", "sortBy": "publishedAt",
              "pageSize": page_size, "apiKey": NEWS_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 429:
            print(f"[Sentiment] Rate limited (429) — switching to synthetic sentiment.")
            return None   # None signals caller to stop all requests immediately
        r.raise_for_status()
        arts = r.json().get("articles", [])
        return [a.get("title", "") + " " + (a.get("description") or "")
                for a in arts if a.get("title")]
    except Exception as e:
        if "429" in str(e):
            print(f"[Sentiment] Rate limited — switching to synthetic sentiment.")
            return None
        print(f"[Sentiment] NewsAPI error: {e}")
        return []


# ── Synthetic fallback ────────────────────────────────────────────────────────

def _synthetic_sentiment(n_days, seed=42):
    rng = np.random.default_rng(seed)
    s   = np.zeros(n_days)
    for i in range(1, n_days):
        s[i] = s[i-1] + 0.1*(0.0 - s[i-1]) + 0.12*rng.normal()
    return np.clip(s, -1, 1)


# ── Public API ────────────────────────────────────────────────────────────────

def build_sentiment_series(df, ticker, use_finbert: bool = False):
    """
    Build a daily sentiment series for the given DataFrame index.

    Uses FinBERT when use_finbert=True and FinBERT is available;
    falls back transparently to TextBlob otherwise.
    Returns a normalised, clipped sentiment Series aligned to df.index.
    """
    dates = df.index
    if NEWS_API_KEY:
        print(f"[Sentiment] Fetching real news for {ticker} ...")
        raw = {}
        # Free tier only serves last 30 days — one request, not a monthly loop
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=30)).normalize()
        recent = dates[dates >= cutoff]
        if len(recent) > 0:
            hl = _fetch_newsapi(ticker,
                                recent[0].strftime("%Y-%m-%d"),
                                recent[-1].strftime("%Y-%m-%d"),
                                page_size=20)
            if hl is None:  # 429 — stop immediately, use synthetic
                series = pd.Series(_synthetic_sentiment(len(dates)), index=dates)
            elif len(hl) > 0:
                sc = score_headlines(hl, use_finbert=use_finbert)
                for d in recent:
                    raw[d] = sc
                series = pd.Series([raw.get(d, 0.0) for d in dates], index=dates)
                scorer = "FinBERT" if (use_finbert and _FINBERT_LOADED and
                                       _FINBERT_MODEL is not None) else "TextBlob"
                print(f"[Sentiment] {len(hl)} headlines scored={sc:.3f} via {scorer}")
            else:
                series = pd.Series(_synthetic_sentiment(len(dates)), index=dates)
        else:
            series = pd.Series(_synthetic_sentiment(len(dates)), index=dates)
    else:
        series = pd.Series(_synthetic_sentiment(len(dates)), index=dates)

    sm = series.ewm(alpha=1-SENTIMENT_DECAY, adjust=False).mean()
    sm = (sm - sm.mean()) / (sm.std() + 1e-9)
    return sm.clip(-3, 3) / 3


def attach_sentiment(df, ticker, use_finbert: bool = False):
    df = df.copy()
    df["sentiment"] = build_sentiment_series(df, ticker, use_finbert).values
    df["sentiment_ma_3"] = df["sentiment"].rolling(3).mean().bfill()
    print(f"[Sentiment] Attached — range [{df['sentiment'].min():.3f}, {df['sentiment'].max():.3f}]")
    return df


def get_realtime_sentiment(ticker, use_finbert: bool = False):
    today = datetime.date.today()
    hl    = _fetch_newsapi(ticker, (today-datetime.timedelta(7)).strftime("%Y-%m-%d"),
                           today.strftime("%Y-%m-%d"), page_size=15)
    return score_headlines(hl, use_finbert=use_finbert) if hl else 0.0
