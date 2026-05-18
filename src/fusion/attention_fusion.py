"""
Dreaming AI v3 — Attention-Based Feature Fusion
Extends the DEBM architecture with a cross-attention mechanism that
dynamically learns how to weight price/technical features vs sentiment features.

The existing BiLSTM encoder + Energy Function + Prediction Head are NOT modified.
This module sits BEFORE the encoder as a pre-fusion step, and is also used
INSIDE the energy function as an additional conditioning signal.

Architecture integration:
  Input (B, W, F)
      ↓
  FeatureSplitter: split into price_feats (B,W,17) and sent_feats (B,W,5)
      ↓
  PriceEmbedding (Linear -> 128)   SentimentEmbedding (Linear -> 128)
      ↓                                   ↓
  CrossAttention: price queries, sentiment keys/values
  + SelfAttention on price
      ↓
  FusedOutput (B, W, 128) -> fed into BiLSTM encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import (FUSION_HEADS, FUSION_DIM, PRICE_FEAT_DIM,
                    SENTIMENT_FEAT_DIM, N_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# Feature splitter
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSplitter(nn.Module):
    """
    Splits the (B, W, N_FEATURES) input tensor into two groups:
      price_feats:    first PRICE_FEAT_DIM columns (OHLCV + 15 indicators)
      sentiment_feats: last SENTIMENT_FEAT_DIM columns (scalar + intensity + vol + 2 extras)

    If actual n_features < PRICE_FEAT_DIM + SENTIMENT_FEAT_DIM, it adapts.
    """
    def __init__(self, n_features: int = N_FEATURES,
                 price_dim: int = PRICE_FEAT_DIM,
                 sent_dim: int = SENTIMENT_FEAT_DIM):
        super().__init__()
        self.price_end = min(price_dim, n_features)
        self.n_features = n_features

    def forward(self, x: torch.Tensor):
        """
        x: (B, W, F)
        returns: price_x (B,W, price_end), sent_x (B,W, F-price_end)
        """
        price_x = x[:, :, :self.price_end]
        sent_x  = x[:, :, self.price_end:]
        # If no sentiment features exist, return zeros
        if sent_x.shape[-1] == 0:
            sent_x = torch.zeros(*x.shape[:2], 1, device=x.device)
        return price_x, sent_x


# ─────────────────────────────────────────────────────────────────────────────
# Cross-attention fusion
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention: price features attend to sentiment features.

    Query:   price embedding  (B, W, FUSION_DIM)
    Key/Val: sentiment embedding (B, W, FUSION_DIM)
    Output:  attended price features (B, W, FUSION_DIM)

    Additionally applies self-attention on price features so each time step
    can attend to other time steps.

    Final output: concatenation of self-attended price + cross-attended price
    projected back to FUSION_DIM.
    """
    def __init__(self, price_in: int, sent_in: int,
                 fusion_dim: int = FUSION_DIM, n_heads: int = FUSION_HEADS):
        super().__init__()
        # Project price and sentiment features to FUSION_DIM
        self.price_proj = nn.Sequential(
            nn.Linear(price_in, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU()
        )
        self.sent_proj = nn.Sequential(
            nn.Linear(sent_in, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU()
        )
        # Self-attention on price
        self.price_self_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=n_heads, dropout=0.1, batch_first=True
        )
        # Cross-attention: price (Q) ← sentiment (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=n_heads, dropout=0.1, batch_first=True
        )
        # Merge self + cross
        self.merge = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        # Attention weight capture (for visualisation)
        self._last_cross_weights = None
        self._last_self_weights  = None

    def forward(self, price_feats: torch.Tensor,
                sent_feats: torch.Tensor) -> torch.Tensor:
        """
        price_feats: (B, W, price_in)
        sent_feats:  (B, W, sent_in)
        returns:     (B, W, FUSION_DIM)  — fused representation
        """
        P = self.price_proj(price_feats)   # (B, W, FUSION_DIM)
        S = self.sent_proj(sent_feats)     # (B, W, FUSION_DIM)

        # Self-attention over price time-steps
        P_self, w_self = self.price_self_attn(P, P, P)
        self._last_self_weights = w_self.detach()

        # Cross-attention: price queries attend to sentiment
        P_cross, w_cross = self.cross_attn(P, S, S)
        self._last_cross_weights = w_cross.detach()

        # Merge
        fused = self.merge(torch.cat([P_self, P_cross], dim=-1))
        return fused   # (B, W, FUSION_DIM)

    def get_attention_weights(self):
        """Return (self_attention, cross_attention) weight matrices for visualisation."""
        return self._last_self_weights, self._last_cross_weights


# ─────────────────────────────────────────────────────────────────────────────
# Full fusion wrapper — plug-and-play with existing DEBM
# ─────────────────────────────────────────────────────────────────────────────

class AttentionFusionLayer(nn.Module):
    """
    Drop-in pre-processing layer that wraps CrossAttentionFusion.
    Input:  raw (B, W, N_FEATURES) tensor  (same as v2 DEBM input)
    Output: (B, W, FUSION_DIM) fused tensor  (replaces raw input to BiLSTM)

    The BiLSTM encoder's input_size must be updated to FUSION_DIM.
    All other v2 DEBM components remain unchanged.
    """
    def __init__(self, n_features: int = N_FEATURES,
                 price_dim: int = PRICE_FEAT_DIM,
                 fusion_dim: int = FUSION_DIM,
                 n_heads: int = FUSION_HEADS):
        super().__init__()
        self.splitter = FeatureSplitter(n_features, price_dim)
        price_in = self.splitter.price_end
        sent_in  = max(n_features - price_in, 1)
        self.fusion  = CrossAttentionFusion(price_in, sent_in, fusion_dim, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        price_x, sent_x = self.splitter(x)
        return self.fusion(price_x, sent_x)

    def get_attention_weights(self):
        return self.fusion.get_attention_weights()


# ─────────────────────────────────────────────────────────────────────────────
# v3 DEBM with fusion — extends v2 DreamingAI
# ─────────────────────────────────────────────────────────────────────────────

class DreamingAIv3(nn.Module):
    """
    Dreaming AI v3: v2 DreamingAI extended with:
      1. AttentionFusionLayer before the BiLSTM encoder
      2. Multi-horizon prediction heads (1-day, 3-day, 5-day)
      3. Timeframe-conditioned energy (optional timeframe embedding)

    The existing v2 components are instantiated INSIDE this class,
    not modified. The v2 DreamingAI class remains available for fallback.
    """
    def __init__(self, n_features: int,
                 fusion_dim: int = FUSION_DIM,
                 latent_dim: int = LATENT_DIM,
                 forecast_horizons: list = None):
        super().__init__()
        if forecast_horizons is None:
            from config import FORECAST_HORIZONS
            forecast_horizons = FORECAST_HORIZONS

        self.forecast_horizons = forecast_horizons

        # ── v3 addition: attention fusion ─────────────────────────────────────
        self.fusion_layer = AttentionFusionLayer(n_features=n_features,
                                                  fusion_dim=fusion_dim)

        # ── v2 components (unchanged) — now receive fused_dim input instead of n_features
        # Re-import here to avoid circular deps
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from src.models.debm import LSTMEncoder, EnergyFunction, PredictionHead
        from config import LATENT_DIM

        self.encoder   = LSTMEncoder(n_features=fusion_dim, latent_dim=latent_dim)
        self.energy_fn = EnergyFunction(latent_dim=latent_dim)
        self.predictor = PredictionHead(latent_dim=latent_dim)

        # ── v3 addition: multi-horizon heads ─────────────────────────────────
        # One head per forecast horizon (in addition to the 1-step predictor above)
        self.horizon_heads = nn.ModuleDict({
            f"h{h}": nn.Sequential(
                nn.Linear(latent_dim, 32), nn.ReLU(),
                nn.Linear(32, 1)
            )
            for h in forecast_horizons if h > 1
        })

        # ── v3 addition: timeframe embedding ──────────────────────────────────
        from config import TIMEFRAMES
        n_tfs = len(TIMEFRAMES)
        self.timeframe_embed  = nn.Embedding(n_tfs, latent_dim)
        self.tf_names         = list(TIMEFRAMES.keys())

    def _tf_idx(self, timeframe: str) -> int:
        return self.tf_names.index(timeframe) if timeframe in self.tf_names else 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,W,F) -> fused -> h: (B, latent_dim)"""
        fused = self.fusion_layer(x)
        return self.encoder(fused)

    def energy(self, h: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:
        return self.energy_fn(h, sentiment)

    def predict(self, x: torch.Tensor,
                timeframe: str = "1d") -> torch.Tensor:
        """Single-step prediction with timeframe conditioning."""
        h  = self.encode(x)
        tf = self.timeframe_embed(
            torch.tensor(self._tf_idx(timeframe), device=x.device)
        ).unsqueeze(0).expand(h.size(0), -1)
        h_cond = h + tf * 0.1   # soft conditioning
        return self.predictor(h_cond)

    def predict_multihorizon(self, x: torch.Tensor,
                              timeframe: str = "1d") -> dict:
        """
        Returns predictions for all forecast horizons.
        Output: {'h1': tensor(B,1), 'h3': tensor(B,1), 'h5': tensor(B,1)}
        """
        h  = self.encode(x)
        tf = self.timeframe_embed(
            torch.tensor(self._tf_idx(timeframe), device=x.device)
        ).unsqueeze(0).expand(h.size(0), -1)
        h_cond = h + tf * 0.1
        out = {"h1": self.predictor(h_cond)}
        for hname, head in self.horizon_heads.items():
            out[hname] = head(h_cond)
        return out

    def forward(self, x: torch.Tensor, sentiment: torch.Tensor,
                timeframe: str = "1d"):
        """Full training forward: returns (energy, pred_h1, latent_h)"""
        h    = self.encode(x)
        tf   = self.timeframe_embed(
            torch.tensor(self._tf_idx(timeframe), device=x.device)
        ).unsqueeze(0).expand(h.size(0), -1)
        h_cond = h + tf * 0.1
        e    = self.energy_fn(h_cond, sentiment)
        pred = self.predictor(h_cond)
        return e, pred, h_cond


def save_v3(model: DreamingAIv3, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[DEBMv3] Saved -> {path}")


def load_v3(path: str, n_features: int) -> DreamingAIv3:
    m = DreamingAIv3(n_features=n_features)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    print(f"[DEBMv3] Loaded <- {path}")
    return m
