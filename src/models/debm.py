"""
Dreaming AI v4 — Deep Energy-Based Model (DEBM)
================================================
KEY UPGRADES OVER v3:
  1. Residual BiLSTM Encoder with LayerNorm
  2. Spectrally-normalised EnergyFunction (fixes RMSE/MAE regression)
  3. GELU activation in PredictionHead (smoother gradients)
  4. Residual shortcut in PredictionHead
  5. Stock Embedding for multi-stock training
  6. Adaptive Langevin with cosine step schedule + gradient clipping
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from config import (LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT, LATENT_DIM,
                    ENERGY_HIDDEN, LANGEVIN_STEPS, LANGEVIN_STEP_SIZE,
                    LANGEVIN_NOISE, DEVICE, DIR_LOSS_WEIGHT)


class ResidualLSTMBlock(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.norm    = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.proj    = (nn.Linear(input_dim, hidden * 2)
                        if input_dim != hidden * 2 else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.norm(self.dropout(out) + self.proj(x))


class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = LSTM_HIDDEN,
                 n_layers: int = LSTM_LAYERS, dropout: float = LSTM_DROPOUT,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_dim = n_features
        for _ in range(n_layers):
            self.blocks.append(ResidualLSTMBlock(in_dim, hidden, dropout))
            in_dim = hidden * 2
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        return self.proj(out[:, -1, :])


class EnergyFunction(nn.Module):
    """Spectrally-normalised energy MLP — prevents CD instability."""
    def __init__(self, latent_dim: int = LATENT_DIM,
                 hidden_dims: list = None, stock_embed_dim: int = 0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = ENERGY_HIDDEN
        input_dim = latent_dim + 1 + stock_embed_dim
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.utils.spectral_norm(nn.Linear(prev, h)),
                       nn.LayerNorm(h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, sentiment: torch.Tensor,
                stock_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [h, sentiment]
        if stock_emb is not None:
            parts.append(stock_emb)
        return self.net(torch.cat(parts, dim=1))


class PredictionHead(nn.Module):
    """Deeper residual MLP with attention-weighted skip for better directional accuracy."""
    def __init__(self, latent_dim: int = LATENT_DIM, stock_embed_dim: int = 0):
        super().__init__()
        in_dim = latent_dim + stock_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128),    nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64),     nn.GELU(),
            nn.Linear(64, 1),
        )
        self.shortcut = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor,
                stock_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if stock_emb is not None:
            h = torch.cat([h, stock_emb], dim=1)
        return self.net(h) + self.shortcut(h)


class StockEmbedding(nn.Module):
    def __init__(self, num_stocks: int, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(num_stocks, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, stock_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(stock_ids)


class DreamingAI(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = LATENT_DIM,
                 num_stocks: int = 0, stock_embed_dim: int = 16):
        super().__init__()
        self.multi_stock     = (num_stocks > 0)
        self.stock_embed_dim = stock_embed_dim if self.multi_stock else 0
        self.encoder   = LSTMEncoder(n_features, latent_dim=latent_dim)
        self.energy_fn = EnergyFunction(latent_dim=latent_dim,
                                        stock_embed_dim=self.stock_embed_dim)
        self.predictor = PredictionHead(latent_dim=latent_dim,
                                        stock_embed_dim=self.stock_embed_dim)
        self.stock_emb = StockEmbedding(num_stocks, stock_embed_dim) if self.multi_stock else None

    def _se(self, ids, device):
        if self.stock_emb is None or ids is None:
            return None
        return self.stock_emb(ids.to(device))

    def encode(self, x): return self.encoder(x)

    def energy(self, h, sentiment, stock_ids=None):
        return self.energy_fn(h, sentiment, self._se(stock_ids, h.device))

    def predict(self, x, stock_ids=None):
        h = self.encoder(x)
        return self.predictor(h, self._se(stock_ids, h.device))

    def forward(self, x, sentiment, stock_ids=None):
        h    = self.encoder(x)
        se   = self._se(stock_ids, h.device)
        e    = self.energy_fn(h, sentiment, se)
        pred = self.predictor(h, se)
        return e, pred, h


class LangevinSampler:
    """PCD sampler with cosine step schedule and gradient clipping."""
    def __init__(self, latent_dim: int, buffer_size: int, device: str = DEVICE):
        self.latent_dim = latent_dim
        self.device     = device
        self.buffer     = torch.randn(buffer_size, latent_dim, device=device)

    def _step(self, energy_fn, h, sentiment, stock_emb, step_size, noise_std):
        h = h.detach().requires_grad_(True)
        with torch.enable_grad():
            e     = energy_fn(h, sentiment, stock_emb)
            grads = torch.autograd.grad(e.sum(), h)[0]
        grads = torch.clamp(grads.detach(), -1.0, 1.0)
        return (h.detach() - 0.5 * step_size * grads
                + noise_std * torch.randn_like(h)).detach()

    def sample(self, energy_fn, batch_size: int, sentiment_val: float = 0.0,
               stock_emb=None, n_steps: int = LANGEVIN_STEPS,
               step_size: float = LANGEVIN_STEP_SIZE,
               noise_std: float = LANGEVIN_NOISE) -> torch.Tensor:
        idx  = torch.randint(0, len(self.buffer), (batch_size,), device=self.device)
        h    = self.buffer[idx].clone()
        sent = torch.full((batch_size, 1), sentiment_val,
                          dtype=torch.float32, device=self.device)
        for step in range(n_steps):
            frac    = step / max(n_steps - 1, 1)
            current = step_size * (0.5 + 0.5 * np.cos(np.pi * frac))
            current_noise = max(noise_std * 0.01, noise_std * (0.99 ** step))
            h = self._step(energy_fn, h, sent, stock_emb, current, current_noise)
        self.buffer[idx] = h.detach()
        return h.detach()


def save_model(model: DreamingAI, path: str):
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[DEBM] Saved -> {path}")


def load_model(path: str, n_features: int, num_stocks: int = 0) -> DreamingAI:
    model = DreamingAI(n_features=n_features, num_stocks=num_stocks)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    from config import N_FEATURES
    B, W, F = 4, 60, N_FEATURES   # Final Checklist item 2: use config N_FEATURES
    m = DreamingAI(n_features=F)
    e, p, h = m(torch.randn(B, W, F), torch.randn(B, 1))
    print(f"single-stock: energy={e.shape} pred={p.shape} latent={h.shape}")
    print(f"n_features={F}  (from config.N_FEATURES = len(FEATURE_COLS))")
    m2 = DreamingAI(n_features=F, num_stocks=5)
    e2, p2, h2 = m2(torch.randn(B, W, F), torch.randn(B, 1), torch.tensor([0,1,2,3]))
    print(f"multi-stock:  energy={e2.shape} pred={p2.shape} latent={h2.shape}")
