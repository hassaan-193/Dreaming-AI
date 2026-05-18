"""
Dreaming AI v7 — Baseline Models (LSTM + GAN)
v7 Additions (Objective 1):
  - AMP (torch.cuda.amp) with GradScaler in LSTM training loop
  - pin_memory=True + num_workers=4 on CUDA DataLoaders
"""
import logging
import os

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from config import (LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
                    LSTM_EPOCHS, LSTM_LR, LSTM_BATCH,
                    GAN_EPOCHS, GAN_LR_G, GAN_LR_D, GAN_BATCH,
                    GAN_NOISE_DIM, MODELS_DIR, DEVICE, AMP_ENABLED)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Baseline
# ─────────────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Standard unidirectional stacked LSTM baseline.
    Identical architecture to many published stock-prediction baselines.
    """
    def __init__(self, n_features: int, hidden: int = 128,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """Forward pass: (B, W, F) -> (B, 1)."""
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_lstm(X_train, y_train, X_val, y_val,
               n_features: int, ticker: str,
               epochs: int = LSTM_EPOCHS,
               lr: float = LSTM_LR,
               batch: int = LSTM_BATCH,
               device: str = DEVICE):
    """
    Train the LSTM baseline with AMP and CUDA DataLoader optimisations.

    Args:
        X_train, y_train: training arrays
        X_val, y_val:     validation arrays
        n_features:       feature dimension
        ticker:           used for checkpoint filename
        epochs, lr, batch: training hyperparams
        device:           'cuda' or 'cpu'

    Returns:
        (model, train_losses, val_losses)
    """
    model = LSTMModel(n_features).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit  = nn.MSELoss()

    # OBJECTIVE 1: AMP
    use_amp = AMP_ENABLED and (device == "cuda")
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # OBJECTIVE 1: pin_memory + num_workers on CUDA
    # num_workers=0 on Windows (nt) to avoid DataLoader subprocess hang
    pin = (device == "cuda")
    nw  = 4 if (pin and os.name != 'nt') else 0
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch, shuffle=True, drop_last=True,
        pin_memory=pin, num_workers=nw,
        persistent_workers=(nw > 0),
    )

    X_v = torch.tensor(X_val, device=device)
    y_v = torch.tensor(y_val, device=device)

    train_losses, val_losses = [], []
    best_val = float("inf")
    ckpt     = os.path.join(MODELS_DIR, f"{ticker}_lstm_best.pth")

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(xb), yb)
            opt.zero_grad()
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(opt)
            amp_scaler.update()
            ep_loss += loss.item()
        tr_l = ep_loss / len(loader)

        model.eval()
        with torch.no_grad():
            vl = crit(model(X_v), y_v.unsqueeze(1)).item()
        sched.step(vl)

        train_losses.append(tr_l)
        val_losses.append(vl)

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), ckpt)

        if ep % 10 == 0 or ep == 1:
            logger.info(f"  [LSTM] Ep {ep:3d}/{epochs}  train={tr_l:.5f}  val={vl:.5f}")

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    logger.info(f"[LSTM] Best val loss: {best_val:.5f}")
    return model, train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# GAN Baseline
# ─────────────────────────────────────────────────────────────────────────────

class GANGenerator(nn.Module):
    """GAN generator: noise -> synthetic price sequence."""
    def __init__(self, noise_dim: int = GAN_NOISE_DIM, n_features: int = 22,
                 window: int = 60):
        super().__init__()
        out_dim = n_features * window
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512),       nn.LeakyReLU(0.2),
            nn.Linear(512, 256),       nn.LeakyReLU(0.2),
            nn.Linear(256, out_dim),   nn.Tanh()
        )
        self.n_features = n_features
        self.window     = window

    def forward(self, z):
        return self.net(z).view(-1, self.window, self.n_features)


class GANDiscriminator(nn.Module):
    """GAN discriminator: real vs synthetic sequence classifier."""
    def __init__(self, n_features: int = 22, window: int = 60):
        super().__init__()
        inp = n_features * window
        self.net = nn.Sequential(
            nn.Linear(inp, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),   nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x.flatten(1))


class GANPredictor(nn.Module):
    """
    GAN-based predictor: uses GAN infrastructure then a dedicated LSTM head
    for price prediction. Common GAN-for-prediction approach.
    """
    def __init__(self, n_features: int = 22, window: int = 60,
                 noise_dim: int = GAN_NOISE_DIM):
        super().__init__()
        self.generator     = GANGenerator(noise_dim, n_features, window)
        self.discriminator = GANDiscriminator(n_features, window)
        self.pred_lstm = nn.LSTM(n_features, 64, 2, batch_first=True, dropout=0.2)
        self.pred_head = nn.Linear(64, 1)

    def predict(self, x):
        """Predict next step from input sequence."""
        out, _ = self.pred_lstm(x)
        return self.pred_head(out[:, -1, :])


def train_gan(X_train, y_train, X_val, y_val,
              n_features: int, window: int, ticker: str,
              epochs: int = GAN_EPOCHS,
              device: str = DEVICE):
    """
    Train GAN (generator + discriminator) then fine-tune prediction head.

    Args:
        X_train, y_train, X_val, y_val: data splits
        n_features, window: architecture params
        ticker: checkpoint name prefix
        epochs: adversarial training epochs
        device: 'cuda' or 'cpu'

    Returns:
        (gan_model, g_losses, val_losses)
    """
    model     = GANPredictor(n_features=n_features, window=window).to(device)
    opt_g     = torch.optim.Adam(model.generator.parameters(),
                                  lr=GAN_LR_G, betas=(0.5, 0.999))
    opt_d     = torch.optim.Adam(model.discriminator.parameters(),
                                  lr=GAN_LR_D, betas=(0.5, 0.999))
    crit_bce  = nn.BCELoss()
    crit_mse  = nn.MSELoss()

    pin = (device == "cuda")
    nw  = 4 if (pin and os.name != 'nt') else 0
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=GAN_BATCH, shuffle=True, drop_last=True,
        pin_memory=pin, num_workers=nw,
        persistent_workers=(nw > 0),
    )

    g_losses, val_losses = [], []
    best_val = float("inf")
    ckpt     = os.path.join(MODELS_DIR, f"{ticker}_gan_best.pth")

    # Phase 1: Adversarial training
    logger.info("[GAN] Phase 1 — Adversarial training …")
    for ep in range(1, epochs + 1):
        model.train()
        ep_g = 0.0
        for xb, _ in loader:
            xb = xb.to(device)
            bs = xb.size(0)
            real_lbl = torch.ones (bs, 1, device=device)
            fake_lbl = torch.zeros(bs, 1, device=device)

            noise = torch.randn(bs, GAN_NOISE_DIM, device=device)
            fake  = model.generator(noise).detach()
            d_loss = (crit_bce(model.discriminator(xb), real_lbl) +
                      crit_bce(model.discriminator(fake), fake_lbl)) * 0.5
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            noise  = torch.randn(bs, GAN_NOISE_DIM, device=device)
            fake   = model.generator(noise)
            g_loss = crit_bce(model.discriminator(fake), real_lbl)
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            ep_g  += g_loss.item()

        g_losses.append(ep_g / len(loader))
        if ep % 10 == 0 or ep == 1:
            logger.info(f"  [GAN] Ep {ep:3d}/{epochs}  g_loss={g_losses[-1]:.5f}")

    # Phase 2: Train prediction head on real data
    logger.info("[GAN] Phase 2 — Training prediction head …")
    opt_pred = torch.optim.Adam(
        list(model.pred_lstm.parameters()) + list(model.pred_head.parameters()),
        lr=1e-3)
    X_v = torch.tensor(X_val, device=device)
    y_v = torch.tensor(y_val, device=device)

    for ep in range(1, 31):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            opt_pred.zero_grad()
            loss = crit_mse(model.predict(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_pred.step()
            ep_loss += loss.item()

        model.eval()
        with torch.no_grad():
            vl = crit_mse(model.predict(X_v), y_v.unsqueeze(1)).item()
        val_losses.append(vl)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), ckpt)
        if ep % 10 == 0:
            logger.info(f"  [GAN-Pred] Ep {ep:2d}/30  val={vl:.5f}")

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    logger.info(f"[GAN] Best val loss: {best_val:.5f}")
    return model, g_losses, val_losses
