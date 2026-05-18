"""
Dreaming AI v7 — DEBM Training Engine
======================================
v7 Production-Grade Additions (Objective 1, 5):
  1. AMP (torch.cuda.amp) with GradScaler — faster GPU throughput
  2. GPU memory monitor — prints allocated + peak VRAM after each epoch
  3. Label smoothing (ε=LABEL_SMOOTHING) applied in _directional_loss()
  4. Early stopping based on validation directional accuracy (not MSE loss)
     with patience=EARLY_STOP_PATIENCE

Retained from v6:
  - Weighted BCE directional loss (Improvement 4)
  - 60/40 Dreaming Phase budget (Improvement 2)
  - pin_memory=True + num_workers=4 on GPU (Final Checklist)
  - Adaptive CD weight (v4 key fix)
  - LR warm-up + cosine anneal
  - EMA weights for evaluation
  - Separate optimisers for energy and prediction heads
  - Extreme condition augmentation in Dreaming Phase
  - Multi-stock support
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from config import (DEBM_EPOCHS, DEBM_BATCH, DEBM_LR, DEBM_WEIGHT_DECAY,
                    CD_WEIGHT, PRED_WEIGHT, GRAD_CLIP, DIR_LOSS_WEIGHT,
                    LANGEVIN_STEPS, LANGEVIN_STEP_SIZE, LANGEVIN_NOISE,
                    DREAM_CYCLES, DREAM_N_SYNTHETIC, DREAM_EPOCHS,
                    DREAM_EXTREME_RATIO, DREAM_NORMAL_RATIO,
                    MODELS_DIR, LATENT_DIM, DEVICE,
                    PREDICT_LOG_RETURN, AMP_ENABLED,
                    LABEL_SMOOTHING, EARLY_STOP_PATIENCE)
from src.models.debm import DreamingAI, LangevinSampler

logger = logging.getLogger(__name__)


# ─── GPU memory helper ───────────────────────────────────────────────────────

def _log_gpu_memory(epoch: int):
    """Print current and peak VRAM usage at the end of an epoch (CUDA only)."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    peak      = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"[GPU] Ep {epoch:3d} | allocated={allocated:.3f} GB | peak={peak:.3f} GB")


# ─── Loss ────────────────────────────────────────────────────────────────────

def _cd_loss(e_real: torch.Tensor, e_fake: torch.Tensor) -> torch.Tensor:
    """
    Contrastive Divergence with L2 regularisation.
    Regularisation keeps energy magnitudes small, stabilising the Langevin sampler.
    """
    cd  = e_real.mean() - e_fake.mean()
    reg = 0.001 * (e_real ** 2 + e_fake ** 2).mean()
    return cd + reg


def _directional_loss(pred: torch.Tensor, target: torch.Tensor,
                       prev_close: torch.Tensor,
                       label_smoothing: float = LABEL_SMOOTHING) -> torch.Tensor:
    """
    Weighted Binary Cross-Entropy on direction with label smoothing (v7).

    Improvements:
    - Rare large moves (crashes/spikes) get higher weight so the model
      focuses on getting extreme days correct.
    - Label smoothing (ε) reduces overconfidence: targets shifted from
      {0,1} -> {ε/2, 1-ε/2} before BCE computation.
    - Works in both price and log-return modes.

    Args:
        pred:           (B, 1) predicted values (scaled)
        target:         (B, 1) actual values (scaled)
        prev_close:     (B,)   last Close in the input window (scaled index 3)
        label_smoothing: float, ε ∈ [0, 1)
    """
    p = pred.squeeze(1)
    t = target.squeeze(1)

    if PREDICT_LOG_RETURN:
        pred_dir   = (p > 0).float()
        target_dir = (t > 0).float()
        move_size  = t.abs()
    else:
        pred_dir   = (p > prev_close).float()
        target_dir = (t > prev_close).float()
        move_size  = (t - prev_close).abs()

    # Per-sample magnitude weight (larger moves -> higher weight)
    weights = 1.0 + move_size / (move_size.mean() + 1e-8)
    weights = weights.detach()

    # Logit scaled to usable range for typical daily stock returns (~0.01)
    if PREDICT_LOG_RETURN:
        logit = p * 200.0
    else:
        logit = (p - prev_close) * 200.0

    # OBJECTIVE 5: Label smoothing — shift targets away from hard 0/1
    if label_smoothing > 0.0:
        eps = label_smoothing
        target_dir_smooth = target_dir * (1.0 - eps) + 0.5 * eps
    else:
        target_dir_smooth = target_dir

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, target_dir_smooth, weight=weights
    )
    return loss


def _adaptive_cd_weight(pred_loss: float, cd_loss: float,
                         base_weight: float = CD_WEIGHT,
                         max_fraction: float = 0.15) -> float:
    """
    Dynamic CD weight capped at max_fraction of the total weighted loss.
    Prevents CD from overwhelming MSE in early training.
    """
    if abs(cd_loss) < 1e-8:
        return base_weight
    target_cd_contribution = max_fraction * abs(pred_loss)
    dynamic_w = target_cd_contribution / (abs(cd_loss) + 1e-8)
    return float(np.clip(dynamic_w, 1e-4, base_weight))


def _compute_val_dir_acc(model: nn.Module,
                          Xv: torch.Tensor,
                          yv: torch.Tensor,
                          sv: torch.Tensor,
                          sid_v) -> float:
    """
    Compute directional accuracy on the validation set (for early stopping).
    Returns a float in [0, 100].
    """
    model.eval()
    vp_list = []
    with torch.no_grad():
        for i in range(0, len(Xv), DEBM_BATCH):
            sid_batch = sid_v[i:i+DEBM_BATCH] if sid_v is not None else None
            _, p, _ = model(Xv[i:i+DEBM_BATCH], sv[i:i+DEBM_BATCH], sid_batch)
            vp_list.append(p)
    vp = torch.cat(vp_list, dim=0)
    vp_flat = vp.squeeze(1).cpu().numpy()
    yv_flat = yv.squeeze(1).cpu().numpy()
    if PREDICT_LOG_RETURN:
        da = float(np.mean(np.sign(vp_flat) == np.sign(yv_flat)) * 100)
    else:
        # Directional change accuracy on sequential targets
        if len(vp_flat) > 1:
            da = float(np.mean(
                np.sign(np.diff(vp_flat)) == np.sign(np.diff(yv_flat))
            ) * 100)
        else:
            da = 50.0
    return da


# ─── EMA helper ──────────────────────────────────────────────────────────────

class EMAModel:
    """Exponential Moving Average of model weights for stable evaluation."""
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model = deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights in-place."""
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


# ─── Main training function ───────────────────────────────────────────────────

def train_debm(X_train, y_train, X_val, y_val,
               n_features, ticker,
               sentiment_train=None, sentiment_val=None,
               stock_ids_train=None, stock_ids_val=None,
               num_stocks: int = 0,
               epochs=DEBM_EPOCHS, device=DEVICE):
    """
    Train DreamingAI DEBM with full v7 production features:

    - AMP (automatic mixed precision) with GradScaler on CUDA
    - GPU memory monitoring per epoch
    - Label smoothing in directional loss (ε=LABEL_SMOOTHING)
    - Early stopping based on validation directional accuracy
    - Adaptive CD weight (v4 key fix)
    - LR warm-up + cosine anneal
    - EMA weights for best-model checkpointing
    - Optional multi-stock embeddings
    - CUDA DataLoader optimisations (pin_memory + num_workers)

    Args:
        X_train:          (N_train, window, n_features)
        y_train:          (N_train,)
        X_val:            (N_val, window, n_features)
        y_val:            (N_val,)
        n_features:       int
        ticker:           str — used for checkpoint filename
        sentiment_train:  (N_train,) or None
        sentiment_val:    (N_val,)   or None
        stock_ids_train:  (N_train,) int64 or None
        stock_ids_val:    (N_val,)   int64 or None
        num_stocks:       int — 0 for single-stock
        epochs:           int
        device:           str — 'cuda' or 'cpu'

    Returns:
        (model, history_dict)
    """
    if sentiment_train is None:
        sentiment_train = np.zeros(len(X_train), dtype=np.float32)
    if sentiment_val is None:
        sentiment_val = np.zeros(len(X_val), dtype=np.float32)

    model   = DreamingAI(n_features=n_features, num_stocks=num_stocks).to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"[GPU] Using {torch.cuda.device_count()} GPUs via DataParallel")

    base_model = model.module if hasattr(model, 'module') else model
    ema     = EMAModel(model, decay=0.995)
    sampler = LangevinSampler(
        latent_dim=LATENT_DIM,
        buffer_size=max(len(X_train), 512),
        device=device
    )

    # Separate LR for energy vs predictor — prevents CD over-regularisation
    energy_params    = list(base_model.energy_fn.parameters())
    predictor_params = list(base_model.predictor.parameters())
    encoder_params   = list(base_model.encoder.parameters())
    other_params     = ([list(base_model.stock_emb.parameters())]
                        if base_model.stock_emb else [[]])
    other_params     = [p for sublist in other_params for p in sublist]

    opt = torch.optim.AdamW([
        {"params": encoder_params,   "lr": DEBM_LR,       "weight_decay": DEBM_WEIGHT_DECAY},
        {"params": predictor_params, "lr": DEBM_LR,       "weight_decay": DEBM_WEIGHT_DECAY},
        {"params": energy_params,    "lr": DEBM_LR * 0.3, "weight_decay": DEBM_WEIGHT_DECAY},
        {"params": other_params,     "lr": DEBM_LR,       "weight_decay": DEBM_WEIGHT_DECAY},
    ])

    # LR schedule: warm-up 10 epochs + cosine anneal (OBJECTIVE 5)
    warmup_epochs = min(10, epochs // 5)
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        frac = (ep - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * frac))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # OBJECTIVE 1: AMP GradScaler — only active on CUDA
    use_amp = AMP_ENABLED and (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        logger.info(f"[GPU] AMP enabled for {ticker} training")

    mse = nn.MSELoss()

    # Build dataset
    tensors = [
        torch.tensor(X_train,         dtype=torch.float32),
        torch.tensor(y_train,         dtype=torch.float32),
        torch.tensor(sentiment_train,  dtype=torch.float32).unsqueeze(1),
    ]
    if stock_ids_train is not None:
        tensors.append(torch.tensor(stock_ids_train, dtype=torch.long))

    # OBJECTIVE 1: pin_memory + num_workers on CUDA
    # num_workers=0 on Windows (nt) to avoid DataLoader subprocess hang
    pin = (device == "cuda")
    nw  = 4 if (pin and os.name != 'nt') else 0
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=DEBM_BATCH, shuffle=True, drop_last=True,
        pin_memory=pin, num_workers=nw,
        persistent_workers=(nw > 0),
    )

    # Validation tensors
    Xv    = torch.tensor(X_val,        dtype=torch.float32, device=device)
    yv    = torch.tensor(y_val,        dtype=torch.float32, device=device).unsqueeze(1)
    sv    = torch.tensor(sentiment_val, dtype=torch.float32, device=device).unsqueeze(1)
    sid_v = (torch.tensor(stock_ids_val, dtype=torch.long, device=device)
             if stock_ids_val is not None else None)

    hist = {"train": [], "val": [], "cd": [], "pred": [], "ema_val": [], "val_dir_acc": []}
    best_val_da  = -1.0       # track best validation directional accuracy
    best_val_mse = float("inf")
    ckpt = os.path.join(MODELS_DIR, f"{ticker}_debm_best.pth")

    # OBJECTIVE 5: Early stopping state
    es_counter  = 0
    es_patience = EARLY_STOP_PATIENCE

    logger.info(
        f"[DEBM] {epochs} epochs | device={device} | AMP={use_amp} | "
        f"batch={DEBM_BATCH} | features={n_features} | "
        f"target={'log_return' if PREDICT_LOG_RETURN else 'Close'} | "
        f"multi_stock={'yes' if num_stocks > 0 else 'no'}"
    )

    for ep in range(1, epochs + 1):
        model.train()
        tot = cd_s = pred_s = dir_s = 0.0

        for batch in loader:
            if len(batch) == 4:
                xb, yb, sb, sid_b = batch
                sid_b = sid_b.to(device)
            else:
                xb, yb, sb = batch
                sid_b = None
            xb, yb, sb = xb.to(device), yb.to(device).unsqueeze(1), sb.to(device)

            # OBJECTIVE 1: AMP forward pass
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Forward (real data)
                e_real, pred, _ = model(xb, sb, sid_b)

                # Langevin negative samples
                avg_s  = float(sb.mean().item())
                h_fake = sampler.sample(
                    base_model.energy_fn, batch_size=xb.size(0),
                    sentiment_val=avg_s,
                    stock_emb=base_model._se(sid_b, device) if num_stocks > 0 else None,
                    n_steps=LANGEVIN_STEPS,
                    step_size=LANGEVIN_STEP_SIZE,
                    noise_std=LANGEVIN_NOISE,
                )
                sf     = torch.full((h_fake.size(0), 1), avg_s,
                                    dtype=torch.float32, device=device)
                e_fake = base_model.energy_fn(
                    h_fake, sf,
                    base_model._se(sid_b, device) if num_stocks > 0 else None
                )

                pred_l = mse(pred, yb)
                cd_l   = _cd_loss(e_real, e_fake)

                # OBJECTIVE 5: Weighted BCE directional loss with label smoothing
                prev_close = xb[:, -1, 3]   # index 3 = Close in FEATURE_COLS
                dir_l  = _directional_loss(pred, yb, prev_close,
                                           label_smoothing=LABEL_SMOOTHING)

                # Dynamic adaptive CD weight (v4 key fix)
                w_cd   = _adaptive_cd_weight(pred_l.item(), cd_l.item())
                loss   = PRED_WEIGHT * pred_l + w_cd * cd_l + DIR_LOSS_WEIGHT * dir_l

            opt.zero_grad()
            scaler.scale(loss).backward()
            # Unscale before clip_grad_norm_ so we clip actual gradients
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            tot    += loss.item()
            cd_s   += cd_l.item()
            pred_s += pred_l.item()
            dir_s  += dir_l.item()

        sched.step()
        N = len(loader)
        hist["train"].append(tot / N)
        hist["cd"].append(cd_s / N)
        hist["pred"].append(pred_s / N)

        # Validation MSE with live model
        model.eval()
        vp_list = []
        with torch.no_grad():
            for i in range(0, len(Xv), DEBM_BATCH):
                sid_batch = sid_v[i:i+DEBM_BATCH] if sid_v is not None else None
                _, p, _ = model(Xv[i:i+DEBM_BATCH], sv[i:i+DEBM_BATCH], sid_batch)
                vp_list.append(p)
            vp = torch.cat(vp_list, dim=0)
            vl = mse(vp, yv).item()
        hist["val"].append(vl)

        # Validation MSE with EMA model (usually better)
        ema.model.eval()
        ema_vp_list = []
        with torch.no_grad():
            for i in range(0, len(Xv), DEBM_BATCH):
                sid_batch = sid_v[i:i+DEBM_BATCH] if sid_v is not None else None
                _, p, _ = ema.model(Xv[i:i+DEBM_BATCH], sv[i:i+DEBM_BATCH], sid_batch)
                ema_vp_list.append(p)
            ema_vp = torch.cat(ema_vp_list, dim=0)
            ema_vl = mse(ema_vp, yv).item()
        hist["ema_val"].append(ema_vl)

        # OBJECTIVE 5: Directional accuracy for early stopping
        val_da = _compute_val_dir_acc(ema.model, Xv, yv, sv, sid_v)
        hist["val_dir_acc"].append(val_da)

        # Save on EMA val MSE improvement (checkpoint)
        if ema_vl < best_val_mse:
            best_val_mse = ema_vl
            best_state = ema.model.module.state_dict() if hasattr(ema.model, 'module') else ema.model.state_dict()
            torch.save(best_state, ckpt)

        # OBJECTIVE 5: Early stopping on directional accuracy
        if val_da > best_val_da:
            best_val_da = val_da
            es_counter  = 0
        else:
            es_counter += 1

        if ep % 10 == 0 or ep == 1:
            logger.info(
                f"  Ep {ep:3d}/{epochs}  "
                f"train={hist['train'][-1]:.5f}  "
                f"val={vl:.5f}  ema_val={ema_vl:.5f}  "
                f"val_da={val_da:.1f}%  "
                f"cd={hist['cd'][-1]:.5f}  "
                f"dir={dir_s/N:.5f}"
            )

        # OBJECTIVE 1: GPU memory monitor
        _log_gpu_memory(ep)

        if es_counter >= es_patience:
            logger.info(
                f"[DEBM] Early stopping at epoch {ep} "
                f"(no val_dir_acc improvement for {es_patience} epochs). "
                f"Best val_dir_acc={best_val_da:.1f}%"
            )
            break

    # Load best EMA weights
    if hasattr(model, 'module'):
        model.module.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    else:
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    logger.info(f"[DEBM] Best EMA val_mse={best_val_mse:.5f}  best_val_dir_acc={best_val_da:.1f}%")
    return model, hist


# ─── Dreaming Phase ──────────────────────────────────────────────────────────

def dreaming_phase(model, X_train, y_train, X_val, y_val,
                   n_features, ticker,
                   sentiment_train=None, sentiment_val=None,
                   stock_ids_train=None, stock_ids_val=None,
                   condition_labels=None,
                   num_stocks: int = 0,
                   device=DEVICE):
    """
    v7 Dreaming Phase:
    - Standard dreaming (nearest-neighbour + noise augmentation)
    - OBJECTIVE 5 / IMPROVEMENT 2: 60/40 extreme-event bias
        n_standard = DREAM_N_SYNTHETIC * DREAM_NORMAL_RATIO  (200 normal)
        n_extreme  = DREAM_N_SYNTHETIC * DREAM_EXTREME_RATIO (300 extreme)
        -> 150 crash + 150 spike sequences per cycle
    - Extreme condition dreaming: perturb real crash/spike sequences
      directly (more realistic than pure Langevin for rare events)
    - AMP and GPU memory monitoring inherited from train_debm()
    """
    logger.info(f"[Dream] Starting {DREAM_CYCLES} Dreaming Phase cycles …")

    # OBJECTIVE 5 / IMPROVEMENT 2: 60/40 extreme bias in synthetic budget
    n_standard      = int(DREAM_N_SYNTHETIC * DREAM_NORMAL_RATIO)    # 200
    n_extreme_total = int(DREAM_N_SYNTHETIC * DREAM_EXTREME_RATIO)   # 300
    n_per_condition = n_extreme_total // 2                           # 150 crash + 150 spike

    rare_total = n_per_condition * 2
    logger.info(
        f"[Dream] Budget: {n_standard} normal + {rare_total} rare "
        f"({rare_total/DREAM_N_SYNTHETIC*100:.1f}% rare)"
    )
    scenario_counts = {
        "normal":            n_standard,
        "crash":             n_per_condition,
        "spike":             n_per_condition,
        "sideways_volatile": 0,
    }
    logger.info(f"[Dream] Scenario breakdown: {scenario_counts}")

    for cycle in range(1, DREAM_CYCLES + 1):
        logger.info(f"[Dream] ════ Cycle {cycle}/{DREAM_CYCLES} ════")

        sampler = LangevinSampler(
            latent_dim=LATENT_DIM,
            buffer_size=max(DREAM_N_SYNTHETIC * 2, 512),
            device=device
        )

        stock_emb_syn = None
        base_model = model.module if hasattr(model, 'module') else model
        if num_stocks > 0 and stock_ids_train is not None:
            chosen_sids = torch.tensor(
                np.random.choice(stock_ids_train, n_standard),
                dtype=torch.long, device=device
            )
            stock_emb_syn = base_model._se(chosen_sids, device)

        # Standard synthetic generation via Langevin (normal market scenarios)
        h_syn = sampler.sample(
            base_model.energy_fn,
            batch_size=n_standard,
            stock_emb=stock_emb_syn,
            n_steps=LANGEVIN_STEPS * 2,
            step_size=LANGEVIN_STEP_SIZE * 0.5,
        )

        # Map latents -> nearest real sequences
        model.eval()
        with torch.no_grad():
            Xt    = torch.tensor(X_train, dtype=torch.float32, device=device)
            h_real_list = []
            for i in range(0, len(Xt), DEBM_BATCH):
                h_real_list.append(base_model.encode(Xt[i:i+DEBM_BATCH]))
            h_real = torch.cat(h_real_list, dim=0)
            hn_s   = nn.functional.normalize(h_syn,  dim=1)
            hn_r   = nn.functional.normalize(h_real, dim=1)
            nn_idx = (hn_s @ hn_r.T).argmax(dim=1).cpu().numpy()

        X_syn = X_train[nn_idx].copy()
        y_syn = y_train[nn_idx].copy()
        X_syn += np.random.randn(*X_syn.shape).astype(np.float32) * X_train.std() * 0.05
        y_syn += np.random.randn(*y_syn.shape).astype(np.float32) * y_train.std() * 0.03

        # ── OBJECTIVE 5 / IMPROVEMENT 2: Extreme condition augmentation ───────
        X_extreme, y_extreme, sid_extreme = [], [], []
        if condition_labels is not None:
            for label_val in [1, 2]:   # 1=crash, 2=spike
                extreme_idx = np.where(condition_labels[:len(X_train)] == label_val)[0]
                if len(extreme_idx) > 0:
                    n_ex   = min(n_per_condition, len(extreme_idx))
                    chosen = np.random.choice(extreme_idx, n_ex, replace=True)
                    Xe = X_train[chosen].copy()
                    ye = y_train[chosen].copy()
                    if stock_ids_train is not None:
                        sid_extreme.append(stock_ids_train[chosen].copy())
                    # Amplify extreme moves by 1.2–1.8× to cover tail risk
                    scale = np.random.uniform(1.2, 1.8, size=(n_ex, 1, 1))
                    Xe   *= scale
                    ye   *= np.random.uniform(1.1, 1.6, size=n_ex)
                    X_extreme.append(Xe)
                    y_extreme.append(ye)
        else:
            # No condition labels — fall back to random amplification
            n_ex   = n_per_condition * 2
            chosen = np.random.choice(len(X_train),
                                      min(n_ex, len(X_train)), replace=True)
            Xe = X_train[chosen].copy()
            ye = y_train[chosen].copy()
            if stock_ids_train is not None:
                sid_extreme.append(stock_ids_train[chosen].copy())
            sc = np.random.uniform(1.2, 1.8, size=(len(chosen), 1, 1))
            Xe *= sc
            ye *= np.random.uniform(1.1, 1.6, size=len(chosen))
            X_extreme.append(Xe)
            y_extreme.append(ye)

        if X_extreme:
            X_extreme = np.concatenate(X_extreme)
            y_extreme = np.concatenate(y_extreme)
            if sid_extreme:
                sid_extreme = np.concatenate(sid_extreme)
            X_syn = np.concatenate([X_syn, X_extreme])
            y_syn = np.concatenate([y_syn, y_extreme])

        X_hyb = np.concatenate([X_train, X_syn])
        y_hyb = np.concatenate([y_train, y_syn])

        # Handle sentiment
        s_hyb = None
        if sentiment_train is not None:
            s_syn = sentiment_train[nn_idx].copy()
            has_extreme = isinstance(X_extreme, np.ndarray) and len(X_extreme) > 0
            if has_extreme:
                n_ex = len(X_extreme)
                s_hyb = np.concatenate([sentiment_train, s_syn,
                                         np.zeros(n_ex, dtype=np.float32)])
            else:
                s_hyb = np.concatenate([sentiment_train, s_syn])

        # Handle stock IDs
        sid_hyb = None
        if stock_ids_train is not None:
            sid_syn = stock_ids_train[nn_idx].copy()
            has_extreme = isinstance(X_extreme, np.ndarray) and len(X_extreme) > 0
            if has_extreme:
                sid_hyb = np.concatenate([stock_ids_train, sid_syn, sid_extreme])
            else:
                sid_hyb = np.concatenate([stock_ids_train, sid_syn])

        logger.info(
            f"[Dream] Hybrid: {len(X_train)} real + {len(X_syn)} synthetic"
            f" = {len(X_hyb)} total"
        )

        model, _ = train_debm(
            X_hyb, y_hyb, X_val, y_val,
            n_features=n_features,
            ticker=f"{ticker}_dream{cycle}",
            sentiment_train=s_hyb,
            sentiment_val=sentiment_val,
            stock_ids_train=sid_hyb,
            stock_ids_val=stock_ids_val,
            num_stocks=num_stocks,
            epochs=DREAM_EPOCHS,
            device=device,
        )

    logger.info("[Dream] Dreaming Phase complete.")
    return model
