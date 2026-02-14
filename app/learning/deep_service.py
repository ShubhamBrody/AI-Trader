from __future__ import annotations

# pyright: reportMissingImports=false

import base64
import hashlib
import io
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.ai.feature_engineering import compute_features
from app.candles.persistence_sql import get_candles
from app.core.db import db_conn
from app.learning.registry import get_registry, slot_key as registry_slot_key


def _default_deep_checkpoint_dir() -> Path:
    # Prefer colocating checkpoints next to the DB so Kaggle/local runs keep artifacts together.
    try:
        from app.core.settings import settings

        db_path = Path(str(getattr(settings, "DATABASE_PATH", "./data/app.db")))
        base = (db_path.parent if db_path.parent else Path("."))
        return (base / "checkpoints" / "deep").resolve()
    except Exception:
        return Path("data/checkpoints/deep").resolve()


def _deep_checkpoint_path(*, model_key: str, checkpoint_dir: str | os.PathLike[str] | None) -> Path:
    base = Path(checkpoint_dir) if checkpoint_dir else _default_deep_checkpoint_dir()
    base.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(model_key.encode("utf-8")).hexdigest()[:20]
    return base / f"{h}.pt"


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    import torch

    torch.save(payload, tmp)
    tmp.replace(path)


def _try_load_torch_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import torch

        return dict(torch.load(path, map_location="cpu"))
    except Exception:
        return None


def _deep_model_key(instrument_key: str, interval: str, horizon_steps: int, model_family: str | None, cap_tier: str | None) -> str:
    fam = (model_family or "generic").lower()
    cap = (cap_tier or "unknown").lower()
    return f"deep::{fam}::{cap}::{instrument_key}::{interval}::h{int(horizon_steps)}"


@dataclass(frozen=True)
class DeepModel:
    model_key: str
    instrument_key: str
    interval: str
    horizon_steps: int
    trained_ts: int
    n_samples: int
    seq_len: int
    feature_dim: int
    norm_mean: list[float]
    norm_std: list[float]
    state_dict_b64: str
    metrics: dict[str, float]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _save_deep_model(m: DeepModel) -> None:
    metrics_json = json.dumps(m.metrics, ensure_ascii=False, separators=(",", ":"))
    model_json = json.dumps(
        {
            "kind": "torch_transformer_regression_v1",
            "seq_len": int(m.seq_len),
            "feature_dim": int(m.feature_dim),
            "norm": {"mean": m.norm_mean, "std": m.norm_std},
            "state_dict_b64": m.state_dict_b64,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO trained_models (model_key, instrument_key, interval, horizon_steps, trained_ts, n_samples, metrics_json, model_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_key) DO UPDATE SET
                trained_ts=excluded.trained_ts,
                n_samples=excluded.n_samples,
                metrics_json=excluded.metrics_json,
                model_json=excluded.model_json
            """,
            (
                m.model_key,
                m.instrument_key,
                m.interval,
                int(m.horizon_steps),
                int(m.trained_ts),
                int(m.n_samples),
                metrics_json,
                model_json,
            ),
        )


def load_deep_model(
    instrument_key: str,
    interval: str,
    horizon_steps: int = 1,
    *,
    model_family: str | None = None,
    cap_tier: str | None = None,
) -> DeepModel | None:
    # Prefer promoted production model if present.
    try:
        slot = registry_slot_key(
            instrument_key=instrument_key,
            interval=interval,
            horizon_steps=int(horizon_steps),
            model_family=model_family,
            cap_tier=cap_tier,
            kind="deep",
        )
        reg = get_registry(slot)
        key = str(reg["model_key"]) if reg and reg.get("model_key") else _deep_model_key(instrument_key, interval, horizon_steps, model_family, cap_tier)
    except Exception:
        key = _deep_model_key(instrument_key, interval, horizon_steps, model_family, cap_tier)
    with db_conn() as conn:
        row = conn.execute(
            "SELECT model_key, instrument_key, interval, horizon_steps, trained_ts, n_samples, metrics_json, model_json FROM trained_models WHERE model_key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None

        metrics = json.loads(row["metrics_json"] or "{}")
        model = json.loads(row["model_json"] or "{}")
        if (model.get("kind") or "").startswith("torch_") is False:
            return None

        norm = model.get("norm") or {}
        mean = norm.get("mean") or []
        std = norm.get("std") or []
        return DeepModel(
            model_key=str(row["model_key"]),
            instrument_key=str(row["instrument_key"]),
            interval=str(row["interval"]),
            horizon_steps=int(row["horizon_steps"]),
            trained_ts=int(row["trained_ts"]),
            n_samples=int(row["n_samples"]),
            seq_len=int(model.get("seq_len") or 120),
            feature_dim=int(model.get("feature_dim") or 3),
            norm_mean=[float(x) for x in mean],
            norm_std=[float(x) for x in std],
            state_dict_b64=str(model.get("state_dict_b64") or ""),
            metrics={k: float(v) for k, v in (metrics or {}).items()},
        )


def _build_features(candles: list[Any]) -> np.ndarray:
    # Features per timestep: log return, hl_range, log volume
    closes = np.asarray([float(c.close) for c in candles], dtype=float)
    highs = np.asarray([float(c.high) for c in candles], dtype=float)
    lows = np.asarray([float(c.low) for c in candles], dtype=float)
    vols = np.asarray([float(c.volume) for c in candles], dtype=float)

    closes = np.where(closes <= 0, 1e-9, closes)
    rets = np.diff(np.log(closes), prepend=np.log(closes[0]))
    hl = (highs - lows) / closes
    lv = np.log(np.where(vols <= 0, 1.0, vols))

    X = np.stack([rets, hl, lv], axis=1)
    return X.astype(np.float32)


def train_deep_model(
    instrument_key: str,
    interval: str = "1d",
    *,
    lookback_days: int = 365,
    horizon_steps: int = 1,
    model_family: str | None = None,
    cap_tier: str | None = None,
    seq_len: int = 120,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    min_samples: int = 500,
    data_fraction: float = 1.0,
    require_cuda: bool = False,
    progress_cb: Any | None = None,
    patience: int = 3,
    resume: bool = False,
    checkpoint_dir: str | None = None,
    epochs_per_run: int | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    if not _torch_available():
        return {"ok": False, "reason": "torch not installed"}

    import torch

    if bool(require_cuda) and not torch.cuda.is_available():
        return {"ok": False, "reason": "cuda not available"}

    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(lookback_days))
    candles = get_candles(instrument_key, interval, int(start.timestamp()), int(now.timestamp()))
    if len(candles) < max(seq_len + horizon_steps + 5, min_samples):
        return {
            "ok": False,
            "reason": "not enough data in DB",
            "needed": int(max(seq_len + horizon_steps + 5, min_samples)),
            "have": int(len(candles)),
        }

    X_all = _build_features(candles)
    closes = [float(c.close) for c in candles]

    # Target: future return over horizon_steps
    y_all = []
    for t in range(len(candles)):
        j = t + int(horizon_steps)
        if j >= len(candles):
            y_all.append(0.0)
            continue
        now_close = float(closes[t])
        fut_close = float(closes[j])
        if now_close <= 0:
            y_all.append(0.0)
        else:
            y_all.append((fut_close - now_close) / now_close)
    y_all = np.asarray(y_all, dtype=np.float32)

    # Build windows
    X_seq: list[np.ndarray] = []
    y_seq: list[float] = []
    for t in range(seq_len, len(candles) - int(horizon_steps) - 1):
        X_seq.append(X_all[t - seq_len : t])
        y_seq.append(float(y_all[t]))

    if len(y_seq) < min_samples:
        return {"ok": False, "reason": "not enough usable samples", "needed": int(min_samples), "have": int(len(y_seq))}

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32)

    frac = float(data_fraction)
    if 0.0 < frac < 1.0 and y_seq.size > 1:
        target_n = int(round(float(y_seq.size) * frac))
        target_n = max(2, min(int(y_seq.size), int(target_n)))
        idx = np.linspace(0, int(y_seq.size) - 1, num=int(target_n), dtype=int)
        idx = np.unique(idx)
        if idx.size >= 2:
            X_seq = X_seq[idx]
            y_seq = y_seq[idx]

    split = int(0.8 * len(y_seq))
    X_tr, y_tr = X_seq[:split], y_seq[:split]
    X_te, y_te = X_seq[split:], y_seq[split:]

    # Normalize per feature across train set
    mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    std = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0)
    std = np.where(std <= 1e-6, 1.0, std)

    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    class SeqDs(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X
            self.y = y

        def __len__(self) -> int:
            return int(self.X.shape[0])

        def __getitem__(self, idx: int):
            return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)

    train_loader = DataLoader(SeqDs(X_tr, y_tr), batch_size=int(batch_size), shuffle=True, drop_last=True)
    val_loader = DataLoader(SeqDs(X_te, y_te), batch_size=int(batch_size), shuffle=False)

    class PosEnc(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TSModel(nn.Module):
        def __init__(self, feature_dim: int, d_model: int = 64, nhead: int = 4, layers: int = 3, dropout: float = 0.1):
            super().__init__()
            self.inp = nn.Linear(feature_dim, d_model)
            self.pos = PosEnc(d_model, max_len=max(512, int(seq_len) + 4))
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, x):
            x = self.inp(x)
            x = self.pos(x)
            x = self.enc(x)
            x_last = x[:, -1, :]
            out = self.head(x_last).squeeze(-1)
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TSModel(feature_dim=int(X_tr.shape[-1])).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.HuberLoss(delta=1.0)

    model_key = _deep_model_key(instrument_key, interval, horizon_steps, model_family, cap_tier)
    ckpt_path = _deep_checkpoint_path(model_key=model_key, checkpoint_dir=checkpoint_dir)
    ckpt = _try_load_torch_checkpoint(ckpt_path) if bool(resume) else None

    # Best-effort resume:
    # 1) Prefer disk checkpoint (includes optimizer + progress)
    # 2) Fall back to DB model weights (no optimizer)
    start_epoch = 0
    best_val = 1e9
    best_state = None
    best_epoch = 0
    no_improve = 0
    if isinstance(ckpt, dict) and ckpt.get("kind") == "deep_train_ckpt_v1" and ckpt.get("model_key") == model_key:
        try:
            msd = ckpt.get("model_state")
            if isinstance(msd, dict):
                model.load_state_dict(msd)
            osd = ckpt.get("opt_state")
            if isinstance(osd, dict):
                opt.load_state_dict(osd)
            start_epoch = int(ckpt.get("epoch") or 0)
            best_val = float(ckpt.get("best_val") or best_val)
            best_state = ckpt.get("best_state") if isinstance(ckpt.get("best_state"), dict) else None
            best_epoch = int(ckpt.get("best_epoch") or 0)
            no_improve = int(ckpt.get("no_improve") or 0)
            # Restore RNG for better continuity (optional)
            try:
                rs = ckpt.get("rng") or {}
                if isinstance(rs, dict):
                    if isinstance(rs.get("python"), tuple):
                        random.setstate(rs["python"])
                    if isinstance(rs.get("numpy"), tuple):
                        np.random.set_state(rs["numpy"])
                    if isinstance(rs.get("torch"), (bytes, bytearray)):
                        torch.set_rng_state(rs["torch"])
                    if torch.cuda.is_available() and isinstance(rs.get("torch_cuda"), list):
                        try:
                            torch.cuda.set_rng_state_all(rs["torch_cuda"])
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception:
            # If checkpoint is corrupt/mismatched, ignore and continue from scratch.
            start_epoch = 0
            best_val = 1e9
            best_state = None
            best_epoch = 0
            no_improve = 0
    elif bool(resume):
        try:
            prev = load_deep_model(
                instrument_key=instrument_key,
                interval=interval,
                horizon_steps=int(horizon_steps),
                model_family=model_family,
                cap_tier=cap_tier,
            )
            if prev and prev.state_dict_b64:
                raw = base64.b64decode(prev.state_dict_b64.encode("ascii"))
                state = torch.load(io.BytesIO(raw), map_location="cpu")
                if isinstance(state, dict):
                    model.load_state_dict(state)
        except Exception:
            pass

    def eval_loss() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                losses.append(float(loss_fn(pred, yb).item()))
        return float(np.mean(losses)) if losses else 0.0

    epoch_times: list[float] = []

    target_epochs = int(epochs)
    if start_epoch >= target_epochs:
        # Nothing to do; still refresh DB entry with existing weights.
        pass

    run_start = datetime.now(timezone.utc)
    max_ep = target_epochs
    if epochs_per_run is not None and int(epochs_per_run) > 0:
        max_ep = min(target_epochs, start_epoch + int(epochs_per_run))

    for ep in range(int(start_epoch), int(max_ep)):
        ep_t0 = datetime.now(timezone.utc)
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                train_losses.append(float(loss.detach().cpu().item()))
            except Exception:
                pass

        v = eval_loss()
        tr = float(np.mean(train_losses)) if train_losses else 0.0
        if v < best_val:
            best_val = v
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            no_improve = 0
        else:
            no_improve += 1

        ep_dt = (datetime.now(timezone.utc) - ep_t0).total_seconds()
        epoch_times.append(float(ep_dt))
        avg_epoch = float(np.mean(epoch_times)) if epoch_times else 0.0
        remaining = max(0, int(target_epochs) - (ep + 1))
        eta_seconds = float(avg_epoch * remaining)

        if progress_cb is not None:
            try:
                progress_cb(
                    float((ep + 1) / max(1, int(target_epochs))),
                    f"epoch {ep + 1}/{int(target_epochs)}",
                    {
                        "epoch": int(ep + 1),
                        "epochs": int(target_epochs),
                        "train_huber": float(tr),
                        "val_huber": float(v),
                        "best_val_huber": float(best_val),
                        "best_epoch": int(best_epoch),
                        "patience": int(patience),
                        "no_improve": int(no_improve),
                        "eta_seconds": float(eta_seconds),
                        "device": device.type,
                    },
                )
            except Exception:
                pass

        # Persist training checkpoint each epoch so runs can be very short.
        try:
            rng = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }
            _atomic_torch_save(
                ckpt_path,
                {
                    "kind": "deep_train_ckpt_v1",
                    "model_key": model_key,
                    "instrument_key": instrument_key,
                    "interval": interval,
                    "horizon_steps": int(horizon_steps),
                    "model_family": model_family,
                    "cap_tier": cap_tier,
                    "seq_len": int(seq_len),
                    "feature_dim": int(X_tr.shape[-1]),
                    "epoch": int(ep + 1),
                    "target_epochs": int(target_epochs),
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                    "no_improve": int(no_improve),
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "best_state": best_state,
                    "opt_state": opt.state_dict(),
                    "rng": rng,
                    "saved_ts": int(datetime.now(timezone.utc).timestamp()),
                },
            )
        except Exception:
            pass

        # Budget stop (time)
        if max_seconds is not None and float(max_seconds) > 0:
            elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()
            if elapsed >= float(max_seconds):
                if progress_cb is not None:
                    try:
                        progress_cb(
                            float((ep + 1) / max(1, int(target_epochs))),
                            "budget stop (max_seconds reached)",
                            {"elapsed_seconds": float(elapsed), "max_seconds": float(max_seconds)},
                        )
                    except Exception:
                        pass
                break

        # Early stopping
        if int(patience) > 0 and no_improve >= int(patience):
            break

    # Basic direction accuracy on val
    model.load_state_dict(best_state or model.state_dict())
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            preds.append(model(xb).detach().cpu().numpy())
    y_hat = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.float32)

    y_true = y_te[: len(y_hat)]
    dir_acc = float(np.mean((y_true >= 0) == (y_hat >= 0))) if y_hat.size else 0.0

    # Serialize
    buf = io.BytesIO()
    torch.save(best_state or model.state_dict(), buf)
    state_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    trained_ts = int(now.timestamp())
    m = DeepModel(
        model_key=model_key,
        instrument_key=instrument_key,
        interval=interval,
        horizon_steps=int(horizon_steps),
        trained_ts=trained_ts,
        n_samples=int(len(y_seq)),
        seq_len=int(seq_len),
        feature_dim=int(X_tr.shape[-1]),
        norm_mean=[float(x) for x in mean.tolist()],
        norm_std=[float(x) for x in std.tolist()],
        state_dict_b64=state_b64,
        metrics={"val_huber": float(best_val), "direction_acc": float(dir_acc), "device": 1.0 if device.type == "cuda" else 0.0},
    )
    _save_deep_model(m)

    return {
        "ok": True,
        "model": {
            "model_key": m.model_key,
            "instrument_key": instrument_key,
            "interval": interval,
            "horizon_steps": int(horizon_steps),
            "trained_ts": trained_ts,
            "n_samples": int(m.n_samples),
            "seq_len": int(seq_len),
            "metrics": m.metrics,
        },
    }


def predict_deep_return(
    model: DeepModel,
    *,
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
) -> dict[str, Any]:
    if not _torch_available():
        return {"ok": False, "reason": "torch not installed"}

    import torch
    import torch.nn as nn

    # Rebuild model architecture
    seq_len = int(model.seq_len)
    feature_dim = int(model.feature_dim)

    if highs is None:
        highs = closes
    if lows is None:
        lows = closes
    if volumes is None:
        volumes = [1.0 for _ in closes]

    # Build candle-like arrays from passed values
    n = min(len(closes), len(highs), len(lows), len(volumes))
    closes = [float(x) for x in closes[-n:]]
    highs = [float(x) for x in highs[-n:]]
    lows = [float(x) for x in lows[-n:]]
    volumes = [float(x) for x in volumes[-n:]]

    if n < seq_len + 2:
        return {"ok": False, "reason": "not enough data", "needed": seq_len + 2, "have": n}

    # Compute feature matrix (same as training)
    closes_a = np.asarray(closes, dtype=float)
    highs_a = np.asarray(highs, dtype=float)
    lows_a = np.asarray(lows, dtype=float)
    vols_a = np.asarray(volumes, dtype=float)
    closes_a = np.where(closes_a <= 0, 1e-9, closes_a)
    rets = np.diff(np.log(closes_a), prepend=np.log(closes_a[0]))
    hl = (highs_a - lows_a) / closes_a
    lv = np.log(np.where(vols_a <= 0, 1.0, vols_a))
    X = np.stack([rets, hl, lv], axis=1).astype(np.float32)

    X = X[-seq_len:, :]
    mean = np.asarray(model.norm_mean, dtype=np.float32)
    std = np.asarray(model.norm_std, dtype=np.float32)
    if mean.size != feature_dim or std.size != feature_dim:
        return {"ok": False, "reason": "normalizer shape mismatch"}

    Xn = (X - mean) / np.where(std <= 1e-6, 1.0, std)

    class PosEnc(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TSModel(nn.Module):
        def __init__(self, feature_dim: int, d_model: int = 64, nhead: int = 4, layers: int = 3, dropout: float = 0.1):
            super().__init__()
            self.inp = nn.Linear(feature_dim, d_model)
            self.pos = PosEnc(d_model, max_len=max(512, int(seq_len) + 4))
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, x):
            x = self.inp(x)
            x = self.pos(x)
            x = self.enc(x)
            x_last = x[:, -1, :]
            out = self.head(x_last).squeeze(-1)
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TSModel(feature_dim=feature_dim).to(device)

    raw = base64.b64decode(model.state_dict_b64.encode("ascii"))
    state = torch.load(io.BytesIO(raw), map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    xb = torch.from_numpy(Xn).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = float(net(xb).detach().cpu().item())

    # Convert into signal + confidence heuristics
    feats = compute_features(closes[-max(60, min(len(closes), 120)) :])
    thresh = 0.002 if model.interval.endswith("m") else 0.01
    if pred > thresh and feats.rsi14 < 70:
        signal = "BUY"
    elif pred < -thresh and feats.rsi14 > 30:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Confidence: scale by |pred| and volatility proxy
    vol = max(float(feats.volatility20), 1e-6)
    confidence = float(max(0.0, min(1.0, (abs(pred) / (vol + 1e-6)) / 10.0)))

    return {
        "ok": True,
        "predicted_return": float(pred),
        "signal": signal,
        "confidence": confidence,
        "device": device.type,
    }
