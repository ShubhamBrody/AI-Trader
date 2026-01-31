from __future__ import annotations

# pyright: reportMissingImports=false

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.ai.feature_engineering import compute_features
from app.candles.persistence_sql import get_candles
from app.core.db import db_conn


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2))) if err.size else 0.0
    mae = float(np.mean(np.abs(err))) if err.size else 0.0
    dir_acc = float(np.mean((y_true >= 0) == (y_pred >= 0))) if err.size else 0.0
    return {"rmse": rmse, "mae": mae, "direction_acc": dir_acc}


def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean

    n_features = X.shape[1]
    A = Xc.T @ Xc + float(l2) * np.eye(n_features)
    b = Xc.T @ yc
    w = np.linalg.solve(A, b)
    bias = y_mean - float(x_mean @ w)
    return w, float(bias)


def build_supervised(closes: list[float], *, horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[list[float]] = []
    y_rows: list[float] = []

    for t in range(30, len(closes) - int(horizon_steps) - 1):
        window = closes[: t + 1]
        feats = compute_features(window)
        if feats.last_close <= 0:
            continue
        future = float(closes[t + int(horizon_steps)])
        now_close = float(closes[t])
        if now_close == 0:
            continue
        target_ret = (future - now_close) / now_close
        X_rows.append([feats.rsi14, feats.sma20, feats.ema20, feats.volatility20, feats.last_close])
        y_rows.append(float(target_ret))

    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float)


def walk_forward_eval_ridge(
    instrument_key: str,
    interval: str,
    *,
    lookback_days: int,
    horizon_steps: int,
    l2: float = 1e-2,
    folds: int = 5,
    min_train: int = 300,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(lookback_days))
    candles = get_candles(instrument_key, interval, int(start.timestamp()), int(now.timestamp()))
    closes = [float(c.close) for c in candles]
    if len(closes) < max(80, min_train + 50):
        return {"ok": False, "reason": "not enough data", "have": len(closes)}

    X, y = build_supervised(closes, horizon_steps=int(horizon_steps))
    if y.size < min_train + 50:
        return {"ok": False, "reason": "not enough supervised samples", "have": int(y.size)}

    n = int(y.size)
    folds = max(2, min(int(folds), 10))
    test_size = max(30, int(0.1 * n))

    fold_metrics: list[dict[str, float]] = []
    # Expanding window: last folds*test_size used for testing slices.
    for i in range(folds):
        test_end = n - i * test_size
        test_start = max(0, test_end - test_size)
        train_end = test_start
        if train_end < min_train:
            break

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        if y_te.size == 0:
            continue

        w, b = _ridge_fit(X_tr, y_tr, float(l2))
        y_hat = X_te @ w + b
        fold_metrics.append(_metrics(y_te, y_hat))

    if not fold_metrics:
        return {"ok": False, "reason": "no folds evaluated"}

    # Aggregate
    rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
    mae = float(np.mean([m["mae"] for m in fold_metrics]))
    dir_acc = float(np.mean([m["direction_acc"] for m in fold_metrics]))

    # Baseline stats for drift
    rets = np.diff(np.asarray(closes, dtype=float))
    rets = rets / np.where(np.asarray(closes[:-1], dtype=float) == 0, 1e-9, np.asarray(closes[:-1], dtype=float))
    base_mean = float(np.mean(rets)) if rets.size else 0.0
    base_std = float(np.std(rets)) if rets.size else 0.0

    return {
        "ok": True,
        "eval_kind": "walk_forward",
        "folds": len(fold_metrics),
        "rmse": rmse,
        "mae": mae,
        "direction_acc": dir_acc,
        "baseline": {"ret_mean": base_mean, "ret_std": base_std},
    }


def walk_forward_eval_deep(
    instrument_key: str,
    interval: str,
    *,
    lookback_days: int,
    horizon_steps: int,
    seq_len: int = 120,
    folds: int = 3,
    epochs: int = 2,
    batch_size: int = 128,
    min_train: int = 400,
) -> dict[str, Any]:
    """True walk-forward evaluation for deep models.

    Notes:
    - Uses an expanding window and evaluates on the most recent slices.
    - Trains lightweight transformer folds (short epochs) to keep runtime reasonable.
    - Returns rmse/mae/direction_acc comparable to ridge walk-forward.
    """

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except Exception:
        return {"ok": False, "reason": "torch not installed"}

    from app.learning.deep_service import _build_features  # internal but stable enough for our use

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(lookback_days))
    candles = get_candles(instrument_key, interval, int(start.timestamp()), int(now.timestamp()))
    if len(candles) < max(int(seq_len) + int(horizon_steps) + 10, int(min_train) + 50):
        return {"ok": False, "reason": "not enough data", "have": int(len(candles))}

    X_all = _build_features(candles)
    closes = [float(c.close) for c in candles]

    # Targets: future return over horizon_steps
    y_all: list[float] = []
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

    # Build sequences
    X_seq: list[np.ndarray] = []
    y_seq: list[float] = []
    for t in range(int(seq_len), len(candles) - int(horizon_steps) - 1):
        X_seq.append(X_all[t - int(seq_len) : t])
        y_seq.append(float(y_all[t]))

    if len(y_seq) < int(min_train) + 50:
        return {"ok": False, "reason": "not enough usable samples", "have": int(len(y_seq))}

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32)

    n = int(y_seq.shape[0])
    folds = max(2, min(int(folds), 6))
    test_size = max(30, int(0.1 * n))

    class SeqDs(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X
            self.y = y

        def __len__(self) -> int:
            return int(self.X.shape[0])

        def __getitem__(self, idx: int):
            return torch.from_numpy(self.X[idx]), torch.tensor(float(self.y[idx]), dtype=torch.float32)

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
        def __init__(self, feature_dim: int, seq_len_: int):
            super().__init__()
            d_model = 48
            self.inp = nn.Linear(feature_dim, d_model)
            self.pos = PosEnc(d_model, max_len=max(256, int(seq_len_) + 4))
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=192, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, x):
            x = self.inp(x)
            x = self.pos(x)
            x = self.enc(x)
            x_last = x[:, -1, :]
            return self.head(x_last).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.HuberLoss(delta=1.0)

    fold_metrics: list[dict[str, float]] = []
    for i in range(int(folds)):
        test_end = n - i * test_size
        test_start = max(0, test_end - test_size)
        train_end = test_start
        if train_end < int(min_train):
            break

        X_tr = X_seq[:train_end]
        y_tr = y_seq[:train_end]
        X_te = X_seq[test_start:test_end]
        y_te = y_seq[test_start:test_end]
        if y_te.size == 0:
            continue

        # Normalize on training data
        mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
        std = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0)
        std = np.where(std <= 1e-6, 1.0, std)
        X_trn = (X_tr - mean) / std
        X_ten = (X_te - mean) / std

        train_loader = DataLoader(SeqDs(X_trn, y_tr), batch_size=int(batch_size), shuffle=True, drop_last=True)
        val_loader = DataLoader(SeqDs(X_ten, y_te), batch_size=int(batch_size), shuffle=False)

        model = TSModel(feature_dim=int(X_tr.shape[-1]), seq_len_=int(seq_len)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

        model.train()
        for _ in range(max(1, int(epochs))):
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        # Predict
        model.eval()
        preds: list[float] = []
        ys: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).detach().cpu().numpy().astype(float)
                preds.extend([float(x) for x in pred.reshape(-1)])
                ys.extend([float(x) for x in yb.numpy().astype(float).reshape(-1)])

        fold_metrics.append(_metrics(np.asarray(ys, dtype=float), np.asarray(preds, dtype=float)))

    if not fold_metrics:
        return {"ok": False, "reason": "no folds evaluated"}

    rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
    mae = float(np.mean([m["mae"] for m in fold_metrics]))
    dir_acc = float(np.mean([m["direction_acc"] for m in fold_metrics]))

    rets = np.diff(np.asarray(closes, dtype=float))
    rets = rets / np.where(np.asarray(closes[:-1], dtype=float) == 0, 1e-9, np.asarray(closes[:-1], dtype=float))
    base_mean = float(np.mean(rets)) if rets.size else 0.0
    base_std = float(np.std(rets)) if rets.size else 0.0

    return {
        "ok": True,
        "eval_kind": "walk_forward_deep",
        "device": device.type,
        "folds": len(fold_metrics),
        "rmse": rmse,
        "mae": mae,
        "direction_acc": dir_acc,
        "baseline": {"ret_mean": base_mean, "ret_std": base_std},
    }


def record_evaluation(*, slot_key: str, model_key: str, kind: str, eval_kind: str, metrics: dict[str, Any]) -> None:
    now = int(datetime.now(timezone.utc).timestamp())
    metrics_json = json.dumps(metrics or {}, ensure_ascii=False, separators=(",", ":"))
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO model_evaluations (ts, slot_key, model_key, kind, eval_kind, metrics_json) VALUES (?, ?, ?, ?, ?, ?)",
            (int(now), str(slot_key), str(model_key), str(kind), str(eval_kind), metrics_json),
        )
