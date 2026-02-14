from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.ai.feature_engineering import compute_features
from app.candles.persistence_sql import get_candles
from app.core.db import db_conn
from app.learning.registry import get_registry, slot_key as registry_slot_key


def _model_key(instrument_key: str, interval: str, horizon_steps: int, model_family: str | None, cap_tier: str | None) -> str:
    fam = (model_family or "generic").lower()
    cap = (cap_tier or "unknown").lower()
    return f"{fam}::{cap}::{instrument_key}::{interval}::h{int(horizon_steps)}"


@dataclass(frozen=True)
class TrainedModel:
    model_key: str
    instrument_key: str
    interval: str
    horizon_steps: int
    trained_ts: int
    n_samples: int
    feature_names: list[str]
    weights: list[float]
    bias: float
    metrics: dict[str, float]

    def predict_return(self, closes: list[float]) -> float:
        feats = compute_features(closes)
        x = np.asarray(
            [
                feats.rsi14,
                feats.sma20,
                feats.ema20,
                feats.volatility20,
                feats.last_close,
            ],
            dtype=float,
        )
        w = np.asarray(self.weights, dtype=float)
        return float(np.dot(x, w) + float(self.bias))


def load_model(
    instrument_key: str,
    interval: str,
    horizon_steps: int = 1,
    *,
    model_family: str | None = None,
    cap_tier: str | None = None,
) -> TrainedModel | None:
    # Prefer promoted production model if present.
    keys: list[str]
    try:
        slot = registry_slot_key(
            instrument_key=instrument_key,
            interval=interval,
            horizon_steps=int(horizon_steps),
            model_family=model_family,
            cap_tier=cap_tier,
            kind="ridge",
        )
        reg = get_registry(slot)
        if reg and reg.get("model_key"):
            keys = [str(reg["model_key"])]
        else:
            raise KeyError("no registry entry")
    except Exception:
        # Backwards compatible: try new key first, then legacy key.
        key_new = _model_key(instrument_key, interval, horizon_steps, model_family, cap_tier)
        keys = [key_new, f"{instrument_key}::{interval}::h{int(horizon_steps)}"]
    with db_conn() as conn:
        row = None
        for k in keys:
            cur = conn.execute(
                "SELECT model_key, instrument_key, interval, horizon_steps, trained_ts, n_samples, metrics_json, model_json FROM trained_models WHERE model_key=?",
                (k,),
            )
            row = cur.fetchone()
            if row is not None:
                break
        if row is None:
            return None

        metrics = json.loads(row["metrics_json"])
        model = json.loads(row["model_json"])
        return TrainedModel(
            model_key=row["model_key"],
            instrument_key=row["instrument_key"],
            interval=row["interval"],
            horizon_steps=int(row["horizon_steps"]),
            trained_ts=int(row["trained_ts"]),
            n_samples=int(row["n_samples"]),
            feature_names=list(model["feature_names"]),
            weights=list(model["weights"]),
            bias=float(model["bias"]),
            metrics={k: float(v) for k, v in metrics.items()},
        )


def _save_model(m: TrainedModel) -> None:
    metrics_json = json.dumps(m.metrics, ensure_ascii=False, separators=(",", ":"))
    model_json = json.dumps(
        {"feature_names": m.feature_names, "weights": m.weights, "bias": m.bias},
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


def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    # Add bias term via explicit intercept solving.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")

    # Centering improves conditioning.
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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2))) if err.size else 0.0
    mae = float(np.mean(np.abs(err))) if err.size else 0.0
    dir_acc = float(np.mean((y_true >= 0) == (y_pred >= 0))) if err.size else 0.0
    return {"rmse": rmse, "mae": mae, "direction_acc": dir_acc}


def train_model(
    instrument_key: str,
    interval: str = "1d",
    *,
    lookback_days: int = 365,
    horizon_steps: int = 1,
    model_family: str | None = None,
    cap_tier: str | None = None,
    l2: float = 1e-2,
    min_samples: int = 200,
    data_fraction: float = 1.0,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    end = now
    start = now - timedelta(days=int(lookback_days))
    candles = get_candles(instrument_key, interval, int(start.timestamp()), int(end.timestamp()))
    closes = [c.close for c in candles]
    if len(closes) < max(60, min_samples):
        return {
            "ok": False,
            "reason": "not enough data in DB",
            "needed": max(60, min_samples),
            "have": len(closes),
        }

    feature_names = ["rsi14", "sma20", "ema20", "volatility20", "last_close"]
    X_rows: list[list[float]] = []
    y_rows: list[float] = []

    # Performance note:
    # The naive approach of calling compute_features(closes[:t+1]) inside the training loop is O(n^2)
    # (EMA alone iterates over the full history each time). For intraday series, that can take a very
    # long time. Here we compute the same feature set in O(n) using rolling windows.
    close_arr = np.asarray(closes, dtype=float)
    n = int(close_arr.size)

    # Precompute SMA20 via cumulative sum.
    csum = np.cumsum(close_arr)
    sma20 = np.empty(n, dtype=float)
    for t in range(n):
        w = 20 if t + 1 >= 20 else (t + 1)
        start_i = t + 1 - w
        total = float(csum[t] - (csum[start_i - 1] if start_i > 0 else 0.0))
        sma20[t] = total / float(w)

    # Precompute EMA20 with constant alpha (matches compute_features for t >= 19).
    # Training loop starts at t=30, so this is behaviorally aligned.
    alpha = 2.0 / (20.0 + 1.0)
    ema20 = np.empty(n, dtype=float)
    ema20[0] = float(close_arr[0])
    for t in range(1, n):
        ema20[t] = alpha * float(close_arr[t]) + (1.0 - alpha) * float(ema20[t - 1])

    # Returns for volatility and RSI.
    prev = np.where(close_arr[:-1] == 0, 1e-9, close_arr[:-1])
    rets = (close_arr[1:] - close_arr[:-1]) / prev
    diffs = close_arr[1:] - close_arr[:-1]

    # Rolling RSI/volatility via cumulative sums (O(n)).
    gains = np.maximum(diffs, 0.0)
    losses = np.maximum(-diffs, 0.0)
    cg = np.cumsum(gains)
    cl = np.cumsum(losses)

    cr = np.cumsum(rets)
    cr2 = np.cumsum(rets * rets)

    rsi14 = np.empty(n, dtype=float)
    vol20 = np.empty(n, dtype=float)
    rsi14[0] = 50.0
    vol20[0] = 0.0
    for t in range(1, n):
        # RSI14 over last min(14, t) diffs ending at t-1
        rsi_w = 14 if t >= 14 else t
        start = t - rsi_w
        end = t  # exclusive in diffs/rets space

        sum_gain = float(cg[end - 1] - (cg[start - 1] if start > 0 else 0.0))
        sum_loss = float(cl[end - 1] - (cl[start - 1] if start > 0 else 0.0))
        avg_gain = sum_gain / float(rsi_w)
        avg_loss = sum_loss / float(rsi_w)
        rs = avg_gain / (avg_loss + 1e-9)
        rsi14[t] = 100.0 - (100.0 / (1.0 + rs))

        # Volatility20 (std of returns) over last min(20, t) returns ending at t-1
        vol_w = 20 if t >= 20 else t
        vstart = t - vol_w
        vend = t
        sum_r = float(cr[vend - 1] - (cr[vstart - 1] if vstart > 0 else 0.0))
        sum_r2 = float(cr2[vend - 1] - (cr2[vstart - 1] if vstart > 0 else 0.0))
        mean = sum_r / float(vol_w)
        mean2 = sum_r2 / float(vol_w)
        var = max(0.0, mean2 - mean * mean)
        vol20[t] = float(np.sqrt(var))

    # Build supervised samples: features at t -> return at t+h
    for t in range(30, n - int(horizon_steps) - 1):
        now_close = float(close_arr[t])
        if now_close <= 0:
            continue

        future = float(close_arr[t + int(horizon_steps)])
        if now_close == 0:
            continue
        target_ret = (future - now_close) / now_close
        X_rows.append([float(rsi14[t]), float(sma20[t]), float(ema20[t]), float(vol20[t]), float(now_close)])
        y_rows.append(float(target_ret))

    if len(y_rows) < min_samples:
        return {
            "ok": False,
            "reason": "not enough usable samples",
            "needed": int(min_samples),
            "have": len(y_rows),
        }

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)

    frac = float(data_fraction)
    if 0.0 < frac < 1.0 and y.size > 1:
        target_n = int(round(float(y.size) * frac))
        target_n = max(2, min(int(y.size), int(target_n)))
        idx = np.linspace(0, int(y.size) - 1, num=int(target_n), dtype=int)
        # Ensure strictly increasing indices to avoid duplicates for small series.
        idx = np.unique(idx)
        if idx.size >= 2:
            X = X[idx]
            y = y[idx]

    # Simple time-based split (80/20) to avoid leakage.
    split = int(0.8 * len(y))
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    w, bias = _ridge_fit(X_tr, y_tr, l2=float(l2))
    y_hat = X_te @ w + bias
    metrics = _metrics(y_te, y_hat)

    trained_ts = int(now.timestamp())
    model = TrainedModel(
        model_key=_model_key(instrument_key, interval, horizon_steps, model_family, cap_tier),
        instrument_key=instrument_key,
        interval=interval,
        horizon_steps=int(horizon_steps),
        trained_ts=trained_ts,
        n_samples=int(y.size),
        feature_names=feature_names,
        weights=[float(v) for v in w.tolist()],
        bias=float(bias),
        metrics=metrics,
    )
    _save_model(model)

    return {
        "ok": True,
        "model": {
            "model_key": model.model_key,
            "instrument_key": instrument_key,
            "interval": interval,
            "horizon_steps": int(horizon_steps),
            "trained_ts": trained_ts,
            "n_samples": int(len(y_rows)),
            "metrics": metrics,
        },
    }
