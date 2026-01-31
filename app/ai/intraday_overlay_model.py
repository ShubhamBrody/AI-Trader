from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.core.db import db_conn


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _i(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _candle_get(c: Any, k: str, default: Any = None) -> Any:
    if isinstance(c, dict):
        return c.get(k, default)
    return getattr(c, k, default)


def _sorted_candles(candles: list[Any]) -> list[Any]:
    return sorted(list(candles or []), key=lambda c: _i(_candle_get(c, "ts", 0)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def compute_atr(candles: list[Any], period: int = 14) -> float:
    cs = _sorted_candles(candles)
    if len(cs) < 2:
        return 0.0
    trs: list[float] = []
    for prev, cur in zip(cs[:-1], cs[1:]):
        pc = _f(_candle_get(prev, "close"))
        ch = _f(_candle_get(cur, "high"))
        cl = _f(_candle_get(cur, "low"))
        tr = max(ch - cl, abs(ch - pc), abs(cl - pc))
        trs.append(float(max(0.0, tr)))
    if not trs:
        return 0.0
    n = min(int(period), len(trs))
    return float(sum(trs[-n:]) / max(1, n))


def _linreg(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    # Return (slope, r2)
    if len(xs) < 2:
        return 0.0, 0.0
    x = xs.astype(float)
    y = ys.astype(float)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 1e-12:
        return 0.0, 0.0
    slope = float((x * (y - y.mean())).sum() / denom)
    yhat = slope * x + y.mean()
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - yhat) ** 2).sum())
    r2 = 0.0 if ss_tot <= 1e-12 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return slope, float(r2)


FEATURES = [
    "ret1",
    "ret5",
    "atr_pct",
    "trend_slope_pct",
    "trend_r2",
    "range_pct",
    "vol_ratio",
    "breakout_up",
    "breakout_down",
]


def featurize(candles: list[Any]) -> dict[str, float]:
    """Compute the feature vector for the *last* candle of a window."""
    cs = _sorted_candles(candles)
    if len(cs) < 35:
        return {k: 0.0 for k in FEATURES}

    last = cs[-1]
    closes = np.array([_f(_candle_get(c, "close")) for c in cs], dtype=float)
    highs = np.array([_f(_candle_get(c, "high")) for c in cs], dtype=float)
    lows = np.array([_f(_candle_get(c, "low")) for c in cs], dtype=float)
    vols = np.array([_f(_candle_get(c, "volume")) for c in cs], dtype=float)

    c0 = float(closes[-1] or 1.0)
    c1 = float(closes[-2] or c0)
    c5 = float(closes[-6] if len(closes) >= 6 else c1)

    ret1 = (c0 - c1) / max(1e-9, c1)
    ret5 = (c0 - c5) / max(1e-9, c5)

    atr = compute_atr(cs[-40:], period=14)
    atr_pct = (atr / max(1e-9, c0))

    # Trend over last 30 closes
    window = closes[-30:]
    xs = np.arange(len(window), dtype=float)
    slope, r2 = _linreg(xs, window)
    trend_slope_pct = (slope / max(1e-9, float(window[-1]))) * 100.0

    # Range over last 30
    range_pct = (float(highs[-30:].max() - lows[-30:].min()) / max(1e-9, c0))

    avg_vol = float(vols[-30:].mean()) if len(vols) >= 30 else float(vols.mean() if len(vols) else 0.0)
    vol_ratio = (float(vols[-1]) / max(1e-9, avg_vol)) if avg_vol > 0 else 1.0
    vol_ratio = float(min(5.0, max(0.0, vol_ratio)))

    recent_high = float(highs[-20:].max())
    recent_low = float(lows[-20:].min())
    breakout_up = 1.0 if c0 >= recent_high else 0.0
    breakout_down = 1.0 if c0 <= recent_low else 0.0

    return {
        "ret1": float(ret1),
        "ret5": float(ret5),
        "atr_pct": float(atr_pct),
        "trend_slope_pct": float(trend_slope_pct),
        "trend_r2": float(r2),
        "range_pct": float(range_pct),
        "vol_ratio": float(vol_ratio),
        "breakout_up": float(breakout_up),
        "breakout_down": float(breakout_down),
    }


def _label_up_move(
    candles: list[Any],
    *,
    horizon_steps: int,
    k_atr: float,
    atr_period: int = 14,
) -> int | None:
    """Label whether an up-move of k*ATR happens before a down-move of k*ATR."""
    cs = _sorted_candles(candles)
    if len(cs) < (atr_period + 5):
        return None
    last = cs[-1]
    entry = _f(_candle_get(last, "close"))
    atr = compute_atr(cs[-max(atr_period + 5, 40) :], period=atr_period)
    if atr <= 0 or entry <= 0:
        return None
    up_thr = entry + k_atr * atr
    dn_thr = entry - k_atr * atr

    # Horizon candles are provided separately by the caller.
    # This function assumes `candles` includes the entry candle only.
    return None


@dataclass(frozen=True)
class OverlayModel:
    instrument_key: str
    interval: str
    created_ts: int
    features: list[str]
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float
    horizon_steps: int
    k_atr: float
    metrics: dict[str, Any]

    def predict_proba_up(self, feats: dict[str, float]) -> float:
        x = np.array([float(feats.get(k, 0.0)) for k in self.features], dtype=float)
        m = np.array(self.mean, dtype=float)
        s = np.array(self.std, dtype=float)
        s = np.where(s <= 1e-9, 1.0, s)
        z = (x - m) / s
        w = np.array(self.weights, dtype=float)
        score = float(np.dot(w, z) + float(self.bias))
        p = float(_sigmoid(np.array([score], dtype=float))[0])
        return float(max(0.0, min(1.0, p)))


def load_model(instrument_key: str, interval: str) -> OverlayModel | None:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT instrument_key, interval, created_ts, model_json FROM intraday_overlay_models WHERE instrument_key=? AND interval=?",
            (str(instrument_key), str(interval)),
        ).fetchone()
        if row is None:
            # fallback to global model if present
            row = conn.execute(
                "SELECT instrument_key, interval, created_ts, model_json FROM intraday_overlay_models WHERE instrument_key='__global__' AND interval=?",
                (str(interval),),
            ).fetchone()
        if row is None:
            return None

    payload = json.loads(row["model_json"])
    return OverlayModel(
        instrument_key=str(row["instrument_key"]),
        interval=str(row["interval"]),
        created_ts=int(row["created_ts"]),
        features=list(payload.get("features") or FEATURES),
        mean=list(payload.get("mean") or [0.0] * len(FEATURES)),
        std=list(payload.get("std") or [1.0] * len(FEATURES)),
        weights=list(payload.get("weights") or [0.0] * len(FEATURES)),
        bias=float(payload.get("bias") or 0.0),
        horizon_steps=int(payload.get("horizon_steps") or 30),
        k_atr=float(payload.get("k_atr") or 1.2),
        metrics=dict(payload.get("metrics") or {}),
    )


def save_model(model: OverlayModel) -> None:
    model_json = json.dumps(
        {
            "version": 1,
            "kind": "logreg",
            "features": model.features,
            "mean": model.mean,
            "std": model.std,
            "weights": model.weights,
            "bias": model.bias,
            "horizon_steps": model.horizon_steps,
            "k_atr": model.k_atr,
            "metrics": model.metrics,
        },
        ensure_ascii=False,
    )
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO intraday_overlay_models (instrument_key, interval, created_ts, model_json) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(instrument_key, interval) DO UPDATE SET created_ts=excluded.created_ts, model_json=excluded.model_json",
            (model.instrument_key, model.interval, int(model.created_ts), model_json),
        )


def _build_dataset(
    candles: list[Any],
    *,
    feature_window: int,
    horizon_steps: int,
    k_atr: float,
) -> tuple[np.ndarray, np.ndarray]:
    cs = _sorted_candles(candles)
    if len(cs) < (feature_window + horizon_steps + 20):
        return np.zeros((0, len(FEATURES))), np.zeros((0,))

    X: list[list[float]] = []
    y: list[int] = []

    for i in range(feature_window, len(cs) - horizon_steps - 1):
        window = cs[i - feature_window : i + 1]
        fut = cs[i + 1 : i + 1 + horizon_steps]

        entry = _f(_candle_get(cs[i], "close"))
        if entry <= 0:
            continue

        atr = compute_atr(window[-40:], period=14)
        if atr <= 0:
            continue

        up_thr = entry + k_atr * atr
        dn_thr = entry - k_atr * atr

        up_hit = False
        dn_hit = False
        for fc in fut:
            hi = _f(_candle_get(fc, "high"))
            lo = _f(_candle_get(fc, "low"))
            if hi >= up_thr:
                up_hit = True
            if lo <= dn_thr:
                dn_hit = True
            # resolve early if only one hit
            if up_hit and not dn_hit:
                break
            if dn_hit and not up_hit:
                break

        if up_hit and dn_hit:
            continue
        if not up_hit and not dn_hit:
            continue

        label = 1 if up_hit else 0
        feats = featurize(window)
        X.append([float(feats[k]) for k in FEATURES])
        y.append(int(label))

    if not X:
        return np.zeros((0, len(FEATURES))), np.zeros((0,))

    return np.array(X, dtype=float), np.array(y, dtype=float)


def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.25,
    epochs: int = 400,
    l2: float = 0.1,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    n, d = X.shape
    if n <= 10:
        return np.zeros((d,), dtype=float), 0.0, {"ok": False, "reason": "not_enough_samples", "n": int(n)}

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std <= 1e-9, 1.0, std)
    Z = (X - mean) / std

    w = np.zeros((d,), dtype=float)
    b = 0.0

    for _ in range(int(epochs)):
        p = _sigmoid(Z @ w + b)
        grad_w = (Z.T @ (p - y)) / n + l2 * w
        grad_b = float((p - y).mean())
        w = w - lr * grad_w
        b = b - lr * grad_b

    p = _sigmoid(Z @ w + b)
    preds = (p >= 0.5).astype(float)
    acc = float((preds == y).mean())
    # logloss
    eps = 1e-9
    loss = float((-y * np.log(p + eps) - (1 - y) * np.log(1 - p + eps)).mean())

    metrics = {
        "ok": True,
        "n": int(n),
        "acc": acc,
        "logloss": loss,
    }
    return w, float(b), {"metrics": metrics, "mean": mean.tolist(), "std": std.tolist()}


def train_and_store(
    *,
    instrument_key: str,
    interval: str,
    candles: list[Any],
    feature_window: int = 60,
    horizon_steps: int = 30,
    k_atr: float = 1.2,
) -> dict[str, Any]:
    X, y = _build_dataset(candles, feature_window=int(feature_window), horizon_steps=int(horizon_steps), k_atr=float(k_atr))
    if X.shape[0] < 15:
        return {"ok": False, "reason": "not_enough_labeled_samples", "n": int(X.shape[0])}

    w, b, pack = train_logreg(X, y)
    metrics = dict((pack.get("metrics") or {}))
    if not metrics.get("ok"):
        return metrics

    created_ts = int(datetime.now(timezone.utc).timestamp())
    model = OverlayModel(
        instrument_key=str(instrument_key),
        interval=str(interval),
        created_ts=created_ts,
        features=list(FEATURES),
        mean=list(pack.get("mean") or [0.0] * len(FEATURES)),
        std=list(pack.get("std") or [1.0] * len(FEATURES)),
        weights=[float(x) for x in w.tolist()],
        bias=float(b),
        horizon_steps=int(horizon_steps),
        k_atr=float(k_atr),
        metrics=metrics,
    )
    save_model(model)
    return {"ok": True, "instrument_key": instrument_key, "interval": interval, "created_ts": created_ts, "metrics": metrics}
