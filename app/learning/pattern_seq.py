from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from app.ai.candlestick_patterns import CATALOG, PatternMatch, detect_patterns
from app.ai.candlestick_patterns import Candle as PatternCandle


PATTERN_NAMES: list[str] = [p.name for p in CATALOG]
PATTERN_INDEX: dict[str, int] = {name: i for i, name in enumerate(PATTERN_NAMES)}


def torch_available() -> bool:
    return torch is not None and nn is not None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _log1p(x: float) -> float:
    try:
        return math.log1p(max(0.0, float(x)))
    except Exception:
        return 0.0


def candles_to_pattern_candles(candles: list[Any]) -> list[PatternCandle]:
    out: list[PatternCandle] = []
    for c in candles:
        out.append(
            PatternCandle(
                ts=int(getattr(c, "ts")),
                open=_safe_float(getattr(c, "open")),
                high=_safe_float(getattr(c, "high")),
                low=_safe_float(getattr(c, "low")),
                close=_safe_float(getattr(c, "close")),
                volume=_safe_float(getattr(c, "volume", 0.0)),
            )
        )
    return out


def label_window(
    candles: list[Any],
    *,
    confidence_threshold: float = 0.35,
) -> tuple[list[int], list[PatternMatch]]:
    """Weak labels for a window using the existing detector.

    Returns a multi-hot vector aligned to PATTERN_NAMES.
    """

    y = [0] * len(PATTERN_NAMES)
    if not candles:
        return y, []

    cs = candles_to_pattern_candles(candles)
    matches = detect_patterns(cs, weights=None, max_results=64)
    for m in matches:
        if float(m.confidence) < float(confidence_threshold):
            continue
        idx = PATTERN_INDEX.get(str(m.name))
        if idx is not None:
            y[idx] = 1

    return y, matches


def featurize_window(candles: list[Any]) -> list[list[float]]:
    """Convert candles to a scale-invariant feature sequence.

    Uses relative/candle-shape features rather than absolute price.
    Output shape: [T][F]
    """

    if not candles:
        return []

    closes = [_safe_float(getattr(c, "close"), 0.0) for c in candles]
    last_close = max(1e-9, closes[-1] if closes else 1.0)

    feats: list[list[float]] = []
    prev_close = closes[0] if closes else last_close

    vols = [_safe_float(getattr(c, "volume", 0.0), 0.0) for c in candles]
    log_vols = [_log1p(v) for v in vols]
    mu = sum(log_vols) / max(1, len(log_vols))
    var = sum((v - mu) ** 2 for v in log_vols) / max(1, len(log_vols))
    sigma = math.sqrt(max(1e-9, var))

    for i, c in enumerate(candles):
        o = _safe_float(getattr(c, "open"), 0.0)
        h = _safe_float(getattr(c, "high"), 0.0)
        l = _safe_float(getattr(c, "low"), 0.0)
        cl = _safe_float(getattr(c, "close"), 0.0)

        # Close-to-close return (scale free)
        r_cc = 0.0
        if i > 0:
            denom = max(1e-9, float(prev_close))
            r_cc = float(cl / denom - 1.0)
        prev_close = cl

        body = float((cl - o) / last_close)
        rng = float((h - l) / last_close)
        upper = float((h - max(o, cl)) / last_close)
        lower = float((min(o, cl) - l) / last_close)

        v = float((log_vols[i] - mu) / sigma) if i < len(log_vols) else 0.0

        feats.append([r_cc, body, rng, upper, lower, v])

    return feats


@dataclass(frozen=True)
class PatternSeqConfig:
    seq_len: int = 64
    feat_dim: int = 6
    hidden: int = 64


def _require_torch() -> None:
    if not torch_available():
        raise RuntimeError("PyTorch is not installed. Install requirements-deep.txt to enable sequence model.")


class PatternSeqNet(nn.Module):
    def __init__(self, feat_dim: int, num_labels: int, hidden: int = 64):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_labels = int(num_labels)
        self.hidden = int(hidden)

        self.net = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(self.hidden, self.num_labels)

    def forward(self, x):
        # x: [B,T,F] -> [B,F,T]
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        return self.head(h)


def _to_tensor(x_seq: list[list[float]]):
    _require_torch()
    return torch.tensor(x_seq, dtype=torch.float32)


def save_model(path: str, model: Any, *, cfg: PatternSeqConfig) -> None:
    _require_torch()
    payload = {
        "state_dict": model.state_dict(),
        "pattern_names": PATTERN_NAMES,
        "seq_len": int(cfg.seq_len),
        "feat_dim": int(cfg.feat_dim),
        "hidden": int(cfg.hidden),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


@lru_cache(maxsize=4)
def load_model(path: str) -> tuple[Any, PatternSeqConfig]:
    _require_torch()
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)

    payload = torch.load(path, map_location="cpu")
    names = list(payload.get("pattern_names") or [])
    if names != PATTERN_NAMES:
        raise RuntimeError("pattern_names mismatch: model was trained for a different catalog")

    cfg = PatternSeqConfig(
        seq_len=int(payload.get("seq_len") or 64),
        feat_dim=int(payload.get("feat_dim") or 6),
        hidden=int(payload.get("hidden") or 64),
    )

    model = PatternSeqNet(cfg.feat_dim, len(PATTERN_NAMES), hidden=cfg.hidden)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg


def predict_patterns_from_candles(
    candles: list[Any],
    *,
    model_path: str,
    max_results: int = 8,
) -> dict[str, Any]:
    """Predict pattern probabilities from the last seq_len candles."""

    model, cfg = load_model(model_path)

    if len(candles) < cfg.seq_len:
        return {
            "ok": False,
            "reason": "not_enough_candles",
            "required": int(cfg.seq_len),
            "available": int(len(candles)),
        }

    window = candles[-cfg.seq_len :]
    feats = featurize_window(window)

    x = _to_tensor(feats).unsqueeze(0)  # [1,T,F]
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).tolist()

    scored = sorted(
        (
            {"name": PATTERN_NAMES[i], "prob": float(probs[i])}
            for i in range(len(PATTERN_NAMES))
        ),
        key=lambda d: d["prob"],
        reverse=True,
    )

    top = scored[: max(1, int(max_results))]

    return {
        "ok": True,
        "seq_len": int(cfg.seq_len),
        "features": int(cfg.feat_dim),
        "count": len(PATTERN_NAMES),
        "top": top,
    }
