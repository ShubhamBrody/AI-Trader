from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from app.candles.persistence_sql import get_candles
from app.learning.pattern_seq import (
    PATTERN_NAMES,
    PatternSeqConfig,
    PatternSeqNet,
    featurize_window,
    label_window,
    save_model,
    torch_available,
)


@dataclass(frozen=True)
class TrainPatternSeqParams:
    interval: str = "1m"
    lookback_days: int = 30
    seq_len: int = 64
    stride: int = 2
    epochs: int = 3
    batch_size: int = 256
    lr: float = 1e-3
    label_threshold: float = 0.35

    max_candles_per_symbol: int = 5000
    max_windows_per_symbol: int = 1500
    max_windows_total: int = 50000


def _clamp_int(v: Any, lo: int, hi: int) -> int:
    try:
        v = int(v)
    except Exception:
        v = lo
    return max(int(lo), min(int(hi), int(v)))


def _clamp_float(v: Any, lo: float, hi: float) -> float:
    try:
        v = float(v)
    except Exception:
        v = lo
    return float(max(float(lo), min(float(hi), float(v))))


def train_pattern_seq_model_for_instruments(
    instrument_keys: list[str],
    *,
    out_path: str,
    params: TrainPatternSeqParams,
    progress_cb: Callable[[float, str, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    """Train a single global multi-label pattern model over many instruments.

    Labels are generated using the existing detector (weak supervision) during training.
    Inference uses only the neural model.
    """

    if not torch_available():
        return {"ok": False, "reason": "torch not installed"}

    import torch
    import torch.nn.functional as F

    keys = [str(k).strip() for k in (instrument_keys or []) if str(k).strip()]
    if not keys:
        return {"ok": False, "reason": "no instruments"}

    p = TrainPatternSeqParams(
        interval=str(params.interval),
        lookback_days=_clamp_int(params.lookback_days, 1, 3650),
        seq_len=_clamp_int(params.seq_len, 16, 512),
        stride=_clamp_int(params.stride, 1, 60),
        epochs=_clamp_int(params.epochs, 1, 200),
        batch_size=_clamp_int(params.batch_size, 16, 4096),
        lr=_clamp_float(params.lr, 1e-6, 1e-1),
        label_threshold=_clamp_float(params.label_threshold, 0.05, 0.95),
        max_candles_per_symbol=_clamp_int(params.max_candles_per_symbol, 200, 200000),
        max_windows_per_symbol=_clamp_int(params.max_windows_per_symbol, 50, 200000),
        max_windows_total=_clamp_int(params.max_windows_total, 500, 500000),
    )

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(p.lookback_days))

    xs: list[list[list[float]]] = []
    ys: list[list[int]] = []

    total_syms = len(keys)
    used_syms = 0

    def emit(frac: float, msg: str, metrics: dict[str, Any] | None = None) -> None:
        if progress_cb is not None:
            progress_cb(float(frac), str(msg), metrics)

    emit(0.01, f"collecting samples from {total_syms} symbols", {"total_symbols": total_syms})

    for idx, instrument_key in enumerate(keys):
        if len(xs) >= p.max_windows_total:
            break

        candles = get_candles(
            instrument_key,
            p.interval,
            int(start.timestamp()),
            int(now.timestamp()),
            limit=int(p.max_candles_per_symbol),
        )

        if len(candles) < p.seq_len + 5:
            emit(
                0.02 + 0.5 * (idx / max(1, total_syms)),
                f"skip {instrument_key} (not enough candles)",
                {"instrument_key": instrument_key, "available": len(candles)},
            )
            continue

        windows_added = 0
        for j in range(p.seq_len, len(candles) + 1, p.stride):
            if len(xs) >= p.max_windows_total or windows_added >= p.max_windows_per_symbol:
                break

            window = candles[j - p.seq_len : j]
            y, _ = label_window(window, confidence_threshold=float(p.label_threshold))
            x = featurize_window(window)
            if not x:
                continue
            xs.append(x)
            ys.append(y)
            windows_added += 1

        used_syms += 1
        emit(
            0.02 + 0.5 * (idx / max(1, total_syms)),
            f"collected {windows_added} windows from {instrument_key}",
            {
                "instrument_key": instrument_key,
                "windows": windows_added,
                "total_windows": len(xs),
                "done_symbols": used_syms,
            },
        )

    if not xs:
        return {"ok": False, "reason": "no samples"}

    emit(0.60, f"training on {len(xs)} samples", {"samples": len(xs), "symbols": used_syms})

    x_t = torch.tensor(xs, dtype=torch.float32)
    y_t = torch.tensor(ys, dtype=torch.float32)

    cfg = PatternSeqConfig(seq_len=int(p.seq_len))
    model = PatternSeqNet(cfg.feat_dim, len(PATTERN_NAMES), hidden=cfg.hidden)

    opt = torch.optim.Adam(model.parameters(), lr=float(p.lr))

    model.train()
    n = int(x_t.shape[0])
    bs = int(p.batch_size)

    losses: list[float] = []
    for epoch in range(1, int(p.epochs) + 1):
        # simple mini-batch loop (no shuffling for determinism; ok for MVP)
        epoch_loss = 0.0
        steps = 0
        for start_i in range(0, n, bs):
            end_i = min(n, start_i + bs)
            xb = x_t[start_i:end_i]
            yb = y_t[start_i:end_i]

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach().cpu().item())
            steps += 1

        avg = epoch_loss / max(1, steps)
        losses.append(avg)
        emit(
            0.60 + 0.38 * (epoch / max(1, int(p.epochs))),
            f"epoch {epoch}/{p.epochs} loss={avg:.4f}",
            {"epoch": epoch, "loss": avg, "samples": n},
        )

    model.eval()
    save_model(out_path, model, cfg=cfg)

    return {
        "ok": True,
        "model_path": out_path,
        "seq_len": int(p.seq_len),
        "labels": len(PATTERN_NAMES),
        "samples": int(n),
        "symbols": int(used_syms),
        "loss": (None if not losses else float(losses[-1])),
        "loss_history": losses[-10:],
    }
