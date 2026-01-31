from __future__ import annotations

import argparse
from datetime import datetime, timezone

from app.candles.service import CandleService
from app.learning.pattern_seq import (
    PATTERN_NAMES,
    PatternSeqConfig,
    PatternSeqNet,
    featurize_window,
    label_window,
    save_model,
    torch_available,
)


def _parse_dt(s: str) -> datetime:
    # ISO-ish: 2026-01-01T09:15:00Z or 2026-01-01T09:15
    s = (s or "").strip()
    if not s:
        raise ValueError("empty datetime")
    if s.endswith("Z"):
        s = s[:-1]
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a sequence-based candlestick pattern model (weak-labeled).")
    ap.add_argument("--instrument", required=True, help="instrument_key or symbol")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start", required=True, help="ISO datetime, e.g. 2026-01-01T00:00")
    ap.add_argument("--end", required=True, help="ISO datetime")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="data/models/pattern_seq.pt")
    ap.add_argument("--label-threshold", type=float, default=0.35)

    args = ap.parse_args()

    if not torch_available():
        raise SystemExit("PyTorch not installed. Install requirements-deep.txt first.")

    import torch
    import torch.nn.functional as F

    svc = CandleService()
    start = _parse_dt(args.start)
    end = _parse_dt(args.end)

    series = svc.load_historical(args.instrument, args.interval, start, end)
    candles = list(series.candles or [])

    if len(candles) < args.seq_len + 10:
        raise SystemExit(f"Not enough candles: {len(candles)}")

    xs = []
    ys = []

    seq_len = int(args.seq_len)
    stride = max(1, int(args.stride))

    for i in range(seq_len, len(candles) + 1, stride):
        window = candles[i - seq_len : i]
        y, _ = label_window(window, confidence_threshold=float(args.label_threshold))
        x = featurize_window(window)
        if not x:
            continue
        xs.append(x)
        ys.append(y)

    if not xs:
        raise SystemExit("No samples built")

    x_t = torch.tensor(xs, dtype=torch.float32)
    y_t = torch.tensor(ys, dtype=torch.float32)

    cfg = PatternSeqConfig(seq_len=seq_len)
    model = PatternSeqNet(cfg.feat_dim, len(PATTERN_NAMES), hidden=cfg.hidden)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # Simple train loop (CPU by default)
    model.train()
    for epoch in range(1, int(args.epochs) + 1):
        opt.zero_grad()
        logits = model(x_t)
        loss = F.binary_cross_entropy_with_logits(logits, y_t)
        loss.backward()
        opt.step()
        print(f"epoch {epoch}/{args.epochs} loss={float(loss):.4f} samples={x_t.shape[0]}")

    model.eval()
    save_model(str(args.out), model, cfg=cfg)
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
