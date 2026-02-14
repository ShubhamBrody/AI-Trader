from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.candles.models import Candle
from app.candles.persistence_sql import get_candles, upsert_candles
from app.candles.service import CandleService
from app.core.db import init_db
from app.learning.service import train_model


def _interval_seconds(interval: str) -> int:
    interval = str(interval)
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    if interval.endswith("h"):
        return int(interval[:-1]) * 3600
    if interval.endswith("d"):
        return int(interval[:-1]) * 86400
    return 60


def _count_db_candles(instrument_key: str, interval: str, start: datetime, end: datetime) -> int:
    rows = get_candles(str(instrument_key), str(interval), int(start.timestamp()), int(end.timestamp()))
    return int(len(rows))


def _generate_synthetic_candles(
    *,
    instrument_key: str,
    interval: str,
    start: datetime,
    end: datetime,
    seed: int,
    start_price: float,
    drift_per_step: float,
    vol_per_step: float,
) -> list[Candle]:
    step = _interval_seconds(interval)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    if end_ts <= start_ts:
        return []

    rng = np.random.default_rng(int(seed))

    candles: list[Candle] = []
    price = float(max(1e-6, start_price))
    ts = start_ts - (start_ts % step)

    while ts <= end_ts:
        # Log-return random walk.
        ret = float(drift_per_step + vol_per_step * rng.standard_normal())
        next_close = float(max(1e-6, price * (1.0 + ret)))

        o = float(price)
        c = float(next_close)
        # Add some intrabar range.
        spread = float(abs(c - o) + (abs(o) * 0.001 + abs(c) * 0.001))
        hi = float(max(o, c) + spread * float(rng.uniform(0.2, 0.8)))
        lo = float(min(o, c) - spread * float(rng.uniform(0.2, 0.8)))
        v = float(max(1.0, rng.lognormal(mean=10.0, sigma=0.25)))

        candles.append(Candle(ts=int(ts), open=o, high=hi, low=lo, close=c, volume=v))

        price = c
        ts += step

    return candles


@dataclass(frozen=True)
class TrainQuickArgs:
    instrument: str
    interval: str
    lookback_days: int
    horizon_steps: int
    min_samples: int
    model_family: str | None
    cap_tier: str | None
    l2: float
    warm_fetch: bool
    allow_synthetic: bool
    synthetic_seed: int


def _parse_args() -> TrainQuickArgs:
    p = argparse.ArgumentParser(
        description=(
            "Quickly train the lightweight ridge model used by AIEngine (no torch, no sklearn). "
            "This is meant to be fast so you can see AI-driven signals immediately."
        )
    )
    p.add_argument("--instrument", required=True, help="Instrument key or symbol query (e.g. NSE_EQ|INE002A01018 or RELIANCE or NIFTY)")
    p.add_argument("--interval", default="1m", help="Candle interval (e.g. 1m, 5m, 1d)")
    p.add_argument("--lookback-days", type=int, default=7, help="Training lookback in days")
    p.add_argument("--horizon-steps", type=int, default=12, help="Forecast horizon steps (e.g. 12 for 12m if interval=1m)")
    p.add_argument("--min-samples", type=int, default=120, help="Minimum supervised samples required")
    p.add_argument("--model-family", default="intraday", help="Model family label stored in DB (intraday/long/etc)")
    p.add_argument("--cap-tier", default=None, help="Optional cap tier override (large/mid/small/unknown)")
    p.add_argument("--l2", type=float, default=1e-2, help="Ridge L2 regularization")

    p.add_argument("--warm-fetch", action=argparse.BooleanOptionalAction, default=True, help="Try to fetch candles via CandleService before training")
    p.add_argument("--allow-synthetic", action=argparse.BooleanOptionalAction, default=True, help="If DB has insufficient candles, generate synthetic candles and train anyway")
    p.add_argument("--synthetic-seed", type=int, default=7, help="Seed for synthetic candles")

    a = p.parse_args()
    return TrainQuickArgs(
        instrument=str(a.instrument),
        interval=str(a.interval),
        lookback_days=int(a.lookback_days),
        horizon_steps=int(a.horizon_steps),
        min_samples=int(a.min_samples),
        model_family=str(a.model_family) if a.model_family else None,
        cap_tier=str(a.cap_tier) if a.cap_tier else None,
        l2=float(a.l2),
        warm_fetch=bool(a.warm_fetch),
        allow_synthetic=bool(a.allow_synthetic),
        synthetic_seed=int(a.synthetic_seed),
    )


def main() -> None:
    args = _parse_args()

    init_db()

    service = CandleService()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(args.lookback_days))
    train_lookback_days = int(args.lookback_days)

    if args.warm_fetch:
        # Best-effort: fetch from Upstox (if configured). If it fails, CandleService falls back to DB.
        try:
            service.load_historical(args.instrument, args.interval, start, end)
        except Exception:
            pass

    instrument_key = service._resolve_instrument_key(args.instrument)  # type: ignore[attr-defined]

    have = _count_db_candles(instrument_key, args.interval, start, end)
    print(f"DB candles in range: {have} ({instrument_key=} {args.interval=} {args.lookback_days=})")

    # Training requires at least max(60, min_samples) closes, plus horizon.
    needed = max(60, int(args.min_samples))
    if have < needed and args.allow_synthetic:
        # Create a reasonably sized synthetic series matching the interval.
        step = _interval_seconds(args.interval)
        approx_steps = int((end.timestamp() - start.timestamp()) // max(1, step))
        # Ensure we exceed needed by margin.
        target_steps = max(needed + int(args.horizon_steps) + 50, approx_steps)
        synth_start = end - timedelta(seconds=int(target_steps * step))

        # Ensure training window is large enough to include the synthetic candles.
        seconds_back = int((end - synth_start).total_seconds())
        implied_days = max(1, int(seconds_back // 86400) + 1)
        train_lookback_days = max(int(train_lookback_days), int(implied_days))

        candles = _generate_synthetic_candles(
            instrument_key=str(instrument_key),
            interval=str(args.interval),
            start=synth_start,
            end=end,
            seed=int(args.synthetic_seed),
            start_price=200.0,
            drift_per_step=0.0001,
            vol_per_step=0.0025,
        )
        inserted = upsert_candles(str(instrument_key), str(args.interval), candles)
        have = _count_db_candles(instrument_key, args.interval, synth_start, end)
        print(f"Inserted synthetic candles: {inserted}; now have: {have}")

    out: dict[str, Any] = train_model(
        instrument_key=str(instrument_key),
        interval=str(args.interval),
        lookback_days=int(train_lookback_days),
        horizon_steps=int(args.horizon_steps),
        model_family=args.model_family,
        cap_tier=args.cap_tier,
        l2=float(args.l2),
        min_samples=int(args.min_samples),
    )

    if not out.get("ok"):
        print("TRAIN FAILED")
        print(out)
        raise SystemExit(2)

    print("TRAIN OK")
    print(out)

    print("\nNext: verify via API")
    print(f"- GET /api/learning/status?instrument_key={instrument_key}&interval={args.interval}&horizon_steps={args.horizon_steps}")
    print(f"- GET /api/learning/predict?instrument_key={instrument_key}&interval={args.interval}&horizon_steps={args.horizon_steps}&lookback=60")
    print(f"- GET /api/recommendations/statistical?instrument_key={instrument_key}&interval={args.interval}&lookback_days={max(1, train_lookback_days)}&horizon_steps={args.horizon_steps}")


if __name__ == "__main__":
    main()
