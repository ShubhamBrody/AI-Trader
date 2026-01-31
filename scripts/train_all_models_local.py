from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterable

from app.core.settings import settings
from app.learning.deep_service import train_deep_model
from app.learning.pattern_seq_trainer import TrainPatternSeqParams, train_pattern_seq_model_for_instruments
from app.learning.service import train_model
from app.universe.service import UniverseService


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_secs(v: Any) -> str:
    try:
        v = int(v)
    except Exception:
        return "--"
    if v < 0:
        return "--"
    h = v // 3600
    m = (v % 3600) // 60
    s = v % 60
    if h > 0:
        return f"{h:d}h{m:02d}m"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _eta_seconds(*, start_ts: float, progress: float) -> int | None:
    p = float(progress)
    if not (0.0 < p < 1.0):
        return None
    elapsed = max(0.0, time.time() - float(start_ts))
    if elapsed < 2.0:
        return None
    eta = int(round(elapsed * (1.0 - p) / max(p, 1e-6)))
    return max(0, min(eta, 60 * 60 * 24 * 30))


def _print_progress(prefix: str, *, start_ts: float, progress: float, msg: str) -> None:
    eta = _eta_seconds(start_ts=start_ts, progress=progress)
    elapsed = int(max(0.0, time.time() - float(start_ts)))
    print(f"{prefix} {progress*100:6.2f}%  elapsed={_fmt_secs(elapsed)}  eta={_fmt_secs(eta)}  {msg}")


def _iter_default_universe() -> list[str]:
    keys = [k.strip() for k in str(settings.DEFAULT_UNIVERSE or "").split(",") if k.strip()]
    return keys


def _iter_nse_eq_keys(*, max_symbols: int, after: str | None, page_size: int) -> list[str]:
    uni = UniverseService()
    total = uni.count(prefix="NSE_EQ|")
    if total <= 0:
        raise RuntimeError("instrument_meta empty; import universe first or upload DB with instrument_meta")

    out: list[str] = []
    cursor = (after or "").strip() or None
    while True:
        page = uni.list_keys_paged(prefix="NSE_EQ|", limit=int(page_size), after=cursor)
        keys = list(page.get("keys") or [])
        cursor = page.get("next_after")
        if not keys:
            break
        for k in keys:
            out.append(str(k))
            if max_symbols and len(out) >= int(max_symbols):
                cursor = None
                break
        if cursor is None:
            break
    return out


def train_ridge_batch(*, min_samples: int, use_presets: bool) -> None:
    keys = _iter_default_universe()
    if not keys:
        print("[ridge] No DEFAULT_UNIVERSE configured; skipping")
        return

    start_ts = time.time()
    total = max(1, len(keys) * (2 if use_presets else 1))
    done = 0

    _print_progress("[ridge]", start_ts=start_ts, progress=0.0, msg=f"starting batch for {len(keys)} instruments")

    for instrument_key in keys:
        if use_presets:
            # long
            done += 1
            _print_progress(
                "[ridge]",
                start_ts=start_ts,
                progress=done / total,
                msg=f"training long {instrument_key} ({settings.TRAIN_LONG_INTERVAL})",
            )
            train_model(
                instrument_key=instrument_key,
                interval=settings.TRAIN_LONG_INTERVAL,
                lookback_days=int(settings.TRAIN_LONG_LOOKBACK_DAYS),
                horizon_steps=int(settings.TRAIN_LONG_HORIZON_STEPS),
                model_family="long",
                cap_tier=None,
                min_samples=int(min_samples),
            )

            # intraday
            done += 1
            _print_progress(
                "[ridge]",
                start_ts=start_ts,
                progress=done / total,
                msg=f"training intraday {instrument_key} ({settings.TRAIN_INTRADAY_INTERVAL})",
            )
            train_model(
                instrument_key=instrument_key,
                interval=settings.TRAIN_INTRADAY_INTERVAL,
                lookback_days=int(settings.TRAIN_INTRADAY_LOOKBACK_DAYS),
                horizon_steps=int(settings.TRAIN_INTRADAY_HORIZON_STEPS),
                model_family="intraday",
                cap_tier=None,
                min_samples=int(min_samples),
            )
        else:
            done += 1
            _print_progress("[ridge]", start_ts=start_ts, progress=done / total, msg=f"training {instrument_key}")
            train_model(
                instrument_key=instrument_key,
                interval=settings.TRAIN_LONG_INTERVAL,
                lookback_days=int(settings.TRAIN_LONG_LOOKBACK_DAYS),
                horizon_steps=int(settings.TRAIN_LONG_HORIZON_STEPS),
                model_family=None,
                cap_tier=None,
                min_samples=int(min_samples),
            )

    _print_progress("[ridge]", start_ts=start_ts, progress=1.0, msg="done")


def train_deep_nse_eq(*, epochs: int, seq_len: int, batch_size: int, min_samples: int, max_symbols: int, after: str | None, page_size: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; deep NSE_EQ training requires GPU")

    keys = _iter_nse_eq_keys(max_symbols=max_symbols, after=after, page_size=page_size)
    if not keys:
        print("[deep] No NSE_EQ keys; skipping")
        return

    start_ts = time.time()
    total_work = max(1, len(keys) * 2)
    done_work = 0

    _print_progress("[deep]", start_ts=start_ts, progress=0.0, msg=f"starting NSE_EQ deep training for {len(keys)} symbols")

    for instrument_key in keys:
        def _map_progress(frac: float, message: str, metrics: dict | None = None) -> None:
            overall = (done_work + max(0.0, min(1.0, float(frac)))) / total_work
            _print_progress("[deep]", start_ts=start_ts, progress=overall, msg=f"{instrument_key}: {message}")

        # long
        _print_progress("[deep]", start_ts=start_ts, progress=done_work / total_work, msg=f"training long {instrument_key}")
        train_deep_model(
            instrument_key=instrument_key,
            interval=settings.TRAIN_LONG_INTERVAL,
            lookback_days=int(settings.TRAIN_LONG_LOOKBACK_DAYS),
            horizon_steps=int(settings.TRAIN_LONG_HORIZON_STEPS),
            model_family="long",
            cap_tier=None,
            seq_len=int(seq_len),
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=2e-4,
            weight_decay=1e-4,
            min_samples=int(min_samples),
            require_cuda=True,
            progress_cb=_map_progress,
        )
        done_work += 1

        # intraday
        _print_progress("[deep]", start_ts=start_ts, progress=done_work / total_work, msg=f"training intraday {instrument_key}")
        train_deep_model(
            instrument_key=instrument_key,
            interval=settings.TRAIN_INTRADAY_INTERVAL,
            lookback_days=int(settings.TRAIN_INTRADAY_LOOKBACK_DAYS),
            horizon_steps=int(settings.TRAIN_INTRADAY_HORIZON_STEPS),
            model_family="intraday",
            cap_tier=None,
            seq_len=int(seq_len),
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=2e-4,
            weight_decay=1e-4,
            min_samples=int(min_samples),
            require_cuda=True,
            progress_cb=_map_progress,
        )
        done_work += 1

    _print_progress("[deep]", start_ts=start_ts, progress=1.0, msg="done")


def train_pattern_seq_nse_eq(
    *,
    interval: str,
    lookback_days: int,
    seq_len: int,
    stride: int,
    epochs: int,
    batch_size: int,
    lr: float,
    label_threshold: float,
    max_candles_per_symbol: int,
    max_windows_per_symbol: int,
    max_windows_total: int,
    max_symbols: int,
    after: str | None,
    page_size: int,
    out_path: str | None,
) -> None:
    keys = _iter_nse_eq_keys(max_symbols=max_symbols, after=after, page_size=page_size)
    if not keys:
        print("[pattern-seq] No NSE_EQ keys; skipping")
        return

    target_path = str(out_path or getattr(settings, "PATTERN_SEQ_MODEL_PATH", "data/models/pattern_seq.pt"))

    params = TrainPatternSeqParams(
        interval=str(interval),
        lookback_days=int(lookback_days),
        seq_len=int(seq_len),
        stride=int(stride),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        label_threshold=float(label_threshold),
        max_candles_per_symbol=int(max_candles_per_symbol),
        max_windows_per_symbol=int(max_windows_per_symbol),
        max_windows_total=int(max_windows_total),
    )

    start_ts = time.time()

    def _cb(frac: float, msg: str, metrics: dict[str, Any] | None = None) -> None:
        _print_progress("[pattern-seq]", start_ts=start_ts, progress=float(frac), msg=msg)

    out = train_pattern_seq_model_for_instruments(keys, out_path=target_path, params=params, progress_cb=_cb)
    if not out.get("ok"):
        raise RuntimeError(str(out.get("reason") or out))

    _print_progress("[pattern-seq]", start_ts=start_ts, progress=1.0, msg=f"saved {target_path}")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Train all models locally (Colab-friendly; no API server required)")

    p.add_argument("--run-ridge-batch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-deep-nse-eq", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-pattern-seq", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--ridge-min-samples", type=int, default=200)
    p.add_argument("--ridge-use-presets", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--deep-epochs", type=int, default=6)
    p.add_argument("--deep-seq-len", type=int, default=120)
    p.add_argument("--deep-batch-size", type=int, default=128)
    p.add_argument("--deep-min-samples", type=int, default=500)

    p.add_argument("--nse-max-symbols", type=int, default=0)
    p.add_argument("--nse-after", default=None)
    p.add_argument("--nse-page-size", type=int, default=500)

    p.add_argument("--ps-interval", default="1m")
    p.add_argument("--ps-lookback-days", type=int, default=30)
    p.add_argument("--ps-seq-len", type=int, default=64)
    p.add_argument("--ps-stride", type=int, default=2)
    p.add_argument("--ps-epochs", type=int, default=3)
    p.add_argument("--ps-batch-size", type=int, default=256)
    p.add_argument("--ps-lr", type=float, default=1e-3)
    p.add_argument("--ps-label-threshold", type=float, default=0.35)
    p.add_argument("--ps-max-candles-per-symbol", type=int, default=5000)
    p.add_argument("--ps-max-windows-per-symbol", type=int, default=1500)
    p.add_argument("--ps-max-windows-total", type=int, default=50000)
    p.add_argument("--ps-out", default=None)

    args = p.parse_args(argv)

    print("Training start (UTC):", _utc_now().isoformat())
    print("DATABASE_PATH:", settings.DATABASE_PATH)
    print("PATTERN_SEQ_MODEL_PATH:", getattr(settings, "PATTERN_SEQ_MODEL_PATH", "data/models/pattern_seq.pt"))

    if bool(args.run_ridge_batch):
        train_ridge_batch(min_samples=int(args.ridge_min_samples), use_presets=bool(args.ridge_use_presets))

    if bool(args.run_deep_nse_eq):
        train_deep_nse_eq(
            epochs=int(args.deep_epochs),
            seq_len=int(args.deep_seq_len),
            batch_size=int(args.deep_batch_size),
            min_samples=int(args.deep_min_samples),
            max_symbols=int(args.nse_max_symbols),
            after=args.nse_after,
            page_size=int(args.nse_page_size),
        )

    if bool(args.run_pattern_seq):
        train_pattern_seq_nse_eq(
            interval=str(args.ps_interval),
            lookback_days=int(args.ps_lookback_days),
            seq_len=int(args.ps_seq_len),
            stride=int(args.ps_stride),
            epochs=int(args.ps_epochs),
            batch_size=int(args.ps_batch_size),
            lr=float(args.ps_lr),
            label_threshold=float(args.ps_label_threshold),
            max_candles_per_symbol=int(args.ps_max_candles_per_symbol),
            max_windows_per_symbol=int(args.ps_max_windows_per_symbol),
            max_windows_total=int(args.ps_max_windows_total),
            max_symbols=int(args.nse_max_symbols),
            after=args.nse_after,
            page_size=int(args.nse_page_size),
            out_path=args.ps_out,
        )

    print("\nArtifacts to download:")
    print("- DB (ridge+deep models live inside):", settings.DATABASE_PATH)
    print("- Pattern sequence model file:", getattr(settings, "PATTERN_SEQ_MODEL_PATH", "data/models/pattern_seq.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
