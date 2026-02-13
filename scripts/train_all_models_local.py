from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
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
            resume=False,
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
            resume=False,
        )
        done_work += 1

    _print_progress("[deep]", start_ts=start_ts, progress=1.0, msg="done")


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_json(path: str | None, obj: dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def train_deep_nse_eq_budgeted(
    *,
    epochs: int,
    epochs_per_run: int | None,
    max_minutes: float | None,
    seq_len: int,
    batch_size: int,
    min_samples: int,
    max_symbols: int,
    after: str | None,
    page_size: int,
    resume: bool,
    checkpoint_dir: str | None,
    state_file: str | None,
    train_long: bool,
    train_intraday: bool,
) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; deep NSE_EQ training requires GPU")

    state = _load_json(state_file) or {}
    effective_after = (after or "").strip() or None
    if effective_after is None:
        s_after = (state.get("nse_after") or "").strip()
        effective_after = s_after or None

    # Determine a per-run chunk size.
    # If max_symbols is set (>0), it caps the chunk size.
    effective_max_symbols = int(max_symbols)
    chunk_symbols = 0
    frac = float(state.get("symbols_fraction_per_run") or 0.0)
    if "symbols_fraction_per_run" in state:
        # Allow state file to override defaults if user wants.
        frac = float(state.get("symbols_fraction_per_run") or 0.0)
    if frac > 0.0 and frac < 1.0:
        try:
            uni = UniverseService()
            total = int(uni.count(prefix="NSE_EQ|"))
            if total > 0:
                chunk_symbols = max(1, int(math.ceil(total * frac)))
        except Exception:
            chunk_symbols = 0

    if chunk_symbols > 0:
        if effective_max_symbols > 0:
            effective_max_symbols = min(effective_max_symbols, int(chunk_symbols))
        else:
            effective_max_symbols = int(chunk_symbols)

    keys = _iter_nse_eq_keys(max_symbols=effective_max_symbols, after=effective_after, page_size=page_size)
    if not keys:
        print("[deep] No NSE_EQ keys; skipping")
        return

    start_ts = time.time()
    budget_seconds = float(max_minutes) * 60.0 if max_minutes is not None and float(max_minutes) > 0 else 0.0
    total_work = max(1, len(keys) * (int(bool(train_long)) + int(bool(train_intraday))))
    done_work = 0

    _print_progress(
        "[deep]",
        start_ts=start_ts,
        progress=0.0,
        msg=f"starting budgeted NSE_EQ deep training for {len(keys)} symbols (after={effective_after} max={effective_max_symbols or 'all'})",
    )

    for instrument_key in keys:
        def _map_progress(frac: float, message: str, metrics: dict | None = None) -> None:
            overall = (done_work + max(0.0, min(1.0, float(frac)))) / total_work
            _print_progress("[deep]", start_ts=start_ts, progress=overall, msg=f"{instrument_key}: {message}")

        # Time budget check (between symbols)
        if budget_seconds > 0 and (time.time() - start_ts) >= budget_seconds:
            print(f"[deep] Budget reached ({max_minutes} min). Saving cursor and exiting.")
            _save_json(
                state_file,
                {"nse_after": instrument_key, "updated_ts": int(_utc_now().timestamp()), "note": "budget_stop"},
            )
            return

        if bool(train_long):
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
                resume=bool(resume),
                checkpoint_dir=checkpoint_dir,
                epochs_per_run=int(epochs_per_run) if epochs_per_run is not None else None,
            )
            done_work += 1

        if bool(train_intraday):
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
                resume=bool(resume),
                checkpoint_dir=checkpoint_dir,
                epochs_per_run=int(epochs_per_run) if epochs_per_run is not None else None,
            )
            done_work += 1

        # Update cursor after completing the symbol
        _save_json(state_file, {"nse_after": instrument_key, "updated_ts": int(_utc_now().timestamp())})

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
    p.add_argument("--exit-after-deep", action=argparse.BooleanOptionalAction, default=False, help="Exit after deep stage completes (useful for chunked runs).")

    p.add_argument(
        "--ridge-universe",
        default="default",
        choices=["default", "nse_eq"],
        help="Ridge batch universe: 'default' uses DEFAULT_UNIVERSE; 'nse_eq' iterates instrument_meta keys with prefix NSE_EQ|",
    )
    p.add_argument("--ridge-min-samples", type=int, default=200)
    p.add_argument("--ridge-use-presets", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--deep-epochs", type=int, default=6)
    p.add_argument("--deep-epochs-per-run", type=int, default=1, help="How many epochs to run per call (resume-friendly).")
    p.add_argument("--deep-resume", action=argparse.BooleanOptionalAction, default=True, help="Resume from checkpoints/DB if available.")
    p.add_argument("--deep-checkpoint-dir", default=None, help="Directory to store deep training checkpoints (default: next to DB).")
    p.add_argument("--deep-max-minutes", type=float, default=0.0, help="Stop deep training after this many minutes (0 disables).")
    p.add_argument("--deep-state-file", default=None, help="JSON file to persist NSE cursor between runs.")
    p.add_argument("--deep-symbol-fraction-per-run", type=float, default=0.0, help="Train only this fraction of NSE_EQ symbols per run (e.g., 0.05 for 5%%).")
    p.add_argument("--deep-train-long", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deep-train-intraday", action=argparse.BooleanOptionalAction, default=True)
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
    print("RIDGE_UNIVERSE:", str(args.ridge_universe))
    print("NSE_AFTER:", args.nse_after)
    print("NSE_MAX_SYMBOLS:", int(args.nse_max_symbols))
    if float(args.deep_max_minutes or 0.0) > 0:
        print("DEEP_MAX_MINUTES:", float(args.deep_max_minutes))
    print("DEEP_EPOCHS_PER_RUN:", int(args.deep_epochs_per_run))
    print("DEEP_RESUME:", bool(args.deep_resume))

    if bool(args.run_ridge_batch):
        if str(args.ridge_universe).lower().strip() == "nse_eq":
            keys = _iter_nse_eq_keys(max_symbols=int(args.nse_max_symbols), after=args.nse_after, page_size=int(args.nse_page_size))
            if not keys:
                print("[ridge] No NSE_EQ keys; skipping")
            else:
                # Reuse ridge trainer by temporarily overriding DEFAULT_UNIVERSE selection.
                # This avoids duplicating the ridge training loop.
                orig = settings.DEFAULT_UNIVERSE
                try:
                    object.__setattr__(settings, "DEFAULT_UNIVERSE", ",".join(keys))
                except Exception:
                    # pydantic BaseSettings may be frozen depending on config; fall back to env var.
                    os.environ["DEFAULT_UNIVERSE"] = ",".join(keys)
                try:
                    train_ridge_batch(min_samples=int(args.ridge_min_samples), use_presets=bool(args.ridge_use_presets))
                finally:
                    try:
                        object.__setattr__(settings, "DEFAULT_UNIVERSE", orig)
                    except Exception:
                        if "DEFAULT_UNIVERSE" in os.environ:
                            os.environ.pop("DEFAULT_UNIVERSE", None)
        else:
            train_ridge_batch(min_samples=int(args.ridge_min_samples), use_presets=bool(args.ridge_use_presets))

    if bool(args.run_deep_nse_eq):
        state_file = str(args.deep_state_file) if args.deep_state_file else str(Path(settings.DATABASE_PATH).parent / "checkpoints" / "deep" / "nse_eq_cursor.json")

        # Persist user knobs into state so repeated runs behave consistently.
        if float(args.deep_symbol_fraction_per_run or 0.0) > 0:
            _save_json(
                state_file,
                {
                    "symbols_fraction_per_run": float(args.deep_symbol_fraction_per_run),
                    "updated_ts": int(_utc_now().timestamp()),
                },
            )

        train_deep_nse_eq_budgeted(
            epochs=int(args.deep_epochs),
            epochs_per_run=int(args.deep_epochs_per_run) if int(args.deep_epochs_per_run) > 0 else None,
            max_minutes=float(args.deep_max_minutes) if float(args.deep_max_minutes or 0.0) > 0 else None,
            seq_len=int(args.deep_seq_len),
            batch_size=int(args.deep_batch_size),
            min_samples=int(args.deep_min_samples),
            max_symbols=int(args.nse_max_symbols),
            after=args.nse_after,
            page_size=int(args.nse_page_size),
            resume=bool(args.deep_resume),
            checkpoint_dir=str(args.deep_checkpoint_dir) if args.deep_checkpoint_dir else None,
            state_file=state_file,
            train_long=bool(args.deep_train_long),
            train_intraday=bool(args.deep_train_intraday),
        )

        if bool(args.exit_after_deep):
            return 0

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
