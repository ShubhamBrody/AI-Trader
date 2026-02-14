from __future__ import annotations

import argparse
import json
import sys
import time
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

# Allow running as a script: `python scripts/train_all_models_lightweight.py`
# by ensuring the repository root (parent of /scripts) is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.db import init_db
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


def _norm_suffix(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s[1:] if s.startswith("_") else s


def _model_family(base: str, suffix: str) -> str:
    suf = _norm_suffix(suffix)
    return f"{base}_{suf}" if suf else base


@dataclass(frozen=True)
class ProgressReporter:
    progress_file: str | None

    def _write(self, payload: dict[str, Any]) -> None:
        if not self.progress_file:
            return
        p = Path(self.progress_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                tmp.replace(p)
            except PermissionError:
                # Windows can deny atomic replace if the target is being watched/opened.
                # Fall back to direct write so training doesn't crash.
                p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                try:
                    if tmp.exists():
                        tmp.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            # Progress reporting should never be fatal.
            return

    def report(self, *, stage: str, progress: float, message: str, start_ts: float, extra: dict[str, Any] | None = None) -> None:
        p = max(0.0, min(1.0, float(progress)))
        elapsed = int(max(0.0, time.time() - float(start_ts)))
        eta = _eta_seconds(start_ts=start_ts, progress=p)
        line = f"[{stage:12s}] {p*100:6.2f}%  elapsed={_fmt_secs(elapsed)}  eta={_fmt_secs(eta)}  {message}"
        print(line)
        payload: dict[str, Any] = {
            "ts_utc": _utc_now().isoformat(),
            "stage": stage,
            "progress": p,
            "message": message,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
        }
        if extra:
            payload["extra"] = dict(extra)
        self._write(payload)


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
        page_any: Any = cast(Any, uni).list_keys_paged(prefix="NSE_EQ|", limit=int(page_size), after=cursor)
        page = cast(dict[str, Any], page_any)
        raw_keys = cast(list[Any], page.get("keys") or [])
        keys = [str(k) for k in raw_keys]
        cursor = cast(str | None, page.get("next_after"))
        if not keys:
            break
        for k in keys:
            out.append(k)
            if max_symbols and len(out) >= int(max_symbols):
                cursor = None
                break
        if cursor is None:
            break
    return out


def train_ridge_batch_lightweight(
    *,
    reporter: ProgressReporter,
    start_ts: float,
    universe: str,
    min_samples: int,
    data_fraction: float,
    model_family_suffix: str,
    nse_max_symbols: int,
    nse_after: str | None,
    nse_page_size: int,
    warm_fetch: bool,
    warm_intraday_sessions: int,
) -> None:
    from datetime import timedelta

    try:
        from app.candles.service import CandleService

        candles = CandleService()
    except Exception:
        candles = None

    if str(universe).lower().strip() == "nse_eq":
        keys = _iter_nse_eq_keys(max_symbols=int(nse_max_symbols), after=nse_after, page_size=int(nse_page_size))
    else:
        keys = _iter_default_universe()

    if not keys:
        reporter.report(stage="ridge", progress=1.0, message="no instruments; skipping", start_ts=start_ts)
        return

    total = max(1, len(keys) * 2)
    done = 0

    reporter.report(stage="ridge", progress=0.0, message=f"starting for {len(keys)} instruments (fraction={data_fraction:.2f})", start_ts=start_ts)

    for instrument_key in keys:
        # Best-effort warm fetch so training has data.
        if candles is not None and bool(warm_fetch):
            try:
                end = _utc_now()
                long_interval = str(settings.TRAIN_LONG_INTERVAL)
                intra_interval = str(settings.TRAIN_INTRADAY_INTERVAL)

                # Long interval: normal historical range fetch.
                start_long = end - timedelta(days=int(settings.TRAIN_LONG_LOOKBACK_DAYS))
                candles.load_historical(instrument_key, long_interval, start_long, end)

                # Intraday: Upstox can be picky about wide ranges; fetch by sessions.
                if intra_interval.endswith("m") or intra_interval.endswith("h"):
                    sessions = max(1, min(30, int(warm_intraday_sessions)))
                    candles.bulk_load_last_sessions(instrument_key, intra_interval, num_trading_sessions=int(sessions))
                else:
                    start_intra = end - timedelta(days=int(settings.TRAIN_INTRADAY_LOOKBACK_DAYS))
                    candles.load_historical(instrument_key, intra_interval, start_intra, end)
            except Exception:
                pass

        # long
        done += 1
        reporter.report(
            stage="ridge",
            progress=done / total,
            message=f"training long {instrument_key} ({settings.TRAIN_LONG_INTERVAL})",
            start_ts=start_ts,
        )
        out_long = train_model(
            instrument_key=instrument_key,
            interval=settings.TRAIN_LONG_INTERVAL,
            lookback_days=int(settings.TRAIN_LONG_LOOKBACK_DAYS),
            horizon_steps=int(settings.TRAIN_LONG_HORIZON_STEPS),
            model_family=_model_family("long", model_family_suffix),
            cap_tier=None,
            min_samples=int(min_samples),
            data_fraction=float(data_fraction),
        )
        if not bool(out_long.get("ok")):
            reporter.report(
                stage="ridge",
                progress=done / total,
                message=f"long {instrument_key}: FAILED ({out_long.get('reason') or 'unknown'})",
                start_ts=start_ts,
                extra={"out": out_long},
            )

        # intraday
        done += 1
        reporter.report(
            stage="ridge",
            progress=done / total,
            message=f"training intraday {instrument_key} ({settings.TRAIN_INTRADAY_INTERVAL})",
            start_ts=start_ts,
        )
        out_intra = train_model(
            instrument_key=instrument_key,
            interval=settings.TRAIN_INTRADAY_INTERVAL,
            lookback_days=int(settings.TRAIN_INTRADAY_LOOKBACK_DAYS),
            horizon_steps=int(settings.TRAIN_INTRADAY_HORIZON_STEPS),
            model_family=_model_family("intraday", model_family_suffix),
            cap_tier=None,
            min_samples=int(min_samples),
            data_fraction=float(data_fraction),
        )
        if not bool(out_intra.get("ok")):
            reporter.report(
                stage="ridge",
                progress=done / total,
                message=f"intraday {instrument_key}: FAILED ({out_intra.get('reason') or 'unknown'})",
                start_ts=start_ts,
                extra={"out": out_intra},
            )

    reporter.report(stage="ridge", progress=1.0, message="done", start_ts=start_ts)


def train_deep_nse_eq_lightweight(
    *,
    reporter: ProgressReporter,
    start_ts: float,
    epochs: int,
    seq_len: int,
    batch_size: int,
    min_samples: int,
    data_fraction: float,
    model_family_suffix: str,
    max_symbols: int,
    after: str | None,
    page_size: int,
    train_long: bool,
    train_intraday: bool,
    require_cuda: bool,
    resume: bool,
    checkpoint_dir: str | None,
    epochs_per_run: int | None,
    warm_fetch: bool,
    warm_intraday_sessions: int,
) -> None:
    try:
        import torch  # noqa: F401

        if bool(require_cuda):
            import torch as _torch

            if not bool(_torch.cuda.is_available()):
                reporter.report(stage="deep", progress=1.0, message="CUDA required but not available; skipping", start_ts=start_ts)
                return
    except Exception as e:
        reporter.report(stage="deep", progress=1.0, message=f"torch unavailable ({e}); skipping", start_ts=start_ts)
        return

    from datetime import timedelta

    try:
        from app.candles.service import CandleService

        candles = CandleService()
    except Exception:
        candles = None

    keys = _iter_nse_eq_keys(max_symbols=int(max_symbols), after=after, page_size=int(page_size))
    if not keys:
        reporter.report(stage="deep", progress=1.0, message="no NSE_EQ keys; skipping", start_ts=start_ts)
        return

    per_symbol = int(bool(train_long)) + int(bool(train_intraday))
    total_work = max(1, len(keys) * max(1, per_symbol))
    done_work = 0

    reporter.report(stage="deep", progress=0.0, message=f"starting for {len(keys)} symbols (fraction={data_fraction:.2f})", start_ts=start_ts)

    for instrument_key in keys:
        def _cb(frac: float, msg: str, metrics: dict[str, Any] | None = None) -> None:
            overall = (done_work + max(0.0, min(1.0, float(frac)))) / total_work
            reporter.report(stage="deep", progress=overall, message=f"{instrument_key}: {msg}", start_ts=start_ts)

        # Best-effort warm fetch so training has data.
        if candles is not None and bool(warm_fetch):
            try:
                end = _utc_now()
                long_interval = str(settings.TRAIN_LONG_INTERVAL)
                intra_interval = str(settings.TRAIN_INTRADAY_INTERVAL)

                start_long = end - timedelta(days=int(settings.TRAIN_LONG_LOOKBACK_DAYS))
                candles.load_historical(instrument_key, long_interval, start_long, end)

                if intra_interval.endswith("m") or intra_interval.endswith("h"):
                    sessions = max(1, min(30, int(warm_intraday_sessions)))
                    candles.bulk_load_last_sessions(instrument_key, intra_interval, num_trading_sessions=int(sessions))
                else:
                    start_intra = end - timedelta(days=int(settings.TRAIN_INTRADAY_LOOKBACK_DAYS))
                    candles.load_historical(instrument_key, intra_interval, start_intra, end)
            except Exception:
                pass

        if bool(train_long):
            reporter.report(stage="deep", progress=done_work / total_work, message=f"training long {instrument_key}", start_ts=start_ts)
            out_long = train_deep_model(
                instrument_key=instrument_key,
                interval=settings.TRAIN_LONG_INTERVAL,
                lookback_days=int(settings.TRAIN_LONG_LOOKBACK_DAYS),
                horizon_steps=int(settings.TRAIN_LONG_HORIZON_STEPS),
                model_family=_model_family("long", model_family_suffix),
                cap_tier=None,
                seq_len=int(seq_len),
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=2e-4,
                weight_decay=1e-4,
                min_samples=int(min_samples),
                data_fraction=float(data_fraction),
                require_cuda=bool(require_cuda),
                progress_cb=_cb,
                resume=bool(resume),
                checkpoint_dir=checkpoint_dir,
                epochs_per_run=(None if epochs_per_run is None else int(epochs_per_run)),
            )
            if not bool(out_long.get("ok")):
                reporter.report(
                    stage="deep",
                    progress=done_work / total_work,
                    message=f"long {instrument_key}: FAILED ({out_long.get('reason') or 'unknown'})",
                    start_ts=start_ts,
                    extra={"out": out_long},
                )
            done_work += 1

        if bool(train_intraday):
            reporter.report(stage="deep", progress=done_work / total_work, message=f"training intraday {instrument_key}", start_ts=start_ts)
            out_intra = train_deep_model(
                instrument_key=instrument_key,
                interval=settings.TRAIN_INTRADAY_INTERVAL,
                lookback_days=int(settings.TRAIN_INTRADAY_LOOKBACK_DAYS),
                horizon_steps=int(settings.TRAIN_INTRADAY_HORIZON_STEPS),
                model_family=_model_family("intraday", model_family_suffix),
                cap_tier=None,
                seq_len=int(seq_len),
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=2e-4,
                weight_decay=1e-4,
                min_samples=int(min_samples),
                data_fraction=float(data_fraction),
                require_cuda=bool(require_cuda),
                progress_cb=_cb,
                resume=bool(resume),
                checkpoint_dir=checkpoint_dir,
                epochs_per_run=(None if epochs_per_run is None else int(epochs_per_run)),
            )
            if not bool(out_intra.get("ok")):
                reporter.report(
                    stage="deep",
                    progress=done_work / total_work,
                    message=f"intraday {instrument_key}: FAILED ({out_intra.get('reason') or 'unknown'})",
                    start_ts=start_ts,
                    extra={"out": out_intra},
                )
            done_work += 1

    reporter.report(stage="deep", progress=1.0, message="done", start_ts=start_ts)


def train_pattern_seq_lightweight(
    *,
    reporter: ProgressReporter,
    start_ts: float,
    interval: str,
    lookback_days: int,
    seq_len: int,
    stride: int,
    epochs: int,
    batch_size: int,
    lr: float,
    label_threshold: float,
    data_fraction: float,
    max_symbols: int,
    after: str | None,
    page_size: int,
    out_path: str | None,
) -> None:
    keys = _iter_nse_eq_keys(max_symbols=int(max_symbols), after=after, page_size=int(page_size))
    if not keys:
        reporter.report(stage="pattern_seq", progress=1.0, message="no NSE_EQ keys; skipping", start_ts=start_ts)
        return

    frac = float(data_fraction)
    # Scale caps down to roughly match requested fraction.
    max_candles_per_symbol = max(200, int(round(5000 * frac)))
    max_windows_per_symbol = max(200, int(round(1500 * frac)))
    max_windows_total = max(2000, int(round(50000 * frac)))

    target_path = str(out_path or "data/models/pattern_seq_lightweight.pt")

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

    reporter.report(stage="pattern_seq", progress=0.0, message=f"starting for {len(keys)} symbols (caps scaled by {frac:.2f})", start_ts=start_ts)

    def _cb(frac2: float, msg: str, metrics: dict[str, Any] | None = None) -> None:
        reporter.report(stage="pattern_seq", progress=float(frac2), message=msg, start_ts=start_ts)

    out = train_pattern_seq_model_for_instruments(keys, out_path=target_path, params=params, progress_cb=_cb)
    if not out.get("ok"):
        reporter.report(
            stage="pattern_seq",
            progress=1.0,
            message=f"skipping: {str(out.get('reason') or out)}",
            start_ts=start_ts,
        )
        return

    reporter.report(stage="pattern_seq", progress=1.0, message=f"saved {target_path}", start_ts=start_ts)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Train lightweight (10%) variants of the models for fast demos + live trading")

    p.add_argument("--data-fraction", type=float, default=0.10, help="Fraction of supervised samples/windows to use (0<frac<=1).")
    p.add_argument("--model-family-suffix", default="lightweight", help="Appended to model_family (long -> long_lightweight).")

    p.add_argument("--progress-file", default="data/training_progress_lightweight.json", help="Optional JSON file updated during training (set empty to disable).")

    p.add_argument("--all", action=argparse.BooleanOptionalAction, default=False, help="Train ridge + deep + pattern-seq (where available)")

    p.add_argument("--warm-fetch", action=argparse.BooleanOptionalAction, default=True, help="Best-effort fetch candles before training to populate DB")
    p.add_argument(
        "--warm-intraday-sessions",
        type=int,
        default=2,
        help="When warming intraday data, fetch this many recent trading sessions per symbol (kept small to avoid rate limits)",
    )

    p.add_argument("--run-ridge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-deep", action=argparse.BooleanOptionalAction, default=False, help="Deep training (torch). Default off.")
    p.add_argument("--run-pattern-seq", action=argparse.BooleanOptionalAction, default=False, help="Pattern sequence model training. Default off.")

    p.add_argument("--ridge-universe", default="default", choices=["default", "nse_eq"], help="Which universe to iterate for ridge.")
    p.add_argument("--ridge-min-samples", type=int, default=200)

    p.add_argument("--nse-max-symbols", type=int, default=50, help="Max NSE_EQ symbols to iterate (0 means all)")
    p.add_argument("--nse-after", default=None)
    p.add_argument("--nse-page-size", type=int, default=500)

    p.add_argument("--deep-epochs", type=int, default=2)
    p.add_argument("--deep-epochs-per-run", type=int, default=1)
    p.add_argument("--deep-resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deep-checkpoint-dir", default=None)
    p.add_argument("--deep-require-cuda", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--deep-seq-len", type=int, default=120)
    p.add_argument("--deep-batch-size", type=int, default=128)
    p.add_argument("--deep-min-samples", type=int, default=500)
    p.add_argument("--deep-train-long", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deep-train-intraday", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--ps-interval", default="1m")
    p.add_argument("--ps-lookback-days", type=int, default=30)
    p.add_argument("--ps-seq-len", type=int, default=64)
    p.add_argument("--ps-stride", type=int, default=2)
    p.add_argument("--ps-epochs", type=int, default=2)
    p.add_argument("--ps-batch-size", type=int, default=256)
    p.add_argument("--ps-lr", type=float, default=1e-3)
    p.add_argument("--ps-label-threshold", type=float, default=0.35)
    p.add_argument("--ps-out", default=None)

    args = p.parse_args(argv)

    if bool(args.all):
        args.run_ridge = True
        args.run_deep = True
        args.run_pattern_seq = True

    frac = float(args.data_fraction)
    if not (0.0 < frac <= 1.0):
        print("ERROR: --data-fraction must satisfy 0 < frac <= 1")
        return 2

    init_db()

    progress_file = str(args.progress_file).strip() or None
    reporter = ProgressReporter(progress_file=progress_file)
    start_ts = time.time()

    print("Training start (UTC):", _utc_now().isoformat())
    print("DATABASE_PATH:", settings.DATABASE_PATH)
    print("data_fraction:", frac)
    print("model_family_suffix:", str(args.model_family_suffix))
    if progress_file:
        print("progress_file:", progress_file)

    if bool(args.run_ridge):
        train_ridge_batch_lightweight(
            reporter=reporter,
            start_ts=start_ts,
            universe=str(args.ridge_universe),
            min_samples=int(args.ridge_min_samples),
            data_fraction=frac,
            model_family_suffix=str(args.model_family_suffix),
            nse_max_symbols=int(args.nse_max_symbols),
            nse_after=(None if args.nse_after is None else str(args.nse_after)),
            nse_page_size=int(args.nse_page_size),
            warm_fetch=bool(args.warm_fetch),
            warm_intraday_sessions=int(args.warm_intraday_sessions),
        )

    if bool(args.run_deep):
        train_deep_nse_eq_lightweight(
            reporter=reporter,
            start_ts=start_ts,
            epochs=int(args.deep_epochs),
            seq_len=int(args.deep_seq_len),
            batch_size=int(args.deep_batch_size),
            min_samples=int(args.deep_min_samples),
            data_fraction=frac,
            model_family_suffix=str(args.model_family_suffix),
            max_symbols=int(args.nse_max_symbols),
            after=(None if args.nse_after is None else str(args.nse_after)),
            page_size=int(args.nse_page_size),
            train_long=bool(args.deep_train_long),
            train_intraday=bool(args.deep_train_intraday),
            require_cuda=bool(args.deep_require_cuda),
            resume=bool(args.deep_resume),
            checkpoint_dir=(None if not args.deep_checkpoint_dir else str(args.deep_checkpoint_dir)),
            epochs_per_run=(None if args.deep_epochs_per_run is None else int(args.deep_epochs_per_run)),
            warm_fetch=bool(args.warm_fetch),
            warm_intraday_sessions=int(args.warm_intraday_sessions),
        )

    if bool(args.run_pattern_seq):
        train_pattern_seq_lightweight(
            reporter=reporter,
            start_ts=start_ts,
            interval=str(args.ps_interval),
            lookback_days=int(args.ps_lookback_days),
            seq_len=int(args.ps_seq_len),
            stride=int(args.ps_stride),
            epochs=int(args.ps_epochs),
            batch_size=int(args.ps_batch_size),
            lr=float(args.ps_lr),
            label_threshold=float(args.ps_label_threshold),
            data_fraction=frac,
            max_symbols=int(args.nse_max_symbols),
            after=(None if args.nse_after is None else str(args.nse_after)),
            page_size=int(args.nse_page_size),
            out_path=(None if not args.ps_out else str(args.ps_out)),
        )

    reporter.report(stage="done", progress=1.0, message="all requested trainings finished", start_ts=start_ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
