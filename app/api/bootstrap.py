from __future__ import annotations

from datetime import datetime, timezone

import gzip
import json
import time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.audit import log_event
from app.core.settings import settings
from app.data.pipeline import make_backfill_plan, run_backfill
from app.learning.jobs import JOBS as TRAINING_JOBS
from app.learning.jobs import TrainingJob, new_job_id
from app.learning.deep_service import train_deep_model
from app.learning.service import train_model
from app.universe.service import UniverseService


router = APIRouter(prefix="/bootstrap", tags=["bootstrap"])


class FirstRunRequest(BaseModel):
    instrument_keys: list[str] | None = None
    cap_tier: str | None = None

    intervals: list[str] = Field(default_factory=lambda: ["1d", "1m"])
    lookback_days_daily: int = Field(default=365, ge=30, le=3650)
    lookback_days_intraday: int = Field(default=30, ge=5, le=365)
    resume_from_db: bool = True

    # Training knobs
    train_min_samples: int = Field(default=200, ge=50, le=5000)
    train_lookback_days_long: int | None = Field(default=None, ge=30, le=3650)
    train_lookback_days_intraday: int | None = Field(default=None, ge=5, le=365)


class NseEqFullDeepTrainRequest(BaseModel):
    # Universe import
    skip_universe_import: bool = False
    import_limit: int = Field(default=0, ge=0)

    # Iteration/resume
    after: str | None = None
    page_size: int = Field(default=200, ge=50, le=2000)
    max_symbols: int = Field(default=0, ge=0)  # 0 means all

    # Backfill
    backfill_daily: bool = True
    backfill_intraday: bool = True
    lookback_days_daily: int = Field(default=1460, ge=30, le=3650)  # ~4y
    lookback_days_intraday: int = Field(default=95, ge=5, le=365)  # ~3 months trading days
    resume_from_db: bool = True

    # Deep training (GPU-only)
    train_long: bool = True
    train_intraday: bool = True
    seq_len: int = Field(default=120, ge=30, le=1000)
    epochs_long: int = Field(default=6, ge=1, le=100)
    epochs_intraday: int = Field(default=6, ge=1, le=100)
    batch_size: int = Field(default=128, ge=16, le=2048)
    min_samples: int = Field(default=500, ge=50, le=50000)

    # Throttling (important for universe-scale Upstox calls)
    sleep_seconds_per_chunk: float = Field(default=0.0, ge=0.0, le=5.0)
    sleep_seconds_per_symbol: float = Field(default=0.0, ge=0.0, le=10.0)


def _import_upstox_nse_eq(*, limit: int = 0) -> dict:
    url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    try:
        r = httpx.get(url, timeout=60.0)
        r.raise_for_status()
        raw = gzip.decompress(r.content)
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as e:
        raise HTTPException(status_code=502, detail={"ok": False, "reason": "failed to fetch/parse instruments", "error": str(e)})

    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail={"ok": False, "reason": "unexpected instruments format"})

    items: list[dict] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        if str(it.get("segment") or "") != "NSE_EQ":
            continue
        if str(it.get("instrument_type") or "") != "EQ":
            continue

        instrument_key = str(it.get("instrument_key") or "").strip()
        if not instrument_key.startswith("NSE_EQ|"):
            continue

        items.append(
            {
                "instrument_key": instrument_key,
                "tradingsymbol": (str(it.get("trading_symbol") or "").strip() or None),
                "cap_tier": "unknown",
                "upstox_token": (str(it.get("exchange_token") or "").strip() or None),
            }
        )
        if int(limit) > 0 and len(items) >= int(limit):
            break

    svc = UniverseService()
    res = svc.bulk_import(items)
    return {"ok": True, "source": url, "parsed": len(items), **res}


def _select_instruments(req: FirstRunRequest) -> list[str]:
    if req.instrument_keys:
        keys = [str(k).strip() for k in req.instrument_keys if str(k).strip()]
        return keys

    uni = UniverseService()

    # If a tier is requested, prefer the DB universe.
    if req.cap_tier:
        items = uni.list(cap_tier=req.cap_tier, limit=2000)
        keys = [str(i["instrument_key"]) for i in items if i.get("instrument_key")]
        if keys:
            return keys

    # Otherwise, if any universe exists in DB, use it.
    items = uni.list(limit=2000)
    keys = [str(i["instrument_key"]) for i in items if i.get("instrument_key")]
    if keys:
        return keys

    # Final fallback: DEFAULT_UNIVERSE env setting.
    keys = [k.strip() for k in str(settings.DEFAULT_UNIVERSE or "").split(",") if k.strip()]
    return keys


@router.post("/first-run-async")
def first_run_async(req: FirstRunRequest) -> dict:
    """One-shot first-time bootstrap.

    Runs:
    1) Candle backfill (default 1d + 1m)
    2) Ridge training for long + intraday presets

    Progress is written to the TrainingJob system so you can watch via:
    - GET /api/learning/jobs/{job_id}
    - WS  /ws/training
    """

    keys = _select_instruments(req)
    if not keys:
        return {"ok": True, "job_id": None, "detail": "no instruments"}

    intervals = [str(i).strip() for i in (req.intervals or []) if str(i).strip()]
    if not intervals:
        intervals = ["1d", "1m"]

    job_id = new_job_id()
    job = TrainingJob(
        job_id=job_id,
        kind="bootstrap_first_run",
        instrument_key=None,
        interval=None,
        horizon_steps=None,
        model_family=None,
        cap_tier=(str(req.cap_tier).lower().strip() if req.cap_tier else None),
    )
    cb = TRAINING_JOBS.progress_cb(job_id)

    train_long_lookback = int(req.train_lookback_days_long) if req.train_lookback_days_long is not None else int(settings.TRAIN_LONG_LOOKBACK_DAYS)
    train_intra_lookback = int(req.train_lookback_days_intraday) if req.train_lookback_days_intraday is not None else int(settings.TRAIN_INTRADAY_LOOKBACK_DAYS)

    def _run() -> dict:
        started = int(datetime.now(timezone.utc).timestamp())

        # Total units: each (instrument, interval) backfill is 1 unit; each model train is 1 unit.
        total_units = max(1, (len(keys) * len(intervals)) + (len(keys) * 2))
        done_units = 0

        inserted_total = 0
        cb(0.01, f"bootstrap queued {len(keys)} instruments", {"instrument_count": len(keys), "intervals": intervals})

        # 1) Backfill
        for instrument_key in keys:
            for interval in intervals:
                lookback = int(req.lookback_days_intraday if str(interval).endswith("m") else req.lookback_days_daily)
                plan = make_backfill_plan(
                    instrument_key,
                    interval,
                    lookback_days=lookback,
                    end=datetime.now(timezone.utc),
                    resume_from_db=bool(req.resume_from_db),
                )

                def _bf_cb(frac: float, message: str, metrics: dict | None = None) -> None:
                    overall = float((done_units + max(0.0, min(1.0, float(frac)))) / total_units)
                    merged = {"phase": "backfill", "inserted_total": inserted_total, "instrument_key": instrument_key, "interval": interval}
                    if metrics:
                        merged.update(metrics)
                    cb(overall, message, merged)

                out = run_backfill(plan, progress_cb=_bf_cb, log_to_terminal=True)
                inserted_total += int(out.get("inserted") or 0)
                done_units += 1
                cb(float(done_units / total_units), "backfill done", {"phase": "backfill", "inserted_total": inserted_total})

        # 2) Train ridge (long + intraday)
        results: list[dict] = []
        for instrument_key in keys:
            cb(
                float(done_units / total_units),
                f"training long {instrument_key}",
                {"phase": "train", "instrument_key": instrument_key, "model_family": "long"},
            )
            results.append(
                train_model(
                    instrument_key=instrument_key,
                    interval=settings.TRAIN_LONG_INTERVAL,
                    lookback_days=train_long_lookback,
                    horizon_steps=int(settings.TRAIN_LONG_HORIZON_STEPS),
                    model_family="long",
                    cap_tier=None,
                    min_samples=int(req.train_min_samples),
                )
            )
            done_units += 1
            cb(
                float(done_units / total_units),
                "training long done",
                {"phase": "train", "instrument_key": instrument_key, "model_family": "long"},
            )

            cb(
                float(done_units / total_units),
                f"training intraday {instrument_key}",
                {"phase": "train", "instrument_key": instrument_key, "model_family": "intraday"},
            )
            results.append(
                train_model(
                    instrument_key=instrument_key,
                    interval=settings.TRAIN_INTRADAY_INTERVAL,
                    lookback_days=train_intra_lookback,
                    horizon_steps=int(settings.TRAIN_INTRADAY_HORIZON_STEPS),
                    model_family="intraday",
                    cap_tier=None,
                    min_samples=int(req.train_min_samples),
                )
            )
            done_units += 1
            cb(
                float(done_units / total_units),
                "training intraday done",
                {"phase": "train", "instrument_key": instrument_key, "model_family": "intraday"},
            )

        ended = int(datetime.now(timezone.utc).timestamp())
        log_event(
            "bootstrap.first_run",
            {
                "job_id": job_id,
                "instrument_count": len(keys),
                "intervals": intervals,
                "inserted": int(inserted_total),
                "started_ts": started,
                "ended_ts": ended,
            },
        )

        cb(1.0, "bootstrap done", {"phase": "done", "inserted_total": inserted_total})
        return {
            "ok": True,
            "job_id": job_id,
            "instrument_count": len(keys),
            "intervals": intervals,
            "inserted": int(inserted_total),
            "results": results,
            "started_ts": started,
            "ended_ts": ended,
        }

    return TRAINING_JOBS.start_threaded(job, _run)


@router.post("/nse-eq-full-deep-async")
def nse_eq_full_deep_async(req: NseEqFullDeepTrainRequest) -> dict:
    """Universe-scale training pipeline (GPU-only deep models) for *all* NSE_EQ equities.

    Runs, per symbol:
    1) backfill daily (1d) for ~4 years
    2) train long deep model
    3) backfill intraday (1m) for ~3 months
    4) train intraday deep model

    Resumable via `after` cursor + `resume_from_db=true`.
    Watch progress via GET /api/learning/jobs/{job_id} (or WS /ws/training).
    """

    # Hard requirement: real candles require a token (otherwise we generate synthetic candles).
    if not settings.UPSTOX_ACCESS_TOKEN:
        raise HTTPException(status_code=400, detail={"ok": False, "reason": "UPSTOX_ACCESS_TOKEN is required for real candles"})

    # Hard requirement: GPU-only.
    try:
        import torch
    except Exception:
        raise HTTPException(status_code=501, detail={"ok": False, "reason": "torch not installed"})
    if not torch.cuda.is_available():
        raise HTTPException(status_code=501, detail={"ok": False, "reason": "cuda not available"})

    if not (bool(req.backfill_daily) or bool(req.backfill_intraday) or bool(req.train_long) or bool(req.train_intraday)):
        return {"ok": False, "reason": "nothing to do"}

    if not bool(req.skip_universe_import):
        # Import may take ~1-2 minutes; do it inside the job so it’s tracked.
        pass

    job_id = new_job_id()
    job = TrainingJob(job_id=job_id, kind="nse_eq_full_deep", instrument_key=None, interval=None, horizon_steps=None, model_family=None, cap_tier=None)
    cb = TRAINING_JOBS.progress_cb(job_id)

    uni = UniverseService()

    max_syms = int(req.max_symbols)
    if max_syms < 0:
        max_syms = 0

    steps_per_symbol = int(bool(req.backfill_daily)) + int(bool(req.train_long)) + int(bool(req.backfill_intraday)) + int(bool(req.train_intraday))
    steps_per_symbol = max(1, steps_per_symbol)

    def _run() -> dict:
        started_ts = int(datetime.now(timezone.utc).timestamp())
        inserted_total = 0
        done_symbols = 0

        trained_long = 0
        trained_intraday = 0
        skipped = 0
        failed = 0

        cursor = (req.after or "").strip() or None
        last_completed_key: str | None = None

        cb(0.01, "starting NSE_EQ full deep pipeline", {"phase": "start"})

        if not bool(req.skip_universe_import):
            cb(0.02, "importing Upstox NSE_EQ universe", {"phase": "import"})
            _import = _import_upstox_nse_eq(limit=int(req.import_limit))
            cb(0.03, "universe import done", {"phase": "import", **_import})

        total_syms = uni.count(prefix="NSE_EQ|")
        if total_syms <= 0:
            return {"ok": False, "reason": "instrument_meta empty; universe import required"}

        target_syms = int(total_syms if max_syms == 0 else min(total_syms, max_syms))
        total_work = max(1, target_syms * steps_per_symbol)
        done_work = 0

        cb(0.05, f"queued {target_syms} NSE_EQ symbols", {"phase": "queued", "total_symbols": target_syms, "steps_per_symbol": steps_per_symbol})

        while True:
            page = uni.list_keys_paged(prefix="NSE_EQ|", limit=int(req.page_size), after=cursor)
            keys = list(page.get("keys") or [])
            cursor = page.get("next_after")
            if not keys:
                break

            for instrument_key in keys:
                if max_syms and done_symbols >= max_syms:
                    cursor = None
                    break

                done_symbols += 1

                def _map_progress(frac: float, message: str, metrics: dict | None = None) -> None:
                    overall = float((done_work + max(0.0, min(1.0, float(frac)))) / total_work)
                    m = {
                        "phase": "running",
                        "instrument_key": instrument_key,
                        "done_symbols": done_symbols,
                        "inserted_total": inserted_total,
                        "trained_long": trained_long,
                        "trained_intraday": trained_intraday,
                        "skipped": skipped,
                        "failed": failed,
                        "last_completed_key": last_completed_key,
                        "next_after": cursor,
                    }
                    if metrics:
                        m.update(metrics)
                    cb(overall, message, m)

                try:
                    # 1) backfill daily
                    if bool(req.backfill_daily):
                        plan = make_backfill_plan(
                            instrument_key,
                            settings.TRAIN_LONG_INTERVAL,
                            lookback_days=int(req.lookback_days_daily),
                            end=datetime.now(timezone.utc),
                            resume_from_db=bool(req.resume_from_db),
                        )
                        out = run_backfill(
                            plan,
                            progress_cb=_map_progress,
                            log_to_terminal=True,
                            sleep_seconds=float(req.sleep_seconds_per_chunk),
                        )
                        inserted_total += int(out.get("inserted") or 0)
                        done_work += 1

                    # 2) train long
                    if bool(req.train_long):
                        _map_progress(0.0, f"training long {instrument_key}", {"model_family": "long"})
                        out = train_deep_model(
                            instrument_key=instrument_key,
                            interval=settings.TRAIN_LONG_INTERVAL,
                            lookback_days=int(req.lookback_days_daily),
                            horizon_steps=int(settings.TRAIN_LONG_HORIZON_STEPS),
                            model_family="long",
                            cap_tier=None,
                            seq_len=int(req.seq_len),
                            epochs=int(req.epochs_long),
                            batch_size=int(req.batch_size),
                            lr=2e-4,
                            weight_decay=1e-4,
                            min_samples=int(req.min_samples),
                            require_cuda=True,
                            progress_cb=_map_progress,
                        )
                        done_work += 1
                        if out.get("ok"):
                            trained_long += 1
                        elif str(out.get("reason") or "") in {"not enough data in DB", "not enough usable samples"}:
                            skipped += 1
                        else:
                            failed += 1

                    # 3) backfill intraday
                    if bool(req.backfill_intraday):
                        plan = make_backfill_plan(
                            instrument_key,
                            settings.TRAIN_INTRADAY_INTERVAL,
                            lookback_days=int(req.lookback_days_intraday),
                            end=datetime.now(timezone.utc),
                            resume_from_db=bool(req.resume_from_db),
                        )
                        out = run_backfill(
                            plan,
                            progress_cb=_map_progress,
                            log_to_terminal=True,
                            sleep_seconds=float(req.sleep_seconds_per_chunk),
                        )
                        inserted_total += int(out.get("inserted") or 0)
                        done_work += 1

                    # 4) train intraday
                    if bool(req.train_intraday):
                        _map_progress(0.0, f"training intraday {instrument_key}", {"model_family": "intraday"})
                        out = train_deep_model(
                            instrument_key=instrument_key,
                            interval=settings.TRAIN_INTRADAY_INTERVAL,
                            lookback_days=int(req.lookback_days_intraday),
                            horizon_steps=int(settings.TRAIN_INTRADAY_HORIZON_STEPS),
                            model_family="intraday",
                            cap_tier=None,
                            seq_len=int(req.seq_len),
                            epochs=int(req.epochs_intraday),
                            batch_size=int(req.batch_size),
                            lr=2e-4,
                            weight_decay=1e-4,
                            min_samples=int(req.min_samples),
                            require_cuda=True,
                            progress_cb=_map_progress,
                        )
                        done_work += 1
                        if out.get("ok"):
                            trained_intraday += 1
                        elif str(out.get("reason") or "") in {"not enough data in DB", "not enough usable samples"}:
                            skipped += 1
                        else:
                            failed += 1

                    last_completed_key = instrument_key
                    _map_progress(1.0, "symbol done", {"phase": "symbol_done", "last_completed_key": last_completed_key})

                    if float(req.sleep_seconds_per_symbol) > 0:
                        time.sleep(float(req.sleep_seconds_per_symbol))
                except Exception as e:
                    failed += 1
                    done_work += steps_per_symbol
                    _map_progress(1.0, f"symbol failed: {instrument_key}", {"phase": "symbol_failed", "error": str(e)[:500]})
                    last_completed_key = instrument_key

            if cursor is None:
                break

        ended_ts = int(datetime.now(timezone.utc).timestamp())
        cb(1.0, "done", {"phase": "done", "done_symbols": done_symbols, "trained_long": trained_long, "trained_intraday": trained_intraday, "skipped": skipped, "failed": failed, "last_completed_key": last_completed_key})
        log_event(
            "bootstrap.nse_eq_full_deep",
            {
                "job_id": job_id,
                "done_symbols": done_symbols,
                "trained_long": trained_long,
                "trained_intraday": trained_intraday,
                "skipped": skipped,
                "failed": failed,
                "started_ts": started_ts,
                "ended_ts": ended_ts,
                "last_completed_key": last_completed_key,
            },
        )
        return {
            "ok": True,
            "job_id": job_id,
            "done_symbols": done_symbols,
            "trained_long": trained_long,
            "trained_intraday": trained_intraday,
            "skipped": skipped,
            "failed": failed,
            "inserted_total": int(inserted_total),
            "started_ts": started_ts,
            "ended_ts": ended_ts,
            "last_completed_key": last_completed_key,
            "next_after": cursor,
        }

    return TRAINING_JOBS.start_threaded(job, _run)
