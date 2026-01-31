from __future__ import annotations

from datetime import datetime, timedelta, timezone

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.candles.persistence_sql import get_candles
from app.core.audit import log_event
from app.learning.service import load_model, train_model
from app.learning.deep_service import load_deep_model, predict_deep_return, train_deep_model
from app.learning.jobs import JOBS, TrainingJob, get_job, list_jobs, new_job_id

router = APIRouter(prefix="/learning", tags=["learning"])


def _artifact_info(name: str, path: str) -> dict:
    ap = os.path.abspath(path)
    exists = os.path.exists(ap)
    size = None
    mtime = None
    if exists:
        try:
            st = os.stat(ap)
            size = int(st.st_size)
            mtime = int(st.st_mtime)
        except Exception:
            size = None
            mtime = None
    return {
        "name": str(name),
        "path": str(path),
        "abs_path": ap,
        "exists": bool(exists),
        "size_bytes": size,
        "mtime_ts": mtime,
    }


@router.get("/artifacts")
def artifacts() -> dict:
    """List model artifacts produced by training, for easy cloud download.

    - Ridge + Deep models are stored inside the SQLite DB at Settings.DATABASE_PATH.
    - Pattern-seq model is a Torch payload file at Settings.PATTERN_SEQ_MODEL_PATH.
    """

    from app.core.settings import settings

    items = [
        _artifact_info("db", settings.DATABASE_PATH),
        _artifact_info("pattern_seq", settings.PATTERN_SEQ_MODEL_PATH),
    ]
    return {"ok": True, "artifacts": items}


@router.get("/artifacts/{name}")
def download_artifact(name: str) -> FileResponse:
    from app.core.settings import settings

    mapping = {
        "db": settings.DATABASE_PATH,
        "pattern_seq": settings.PATTERN_SEQ_MODEL_PATH,
    }
    if name not in mapping:
        raise HTTPException(status_code=404, detail="unknown artifact")

    path = os.path.abspath(str(mapping[name]))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail={"ok": False, "reason": "artifact not found", "name": name, "path": path})

    filename = os.path.basename(path)
    return FileResponse(path=path, filename=filename)


class TrainRequest(BaseModel):
    instrument_key: str
    interval: str = "1d"
    lookback_days: int = Field(default=365, ge=30, le=3650)
    horizon_steps: int = Field(default=1, ge=1, le=50)
    model_family: str | None = None
    cap_tier: str | None = None
    l2: float = Field(default=1e-2, ge=0)
    min_samples: int = Field(default=200, ge=50, le=5000)


class TrainDeepRequest(BaseModel):
    instrument_key: str
    interval: str = "1d"
    lookback_days: int = Field(default=365, ge=30, le=3650)
    horizon_steps: int = Field(default=1, ge=1, le=200)
    model_family: str | None = None
    cap_tier: str | None = None
    seq_len: int = Field(default=120, ge=30, le=1000)
    epochs: int = Field(default=8, ge=1, le=100)
    batch_size: int = Field(default=128, ge=16, le=2048)
    lr: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    min_samples: int = Field(default=500, ge=100, le=50000)


class BatchTrainRequest(BaseModel):
    instrument_keys: list[str] | None = None
    # If true, trains long + intraday using the Settings TRAIN_* presets.
    use_presets: bool = True
    # Optional overrides for faster first-run experimentation.
    lookback_days_long: int | None = Field(default=None, ge=30, le=3650)
    lookback_days_intraday: int | None = Field(default=None, ge=5, le=365)
    min_samples: int = Field(default=200, ge=50, le=5000)


@router.post("/train")
def train(req: TrainRequest) -> dict:
    res = train_model(
        instrument_key=req.instrument_key,
        interval=req.interval,
        lookback_days=req.lookback_days,
        horizon_steps=req.horizon_steps,
        model_family=req.model_family,
        cap_tier=req.cap_tier,
        l2=req.l2,
        min_samples=req.min_samples,
    )
    log_event("learning.train", {"request": req.model_dump(), "result": res})
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res)
    return res


@router.post("/train-async")
def train_async(req: TrainRequest) -> dict:
    """Async wrapper around classic ridge training (runs in a background thread)."""
    job_id = new_job_id()
    job = TrainingJob(
        job_id=job_id,
        kind="ridge",
        instrument_key=req.instrument_key,
        interval=req.interval,
        horizon_steps=int(req.horizon_steps),
        model_family=req.model_family,
        cap_tier=req.cap_tier,
    )

    cb = JOBS.progress_cb(job_id)

    def _run() -> dict:
        cb(0.05, "loading data")
        out = train_model(
            instrument_key=req.instrument_key,
            interval=req.interval,
            lookback_days=req.lookback_days,
            horizon_steps=req.horizon_steps,
            model_family=req.model_family,
            cap_tier=req.cap_tier,
            l2=req.l2,
            min_samples=req.min_samples,
        )
        cb(0.95, "saving")
        return out

    return JOBS.start_threaded(job, _run)


@router.get("/status")
def status(instrument_key: str, interval: str = "1d", horizon_steps: int = 1) -> dict:
    m = load_model(instrument_key, interval, horizon_steps=horizon_steps)
    return {
        "exists": m is not None,
        "model": None
        if m is None
        else {
            "model_key": m.model_key,
            "trained_ts": m.trained_ts,
            "n_samples": m.n_samples,
            "metrics": m.metrics,
            "feature_names": m.feature_names,
        },
    }


@router.get("/status-deep")
def status_deep(instrument_key: str, interval: str = "1d", horizon_steps: int = 1, model_family: str | None = None, cap_tier: str | None = None) -> dict:
    m = load_deep_model(instrument_key, interval, horizon_steps=horizon_steps, model_family=model_family, cap_tier=cap_tier)
    return {
        "exists": m is not None,
        "model": None
        if m is None
        else {
            "model_key": m.model_key,
            "trained_ts": m.trained_ts,
            "n_samples": m.n_samples,
            "seq_len": m.seq_len,
            "metrics": m.metrics,
        },
    }


@router.get("/predict")
def predict(instrument_key: str, interval: str = "1d", horizon_steps: int = 1, lookback: int = 60) -> dict:
    m = load_model(instrument_key, interval, horizon_steps=horizon_steps)
    if m is None:
        raise HTTPException(status_code=404, detail="no trained model")

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365)
    candles = get_candles(instrument_key, interval, int(start.timestamp()), int(now.timestamp()))
    closes = [c.close for c in candles][-int(lookback) :]
    if len(closes) < 10:
        raise HTTPException(status_code=400, detail="not enough candles in DB")

    pred_ret = m.predict_return(closes)
    last = float(closes[-1])
    pred_price = last * (1.0 + float(pred_ret))
    return {
        "instrument_key": instrument_key,
        "interval": interval,
        "horizon_steps": int(horizon_steps),
        "predicted_return": float(pred_ret),
        "last_close": last,
        "predicted_close": float(pred_price),
        "model": {
            "model_key": m.model_key,
            "trained_ts": m.trained_ts,
            "metrics": m.metrics,
        },
    }


@router.post("/train-deep")
def train_deep(req: TrainDeepRequest) -> dict:
    res = train_deep_model(
        instrument_key=req.instrument_key,
        interval=req.interval,
        lookback_days=req.lookback_days,
        horizon_steps=req.horizon_steps,
        model_family=req.model_family,
        cap_tier=req.cap_tier,
        seq_len=req.seq_len,
        epochs=req.epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        weight_decay=req.weight_decay,
        min_samples=req.min_samples,
    )
    log_event("learning.train_deep", {"request": req.model_dump(), "result": res})
    if not res.get("ok"):
        raise HTTPException(status_code=501 if res.get("reason") == "torch not installed" else 400, detail=res)
    return res


@router.post("/train-deep-async")
def train_deep_async(req: TrainDeepRequest) -> dict:
    """Async deep training job (threaded). Emits live progress on the training WebSocket."""
    job_id = new_job_id()
    job = TrainingJob(
        job_id=job_id,
        kind="deep",
        instrument_key=req.instrument_key,
        interval=req.interval,
        horizon_steps=int(req.horizon_steps),
        model_family=req.model_family,
        cap_tier=req.cap_tier,
    )
    cb = JOBS.progress_cb(job_id)

    def _run() -> dict:
        cb(0.02, "loading data")
        out = train_deep_model(
            instrument_key=req.instrument_key,
            interval=req.interval,
            lookback_days=req.lookback_days,
            horizon_steps=req.horizon_steps,
            model_family=req.model_family,
            cap_tier=req.cap_tier,
            seq_len=req.seq_len,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            weight_decay=req.weight_decay,
            min_samples=req.min_samples,
            progress_cb=cb,
        )
        cb(0.98, "finalizing")
        return out

    # If torch isn't installed, fail fast instead of creating a job that cannot run.
    probe = train_deep_model(
        instrument_key=req.instrument_key,
        interval=req.interval,
        lookback_days=1,
        horizon_steps=1,
        model_family=req.model_family,
        cap_tier=req.cap_tier,
        seq_len=30,
        epochs=1,
        batch_size=16,
        lr=req.lr,
        weight_decay=req.weight_decay,
        min_samples=10,
    )
    if not probe.get("ok") and probe.get("reason") == "torch not installed":
        raise HTTPException(status_code=501, detail={"ok": False, "reason": "torch not installed"})

    return JOBS.start_threaded(job, _run)


@router.post("/train-batch-async")
def train_batch_async(req: BatchTrainRequest) -> dict:
    """First-time bootstrap training for multiple instruments.

    By default, trains both:
    - long model (Settings TRAIN_LONG_*)
    - intraday model (Settings TRAIN_INTRADAY_*)

    If instrument_keys is omitted, uses Settings.DEFAULT_UNIVERSE.
    """

    keys = [str(k).strip() for k in (req.instrument_keys or []) if str(k).strip()]
    if not keys:
        from app.core.settings import settings as _s

        keys = [k.strip() for k in str(_s.DEFAULT_UNIVERSE or "").split(",") if k.strip()]

    if not keys:
        return {"ok": True, "job_id": None, "detail": "no instruments"}

    from app.core.settings import settings as _s

    job_id = new_job_id()
    job = TrainingJob(job_id=job_id, kind="batch_ridge", instrument_key=None, interval=None, horizon_steps=None, model_family=None, cap_tier=None)
    cb = JOBS.progress_cb(job_id)

    long_lookback = int(req.lookback_days_long) if req.lookback_days_long is not None else int(_s.TRAIN_LONG_LOOKBACK_DAYS)
    intra_lookback = int(req.lookback_days_intraday) if req.lookback_days_intraday is not None else int(_s.TRAIN_INTRADAY_LOOKBACK_DAYS)
    min_samples = int(req.min_samples)

    def _run() -> dict:
        results: list[dict] = []
        total = max(1, len(keys) * (2 if req.use_presets else 1))
        done = 0
        cb(0.01, f"queued {len(keys)} instruments")

        for instrument_key in keys:
            if req.use_presets:
                # long
                done += 1
                cb(float(done / total), f"training long {instrument_key}", {"instrument_key": instrument_key, "model_family": "long"})
                results.append(
                    train_model(
                        instrument_key=instrument_key,
                        interval=_s.TRAIN_LONG_INTERVAL,
                        lookback_days=long_lookback,
                        horizon_steps=int(_s.TRAIN_LONG_HORIZON_STEPS),
                        model_family="long",
                        cap_tier=None,
                        min_samples=min_samples,
                    )
                )

                # intraday
                done += 1
                cb(float(done / total), f"training intraday {instrument_key}", {"instrument_key": instrument_key, "model_family": "intraday"})
                results.append(
                    train_model(
                        instrument_key=instrument_key,
                        interval=_s.TRAIN_INTRADAY_INTERVAL,
                        lookback_days=intra_lookback,
                        horizon_steps=int(_s.TRAIN_INTRADAY_HORIZON_STEPS),
                        model_family="intraday",
                        cap_tier=None,
                        min_samples=min_samples,
                    )
                )
            else:
                done += 1
                cb(float(done / total), f"training {instrument_key}", {"instrument_key": instrument_key})
                results.append(
                    train_model(
                        instrument_key=instrument_key,
                        interval=_s.TRAIN_LONG_INTERVAL,
                        lookback_days=long_lookback,
                        horizon_steps=int(_s.TRAIN_LONG_HORIZON_STEPS),
                        model_family=None,
                        cap_tier=None,
                        min_samples=min_samples,
                    )
                )

        cb(1.0, "done")
        log_event("learning.train_batch", {"job_id": job_id, "instrument_count": len(keys)})
        return {"ok": True, "results": results}

    return JOBS.start_threaded(job, _run)


@router.get("/jobs")
def jobs(limit: int = 50, status: str | None = None) -> dict:
    return {"ok": True, "jobs": list_jobs(limit=limit, status=status)}


@router.get("/jobs/{job_id}")
def job(job_id: str) -> dict:
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True, "job": row}


@router.get("/predict-deep")
def predict_deep(
    instrument_key: str,
    interval: str = "1d",
    horizon_steps: int = 1,
    lookback: int = 240,
    model_family: str | None = None,
    cap_tier: str | None = None,
) -> dict:
    m = load_deep_model(instrument_key, interval, horizon_steps=horizon_steps, model_family=model_family, cap_tier=cap_tier)
    if m is None:
        raise HTTPException(status_code=404, detail="no deep model")

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365)
    candles = get_candles(instrument_key, interval, int(start.timestamp()), int(now.timestamp()))
    candles = candles[-int(lookback) :]
    if len(candles) < 10:
        raise HTTPException(status_code=400, detail="not enough candles in DB")

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    vols = [c.volume for c in candles]

    out = predict_deep_return(m, closes=closes, highs=highs, lows=lows, volumes=vols)
    if not out.get("ok"):
        raise HTTPException(status_code=501 if out.get("reason") == "torch not installed" else 400, detail=out)

    last = float(closes[-1])
    pred_ret = float(out["predicted_return"])
    pred_price = last * (1.0 + pred_ret)
    return {
        "instrument_key": instrument_key,
        "interval": interval,
        "horizon_steps": int(horizon_steps),
        "predicted_return": pred_ret,
        "last_close": last,
        "predicted_close": float(pred_price),
        "signal": out.get("signal"),
        "confidence": out.get("confidence"),
        "device": out.get("device"),
        "model": {"model_key": m.model_key, "trained_ts": m.trained_ts, "metrics": m.metrics},
    }


@router.post("/train-presets")
def train_presets() -> dict:
    """Train cap-tiered long-term + intraday models.

    Uses instrument_meta.cap_tier (large/mid/small). If none exists, does nothing.
    """
    from app.core.settings import settings
    from app.universe.service import UniverseService

    uni = UniverseService()
    out: dict[str, list[dict]] = {"large": [], "mid": [], "small": []}

    for tier in ("large", "mid", "small"):
        items = uni.list(cap_tier=tier, limit=500)
        keys = [i["instrument_key"] for i in items if i.get("instrument_key")]
        for instrument_key in keys:
            # long-term model
            out[tier].append(
                train_model(
                    instrument_key=instrument_key,
                    interval=settings.TRAIN_LONG_INTERVAL,
                    lookback_days=settings.TRAIN_LONG_LOOKBACK_DAYS,
                    horizon_steps=settings.TRAIN_LONG_HORIZON_STEPS,
                    model_family="long",
                    cap_tier=tier,
                    min_samples=200,
                )
            )
            # intraday model
            out[tier].append(
                train_model(
                    instrument_key=instrument_key,
                    interval=settings.TRAIN_INTRADAY_INTERVAL,
                    lookback_days=settings.TRAIN_INTRADAY_LOOKBACK_DAYS,
                    horizon_steps=settings.TRAIN_INTRADAY_HORIZON_STEPS,
                    model_family="intraday",
                    cap_tier=tier,
                    min_samples=200,
                )
            )

    log_event("learning.train_presets", {"result_counts": {k: len(v) for k, v in out.items()}})
    return {"ok": True, "results": out}


@router.post("/train-presets-deep-async")
def train_presets_deep_async(epochs: int = 6, seq_len: int = 120, batch_size: int = 128) -> dict:
    """Async deep training for universe cap tiers.

    Trains both long + intraday deep models for each symbol in `instrument_meta`.
    Streams live progress on `/api/ws/training`.
    """
    from app.core.settings import settings
    from app.universe.service import UniverseService

    uni = UniverseService()

    job_id = new_job_id()
    job = TrainingJob(job_id=job_id, kind="presets_deep", instrument_key=None, interval=None, horizon_steps=None, model_family=None, cap_tier=None)
    cb = JOBS.progress_cb(job_id)

    # fail fast if torch missing
    probe = train_deep_model(
        instrument_key="NSE_EQ|DEMO",
        interval="1d",
        lookback_days=1,
        horizon_steps=1,
        seq_len=30,
        epochs=1,
        batch_size=16,
        lr=2e-4,
        weight_decay=1e-4,
        min_samples=10,
    )
    if not probe.get("ok") and probe.get("reason") == "torch not installed":
        raise HTTPException(status_code=501, detail={"ok": False, "reason": "torch not installed"})

    def _run() -> dict:
        results: dict[str, list[dict]] = {"large": [], "mid": [], "small": []}
        tiers = ("large", "mid", "small")

        # Build work list up front.
        work: list[tuple[str, str, str]] = []
        for tier in tiers:
            items = uni.list(cap_tier=tier, limit=500)
            keys = [i["instrument_key"] for i in items if i.get("instrument_key")]
            for k in keys:
                work.append((tier, "long", k))
                work.append((tier, "intraday", k))

        total = max(1, len(work))
        done = 0
        cb(0.01, f"queued {len(work)} trainings")

        for tier, fam, instrument_key in work:
            done += 1
            frac = float(done / total)
            if fam == "long":
                interval = settings.TRAIN_LONG_INTERVAL
                lookback_days = settings.TRAIN_LONG_LOOKBACK_DAYS
                horizon_steps = settings.TRAIN_LONG_HORIZON_STEPS
            else:
                interval = settings.TRAIN_INTRADAY_INTERVAL
                lookback_days = settings.TRAIN_INTRADAY_LOOKBACK_DAYS
                horizon_steps = settings.TRAIN_INTRADAY_HORIZON_STEPS

            cb(frac, f"training {fam} {tier} {instrument_key}", {"instrument_key": instrument_key, "model_family": fam, "cap_tier": tier})
            out = train_deep_model(
                instrument_key=instrument_key,
                interval=interval,
                lookback_days=int(lookback_days),
                horizon_steps=int(horizon_steps),
                model_family=fam,
                cap_tier=tier,
                seq_len=int(seq_len),
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=2e-4,
                weight_decay=1e-4,
                min_samples=500,
                progress_cb=cb,
            )
            results[tier].append(out)

        cb(1.0, "done")
        log_event("learning.train_presets_deep", {"job_id": job_id, "counts": {k: len(v) for k, v in results.items()}})
        return {"ok": True, "results": results}

    return JOBS.start_threaded(job, _run)


@router.post("/train-nse-eq-deep-async")
def train_nse_eq_deep_async(
    epochs: int = 6,
    seq_len: int = 120,
    batch_size: int = 128,
    train_long: bool = True,
    train_intraday: bool = True,
    lookback_days_long: int | None = None,
    lookback_days_intraday: int | None = None,
    horizon_steps_long: int | None = None,
    horizon_steps_intraday: int | None = None,
    min_samples: int = 500,
    max_symbols: int = 0,
    after: str | None = None,
    page_size: int = 500,
) -> dict:
    """Train deep models for the entire NSE_EQ equity universe using GPU.

    Requirements:
    - torch installed
    - CUDA available (will FAIL if GPU is not available; no CPU fallback)
    - instrument_meta populated (use /api/universe/import-upstox-nse-eq)

    Trains per symbol:
    - long (1d)
    - intraday (1m)
    """
    from app.core.settings import settings
    from app.universe.service import UniverseService

    uni = UniverseService()

    # Fail fast if torch missing or CUDA missing (without requiring candle data).
    try:
        import torch
    except Exception:
        raise HTTPException(status_code=501, detail={"ok": False, "reason": "torch not installed"})
    if not torch.cuda.is_available():
        raise HTTPException(status_code=501, detail={"ok": False, "reason": "cuda not available"})

    total_syms = uni.count(prefix="NSE_EQ|")
    if total_syms <= 0:
        return {"ok": False, "reason": "instrument_meta empty; import universe first"}

    if not bool(train_long) and not bool(train_intraday):
        return {"ok": False, "reason": "nothing to train; set train_long and/or train_intraday"}

    job_id = new_job_id()
    job = TrainingJob(job_id=job_id, kind="nse_eq_deep", instrument_key=None, interval=None, horizon_steps=None, model_family=None, cap_tier=None)
    cb = JOBS.progress_cb(job_id)

    max_syms = int(max_symbols)
    if max_syms < 0:
        max_syms = 0

    def _run() -> dict:
        done_symbols = 0
        trained = 0
        skipped = 0
        failed = 0

        long_lookback = int(settings.TRAIN_LONG_LOOKBACK_DAYS if lookback_days_long is None else lookback_days_long)
        intra_lookback = int(settings.TRAIN_INTRADAY_LOOKBACK_DAYS if lookback_days_intraday is None else lookback_days_intraday)
        long_h = int(settings.TRAIN_LONG_HORIZON_STEPS if horizon_steps_long is None else horizon_steps_long)
        intra_h = int(settings.TRAIN_INTRADAY_HORIZON_STEPS if horizon_steps_intraday is None else horizon_steps_intraday)
        ms = max(50, int(min_samples))

        total_target = int(total_syms if max_syms == 0 else min(total_syms, max_syms))
        per_symbol = int(bool(train_long)) + int(bool(train_intraday))
        total_work = max(1, total_target * max(1, per_symbol))
        done_work = 0

        cursor = (after or "").strip() or None
        cb(0.01, f"queued NSE_EQ deep training for {total_target} symbols", {"total_symbols": total_target})

        results: list[dict] = []

        while True:
            page = uni.list_keys_paged(prefix="NSE_EQ|", limit=int(page_size), after=cursor)
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
                    m = {"instrument_key": instrument_key, "done_symbols": done_symbols, "trained": trained, "skipped": skipped, "failed": failed}
                    if metrics:
                        m.update(metrics)
                    cb(overall, message, m)

                if bool(train_long):
                    cb(float(done_work / total_work), f"training long {instrument_key}", {"instrument_key": instrument_key, "model_family": "long"})
                    out1 = train_deep_model(
                        instrument_key=instrument_key,
                        interval=settings.TRAIN_LONG_INTERVAL,
                        lookback_days=int(long_lookback),
                        horizon_steps=int(long_h),
                        model_family="long",
                        cap_tier=None,
                        seq_len=int(seq_len),
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        lr=2e-4,
                        weight_decay=1e-4,
                        min_samples=int(ms),
                        require_cuda=True,
                        progress_cb=_map_progress,
                    )
                    results.append(out1)
                    done_work += 1
                    if out1.get("ok"):
                        trained += 1
                    elif str(out1.get("reason") or "") in {"not enough data in DB", "not enough usable samples"}:
                        skipped += 1
                    else:
                        failed += 1

                if bool(train_intraday):
                    cb(float(done_work / total_work), f"training intraday {instrument_key}", {"instrument_key": instrument_key, "model_family": "intraday"})
                    out2 = train_deep_model(
                        instrument_key=instrument_key,
                        interval=settings.TRAIN_INTRADAY_INTERVAL,
                        lookback_days=int(intra_lookback),
                        horizon_steps=int(intra_h),
                        model_family="intraday",
                        cap_tier=None,
                        seq_len=int(seq_len),
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        lr=2e-4,
                        weight_decay=1e-4,
                        min_samples=int(ms),
                        require_cuda=True,
                        progress_cb=_map_progress,
                    )
                    results.append(out2)
                    done_work += 1
                    if out2.get("ok"):
                        trained += 1
                    elif str(out2.get("reason") or "") in {"not enough data in DB", "not enough usable samples"}:
                        skipped += 1
                    else:
                        failed += 1

            if cursor is None:
                break

        cb(1.0, "done", {"trained": trained, "skipped": skipped, "failed": failed, "done_symbols": done_symbols})
        log_event("learning.train_nse_eq_deep", {"job_id": job_id, "trained": trained, "skipped": skipped, "failed": failed, "done_symbols": done_symbols, "after": after})
        return {"ok": True, "trained": trained, "skipped": skipped, "failed": failed, "done_symbols": done_symbols, "next_after": cursor, "results": results}

    return JOBS.start_threaded(job, _run)


@router.post("/train-nse-eq-pattern-seq-async")
def train_nse_eq_pattern_seq_async(
    interval: str = "1m",
    lookback_days: int = 30,
    seq_len: int = 64,
    stride: int = 2,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    label_threshold: float = 0.35,
    max_candles_per_symbol: int = 5000,
    max_windows_per_symbol: int = 1500,
    max_windows_total: int = 50000,
    max_symbols: int = 0,
    after: str | None = None,
    page_size: int = 500,
    out_path: str | None = None,
) -> dict:
    """Train a global sequence-based candlestick pattern model over the NSE_EQ universe.

    This trains a multi-label model that learns pattern shapes from sequences,
    using weak labels generated from the existing detector.

    Requirements:
    - torch installed (CPU is fine)
    - instrument_meta populated
    """

    from app.core.settings import settings
    from app.universe.service import UniverseService

    # Fail fast if torch missing.
    try:
        from app.learning.pattern_seq import torch_available

        if not torch_available():
            raise HTTPException(status_code=501, detail={"ok": False, "reason": "torch not installed"})
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=501, detail={"ok": False, "reason": "torch not installed"})

    uni = UniverseService()
    total_syms = uni.count(prefix="NSE_EQ|")
    if total_syms <= 0:
        return {"ok": False, "reason": "instrument_meta empty; import universe first"}

    job_id = new_job_id()
    job = TrainingJob(job_id=job_id, kind="nse_eq_pattern_seq", instrument_key=None, interval=str(interval), horizon_steps=None, model_family=None, cap_tier=None)
    cb = JOBS.progress_cb(job_id)

    max_syms = int(max_symbols)
    if max_syms < 0:
        max_syms = 0

    def _run() -> dict:
        cursor = (after or "").strip() or None
        keys: list[str] = []

        cb(0.01, "enumerating NSE_EQ symbols")
        while True:
            page = uni.list_keys_paged(prefix="NSE_EQ|", limit=int(page_size), after=cursor)
            page_keys = [str(k).strip() for k in (page.get("keys") or []) if str(k).strip()]
            cursor = page.get("next_after")
            if not page_keys:
                break

            for k in page_keys:
                if max_syms and len(keys) >= max_syms:
                    cursor = None
                    break
                keys.append(k)
            if cursor is None:
                break

        if not keys:
            return {"ok": False, "reason": "no instruments"}

        target_path = str(out_path or getattr(settings, "PATTERN_SEQ_MODEL_PATH", "data/models/pattern_seq.pt"))
        from app.learning.pattern_seq_trainer import TrainPatternSeqParams, train_pattern_seq_model_for_instruments

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

        def _inner_cb(frac: float, msg: str, metrics: dict | None = None) -> None:
            cb(max(0.02, min(0.99, float(frac))), msg, metrics)

        return train_pattern_seq_model_for_instruments(keys, out_path=target_path, params=params, progress_cb=_inner_cb)

    return JOBS.start_threaded(job, _run)
