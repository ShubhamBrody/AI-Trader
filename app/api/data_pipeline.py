from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.settings import settings
from app.data.continuous import INGESTION
from app.data.jobs import JOBS, PipelineJob, get_job, list_jobs, new_job_id
from app.data.pipeline import make_backfill_plan, run_backfill
from app.universe.service import UniverseService


router = APIRouter(prefix="/data", tags=["data"])


class BackfillRequest(BaseModel):
    instrument_keys: list[str] | None = None
    cap_tier: str | None = None
    intervals: list[str] = Field(default_factory=lambda: ["1d", "1m"])
    lookback_days_daily: int = Field(default=1460, ge=30, le=3650)
    lookback_days_intraday: int = Field(default=95, ge=5, le=365)
    resume_from_db: bool = True


class BackfillNseEqRequest(BaseModel):
    intervals: list[str] = Field(default_factory=lambda: ["1d", "1m"])
    lookback_days_daily: int = Field(default=365, ge=30, le=3650)
    lookback_days_intraday: int = Field(default=5, ge=1, le=365)
    resume_from_db: bool = True
    page_size: int = Field(default=500, ge=50, le=5000)
    after: str | None = None
    max_symbols: int = Field(default=0, ge=0)


@router.post("/backfill-async")
def backfill_async(req: BackfillRequest) -> dict:
    if not settings.UPSTOX_ACCESS_TOKEN and settings.UPSTOX_STRICT:
        raise HTTPException(status_code=400, detail="UPSTOX_STRICT=true but no UPSTOX_ACCESS_TOKEN")

    uni = UniverseService()
    if req.instrument_keys:
        keys = [str(k) for k in req.instrument_keys]
    else:
        items = uni.list(cap_tier=req.cap_tier, limit=2000)
        keys = [str(i["instrument_key"]) for i in items if i.get("instrument_key")]
    if not keys:
        return {"ok": True, "job_id": None, "detail": "no instruments"}

    job_id = new_job_id()
    job = PipelineJob(job_id=job_id, kind="backfill", params=req.model_dump())
    cb = JOBS.progress_cb(job_id)

    def _run() -> dict:
        started = int(datetime.now(timezone.utc).timestamp())
        inserted_total = 0
        done = 0
        total = max(1, len(keys) * max(1, len(req.intervals)))

        for instrument_key in keys:
            for interval in req.intervals:
                lookback = int(req.lookback_days_intraday if str(interval).endswith("m") else req.lookback_days_daily)
                plan = make_backfill_plan(
                    instrument_key,
                    interval,
                    lookback_days=lookback,
                    end=datetime.now(timezone.utc),
                    resume_from_db=bool(req.resume_from_db),
                )
                out = run_backfill(plan, progress_cb=cb, log_to_terminal=True)
                inserted_total += int(out.get("inserted") or 0)
                done += 1
                cb(float(done / total), "backfill progress", {"inserted_total": inserted_total})

        ended = int(datetime.now(timezone.utc).timestamp())
        return {"ok": True, "inserted": int(inserted_total), "started_ts": started, "ended_ts": ended}

    return JOBS.start_threaded(job, _run)


@router.post("/backfill-nse-eq-async")
def backfill_nse_eq_async(req: BackfillNseEqRequest) -> dict:
    """Backfill candles for all NSE_EQ equities in instrument_meta (paged).

    Note: this can be extremely large for 1m candles across the full universe.
    For a realistic rollout, start with daily-only first.
    """

    if not settings.UPSTOX_ACCESS_TOKEN and settings.UPSTOX_STRICT:
        raise HTTPException(status_code=400, detail="UPSTOX_STRICT=true but no UPSTOX_ACCESS_TOKEN")

    uni = UniverseService()
    total_syms = uni.count(prefix="NSE_EQ|")
    if total_syms <= 0:
        return {"ok": False, "reason": "instrument_meta empty; import universe first"}

    job_id = new_job_id()
    job = PipelineJob(
        job_id=job_id,
        kind="backfill_nse_eq",
        params=req.model_dump(),
    )
    cb = JOBS.progress_cb(job_id)

    max_syms = int(req.max_symbols)
    if max_syms < 0:
        max_syms = 0

    def _run() -> dict:
        started = int(datetime.now(timezone.utc).timestamp())
        inserted_total = 0
        done_symbols = 0

        target_syms = int(total_syms if max_syms == 0 else min(total_syms, max_syms))
        work_total = max(1, target_syms * max(1, len(req.intervals)))
        work_done = 0

        cursor = (req.after or "").strip() or None
        cb(0.01, f"queued backfill for {target_syms} NSE_EQ symbols", {"total_symbols": target_syms, "intervals": req.intervals})

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
                for interval in req.intervals:
                    lookback = int(req.lookback_days_intraday if str(interval).endswith("m") else req.lookback_days_daily)

                    def _bf_cb(frac: float, message: str, stats: dict | None = None) -> None:
                        overall = float((work_done + max(0.0, min(1.0, float(frac)))) / work_total)
                        merged = {"inserted_total": inserted_total, "done_symbols": done_symbols, "interval": interval, "instrument_key": instrument_key}
                        if stats:
                            merged.update(stats)
                        cb(overall, message, merged)

                    plan = make_backfill_plan(
                        instrument_key,
                        interval,
                        lookback_days=lookback,
                        end=datetime.now(timezone.utc),
                        resume_from_db=bool(req.resume_from_db),
                    )
                    out = run_backfill(plan, progress_cb=_bf_cb, log_to_terminal=True)
                    inserted_total += int(out.get("inserted") or 0)
                    work_done += 1
                    cb(float(work_done / work_total), "backfill progress", {"inserted_total": inserted_total, "done_symbols": done_symbols})

            if cursor is None:
                break

        ended = int(datetime.now(timezone.utc).timestamp())
        cb(1.0, "done", {"inserted_total": inserted_total, "done_symbols": done_symbols})
        return {"ok": True, "inserted": int(inserted_total), "done_symbols": int(done_symbols), "started_ts": started, "ended_ts": ended, "next_after": cursor}

    return JOBS.start_threaded(job, _run)


@router.get("/jobs")
def pipeline_jobs(limit: int = 50, status: str | None = None) -> dict:
    return {"ok": True, "jobs": list_jobs(limit=limit, status=status)}


@router.get("/jobs/{job_id}")
def pipeline_job(job_id: str) -> dict:
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True, "job": row}


@router.get("/ingestion/status")
def ingestion_status() -> dict:
    return {"ok": True, **INGESTION.status()}


@router.post("/ingestion/start")
async def ingestion_start() -> dict:
    res = await INGESTION.start()
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res)
    return res


@router.post("/ingestion/stop")
async def ingestion_stop() -> dict:
    return await INGESTION.stop()
