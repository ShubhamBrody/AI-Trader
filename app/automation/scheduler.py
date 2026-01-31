from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from loguru import logger

from app.candles.service import CandleService
from app.core.settings import settings
from app.automation.runs import start_run as start_automation_run
from app.automation.runs import update_run as update_automation_run
from app.learning.deep_service import train_deep_model
from app.learning.drift import drift_check_recent
from app.learning.evaluation import record_evaluation, walk_forward_eval_deep, walk_forward_eval_ridge
from app.learning.jobs import JOBS as TRAIN_JOBS
from app.learning.jobs import TrainingJob, new_job_id
from app.learning.registry import promote, slot_key
from app.learning.service import train_model
from app.realtime.bus import publish_sync
from app.universe.service import UniverseService

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


@dataclass(frozen=True)
class AutoCycleResult:
    trained: int
    evaluated: int
    promoted: int
    drifted: int


def _local_now() -> datetime:
    if ZoneInfo is None:
        return datetime.now(timezone.utc)
    try:
        tz = ZoneInfo(str(settings.TIMEZONE))
    except Exception:
        tz = timezone.utc
    return datetime.now(tz)


def _is_due(now_local: datetime, *, hour: int, minute: int) -> bool:
    return (now_local.hour, now_local.minute) >= (int(hour), int(minute))


class AutoRetrainScheduler:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_error: str | None = None
        self._last_cycle_ts: int | None = None
        self._last_long_date: date | None = None
        self._last_intraday_date: date | None = None
        self._lock = asyncio.Lock()

        self._candles = CandleService()
        self._uni = UniverseService()

    def status(self) -> dict[str, Any]:
        return {
            "running": bool(self._running and self._task and not self._task.done()),
            "last_cycle_ts": self._last_cycle_ts,
            "last_error": self._last_error,
            "enabled": bool(settings.AUTO_RETRAIN_ENABLED),
            "poll_seconds": int(settings.AUTO_RETRAIN_POLL_SECONDS),
            "last_long_date": (None if self._last_long_date is None else self._last_long_date.isoformat()),
            "last_intraday_date": (None if self._last_intraday_date is None else self._last_intraday_date.isoformat()),
        }

    async def start(self) -> dict[str, Any]:
        if self._task and not self._task.done():
            return {"ok": True, "status": "already_running"}
        if not settings.AUTO_RETRAIN_ENABLED:
            return {"ok": False, "detail": "AUTO_RETRAIN_ENABLED=false"}

        self._running = True
        self._last_error = None
        self._task = asyncio.create_task(self._run())
        publish_sync("training", "automation.start", self.status())
        return {"ok": True}

    async def stop(self) -> dict[str, Any]:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        publish_sync("training", "automation.stop", {})
        return {"ok": True}

    async def run_once(self, *, force: bool = False) -> dict[str, Any]:
        # Protect against overlapping cycles.
        async with self._lock:
            return await asyncio.to_thread(self._run_cycle_sync, force)

    async def _run(self) -> None:
        try:
            while self._running:
                try:
                    await self._tick()
                    await asyncio.sleep(max(10, int(settings.AUTO_RETRAIN_POLL_SECONDS)))
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    self._last_error = str(e)
                    publish_sync("training", "automation.error", {"error": str(e)})
                    await asyncio.sleep(5)
        finally:
            self._running = False

    async def _tick(self) -> None:
        now_local = _local_now()
        today = now_local.date()

        should_long = _is_due(now_local, hour=settings.AUTO_RETRAIN_LONG_HOUR_LOCAL, minute=settings.AUTO_RETRAIN_LONG_MINUTE_LOCAL)
        should_intra = _is_due(now_local, hour=settings.AUTO_RETRAIN_INTRADAY_HOUR_LOCAL, minute=settings.AUTO_RETRAIN_INTRADAY_MINUTE_LOCAL)

        do_long = bool(should_long and self._last_long_date != today)
        do_intra = bool(should_intra and self._last_intraday_date != today)

        if not (do_long or do_intra):
            return

        async with self._lock:
            await asyncio.to_thread(self._run_cycle_sync, False, do_long, do_intra)

    def _run_cycle_sync(self, force: bool, do_long: bool | None = None, do_intraday: bool | None = None) -> dict[str, Any]:
        started_ts = int(datetime.now(timezone.utc).timestamp())
        now_local = _local_now()
        today = now_local.date()

        if do_long is None or do_intraday is None:
            # If invoked manually, run both families.
            do_long = True
            do_intraday = True
        if force:
            do_long = True
            do_intraday = True

        job_id = new_job_id()

        # Record automation run separately for orchestration visibility.
        start_automation_run(
            job_id=job_id,
            kind="auto_cycle",
            params={
                "force": bool(force),
                "do_long": bool(do_long),
                "do_intraday": bool(do_intraday),
                "max_symbols": int(settings.AUTO_RETRAIN_MAX_SYMBOLS),
                "local_date": today.isoformat(),
            },
        )

        job = TrainingJob(
            job_id=job_id,
            kind="auto_cycle",
            instrument_key=None,
            interval=None,
            horizon_steps=None,
            model_family=None,
            cap_tier=None,
        )

        cb = TRAIN_JOBS.progress_cb(job_id)

        # Run cycle *inside* the TrainingJobManager thread wrapper so it is recorded in DB.
        def _run() -> dict[str, Any]:
            try:
                publish_sync(
                    "training",
                    "automation.cycle.started",
                    {
                        "job_id": job_id,
                        "do_long": bool(do_long),
                        "do_intraday": bool(do_intraday),
                        "max_symbols": int(settings.AUTO_RETRAIN_MAX_SYMBOLS),
                        "local_date": today.isoformat(),
                    },
                )

                items = self._uni.list(limit=int(settings.AUTO_RETRAIN_MAX_SYMBOLS))
                keys = [str(i.get("instrument_key")) for i in items if i.get("instrument_key")]

                # Progress accounting: ridge long/intra, deep long/intra (if available).
                planned = 0
                if do_long:
                    planned += len(keys)
                if do_intraday:
                    planned += len(keys)

                # Deep is optional (if torch missing, train_deep_model returns ok=false quickly).
                planned_deep = planned
                planned_total = max(1, planned + planned_deep)

                trained = 0
                evaluated = 0
                promoted_n = 0
                drifted = 0
                done = 0

                def _log_progress(frac: float, msg: str, extra: dict[str, Any] | None = None) -> None:
                    cb(float(frac), msg, extra)
                    update_automation_run(job_id, progress=float(frac), message=str(msg))
                    try:
                        if extra and "eta_seconds" in extra:
                            logger.info(
                                "[AUTO] {} | {}",
                                msg,
                                {k: extra.get(k) for k in ("instrument_key", "model_kind", "model_family", "eta_seconds")},
                            )
                        else:
                            logger.info("[AUTO] {}", msg)
                    except Exception:
                        pass

                _log_progress(0.01, f"queued {len(keys)} symbols")

                def _maybe_promote(
                    kind: str,
                    fam: str,
                    cap: str | None,
                    instrument_key: str,
                    interval: str,
                    horizon_steps: int,
                    model_key: str,
                    eval_metrics: dict[str, Any],
                    drift: dict[str, Any] | None,
                ) -> bool:
                    nonlocal promoted_n, drifted

                    direction_acc = float(eval_metrics.get("direction_acc") or 0.0)
                    rmse = float(eval_metrics.get("rmse") or eval_metrics.get("val_huber") or 0.0)

                    # Drift gating
                    if drift and drift.get("ok"):
                        z = float(drift.get("z") or 0.0)
                        if z >= float(settings.DRIFT_Z_THRESHOLD):
                            drifted += 1
                            publish_sync(
                                "training",
                                "automation.drift.detected",
                                {
                                    "job_id": job_id,
                                    "kind": kind,
                                    "instrument_key": instrument_key,
                                    "interval": interval,
                                    "model_family": fam,
                                    "z": z,
                                    "threshold": float(settings.DRIFT_Z_THRESHOLD),
                                },
                            )
                            return False

                    ok = (direction_acc >= float(settings.PROMOTION_MIN_DIRECTION_ACC)) and (rmse <= float(settings.PROMOTION_MAX_RMSE))
                    if not ok:
                        return False

                    sk = slot_key(
                        instrument_key=instrument_key,
                        interval=interval,
                        horizon_steps=int(horizon_steps),
                        model_family=fam,
                        cap_tier=cap,
                        kind=kind,
                    )
                    promote(slot_key=sk, model_key=model_key, kind=kind, stage="production", metrics=dict(eval_metrics))
                    promoted_n += 1
                    publish_sync(
                        "training",
                        "automation.model.promoted",
                        {
                            "job_id": job_id,
                            "slot_key": sk,
                            "model_key": model_key,
                            "kind": kind,
                            "instrument_key": instrument_key,
                            "interval": interval,
                            "model_family": fam,
                            "metrics": eval_metrics,
                        },
                    )
                    return True

                def _do_one_ridge(instrument_key: str, fam: str) -> None:
                    nonlocal trained, evaluated, done

                    if fam == "long":
                        interval = str(settings.TRAIN_LONG_INTERVAL)
                        lookback = int(settings.TRAIN_LONG_LOOKBACK_DAYS)
                        horizon = int(settings.TRAIN_LONG_HORIZON_STEPS)
                    else:
                        interval = str(settings.TRAIN_INTRADAY_INTERVAL)
                        lookback = int(settings.TRAIN_INTRADAY_LOOKBACK_DAYS)
                        horizon = int(settings.TRAIN_INTRADAY_HORIZON_STEPS)

                    # Freshen data (cheap).
                    try:
                        if interval.endswith("m"):
                            self._candles.poll_intraday(instrument_key, interval, lookback_minutes=int(settings.DATA_PIPELINE_LOOKBACK_MINUTES))
                        else:
                            self._candles.load_historical(
                                instrument_key,
                                interval,
                                datetime.now(timezone.utc) - timedelta(days=max(30, lookback)),
                                datetime.now(timezone.utc),
                            )
                    except Exception:
                        pass

                    out = train_model(
                        instrument_key=instrument_key,
                        interval=interval,
                        lookback_days=lookback,
                        horizon_steps=horizon,
                        model_family=fam,
                    )
                    trained += 1

                    model_key = (((out.get("model") or {}).get("model_key")) if out.get("ok") else None)
                    done += 1
                    _log_progress(
                        float(done / planned_total),
                        f"ridge trained {fam} {instrument_key}",
                        {"instrument_key": instrument_key, "model_kind": "ridge", "model_family": fam, "ok": bool(out.get("ok"))},
                    )

                    if not out.get("ok") or not model_key:
                        return

                    ev = walk_forward_eval_ridge(instrument_key, interval, lookback_days=lookback, horizon_steps=horizon)
                    if not ev.get("ok"):
                        publish_sync(
                            "training",
                            "automation.eval.failed",
                            {"job_id": job_id, "instrument_key": instrument_key, "kind": "ridge", "reason": ev.get("reason")},
                        )
                        return

                    evaluated += 1
                    sk = slot_key(instrument_key=instrument_key, interval=interval, horizon_steps=horizon, model_family=fam, cap_tier=None, kind="ridge")
                    record_evaluation(slot_key=sk, model_key=str(model_key), kind="ridge", eval_kind=str(ev.get("eval_kind") or "walk_forward"), metrics=dict(ev))

                    drift = drift_check_recent(instrument_key, interval, baseline=dict(ev.get("baseline") or {}), recent_minutes=240, recent_days=10)
                    _maybe_promote("ridge", fam, None, instrument_key, interval, horizon, str(model_key), ev, drift)

                def _do_one_deep(instrument_key: str, fam: str) -> None:
                    nonlocal trained, evaluated, done

                    if fam == "long":
                        interval = str(settings.TRAIN_LONG_INTERVAL)
                        lookback = int(settings.TRAIN_LONG_LOOKBACK_DAYS)
                        horizon = int(settings.TRAIN_LONG_HORIZON_STEPS)
                        seq_len = 120
                        epochs = 6
                        batch = 128
                    else:
                        interval = str(settings.TRAIN_INTRADAY_INTERVAL)
                        lookback = int(settings.TRAIN_INTRADAY_LOOKBACK_DAYS)
                        horizon = int(settings.TRAIN_INTRADAY_HORIZON_STEPS)
                        seq_len = 180
                        epochs = 4
                        batch = 256

                    def _deep_cb(p: float, msg: str, metrics: dict[str, Any] | None = None) -> None:
                        base = float(done / planned_total)
                        span = 1.0 / float(planned_total)
                        frac = min(0.999, base + span * float(p))
                        m = dict(metrics or {})
                        m.update({"instrument_key": instrument_key, "model_kind": "deep", "model_family": fam})
                        _log_progress(frac, f"deep {fam} {instrument_key}: {msg}", m)

                    out = train_deep_model(
                        instrument_key=instrument_key,
                        interval=interval,
                        lookback_days=lookback,
                        horizon_steps=horizon,
                        model_family=fam,
                        seq_len=int(seq_len),
                        epochs=int(epochs),
                        batch_size=int(batch),
                        min_samples=500 if fam == "long" else 1000,
                        progress_cb=_deep_cb,
                        patience=3,
                    )

                    trained += 1
                    model_key = (((out.get("model") or {}).get("model_key")) if out.get("ok") else None)

                    done += 1
                    _log_progress(
                        float(done / planned_total),
                        f"deep trained {fam} {instrument_key}",
                        {"instrument_key": instrument_key, "model_kind": "deep", "model_family": fam, "ok": bool(out.get("ok")), "reason": out.get("reason")},
                    )

                    if not out.get("ok") or not model_key:
                        return

                    evaluated += 1
                    metrics = dict((out.get("model") or {}).get("metrics") or {})
                    ev_wf = walk_forward_eval_deep(
                        instrument_key,
                        interval,
                        lookback_days=min(lookback, 365),
                        horizon_steps=horizon,
                        seq_len=int(seq_len),
                        folds=3,
                        epochs=2,
                        batch_size=min(128, int(batch)),
                        min_train=400,
                    )
                    if ev_wf.get("ok"):
                        ev = dict(ev_wf)
                        eval_kind = str(ev.get("eval_kind") or "walk_forward")
                    else:
                        ev = {"ok": True, "eval_kind": "holdout", **metrics}
                        eval_kind = "holdout"

                    sk = slot_key(instrument_key=instrument_key, interval=interval, horizon_steps=horizon, model_family=fam, cap_tier=None, kind="deep")
                    record_evaluation(slot_key=sk, model_key=str(model_key), kind="deep", eval_kind=eval_kind, metrics=ev)

                    base = walk_forward_eval_ridge(
                        instrument_key,
                        interval,
                        lookback_days=min(lookback, 180),
                        horizon_steps=max(1, min(horizon, 20)),
                    )
                    baseline = dict((base.get("baseline") or {}) if base.get("ok") else {})
                    drift = drift_check_recent(instrument_key, interval, baseline=baseline, recent_minutes=240, recent_days=10)
                    _maybe_promote("deep", fam, None, instrument_key, interval, horizon, str(model_key), ev, drift)

                for instrument_key in keys:
                    if do_long:
                        _do_one_ridge(instrument_key, "long")
                    if do_intraday:
                        _do_one_ridge(instrument_key, "intraday")

                deep_available = True
                for instrument_key in keys:
                    if not deep_available:
                        break
                    if do_long:
                        out_probe = train_deep_model(
                            instrument_key=instrument_key,
                            interval="1d",
                            lookback_days=1,
                            horizon_steps=1,
                            seq_len=30,
                            epochs=1,
                            batch_size=16,
                            min_samples=10,
                        )
                        if not out_probe.get("ok") and out_probe.get("reason") == "torch not installed":
                            deep_available = False
                            publish_sync("training", "automation.deep.disabled", {"job_id": job_id, "reason": "torch not installed"})
                            break
                        _do_one_deep(instrument_key, "long")
                    if do_intraday and deep_available:
                        _do_one_deep(instrument_key, "intraday")

                publish_sync(
                    "training",
                    "automation.cycle.finished",
                    {"job_id": job_id, "trained": int(trained), "evaluated": int(evaluated), "promoted": int(promoted_n), "drifted": int(drifted)},
                )
                update_automation_run(
                    job_id,
                    progress=1.0,
                    status="SUCCEEDED",
                    stats={"trained": int(trained), "evaluated": int(evaluated), "promoted": int(promoted_n), "drifted": int(drifted)},
                )
                return {"ok": True, "trained": int(trained), "evaluated": int(evaluated), "promoted": int(promoted_n), "drifted": int(drifted)}
            except Exception as e:
                update_automation_run(job_id, status="FAILED", error=str(e))
                return {"ok": False, "reason": str(e)}

        # Actually run under the existing training job infrastructure.
        out = TRAIN_JOBS.start_threaded(job, _run)
        # Mark dates immediately to avoid duplicate triggering; the job runs async.
        if do_long:
            self._last_long_date = today
        if do_intraday:
            self._last_intraday_date = today

        self._last_cycle_ts = started_ts
        publish_sync("training", "automation.cycle.queued", {"job_id": job_id, "started_ts": started_ts})
        return {"ok": True, "job_id": out.get("job_id"), "queued": True}


AUTORETRAIN = AutoRetrainScheduler()
