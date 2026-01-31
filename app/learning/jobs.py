from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.core.db import db_conn
from app.realtime.bus import publish_sync


@dataclass(frozen=True)
class TrainingJob:
    job_id: str
    kind: str
    instrument_key: str | None
    interval: str | None
    horizon_steps: int | None
    model_family: str | None
    cap_tier: str | None


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _job_row(job_id: str) -> dict[str, Any] | None:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT job_id, ts_start, ts_end, status, kind, instrument_key, interval, horizon_steps, model_family, cap_tier, progress, message, metrics_json, error FROM training_runs WHERE job_id=?",
            (str(job_id),),
        ).fetchone()
        if not row:
            return None
        try:
            metrics = json.loads(row["metrics_json"] or "null")
        except Exception:
            metrics = None
        return {
            "job_id": row["job_id"],
            "ts_start": int(row["ts_start"]),
            "ts_end": (None if row["ts_end"] is None else int(row["ts_end"])),
            "status": row["status"],
            "kind": row["kind"],
            "instrument_key": row["instrument_key"],
            "interval": row["interval"],
            "horizon_steps": (None if row["horizon_steps"] is None else int(row["horizon_steps"])),
            "model_family": row["model_family"],
            "cap_tier": row["cap_tier"],
            "progress": float(row["progress"]),
            "message": row["message"],
            "metrics": metrics,
            "error": row["error"],
        }


def get_job(job_id: str) -> dict[str, Any] | None:
    return _job_row(job_id)


def list_jobs(*, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    q = "SELECT job_id FROM training_runs"
    params: list[Any] = []
    if status:
        q += " WHERE status=?"
        params.append(str(status))
    q += " ORDER BY ts_start DESC LIMIT ?"
    params.append(int(limit))

    out: list[dict[str, Any]] = []
    with db_conn() as conn:
        cur = conn.execute(q, tuple(params))
        for r in cur.fetchall():
            row = _job_row(str(r["job_id"]))
            if row:
                out.append(row)
    return out


def _insert_job(job: TrainingJob) -> None:
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO training_runs (job_id, ts_start, ts_end, status, kind, instrument_key, interval, horizon_steps, model_family, cap_tier, progress, message, metrics_json, error) "
            "VALUES (?, ?, NULL, 'RUNNING', ?, ?, ?, ?, ?, ?, 0.0, NULL, NULL, NULL)",
            (
                job.job_id,
                _now_ts(),
                job.kind,
                job.instrument_key,
                job.interval,
                job.horizon_steps,
                job.model_family,
                job.cap_tier,
            ),
        )


def _update_job(job_id: str, *, progress: float | None = None, message: str | None = None, metrics: dict[str, Any] | None = None, status: str | None = None, error: str | None = None) -> None:
    sets: list[str] = []
    params: list[Any] = []

    if progress is not None:
        sets.append("progress=?")
        params.append(float(progress))
    if message is not None:
        sets.append("message=?")
        params.append(str(message))
    if metrics is not None:
        sets.append("metrics_json=?")
        params.append(json.dumps(metrics, ensure_ascii=False, separators=(",", ":")))
    if status is not None:
        sets.append("status=?")
        params.append(str(status))
        if str(status).upper() in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            sets.append("ts_end=?")
            params.append(_now_ts())
    if error is not None:
        sets.append("error=?")
        params.append(str(error)[:2000])

    if not sets:
        return

    params.append(str(job_id))
    with db_conn() as conn:
        conn.execute(f"UPDATE training_runs SET {', '.join(sets)} WHERE job_id=?", tuple(params))


class TrainingJobManager:
    def __init__(self) -> None:
        self._threads: dict[str, threading.Thread] = {}

    def start_threaded(self, job: TrainingJob, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        _insert_job(job)
        publish_sync("training", "training.started", {"job_id": job.job_id, "kind": job.kind, "instrument_key": job.instrument_key, "interval": job.interval})

        def _runner() -> None:
            try:
                out = fn() or {}
                ok = bool(out.get("ok"))
                if ok:
                    _update_job(job.job_id, progress=1.0, status="SUCCEEDED", metrics=out)
                    publish_sync("training", "training.succeeded", {"job_id": job.job_id, "result": out})
                else:
                    _update_job(job.job_id, status="FAILED", error=str(out.get("reason") or out.get("error") or "failed"), metrics=out)
                    publish_sync("training", "training.failed", {"job_id": job.job_id, "result": out})
            except Exception as e:
                _update_job(job.job_id, status="FAILED", error=str(e))
                publish_sync("training", "training.failed", {"job_id": job.job_id, "error": str(e)})

        th = threading.Thread(target=_runner, name=f"training:{job.job_id}", daemon=True)
        self._threads[job.job_id] = th
        th.start()
        return {"ok": True, "job_id": job.job_id}

    def progress_cb(self, job_id: str) -> Callable[[float, str, dict[str, Any] | None], None]:
        start_wall_ts = _now_ts()

        def _cb(progress: float, message: str, metrics: dict[str, Any] | None = None) -> None:
            # Enrich metrics with a best-effort ETA estimation.
            # This is intentionally simple (linear extrapolation) but works well for long batch jobs.
            p = max(0.0, min(1.0, float(progress)))
            now_ts = _now_ts()
            elapsed = max(0, int(now_ts - start_wall_ts))

            merged: dict[str, Any] = {}
            if metrics:
                merged.update(metrics)

            merged["elapsed_seconds"] = int(elapsed)
            if 0.0 < p < 1.0 and elapsed >= 2:
                eta = int(round(elapsed * (1.0 - p) / max(p, 1e-6)))
                # clamp to a sensible range to avoid absurd spikes from tiny progress.
                eta = max(0, min(eta, 60 * 60 * 24 * 30))
                merged["eta_seconds"] = int(eta)
                merged["eta_ts_end"] = int(now_ts + eta)
            else:
                merged["eta_seconds"] = None
                merged["eta_ts_end"] = None

            _update_job(job_id, progress=p, message=message, metrics=merged)
            publish_sync(
                "training",
                "training.progress",
                {"job_id": job_id, "progress": p, "message": message, "metrics": merged},
            )

        return _cb


JOBS = TrainingJobManager()


def new_job_id() -> str:
    return str(uuid.uuid4())
