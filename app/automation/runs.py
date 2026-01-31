from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.db import db_conn
from app.realtime.bus import publish_sync


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


@dataclass(frozen=True)
class AutomationRun:
    job_id: str
    kind: str


def _row(job_id: str) -> dict[str, Any] | None:
    with db_conn() as conn:
        r = conn.execute(
            "SELECT job_id, ts_start, ts_end, status, kind, progress, message, params_json, stats_json, error FROM automation_runs WHERE job_id=?",
            (str(job_id),),
        ).fetchone()
        if not r:
            return None

        def _loads(s: str | None) -> Any:
            try:
                return json.loads(s) if s else None
            except Exception:
                return None

        return {
            "job_id": str(r["job_id"]),
            "ts_start": int(r["ts_start"]),
            "ts_end": (None if r["ts_end"] is None else int(r["ts_end"])),
            "status": str(r["status"]),
            "kind": str(r["kind"]),
            "progress": float(r["progress"]),
            "message": r["message"],
            "params": _loads(r["params_json"]),
            "stats": _loads(r["stats_json"]),
            "error": r["error"],
        }


def get_run(job_id: str) -> dict[str, Any] | None:
    return _row(job_id)


def list_runs(*, limit: int = 50, status: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    q = "SELECT job_id FROM automation_runs"
    params: list[Any] = []
    where: list[str] = []
    if status:
        where.append("status=?")
        params.append(str(status).upper())
    if kind:
        where.append("kind=?")
        params.append(str(kind))
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY ts_start DESC LIMIT ?"
    params.append(int(limit))

    out: list[dict[str, Any]] = []
    with db_conn() as conn:
        for r in conn.execute(q, tuple(params)).fetchall():
            row = _row(str(r["job_id"]))
            if row:
                out.append(row)
    return out


def start_run(*, job_id: str, kind: str, params: dict[str, Any] | None = None) -> None:
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO automation_runs (job_id, ts_start, ts_end, status, kind, progress, message, params_json, stats_json, error) "
            "VALUES (?, ?, NULL, 'RUNNING', ?, 0.0, NULL, ?, NULL, NULL)",
            (
                str(job_id),
                _now_ts(),
                str(kind),
                json.dumps(params or {}, ensure_ascii=False, separators=(",", ":")),
            ),
        )
    publish_sync("automation", "automation_run.started", {"job_id": str(job_id), "kind": str(kind), "params": params or {}})


def update_run(
    job_id: str,
    *,
    progress: float | None = None,
    message: str | None = None,
    status: str | None = None,
    stats: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    sets: list[str] = []
    params: list[Any] = []

    if progress is not None:
        sets.append("progress=?")
        params.append(float(progress))
    if message is not None:
        sets.append("message=?")
        params.append(str(message))
    if stats is not None:
        sets.append("stats_json=?")
        params.append(json.dumps(stats or {}, ensure_ascii=False, separators=(",", ":")))
    if status is not None:
        st = str(status).upper()
        sets.append("status=?")
        params.append(st)
        if st in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            sets.append("ts_end=?")
            params.append(_now_ts())
    if error is not None:
        sets.append("error=?")
        params.append(str(error)[:2000])

    if not sets:
        return

    params.append(str(job_id))
    with db_conn() as conn:
        conn.execute(f"UPDATE automation_runs SET {', '.join(sets)} WHERE job_id=?", tuple(params))

    # Best-effort realtime.
    try:
        payload = {"job_id": str(job_id)}
        if progress is not None:
            payload["progress"] = float(progress)
        if message is not None:
            payload["message"] = str(message)
        if status is not None:
            payload["status"] = str(status)
        if stats is not None:
            payload["stats"] = stats
        if error is not None:
            payload["error"] = str(error)
        publish_sync("automation", "automation_run.updated", payload)
    except Exception:
        pass
