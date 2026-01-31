from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.core.db import db_conn


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def insert_prediction(
    *,
    ts_pred: int,
    instrument_key: str,
    interval: str,
    horizon_steps: int,
    model_kind: str,
    model_key: str | None,
    pred_ret: float,
    pred_ret_adj: float,
    calib_bias: float,
    ts_target: int,
    meta: dict[str, Any] | None,
) -> int:
    meta_json = json.dumps(meta or {}, ensure_ascii=False, separators=(",", ":"))
    with db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO prediction_events (ts_pred, instrument_key, interval, horizon_steps, model_kind, model_key, pred_ret, pred_ret_adj, calib_bias, ts_target, status, ts_resolved, actual_ret, error, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', NULL, NULL, NULL, ?)",
            (
                int(ts_pred),
                str(instrument_key),
                str(interval),
                int(horizon_steps),
                str(model_kind),
                (None if model_key is None else str(model_key)),
                float(pred_ret),
                float(pred_ret_adj),
                float(calib_bias),
                int(ts_target),
                meta_json,
            ),
        )
        return int(cur.lastrowid)


def list_predictions(*, limit: int = 200, instrument_key: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 2000))
    q = "SELECT * FROM prediction_events"
    params: list[Any] = []
    where: list[str] = []
    if instrument_key:
        where.append("instrument_key=?")
        params.append(str(instrument_key))
    if status:
        where.append("status=?")
        params.append(str(status).upper())
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))

    out: list[dict[str, Any]] = []
    with db_conn() as conn:
        for r in conn.execute(q, tuple(params)).fetchall():
            try:
                meta = json.loads(r["meta_json"] or "{}")
            except Exception:
                meta = {}
            out.append(
                {
                    "id": int(r["id"]),
                    "ts_pred": int(r["ts_pred"]),
                    "instrument_key": r["instrument_key"],
                    "interval": r["interval"],
                    "horizon_steps": int(r["horizon_steps"]),
                    "model_kind": r["model_kind"],
                    "model_key": r["model_key"],
                    "pred_ret": float(r["pred_ret"]),
                    "pred_ret_adj": float(r["pred_ret_adj"]),
                    "calib_bias": float(r["calib_bias"]),
                    "ts_target": int(r["ts_target"]),
                    "status": r["status"],
                    "ts_resolved": (None if r["ts_resolved"] is None else int(r["ts_resolved"])),
                    "actual_ret": (None if r["actual_ret"] is None else float(r["actual_ret"])),
                    "error": (None if r["error"] is None else float(r["error"])),
                    "meta": meta,
                }
            )
    return out


def fetch_due_pending(*, now_ts: int, limit: int = 500) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 5000))
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM prediction_events WHERE status='PENDING' AND ts_target<=? ORDER BY ts_target ASC LIMIT ?",
            (int(now_ts), int(limit)),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": int(r["id"]),
                "instrument_key": r["instrument_key"],
                "interval": r["interval"],
                "horizon_steps": int(r["horizon_steps"]),
                "ts_pred": int(r["ts_pred"]),
                "ts_target": int(r["ts_target"]),
                "pred_ret_adj": float(r["pred_ret_adj"]),
                "pred_ret": float(r["pred_ret"]),
                "model_kind": r["model_kind"],
                "model_key": r["model_key"],
            }
        )
    return out


def resolve_prediction(*, pred_id: int, actual_ret: float) -> None:
    # error stored as (pred_adj - actual)
    with db_conn() as conn:
        row = conn.execute("SELECT pred_ret_adj FROM prediction_events WHERE id=?", (int(pred_id),)).fetchone()
        pred_adj = float(row["pred_ret_adj"]) if row else 0.0
        err = float(pred_adj - float(actual_ret))
        conn.execute(
            "UPDATE prediction_events SET status='RESOLVED', ts_resolved=?, actual_ret=?, error=? WHERE id=?",
            (_now_ts(), float(actual_ret), float(err), int(pred_id)),
        )
