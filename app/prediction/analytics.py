from __future__ import annotations

import math
import time
from typing import Any

from app.core.db import db_conn


def _now_ts() -> int:
    return int(time.time())


def _sign_up(v: float) -> bool:
    # Treat 0 as up/positive to keep bins deterministic.
    return float(v) >= 0.0


def rollups(
    *,
    since_days: int = 30,
    limit: int = 200,
    instrument_key: str | None = None,
    interval: str | None = None,
    horizon_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregate resolved prediction quality by instrument/interval/horizon."""

    since_days = max(1, min(int(since_days), 3650))
    limit = max(1, min(int(limit), 2000))
    since_ts = _now_ts() - since_days * 86400

    q = (
        "SELECT instrument_key, interval, horizon_steps, "
        "COUNT(*) AS n, "
        "AVG(error) AS mean_error, "
        "AVG(ABS(error)) AS mae, "
        "SUM(error * error) AS sse, "
        "SUM(CASE WHEN pred_ret_adj >= 0 AND actual_ret >= 0 THEN 1 ELSE 0 END) AS tp, "
        "SUM(CASE WHEN pred_ret_adj < 0 AND actual_ret < 0 THEN 1 ELSE 0 END) AS tn, "
        "SUM(CASE WHEN pred_ret_adj >= 0 AND actual_ret < 0 THEN 1 ELSE 0 END) AS fp, "
        "SUM(CASE WHEN pred_ret_adj < 0 AND actual_ret >= 0 THEN 1 ELSE 0 END) AS fn, "
        "AVG(pred_ret_adj) AS avg_pred, "
        "AVG(actual_ret) AS avg_actual "
        "FROM prediction_events "
        "WHERE status='RESOLVED' AND ts_pred >= ? "
    )
    params: list[Any] = [int(since_ts)]

    if instrument_key:
        q += " AND instrument_key=?"
        params.append(str(instrument_key))
    if interval:
        q += " AND interval=?"
        params.append(str(interval))
    if horizon_steps is not None:
        q += " AND horizon_steps=?"
        params.append(int(horizon_steps))

    q += " GROUP BY instrument_key, interval, horizon_steps ORDER BY n DESC LIMIT ?"
    params.append(int(limit))

    rows: list[dict[str, Any]] = []
    with db_conn() as conn:
        for r in conn.execute(q, tuple(params)).fetchall():
            n = int(r["n"] or 0)
            tp = int(r["tp"] or 0)
            tn = int(r["tn"] or 0)
            fp = int(r["fp"] or 0)
            fn = int(r["fn"] or 0)
            sse = float(r["sse"] or 0.0)
            mae = float(r["mae"] or 0.0)
            mean_error = float(r["mean_error"] or 0.0)
            rmse = (math.sqrt(sse / n) if n > 0 else 0.0)
            acc = float((tp + tn) / n) if n > 0 else 0.0

            rows.append(
                {
                    "instrument_key": str(r["instrument_key"]),
                    "interval": str(r["interval"]),
                    "horizon_steps": int(r["horizon_steps"]),
                    "n": int(n),
                    "accuracy": float(acc),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "mean_error": float(mean_error),
                    "avg_pred": float(r["avg_pred"] or 0.0),
                    "avg_actual": float(r["avg_actual"] or 0.0),
                    "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                }
            )

    return rows


def reliability_bins(
    *,
    since_days: int = 30,
    instrument_key: str | None = None,
    interval: str | None = None,
    horizon_steps: int | None = None,
    bins: list[float] | None = None,
    limit_rows: int = 20000,
) -> list[dict[str, Any]]:
    """Bucket reliability by |pred_ret_adj| magnitude.

    This is a simple UI-friendly proxy for a reliability curve.
    """

    since_days = max(1, min(int(since_days), 3650))
    limit_rows = max(1, min(int(limit_rows), 100000))
    since_ts = _now_ts() - since_days * 86400

    edges = bins or [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 1.0]
    edges = sorted({float(x) for x in edges if x is not None and float(x) >= 0.0})
    if not edges or edges[0] != 0.0:
        edges = [0.0] + edges
    if len(edges) < 2:
        edges = [0.0, 1.0]

    # Ensure last edge is large enough.
    if edges[-1] < 1.0:
        edges.append(1.0)

    agg = []
    for i in range(len(edges) - 1):
        agg.append(
            {
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "n": 0,
                "correct": 0,
                "sum_abs_err": 0.0,
                "sum_err": 0.0,
            }
        )

    q = "SELECT pred_ret_adj, actual_ret, error FROM prediction_events WHERE status='RESOLVED' AND ts_pred >= ?"
    params: list[Any] = [int(since_ts)]
    if instrument_key:
        q += " AND instrument_key=?"
        params.append(str(instrument_key))
    if interval:
        q += " AND interval=?"
        params.append(str(interval))
    if horizon_steps is not None:
        q += " AND horizon_steps=?"
        params.append(int(horizon_steps))
    q += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit_rows))

    with db_conn() as conn:
        rows = conn.execute(q, tuple(params)).fetchall()

    for r in rows:
        pred = float(r["pred_ret_adj"] or 0.0)
        actual = float(r["actual_ret"] or 0.0)
        err = float(r["error"] or 0.0)

        mag = abs(pred)
        idx = None
        for i, b in enumerate(agg):
            if float(b["lo"]) <= mag < float(b["hi"]):
                idx = i
                break
        if idx is None:
            continue

        b = agg[idx]
        b["n"] += 1
        if _sign_up(pred) == _sign_up(actual):
            b["correct"] += 1
        b["sum_abs_err"] += abs(err)
        b["sum_err"] += err

    out: list[dict[str, Any]] = []
    for b in agg:
        n = int(b["n"])
        if n <= 0:
            out.append({"lo": b["lo"], "hi": b["hi"], "n": 0, "accuracy": 0.0, "mae": 0.0, "mean_error": 0.0})
            continue
        out.append(
            {
                "lo": float(b["lo"]),
                "hi": float(b["hi"]),
                "n": n,
                "accuracy": float(int(b["correct"]) / n),
                "mae": float(float(b["sum_abs_err"]) / n),
                "mean_error": float(float(b["sum_err"]) / n),
            }
        )
    return out


def drift(
    *,
    baseline_days: int = 30,
    recent_days: int = 7,
    limit: int = 200,
    instrument_key: str | None = None,
    interval: str | None = None,
    horizon_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Compare recent vs baseline quality to spot drift."""

    baseline_days = max(2, min(int(baseline_days), 3650))
    recent_days = max(1, min(int(recent_days), baseline_days - 1))

    now = _now_ts()
    recent_start = now - recent_days * 86400
    baseline_start = now - baseline_days * 86400

    base = _rollups_for_range(
        start_ts=baseline_start,
        end_ts=recent_start,
        limit=limit,
        instrument_key=instrument_key,
        interval=interval,
        horizon_steps=horizon_steps,
    )
    recent = _rollups_for_range(
        start_ts=recent_start,
        end_ts=now + 1,
        limit=limit,
        instrument_key=instrument_key,
        interval=interval,
        horizon_steps=horizon_steps,
    )

    def _k(x: dict[str, Any]) -> str:
        return f"{x['instrument_key']}::{x['interval']}::h{int(x['horizon_steps'])}"

    base_map = {_k(x): x for x in base}
    recent_map = {_k(x): x for x in recent}

    keys = sorted(set(base_map.keys()) | set(recent_map.keys()))
    out: list[dict[str, Any]] = []
    for k in keys[: int(limit)]:
        b = base_map.get(k)
        r = recent_map.get(k)
        out.append(
            {
                "key": k,
                "baseline": b,
                "recent": r,
                "delta": {
                    "accuracy": (None if not b or not r else float(r.get("accuracy", 0.0) - b.get("accuracy", 0.0))),
                    "mae": (None if not b or not r else float(r.get("mae", 0.0) - b.get("mae", 0.0))),
                    "rmse": (None if not b or not r else float(r.get("rmse", 0.0) - b.get("rmse", 0.0))),
                    "mean_error": (None if not b or not r else float(r.get("mean_error", 0.0) - b.get("mean_error", 0.0))),
                },
            }
        )

    return out


def _rollups_for_range(
    *,
    start_ts: int,
    end_ts: int,
    limit: int,
    instrument_key: str | None,
    interval: str | None,
    horizon_steps: int | None,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 2000))

    q = (
        "SELECT instrument_key, interval, horizon_steps, "
        "COUNT(*) AS n, "
        "AVG(error) AS mean_error, "
        "AVG(ABS(error)) AS mae, "
        "SUM(error * error) AS sse, "
        "SUM(CASE WHEN pred_ret_adj >= 0 AND actual_ret >= 0 THEN 1 ELSE 0 END) AS tp, "
        "SUM(CASE WHEN pred_ret_adj < 0 AND actual_ret < 0 THEN 1 ELSE 0 END) AS tn, "
        "SUM(CASE WHEN pred_ret_adj >= 0 AND actual_ret < 0 THEN 1 ELSE 0 END) AS fp, "
        "SUM(CASE WHEN pred_ret_adj < 0 AND actual_ret >= 0 THEN 1 ELSE 0 END) AS fn, "
        "AVG(pred_ret_adj) AS avg_pred, "
        "AVG(actual_ret) AS avg_actual "
        "FROM prediction_events "
        "WHERE status='RESOLVED' AND ts_pred >= ? AND ts_pred < ?"
    )
    params: list[Any] = [int(start_ts), int(end_ts)]

    if instrument_key:
        q += " AND instrument_key=?"
        params.append(str(instrument_key))
    if interval:
        q += " AND interval=?"
        params.append(str(interval))
    if horizon_steps is not None:
        q += " AND horizon_steps=?"
        params.append(int(horizon_steps))

    q += " GROUP BY instrument_key, interval, horizon_steps ORDER BY n DESC LIMIT ?"
    params.append(int(limit))

    out: list[dict[str, Any]] = []
    with db_conn() as conn:
        for r in conn.execute(q, tuple(params)).fetchall():
            n = int(r["n"] or 0)
            tp = int(r["tp"] or 0)
            tn = int(r["tn"] or 0)
            fp = int(r["fp"] or 0)
            fn = int(r["fn"] or 0)
            sse = float(r["sse"] or 0.0)
            mae = float(r["mae"] or 0.0)
            mean_error = float(r["mean_error"] or 0.0)
            rmse = (math.sqrt(sse / n) if n > 0 else 0.0)
            acc = float((tp + tn) / n) if n > 0 else 0.0

            out.append(
                {
                    "instrument_key": str(r["instrument_key"]),
                    "interval": str(r["interval"]),
                    "horizon_steps": int(r["horizon_steps"]),
                    "n": int(n),
                    "accuracy": float(acc),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "mean_error": float(mean_error),
                    "avg_pred": float(r["avg_pred"] or 0.0),
                    "avg_actual": float(r["avg_actual"] or 0.0),
                    "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                }
            )

    return out


def list_calibrations(
    *,
    instrument_key: str | None = None,
    interval: str | None = None,
    horizon_steps: int | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 5000))

    where: list[str] = []
    params: list[Any] = []

    like = None
    if instrument_key and interval and horizon_steps is not None:
        like = f"{instrument_key}::{interval}::h{int(horizon_steps)}"
        where.append("key=?")
        params.append(str(like))
    elif instrument_key and interval:
        like = f"{instrument_key}::{interval}::h%"
        where.append("key LIKE ?")
        params.append(str(like))
    elif instrument_key:
        like = f"{instrument_key}::%"
        where.append("key LIKE ?")
        params.append(str(like))

    q = "SELECT key, updated_ts, n, bias, mae FROM online_calibration"
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY updated_ts DESC LIMIT ?"
    params.append(int(limit))

    out: list[dict[str, Any]] = []
    with db_conn() as conn:
        for r in conn.execute(q, tuple(params)).fetchall():
            out.append(
                {
                    "key": str(r["key"]),
                    "updated_ts": int(r["updated_ts"]),
                    "n": int(r["n"]),
                    "bias": float(r["bias"]),
                    "mae": float(r["mae"]),
                }
            )
    return out
