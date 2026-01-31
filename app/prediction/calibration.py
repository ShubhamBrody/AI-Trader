from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.db import db_conn


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def get_calibration(key: str) -> dict[str, Any] | None:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT key, updated_ts, n, bias, mae FROM online_calibration WHERE key=?",
            (str(key),),
        ).fetchone()
        if not row:
            return None
        return {
            "key": row["key"],
            "updated_ts": int(row["updated_ts"]),
            "n": int(row["n"]),
            "bias": float(row["bias"]),
            "mae": float(row["mae"]),
        }


def upsert_calibration(key: str, *, n: int, bias: float, mae: float) -> None:
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO online_calibration (key, updated_ts, n, bias, mae) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET updated_ts=excluded.updated_ts, n=excluded.n, bias=excluded.bias, mae=excluded.mae",
            (str(key), _now_ts(), int(n), float(bias), float(mae)),
        )


def update_ema(key: str, *, residual: float, abs_error: float, alpha: float = 0.05) -> dict[str, Any]:
    """Online calibration using EMA of residual and MAE.

    residual = pred_ret - actual_ret. We subtract bias from raw predictions.
    """

    alpha = float(max(0.001, min(0.5, alpha)))

    cur = get_calibration(key)
    if not cur:
        n = 1
        bias = float(residual)
        mae = float(abs_error)
    else:
        n = int(cur.get("n") or 0) + 1
        bias0 = float(cur.get("bias") or 0.0)
        mae0 = float(cur.get("mae") or 0.0)
        bias = (1.0 - alpha) * bias0 + alpha * float(residual)
        mae = (1.0 - alpha) * mae0 + alpha * float(abs_error)

    upsert_calibration(key, n=n, bias=bias, mae=mae)
    return {"key": str(key), "n": int(n), "bias": float(bias), "mae": float(mae), "updated_ts": _now_ts()}
