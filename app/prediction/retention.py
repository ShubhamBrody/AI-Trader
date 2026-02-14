from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass(frozen=True)
class PredictionRetentionResult:
    enabled: bool
    max_age_days: int
    max_rows: int
    deleted_by_age: int
    deleted_by_rows: int
    remaining_rows: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def cleanup_prediction_events(
    *,
    conn: Any,
    enabled: bool,
    max_age_days: int,
    max_rows: int,
    now: datetime | None = None,
) -> PredictionRetentionResult:
    """Apply retention policy to prediction_events.

    Policy:
    - If enabled is False: no-op.
    - If max_age_days > 0: delete rows with ts_pred < cutoff.
    - If max_rows > 0: keep only newest max_rows rows by ts_pred.

    Returns counts and remaining rows.

    Notes:
    - Designed for SQLite but only relies on basic SQL.
    - Uses ts_pred as the event timestamp. ts_pred is stored as an integer (epoch seconds).
    """

    now = now or _utc_now()

    deleted_by_age = 0
    deleted_by_rows = 0

    cur = conn.cursor()

    if not enabled:
        cur.execute("SELECT COUNT(*) FROM prediction_events")
        remaining = int(cur.fetchone()[0])
        return PredictionRetentionResult(
            enabled=False,
            max_age_days=max_age_days,
            max_rows=max_rows,
            deleted_by_age=0,
            deleted_by_rows=0,
            remaining_rows=remaining,
        )

    if max_age_days and max_age_days > 0:
        cutoff_ts = int((now - timedelta(days=max_age_days)).timestamp())
        cur.execute("DELETE FROM prediction_events WHERE ts_pred < ?", (cutoff_ts,))
        deleted_by_age = int(getattr(cur, "rowcount", 0) or 0)

    if max_rows and max_rows > 0:
        # Delete everything except the newest N rows.
        # Uses a subquery with ORDER BY/LIMIT to compute the kept set.
        cur.execute(
            """
            DELETE FROM prediction_events
            WHERE id NOT IN (
                SELECT id
                FROM prediction_events
                ORDER BY ts_pred DESC, id DESC
                LIMIT ?
            )
            """,
            (int(max_rows),),
        )
        deleted_by_rows = int(getattr(cur, "rowcount", 0) or 0)

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM prediction_events")
    remaining = int(cur.fetchone()[0])

    return PredictionRetentionResult(
        enabled=True,
        max_age_days=int(max_age_days or 0),
        max_rows=int(max_rows or 0),
        deleted_by_age=deleted_by_age,
        deleted_by_rows=deleted_by_rows,
        remaining_rows=remaining,
    )
