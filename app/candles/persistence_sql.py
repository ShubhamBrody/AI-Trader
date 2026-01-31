from __future__ import annotations

from typing import Iterable

from app.core.db import db_conn
from app.candles.models import Candle
from app.utils.perf import perf_span


def upsert_candles(instrument_key: str, interval: str, candles: Iterable[Candle]) -> int:
    with perf_span("db.candles.upsert", instrument_key=instrument_key, interval=interval):
        with db_conn() as conn:
            cur = conn.cursor()
            rows = 0
            for c in candles:
                cur.execute(
                    """
                    INSERT INTO candles (instrument_key, interval, ts, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(instrument_key, interval, ts) DO UPDATE SET
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        volume=excluded.volume
                    """,
                    (instrument_key, interval, int(c.ts), float(c.open), float(c.high), float(c.low), float(c.close), float(c.volume)),
                )
                rows += 1
            return rows


def get_candles(
    instrument_key: str,
    interval: str,
    start_ts: int,
    end_ts: int,
    *,
    limit: int | None = None,
) -> list[Candle]:
    with perf_span(
        "db.candles.get",
        instrument_key=instrument_key,
        interval=interval,
        start_ts=int(start_ts),
        end_ts=int(end_ts),
        limit=None if limit is None else int(limit),
    ):
        with db_conn() as conn:
            limit_n = None
            try:
                if limit is not None and int(limit) > 0:
                    limit_n = int(limit)
            except Exception:
                limit_n = None

            if limit_n is None:
                cur = conn.execute(
                    """
                    SELECT ts, open, high, low, close, volume
                    FROM candles
                    WHERE instrument_key=? AND interval=? AND ts BETWEEN ? AND ?
                    ORDER BY ts ASC
                    """,
                    (instrument_key, interval, int(start_ts), int(end_ts)),
                )
            else:
                # Fetch last N candles in the window efficiently, then reverse back to ASC.
                cur = conn.execute(
                    """
                    SELECT ts, open, high, low, close, volume
                    FROM candles
                    WHERE instrument_key=? AND interval=? AND ts BETWEEN ? AND ?
                    ORDER BY ts DESC
                    LIMIT ?
                    """,
                    (instrument_key, interval, int(start_ts), int(end_ts), limit_n),
                )

            out: list[Candle] = []
            for r in cur:
                out.append(
                    Candle(
                        ts=int(r["ts"]),
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                        volume=float(r["volume"]),
                    )
                )
            if limit_n is not None:
                out.reverse()
            return out


def latest_ts(instrument_key: str, interval: str) -> int | None:
    with perf_span("db.candles.latest_ts", instrument_key=instrument_key, interval=interval):
        with db_conn() as conn:
            cur = conn.execute(
                """
                SELECT ts
                FROM candles
                WHERE instrument_key=? AND interval=?
                ORDER BY ts DESC
                LIMIT 1
                """,
                (instrument_key, interval),
            )
            row = cur.fetchone()
            if row is None or row["ts"] is None:
                return None
            return int(row["ts"])
