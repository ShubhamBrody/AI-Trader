from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from threading import Lock

from zoneinfo import ZoneInfo

from app.candles.adapters.upstox import fetch_candles, fetch_intraday
from app.candles.models import CandleSeries
from app.candles.persistence_sql import get_candles, latest_ts, upsert_candles
from app.core.db import db_conn
from app.core.settings import settings
from app.markets.nse.calendar import last_n_trading_days, load_holidays
from app.utils.perf import perf_span


def _to_epoch(dt: datetime) -> int:
    return int(dt.timestamp())


class CandleService:
    _poll_cache_lock = Lock()
    _poll_cache: dict[tuple[str, str, int], tuple[float, CandleSeries]] = {}

    @staticmethod
    def _resolve_instrument_key(query: str) -> str:
        q = str(query or "").strip()
        if not q:
            return q
        if "|" in q:
            return q

        # Best-effort resolve via instrument master table (loaded separately).
        # This lets the UI pass symbols like "RELIANCE" or "RELIANCE-EQ".
        try:
            like = q.upper()
            candidates = [like]
            if not like.endswith("-EQ"):
                candidates.append(f"{like}-EQ")

            with db_conn() as conn:
                for sym in candidates:
                    row = conn.execute(
                        "SELECT instrument_key FROM instrument_meta WHERE UPPER(tradingsymbol)=? LIMIT 1",
                        (sym,),
                    ).fetchone()
                    if row is not None and row["instrument_key"]:
                        return str(row["instrument_key"])

                # Fallback: prefix match
                row = conn.execute(
                    "SELECT instrument_key FROM instrument_meta WHERE UPPER(tradingsymbol) LIKE ? ORDER BY updated_ts DESC LIMIT 1",
                    (f"{like}%",),
                ).fetchone()
                if row is not None and row["instrument_key"]:
                    return str(row["instrument_key"])
        except Exception:
            pass

        return q

    def load_historical(self, instrument_key: str, interval: str, start: datetime, end: datetime) -> CandleSeries:
        instrument_key = self._resolve_instrument_key(instrument_key)
        try:
            candles = fetch_candles(instrument_key, interval, start, end)
            upsert_candles(instrument_key, interval, candles)
            return CandleSeries(instrument_key=instrument_key, interval=interval, candles=candles)
        except Exception:
            # If upstream fetch fails (e.g., invalid instrument key / circuit breaker),
            # fall back to whatever is already cached. Do not crash the API.
            candles = get_candles(instrument_key, interval, _to_epoch(start), _to_epoch(end))
            return CandleSeries(instrument_key=instrument_key, interval=interval, candles=candles)

    def bulk_load_last_sessions(
        self,
        instrument_key: str,
        interval: str,
        num_trading_sessions: int,
        end_date: datetime | None = None,
    ) -> dict:
        instrument_key = self._resolve_instrument_key(instrument_key)
        tz = ZoneInfo(settings.TIMEZONE)
        end_local = (end_date or datetime.now(timezone.utc).astimezone(tz))

        holidays = load_holidays("app/config/nse_holidays.yaml")
        days = last_n_trading_days(end_local.date(), num_trading_sessions, holidays)
        if not days:
            return {"inserted": 0, "sessions": 0}

        inserted = 0
        for d in days:
            start = datetime(d.year, d.month, d.day, 9, 15, tzinfo=tz)
            end = datetime(d.year, d.month, d.day, 15, 29, tzinfo=tz)
            candles = fetch_candles(instrument_key, interval, start, end)
            inserted += upsert_candles(instrument_key, interval, candles)

        return {"inserted": inserted, "sessions": len(days)}

    def get_historical(
        self,
        instrument_key: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        limit: int | None = None,
    ) -> CandleSeries:
        instrument_key = self._resolve_instrument_key(instrument_key)

        start_ts = _to_epoch(start)
        end_ts = _to_epoch(end)
        candles = get_candles(instrument_key, interval, start_ts, end_ts, limit=limit)
        if candles:
            return CandleSeries(instrument_key=instrument_key, interval=interval, candles=candles)

        # Auto-warm on cache miss: fetch and persist so the UI doesn't show empty charts by default.
        try:
            fetched = fetch_candles(instrument_key, interval, start, end)
            if fetched:
                upsert_candles(instrument_key, interval, fetched)
                return CandleSeries(instrument_key=instrument_key, interval=interval, candles=fetched)
        except Exception:
            pass

        return CandleSeries(instrument_key=instrument_key, interval=interval, candles=[])

    def poll_intraday(self, instrument_key: str, interval: str, lookback_minutes: int) -> CandleSeries:
        instrument_key = self._resolve_instrument_key(instrument_key)
        with perf_span("candles.poll_intraday", instrument_key=instrument_key, interval=interval, lookback_minutes=int(lookback_minutes)):
            dedup_s = max(0, int(getattr(settings, "CANDLES_POLL_DEDUP_MS", 0) or 0)) / 1000.0
            cache_key = (str(instrument_key), str(interval), int(lookback_minutes))
            now = time.monotonic()

            if dedup_s > 0:
                with self._poll_cache_lock:
                    hit = self._poll_cache.get(cache_key)
                    if hit is not None:
                        ts, series = hit
                        if (now - float(ts)) < dedup_s:
                            return series

            series = self._poll_intraday_impl(instrument_key, interval, lookback_minutes)
            if dedup_s > 0:
                with self._poll_cache_lock:
                    self._poll_cache[cache_key] = (now, series)
            return series

    def _poll_intraday_impl(self, instrument_key: str, interval: str, lookback_minutes: int) -> CandleSeries:
        tz = ZoneInfo(settings.TIMEZONE)
        now_local = datetime.now(timezone.utc).astimezone(tz)
        window_start = now_local - timedelta(minutes=lookback_minutes)
        window_end = now_local
        fetch_start = window_start

        # If DB already has candles, only fetch newer ones.
        mx = latest_ts(instrument_key, interval)
        if mx is not None:
            fetch_start = max(fetch_start, datetime.fromtimestamp(mx, tz=tz) + timedelta(seconds=1))

        start_ts = int(window_start.timestamp())
        end_ts = int(window_end.timestamp())

        # Prefer broker intraday API when configured.
        try:
            candles = fetch_intraday(instrument_key, interval)
            # Filter to requested lookback.
            candles = [c for c in candles if start_ts <= int(c.ts) <= end_ts]
            upsert_candles(instrument_key, interval, candles)
            return CandleSeries(instrument_key=instrument_key, interval=interval, candles=candles)
        except Exception:
            # Fall back to historical fetch; if that also fails (strict mode / invalid key),
            # fall back to cached DB candles. Never raise from here.
            try:
                candles = fetch_candles(instrument_key, interval, fetch_start, window_end)
                candles = [c for c in candles if start_ts <= int(c.ts) <= end_ts]
                upsert_candles(instrument_key, interval, candles)
                return CandleSeries(instrument_key=instrument_key, interval=interval, candles=candles)
            except Exception:
                candles = get_candles(instrument_key, interval, start_ts, end_ts)
                return CandleSeries(instrument_key=instrument_key, interval=interval, candles=candles)
