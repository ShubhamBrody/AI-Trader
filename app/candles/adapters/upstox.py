from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
from loguru import logger

from app.candles.models import Candle
from app.core.settings import settings
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError, parse_upstox_candles
from app.utils.perf import perf_span


def _seed_from_key(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _generate_ohlc_series(start: datetime, end: datetime, interval_seconds: int, seed: int) -> list[Candle]:
    if end <= start:
        return []

    rng = np.random.default_rng(seed)

    # Build timestamps (epoch seconds)
    times: list[int] = []
    cur = start
    while cur <= end:
        times.append(int(cur.timestamp()))
        cur += timedelta(seconds=interval_seconds)

    if not times:
        return []

    base = 100.0 + (seed % 500) / 10.0
    prices = base + np.cumsum(rng.normal(0, 0.3, size=len(times)))

    candles: list[Candle] = []
    prev_close = float(prices[0])

    for i, ts in enumerate(times):
        o = prev_close
        delta = float(prices[i] - prev_close)
        c = o + delta
        hi = max(o, c) + float(abs(rng.normal(0, 0.15)))
        lo = min(o, c) - float(abs(rng.normal(0, 0.15)))
        v = float(abs(rng.normal(1_000_000, 250_000)))
        candles.append(Candle(ts=ts, open=o, high=hi, low=lo, close=c, volume=v))
        prev_close = c

    return candles


def _parse_interval(interval: str) -> tuple[str, int] | None:
    """Map our internal interval strings to Upstox V3 (unit, interval)."""
    interval = interval.strip().lower()
    mapping = {
        "1m": ("minutes", 1),
        "3m": ("minutes", 3),
        "5m": ("minutes", 5),
        "10m": ("minutes", 10),
        "15m": ("minutes", 15),
        "30m": ("minutes", 30),
        "45m": ("minutes", 45),
        "1h": ("hours", 1),
        "2h": ("hours", 2),
        "4h": ("hours", 4),
        "1d": ("days", 1),
    }
    return mapping.get(interval)


def _fetch_from_upstox_v3(instrument_key: str, interval: str, start: datetime, end: datetime) -> list[Candle]:
    with perf_span(
        "upstox.v3.fetch",
        instrument_key=instrument_key,
        interval=interval,
        start=int(start.timestamp()),
        end=int(end.timestamp()),
    ):
        parsed = _parse_interval(interval)
        if parsed is None:
            raise UpstoxError(f"Unsupported interval for Upstox V3 mapping: {interval}")
        unit, intv = parsed

        cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
        client = UpstoxClient(cfg)
        try:
            # Upstox expects dates (YYYY-MM-DD) in exchange timezone.
            # Using UTC dates can shift the requested window across two dates and often returns empty.
            tz = ZoneInfo(settings.TIMEZONE)
            to_d = end.astimezone(tz).date()
            from_d = start.astimezone(tz).date()

            # For intraday, Upstox provides a dedicated intraday endpoint for the *current* session.
            # The date-based historical endpoint frequently returns empty for "today".
            today = datetime.now(timezone.utc).astimezone(tz).date()

            rows: list[list[object]] = []
            if interval.strip().lower() != "1d" and to_d == today:
                # Case A: window entirely within today -> intraday endpoint only.
                if from_d == to_d:
                    rows = client.intraday_candles_v3(instrument_key=instrument_key, unit=unit, interval=intv)
                else:
                    # Case B: range spans earlier dates + today.
                    # Pull historical up to yesterday and add intraday for today.
                    yesterday = today - timedelta(days=1)
                    if from_d <= yesterday:
                        rows = client.historical_candles_v3(
                            instrument_key=instrument_key,
                            unit=unit,
                            interval=intv,
                            to_date=yesterday,
                            from_date=from_d,
                        )
                    rows = rows + client.intraday_candles_v3(instrument_key=instrument_key, unit=unit, interval=intv)
            else:
                rows = client.historical_candles_v3(
                    instrument_key=instrument_key,
                    unit=unit,
                    interval=intv,
                    to_date=to_d,
                    from_date=from_d,
                )
            parsed_rows = parse_upstox_candles(rows)

            # Filter down to requested start/end (in case Upstox includes extra edges)
            start_ts = int(start.timestamp())
            end_ts = int(end.timestamp())
            parsed_rows = [r for r in parsed_rows if start_ts <= int(r["ts"]) <= end_ts]
            return [Candle(**r) for r in parsed_rows]
        finally:
            client.close()


def fetch_intraday(instrument_key: str, interval: str) -> list[Candle]:
    with perf_span("upstox.fetch_intraday", instrument_key=instrument_key, interval=interval):
        # Never hit external APIs during tests.
        if getattr(settings, "APP_ENV", "") == "test":
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=120)
            interval_seconds = {
                "1m": 60,
                "3m": 180,
                "5m": 300,
                "10m": 600,
                "15m": 900,
                "30m": 1800,
                "45m": 2700,
                "1h": 3600,
                "2h": 7200,
                "4h": 14400,
            }.get(interval.strip().lower(), 300)
            seed = _seed_from_key(f"{instrument_key}:{interval}:intraday")
            return _generate_ohlc_series(start, end, interval_seconds=interval_seconds, seed=seed)

        parsed = _parse_interval(interval)
        if parsed is None:
            raise UpstoxError(f"Unsupported interval for Upstox V3 mapping: {interval}")
        unit, intv = parsed

        cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
        client = UpstoxClient(cfg)
        try:
            rows = client.intraday_candles_v3(instrument_key=instrument_key, unit=unit, interval=intv)
            parsed_rows = parse_upstox_candles(rows)
            return [Candle(**r) for r in parsed_rows]
        finally:
            client.close()


def fetch_candles(instrument_key: str, interval: str, start: datetime, end: datetime) -> list[Candle]:
    """Fetch candles.

    - If `UPSTOX_ACCESS_TOKEN` is configured, uses **Upstox V3 historical candle API**.
    - If not configured (or on failure), falls back to deterministic synthetic candles unless `UPSTOX_STRICT=true`.
    """

    with perf_span("upstox.fetch_candles", instrument_key=instrument_key, interval=interval):
        # Never hit external APIs during tests.
        if getattr(settings, "APP_ENV", "") == "test":
            interval_seconds = {
                "1d": 24 * 3600,
                "1h": 3600,
                "30m": 1800,
                "15m": 900,
                "5m": 300,
                "1m": 60,
            }.get(interval, 300)

            seed = _seed_from_key(f"{instrument_key}:{interval}:test")
            return _generate_ohlc_series(start, end, interval_seconds=interval_seconds, seed=seed)

        # If the user is authenticated via the OAuth flow, the token is stored in `token_store`.
        # UpstoxClient already checks token_store + env, so don't gate on UPSTOX_ACCESS_TOKEN here.
        try:
            return _fetch_from_upstox_v3(instrument_key, interval, start, end)
        except Exception as e:
            logger.warning("Upstox candle fetch failed (will{} fallback): {}", " NOT" if settings.UPSTOX_STRICT else "", str(e))
            if settings.UPSTOX_STRICT:
                raise

        # Fallback deterministic candles
        interval_seconds = {
            "1d": 24 * 3600,
            "1h": 3600,
            "30m": 1800,
            "15m": 900,
            "5m": 300,
            "1m": 60,
        }.get(interval, 300)

        seed = _seed_from_key(f"{instrument_key}:{interval}")
        return _generate_ohlc_series(start, end, interval_seconds=interval_seconds, seed=seed)
