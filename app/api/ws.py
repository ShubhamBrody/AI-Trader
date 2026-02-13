from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from loguru import logger

from app.candles.service import CandleService
from app.portfolio.service import PortfolioService
from app.realtime.bus import BUS
from app.core.settings import settings
from app.ai.intraday_overlays import analyze_intraday

router = APIRouter()


def _clamp_int(v: str | None, default: int, lo: int, hi: int) -> int:
    try:
        n = int(v) if v is not None else int(default)
    except Exception:
        n = int(default)
    return max(int(lo), min(int(hi), int(n)))


def _perf_log(op: str, dt_ms: float, **extra) -> None:
    if not getattr(settings, "PERF_LOG_ENABLED", True):
        return
    slow_ms = int(getattr(settings, "PERF_LOG_SLOW_MS", 250) or 250)
    if dt_ms < float(slow_ms):
        return
    logger.warning("PERF {op} slow: {ms:.1f}ms {extra}", op=op, ms=dt_ms, extra=extra)


async def _ws_auth(ws: WebSocket) -> bool:
    """Return True if the WS connection is authorized, else close and return False."""

    if not settings.REQUIRE_API_KEY_WEBSOCKETS:
        return True

    expected = settings.API_KEY_READ or settings.API_KEY
    if not expected:
        await ws.close(code=1011)
        return False

    supplied = ws.headers.get("x-api-key") or ws.query_params.get("api_key")
    if supplied != expected:
        await ws.close(code=1008)
        return False

    return True


@router.websocket("/ws/candles")
async def ws_candles(ws: WebSocket):
    if not await _ws_auth(ws):
        return
    await ws.accept()
    try:
        default_key = (settings.DEFAULT_UNIVERSE.split(",")[0].strip() if settings.DEFAULT_UNIVERSE else "NSE_EQ|INE002A01018")
        instrument_key = ws.query_params.get("instrument_key", default_key) or default_key
        interval = ws.query_params.get("interval", "1m")
        lookback_minutes = int(ws.query_params.get("lookback_minutes", "10"))
        poll_seconds = _clamp_int(ws.query_params.get("poll_seconds"), 5, 1, 60)
        svc = CandleService()

        # Stream last candle update (throttled). Note: upstream fetch is blocking IO; run in a thread.
        while True:
            end = datetime.now(timezone.utc)
            err: str | None = None
            try:
                t0 = time.perf_counter()
                series = await asyncio.to_thread(svc.poll_intraday, instrument_key, interval, lookback_minutes)
                _perf_log("ws.candles.poll_intraday", (time.perf_counter() - t0) * 1000.0, instrument_key=instrument_key, interval=interval)
                candles = series.candles
            except Exception as e:
                candles = []
                err = str(e)[:200]

            if err:
                await ws.send_json({"type": "error", "detail": err, "server_ts": int(end.timestamp())})
            elif candles:
                c = candles[-1]
                ts = int(getattr(c, "ts"))
                candle = {
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "open": float(c.open),
                    "high": float(c.high),
                    "low": float(c.low),
                    "close": float(c.close),
                    "volume": float(c.volume),
                }
                await ws.send_json(
                    {
                        "type": "candle_update",
                        "instrument_key": instrument_key,
                        "interval": interval,
                        "candle": candle,
                        "server_ts": int(end.timestamp()),
                    }
                )
            await asyncio.sleep(poll_seconds)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/intraday-overlays")
async def ws_intraday_overlays(ws: WebSocket):
    """Realtime intraday AI overlays computed from latest candles.

    Query params:
    - instrument_key (default: first DEFAULT_UNIVERSE item)
    - interval (default: 1m)
    - lookback_minutes (default: 90)

    Sends:
      {type:'intraday_overlays', instrument_key, interval, analysis, server_ts}
    """

    if not await _ws_auth(ws):
        return
    await ws.accept()
    try:
        default_key = (settings.DEFAULT_UNIVERSE.split(",")[0].strip() if settings.DEFAULT_UNIVERSE else "NSE_EQ|INE002A01018")
        instrument_key = ws.query_params.get("instrument_key", default_key) or default_key
        interval = ws.query_params.get("interval", "1m")
        lookback_minutes = int(ws.query_params.get("lookback_minutes", "90"))
        poll_seconds = _clamp_int(ws.query_params.get("poll_seconds"), 5, 1, 60)
        svc = CandleService()

        last_candle_ts: int | None = None
        last_analysis: dict | None = None

        while True:
            end = datetime.now(timezone.utc)
            err: str | None = None
            try:
                # Upstream fetch + DB upsert are blocking; keep event loop responsive.
                t0 = time.perf_counter()
                series = await asyncio.to_thread(svc.poll_intraday, instrument_key, interval, lookback_minutes)
                _perf_log("ws.overlays.poll_intraday", (time.perf_counter() - t0) * 1000.0, instrument_key=instrument_key, interval=interval)
                candles = list(series.candles or [])
            except Exception as e:
                candles = []
                err = str(e)[:200]

            if err:
                await ws.send_json({"type": "error", "detail": err, "server_ts": int(end.timestamp())})
            else:
                cur_ts = None
                try:
                    cur_ts = (int(getattr(candles[-1], "ts")) if candles else None)
                except Exception:
                    cur_ts = None

                if last_analysis is not None and cur_ts is not None and last_candle_ts == cur_ts:
                    analysis = last_analysis
                else:
                    t1 = time.perf_counter()
                    analysis = await asyncio.to_thread(
                        analyze_intraday,
                        instrument_key=instrument_key,
                        interval=interval,
                        candles=candles,
                    )
                    _perf_log("ws.overlays.analyze_intraday", (time.perf_counter() - t1) * 1000.0, instrument_key=instrument_key, interval=interval, n=len(candles or []))
                    last_analysis = analysis
                    last_candle_ts = cur_ts
                await ws.send_json(
                    {
                        "type": "intraday_overlays",
                        "instrument_key": instrument_key,
                        "interval": interval,
                        "analysis": analysis,
                        "server_ts": int(end.timestamp()),
                    }
                )

            await asyncio.sleep(poll_seconds)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    if not await _ws_auth(ws):
        return
    await ws.accept()
    svc = PortfolioService()
    try:
        source = ws.query_params.get("source")  # auto|paper|upstox
        poll_seconds = max(1, min(int(ws.query_params.get("poll_seconds", "5")), 60))

        while True:
            bal = svc.balance(source=source)
            payload = {
                "type": "balance",
                "source": bal.get("source"),
                "balance": bal.get("balance"),
                "segment": bal.get("segment"),
                "alert": bal.get("alert"),
                "server_ts": int(datetime.now(timezone.utc).timestamp()),
            }
            await ws.send_json(payload)
            await asyncio.sleep(poll_seconds)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/alerts-stream")
async def ws_alerts_stream(ws: WebSocket):
    """Live alert events via EventBus.

    Query params:
    - since_id: only send events after this id
    - history: number of recent events to replay on connect (default 200)
    """

    if not await _ws_auth(ws):
        return
    await ws.accept()
    try:
        since_id = ws.query_params.get("since_id")
        history = int(ws.query_params.get("history", "200"))
        since = (None if since_id is None else int(since_id))

        recent = await BUS.recent("alerts", limit=history, since_id=since)
        for ev in recent:
            await ws.send_json(ev.to_dict())

        q = await BUS.subscribe("alerts")
        try:
            while True:
                ev = await q.get()
                await ws.send_json(ev.to_dict())
        finally:
            await BUS.unsubscribe("alerts", q)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/training")
async def ws_training(ws: WebSocket):
    """Live training progress events.

    Query params:
    - since_id: only send events after this id
    - history: number of recent events to replay on connect (default 200)
    """
    if not await _ws_auth(ws):
        return
    await ws.accept()
    try:
        since_id = ws.query_params.get("since_id")
        history = int(ws.query_params.get("history", "200"))
        since = (None if since_id is None else int(since_id))

        recent = await BUS.recent("training", limit=history, since_id=since)
        for ev in recent:
            await ws.send_json(ev.to_dict())

        q = await BUS.subscribe("training")
        try:
            while True:
                ev = await q.get()
                await ws.send_json(ev.to_dict())
        finally:
            await BUS.unsubscribe("training", q)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/hft")
async def ws_hft(ws: WebSocket):
    """Live HFT events via EventBus (channel: hft)."""
    if not await _ws_auth(ws):
        return
    await ws.accept()
    try:
        since_id = ws.query_params.get("since_id")
        history = int(ws.query_params.get("history", "200"))
        since = (None if since_id is None else int(since_id))

        recent = await BUS.recent("hft", limit=history, since_id=since)
        for ev in recent:
            await ws.send_json(ev.to_dict())

        q = await BUS.subscribe("hft")
        try:
            while True:
                ev = await q.get()
                await ws.send_json(ev.to_dict())
        finally:
            await BUS.unsubscribe("hft", q)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/data")
async def ws_data(ws: WebSocket):
    """Live data pipeline events.

    Query params:
    - since_id: only send events after this id
    - history: number of recent events to replay on connect (default 200)
    """
    if not await _ws_auth(ws):
        return
    await ws.accept()
    try:
        since_id = ws.query_params.get("since_id")
        history = int(ws.query_params.get("history", "200"))
        since = (None if since_id is None else int(since_id))

        recent = await BUS.recent("data", limit=history, since_id=since)
        for ev in recent:
            await ws.send_json(ev.to_dict())

        q = await BUS.subscribe("data")
        try:
            while True:
                ev = await q.get()
                await ws.send_json(ev.to_dict())
        finally:
            await BUS.unsubscribe("data", q)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/agent")
async def ws_agent(ws: WebSocket):
    """Live agent decision + trade lifecycle events."""
    if not await _ws_auth(ws):
        return
    await ws.accept()
    try:
        since_id = ws.query_params.get("since_id")
        history = int(ws.query_params.get("history", "200"))
        since = (None if since_id is None else int(since_id))

        recent = await BUS.recent("agent", limit=history, since_id=since)
        for ev in recent:
            await ws.send_json(ev.to_dict())

        q = await BUS.subscribe("agent")
        try:
            while True:
                ev = await q.get()
                await ws.send_json(ev.to_dict())
        finally:
            await BUS.unsubscribe("agent", q)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/prediction")
async def ws_prediction(ws: WebSocket):
    """Live prediction events (created/resolved/cycle).

    Query params:
    - since_id: only send events after this id
    - history: number of recent events to replay on connect (default 200)
    """
    if not await _ws_auth(ws):
        return
    await ws.accept()
    try:
        since_id = ws.query_params.get("since_id")
        history = int(ws.query_params.get("history", "200"))
        since = (None if since_id is None else int(since_id))

        recent = await BUS.recent("prediction", limit=history, since_id=since)
        for ev in recent:
            await ws.send_json(ev.to_dict())

        q = await BUS.subscribe("prediction")
        try:
            while True:
                ev = await q.get()
                await ws.send_json(ev.to_dict())
        finally:
            await BUS.unsubscribe("prediction", q)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/recommendations")
async def ws_recommendations(ws: WebSocket):
    """Live recommendation refresh events.

    Event channel: 'recommendations'

    Query params:
    - since_id: only send events after this id
    - history: number of recent events to replay on connect (default 200)
    """

    if not await _ws_auth(ws):
        return
    await ws.accept()
    try:
        since_id = ws.query_params.get("since_id")
        history = int(ws.query_params.get("history", "200"))
        since = (None if since_id is None else int(since_id))

        recent = await BUS.recent("recommendations", limit=history, since_id=since)
        for ev in recent:
            await ws.send_json(ev.to_dict())

        q = await BUS.subscribe("recommendations")
        try:
            while True:
                ev = await q.get()
                await ws.send_json(ev.to_dict())
        finally:
            await BUS.unsubscribe("recommendations", q)
    except WebSocketDisconnect:
        return


@router.websocket("/ws/automation")
async def ws_automation(ws: WebSocket):
    """Live automation run events.

    Query params:
    - since_id: only send events after this id
    - history: number of recent events to replay on connect (default 200)
    """
    if not await _ws_auth(ws):
        return
    await ws.accept()
    try:
        since_id = ws.query_params.get("since_id")
        history = int(ws.query_params.get("history", "200"))
        since = (None if since_id is None else int(since_id))

        recent = await BUS.recent("automation", limit=history, since_id=since)
        for ev in recent:
            await ws.send_json(ev.to_dict())

        q = await BUS.subscribe("automation")
        try:
            while True:
                ev = await q.get()
                await ws.send_json(ev.to_dict())
        finally:
            await BUS.unsubscribe("automation", q)
    except WebSocketDisconnect:
        return
