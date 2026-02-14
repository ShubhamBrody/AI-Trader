from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WsEnvelope(BaseModel):
    """Stable WebSocket message envelope.

    This matches the EventBus schema (see app.realtime.bus.RealtimeEvent.to_dict) and is intended to be
    UI-friendly and consistent across all websocket channels.

    - `channel`: stream name (e.g. "prediction", "automation", "hft")
    - `type`: event type within the channel (e.g. "progress", "started", "trade_opened")
    - `payload`: event-specific object
    - `ts`: server timestamp (UTC epoch seconds)
    - `id`: optional monotonic id (present for EventBus-backed channels)

    Notes:
    - Some legacy WS endpoints also send top-level fields for backwards compatibility.
    """

    channel: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: int
    id: int | None = None


def wrap_legacy_ws_event(*, channel: str, type: str, payload: dict[str, Any], ts: int, legacy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a stable envelope while preserving legacy top-level fields.

    Returns a dict ready for `ws.send_json()`.
    """

    env = WsEnvelope(channel=str(channel), type=str(type), payload=dict(payload or {}), ts=int(ts), id=None).model_dump()
    if legacy:
        env.update(dict(legacy))
    return env
