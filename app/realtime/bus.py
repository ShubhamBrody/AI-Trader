from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class RealtimeEvent:
    id: int
    channel: str
    type: str
    payload: dict[str, Any]
    ts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "channel": self.channel,
            "type": self.type,
            "payload": self.payload,
            "ts": int(self.ts),
        }


class EventBus:
    def __init__(self, *, max_history: int = 2000) -> None:
        self._max_history = int(max_history)
        self._lock = asyncio.Lock()
        self._next_id = 1
        self._history: dict[str, deque[RealtimeEvent]] = defaultdict(lambda: deque(maxlen=self._max_history))
        self._subs: dict[str, set[asyncio.Queue[RealtimeEvent]]] = defaultdict(set)

    async def publish(self, channel: str, type: str, payload: dict[str, Any]) -> RealtimeEvent:
        async with self._lock:
            ev = RealtimeEvent(
                id=self._next_id,
                channel=str(channel),
                type=str(type),
                payload=dict(payload or {}),
                ts=int(datetime.now(timezone.utc).timestamp()),
            )
            self._next_id += 1
            self._history[ev.channel].append(ev)
            subs = list(self._subs.get(ev.channel, set()))

        for q in subs:
            try:
                q.put_nowait(ev)
            except Exception:
                # Best-effort; slow consumers can miss events.
                pass
        return ev

    async def subscribe(self, channel: str, *, max_queue: int = 500) -> asyncio.Queue[RealtimeEvent]:
        q: asyncio.Queue[RealtimeEvent] = asyncio.Queue(maxsize=int(max_queue))
        async with self._lock:
            self._subs[str(channel)].add(q)
        return q

    async def unsubscribe(self, channel: str, q: asyncio.Queue[RealtimeEvent]) -> None:
        async with self._lock:
            self._subs.get(str(channel), set()).discard(q)

    async def recent(self, channel: str, *, limit: int = 200, since_id: int | None = None) -> list[RealtimeEvent]:
        limit = max(1, min(int(limit), self._max_history))
        async with self._lock:
            items = list(self._history.get(str(channel), deque()))
        if since_id is not None:
            items = [x for x in items if int(x.id) > int(since_id)]
        return items[-limit:]


BUS = EventBus(max_history=5000)


def publish_sync(channel: str, type: str, payload: dict[str, Any]) -> None:
    """Fire-and-forget publish usable from sync code.

    If no event loop is running (unit tests), this becomes a no-op.
    """

    try:
        loop = asyncio.get_running_loop()
    except Exception:
        return

    loop.create_task(BUS.publish(channel, type, payload))
