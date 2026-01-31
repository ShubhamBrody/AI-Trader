from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.db import db_conn


_CATALOG_READY = False


@dataclass(frozen=True)
class PatternStat:
    name: str
    weight: float
    seen: int
    wins: int
    losses: int
    ema_winrate: float


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def ensure_catalog_rows(patterns: list[dict[str, Any]]) -> None:
    ts = _now_ts()
    with db_conn() as conn:
        for p in patterns:
            name = str(p.get("name") or "").strip()
            if not name:
                continue
            family = str(p.get("family") or "")
            side = str(p.get("side") or "")
            base_rel = float(p.get("base_reliability") or 0.7)
            params = p.get("params")
            params_json = json.dumps(params or {}, ensure_ascii=False, separators=(",", ":"))

            conn.execute(
                "INSERT OR IGNORE INTO candle_pattern_stats (pattern, family, side, base_reliability, weight, seen, wins, losses, ema_winrate, updated_ts) "
                "VALUES (?, ?, ?, ?, 1.0, 0, 0, 0, 0.5, ?)",
                (name, family, side, float(base_rel), int(ts)),
            )
            conn.execute(
                "INSERT OR IGNORE INTO candle_pattern_params (pattern, params_json, updated_ts) VALUES (?, ?, ?)",
                (name, params_json, int(ts)),
            )


def ensure_all_catalog_rows() -> None:
    global _CATALOG_READY
    if _CATALOG_READY:
        return
    # Local import to avoid any import-order surprises.
    from app.ai.candlestick_patterns import CATALOG

    ensure_catalog_rows(
        [
            {
                "name": p.name,
                "family": p.family,
                "side": p.side,
                "base_reliability": float(p.base_reliability),
                "params": {"min_window": int(p.min_window), "best_timeframe": p.best_timeframe},
            }
            for p in CATALOG
        ]
    )
    _CATALOG_READY = True


def get_weights() -> dict[str, float]:
    with db_conn() as conn:
        rows = conn.execute("SELECT pattern, weight FROM candle_pattern_stats").fetchall()
        return {str(r["pattern"]): float(r["weight"] or 1.0) for r in rows}


_WEIGHTS_CACHE: dict[str, float] | None = None
_WEIGHTS_CACHE_TS: int = 0


def get_weights_cached(ttl_seconds: int = 30) -> dict[str, float]:
    global _WEIGHTS_CACHE, _WEIGHTS_CACHE_TS
    ttl_seconds = max(1, min(int(ttl_seconds), 600))
    now = _now_ts()
    if _WEIGHTS_CACHE is not None and (now - int(_WEIGHTS_CACHE_TS)) <= ttl_seconds:
        return dict(_WEIGHTS_CACHE)
    w = get_weights()
    _WEIGHTS_CACHE = dict(w)
    _WEIGHTS_CACHE_TS = int(now)
    return w


def get_stats(limit: int = 200) -> list[PatternStat]:
    limit = max(1, min(int(limit), 5000))
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT pattern, weight, seen, wins, losses, ema_winrate FROM candle_pattern_stats ORDER BY (wins * 1.0 / NULLIF(seen,0)) DESC, seen DESC LIMIT ?",
            (limit,),
        )
        out: list[PatternStat] = []
        for r in cur.fetchall():
            out.append(
                PatternStat(
                    name=str(r["pattern"]),
                    weight=float(r["weight"] or 1.0),
                    seen=int(r["seen"] or 0),
                    wins=int(r["wins"] or 0),
                    losses=int(r["losses"] or 0),
                    ema_winrate=float(r["ema_winrate"] or 0.5),
                )
            )
        return out


def apply_feedback(
    *,
    pattern: str,
    good: bool,
    magnitude: float = 1.0,
    note: str | None = None,
    instrument_key: str | None = None,
    interval: str | None = None,
    asof_ts: int | None = None,
) -> PatternStat | None:
    name = str(pattern or "").strip()
    if not name:
        return None

    magnitude = float(max(0.2, min(float(magnitude), 5.0)))
    ts = _now_ts()

    with db_conn() as conn:
        row = conn.execute(
            "SELECT weight, seen, wins, losses, ema_winrate FROM candle_pattern_stats WHERE pattern=?",
            (name,),
        ).fetchone()
        if not row:
            return None

        weight = float(row["weight"] or 1.0)
        seen = int(row["seen"] or 0)
        wins = int(row["wins"] or 0)
        losses = int(row["losses"] or 0)
        ema = float(row["ema_winrate"] or 0.5)

        seen += 1
        if good:
            wins += 1
        else:
            losses += 1

        # EMA update: small step, scaled by magnitude but clamped.
        alpha = float(max(0.01, min(0.08, 0.03 * magnitude)))
        target = 1.0 if good else 0.0
        ema = float(max(0.0, min(1.0, (1.0 - alpha) * ema + alpha * target)))

        # Weight follows EMA gently. Clamp so it doesn't drift wildly.
        # ema=0.5 => weight ~1.0
        weight = float(1.0 + (ema - 0.5) * 0.9)
        weight = float(max(0.55, min(1.45, weight)))

        conn.execute(
            "UPDATE candle_pattern_stats SET weight=?, seen=?, wins=?, losses=?, ema_winrate=?, updated_ts=? WHERE pattern=?",
            (float(weight), int(seen), int(wins), int(losses), float(ema), int(ts), name),
        )

        conn.execute(
            "INSERT INTO candle_pattern_feedback (ts, pattern, good, magnitude, instrument_key, interval, asof_ts, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (int(ts), name, 1 if good else 0, float(magnitude), instrument_key, interval, (None if asof_ts is None else int(asof_ts)), (note or None)),
        )

        return PatternStat(name=name, weight=weight, seen=seen, wins=wins, losses=losses, ema_winrate=ema)


def learn_from_trade_outcome(
    *,
    instrument_key: str | None,
    side: str,
    pnl: float | None,
    meta: dict[str, Any] | None,
    close_reason: str | None = None,
) -> int:
    """Best-effort learning hook.

    Expects patterns to be snapshot in trade meta under:
      meta['overlays']['candle_patterns']  (list)
      meta['overlays']['trade']['pattern_hint'] (dict)

    Returns number of feedback events recorded.
    """

    try:
        side_u = str(side or "").upper()
        if side_u not in {"BUY", "SELL"}:
            return 0
        if pnl is None:
            return 0
        pnl_f = float(pnl)
    except Exception:
        return 0

    # Avoid learning from non-market outcomes.
    reason = str(close_reason or "")
    if reason.startswith("entry_"):
        return 0

    good = pnl_f > 0

    overlays = (meta or {}).get("overlays") if isinstance(meta, dict) else None
    if not isinstance(overlays, dict):
        overlays = {}

    interval = overlays.get("interval")
    asof_ts = overlays.get("asof_ts")

    patterns: list[tuple[str, float]] = []

    # Primary: top candle patterns (up to 3).
    cps = overlays.get("candle_patterns")
    if isinstance(cps, list):
        for p in cps[:3]:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or "").strip()
            if not name:
                continue
            conf = p.get("confidence")
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.6
            patterns.append((name, conf_f))

    # Fallback: trade hint.
    tr = overlays.get("trade")
    if isinstance(tr, dict):
        hint = tr.get("pattern_hint")
        if isinstance(hint, dict):
            name = str(hint.get("name") or "").strip()
            if name and all(name != n for n, _ in patterns):
                try:
                    conf_f = float(hint.get("confidence") or 0.6)
                except Exception:
                    conf_f = 0.6
                patterns.append((name, conf_f))

    if not patterns:
        return 0

    ensure_all_catalog_rows()

    recorded = 0
    for name, conf in patterns:
        magnitude = max(0.2, min(5.0, 0.8 + 2.2 * max(0.0, min(1.0, float(conf)))))
        res = apply_feedback(
            pattern=name,
            good=good,
            magnitude=magnitude,
            note=(f"auto:{reason}" if reason else "auto"),
            instrument_key=(instrument_key or None),
            interval=(str(interval) if interval else None),
            asof_ts=(int(asof_ts) if asof_ts is not None else None),
        )
        if res is not None:
            recorded += 1
    return recorded
