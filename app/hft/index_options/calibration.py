from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any


@dataclass
class CalibrationState:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_sum: float = 0.0
    # Adaptive knob: add to base min confidence (can be negative)
    confidence_delta: float = 0.0


_lock = Lock()


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def load_state(path: str) -> CalibrationState:
    if not path:
        return CalibrationState()
    try:
        if not os.path.exists(path):
            return CalibrationState()
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return CalibrationState()
        return CalibrationState(
            trades=int(raw.get("trades") or 0),
            wins=int(raw.get("wins") or 0),
            losses=int(raw.get("losses") or 0),
            pnl_sum=_safe_float(raw.get("pnl_sum"), 0.0),
            confidence_delta=_safe_float(raw.get("confidence_delta"), 0.0),
        )
    except Exception:
        return CalibrationState()


def save_state(path: str, st: CalibrationState) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(st), f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception:
        return


def update_on_trade_close(*, path: str, pnl: float) -> CalibrationState:
    """Update rolling calibration based on realized pnl.

    Conservative rule:
    - losses push confidence_delta up (be more selective)
    - wins gently pull confidence_delta down
    """

    with _lock:
        st = load_state(path)
        st.trades += 1
        st.pnl_sum = float(st.pnl_sum) + float(pnl)

        if pnl > 0:
            st.wins += 1
            st.confidence_delta = float(st.confidence_delta) - 0.01
        else:
            st.losses += 1
            st.confidence_delta = float(st.confidence_delta) + 0.02

        # Clamp to sane bounds.
        st.confidence_delta = max(-0.10, min(0.20, float(st.confidence_delta)))

        save_state(path, st)
        return st


def snapshot(*, path: str) -> dict[str, Any]:
    with _lock:
        st = load_state(path)
        return asdict(st)
