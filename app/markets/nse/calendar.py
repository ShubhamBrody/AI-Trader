from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import yaml


@dataclass(frozen=True)
class MarketSession:
    tz: str
    open_time: time
    close_time: time


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_market_session(config_path: str) -> MarketSession:
    cfg = _load_yaml(config_path)
    tz = cfg.get("timezone", "Asia/Kolkata")

    market = cfg.get("market_hours", {})
    open_s = market.get("open", "09:15")
    close_s = market.get("close", "15:29")

    open_h, open_m = [int(x) for x in open_s.split(":")]
    close_h, close_m = [int(x) for x in close_s.split(":")]

    return MarketSession(tz=tz, open_time=time(open_h, open_m), close_time=time(close_h, close_m))


def load_holidays(holidays_path: str) -> set[date]:
    cfg = _load_yaml(holidays_path)
    dates = cfg.get("holidays") or []
    out: set[date] = set()
    for d in dates:
        out.add(date.fromisoformat(str(d)))
    return out


def is_trading_day(d: date, holidays: set[date]) -> bool:
    if d.weekday() >= 5:
        return False
    if d in holidays:
        return False
    return True


def iter_trading_days(start: date, end: date, holidays: set[date]):
    cur = start
    while cur <= end:
        if is_trading_day(cur, holidays):
            yield cur
        cur += timedelta(days=1)


def last_n_trading_days(end: date, n: int, holidays: set[date]) -> list[date]:
    days: list[date] = []
    cur = end
    while len(days) < n:
        if is_trading_day(cur, holidays):
            days.append(cur)
        cur -= timedelta(days=1)
    days.reverse()
    return days


def market_state(now_local: datetime, session: MarketSession, is_day_open: bool) -> str:
    if not is_day_open:
        return "CLOSED"

    t = now_local.time()
    if t < session.open_time:
        return "PRE_MARKET"
    if session.open_time <= t <= session.close_time:
        return "LIVE"
    return "POST_MARKET"
