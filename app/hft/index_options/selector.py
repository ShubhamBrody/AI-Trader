from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal

from app.core.db import db_conn


OptionType = Literal["CE", "PE"]


@dataclass(frozen=True)
class OptionContract:
    instrument_key: str
    tradingsymbol: str | None
    upstox_token: str | None
    underlying_symbol: str | None
    expiry: date | None
    strike: float | None
    option_type: str | None


def _parse_expiry(raw: object) -> date | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Upstox exchange instruments often store expiry as epoch millis (string).
    # Example: "1771957799000".
    try:
        if s.isdigit() and len(s) >= 10:
            n = int(s)
            # Heuristic: >= 1e12 => milliseconds; else seconds.
            ts = (n / 1000.0) if n >= 1_000_000_000_000 else float(n)
            return datetime.fromtimestamp(ts, tz=timezone.utc).date()
    except Exception:
        pass
    # Upstox instrument JSON often uses YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None


def find_atm_option(
    *,
    underlying_symbol: str,
    option_type: OptionType,
    spot: float,
    asof: date | None = None,
    max_expiry_days: int = 14,
) -> OptionContract | None:
    """Find nearest-expiry, closest-strike option for an underlying.

    Requires instrument_extra to be populated via /api/universe/import-upstox-exchange.
    """

    u = str(underlying_symbol or "").strip().upper()
    ot = str(option_type or "").strip().upper()
    if not u or ot not in {"CE", "PE"}:
        return None

    if not (spot and float(spot) > 0):
        return None

    asof_d = asof or date.today()

    # NOTE: instrument_extra.expiry is stored as TEXT and may be ISO date OR epoch millis.
    # We avoid SQL DATE() filters and instead parse/filter in Python for robustness.
    sql = """
    SELECT
        m.instrument_key AS instrument_key,
        m.tradingsymbol AS tradingsymbol,
        m.upstox_token AS upstox_token,
        e.underlying_symbol AS underlying_symbol,
        e.expiry AS expiry,
        e.strike AS strike,
        COALESCE(e.option_type, e.instrument_type) AS option_type
    FROM instrument_extra e
    JOIN instrument_meta m ON m.instrument_key = e.instrument_key
    WHERE
        UPPER(COALESCE(e.underlying_symbol,'')) = ?
        AND UPPER(COALESCE(e.option_type, e.instrument_type,'')) = ?
        AND e.expiry IS NOT NULL AND TRIM(e.expiry) <> ''
        AND e.strike IS NOT NULL
    ORDER BY
        e.expiry ASC,
        ABS(e.strike - ?) ASC
    LIMIT 200
    """

    rows: list[Any]
    with db_conn() as conn:
        rows = list(conn.execute(sql, (u, ot, float(spot))).fetchall())

    if not rows:
        return None

    # Apply max_expiry_days filter in Python for robustness (expiry format inconsistencies).
    max_days = max(0, int(max_expiry_days))

    best: OptionContract | None = None
    for r in rows:
        exp = _parse_expiry(r[4])
        if exp is None:
            continue
        try:
            d_days = (exp - asof_d).days
        except Exception:
            continue
        if d_days < 0 or d_days > max_days:
            continue

        strike = None
        try:
            strike = (None if r[5] is None else float(r[5]))
        except Exception:
            strike = None

        best = OptionContract(
            instrument_key=str(r[0]),
            tradingsymbol=(None if r[1] is None else str(r[1])),
            upstox_token=(None if r[2] is None else str(r[2])),
            underlying_symbol=(None if r[3] is None else str(r[3])),
            expiry=exp,
            strike=strike,
            option_type=(None if r[6] is None else str(r[6])),
        )
        break

    return best
