from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter
from fastapi import Query

from app.core.db import db_conn

router = APIRouter(prefix="/smart-search", tags=["smart-search"])


def _normalize_query(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q


def _tokens(q: str) -> list[str]:
    q = q.upper()
    parts = re.split(r"[^A-Z0-9]+", q)
    return [p for p in parts if p]


THEME_SYNONYMS: dict[str, list[str]] = {
    "SILVER": ["SILVER", "SILVERMIC", "SILVERM", "XAG", "AG"],
    "GOLD": ["GOLD", "GOLDM", "GOLDMIC", "XAU"],
    "CRUDE": ["CRUDE", "CRUDEOIL"],
    "NIFTY": ["NIFTY", "NIFTY 50", "NIFTY50"],
    "BANKNIFTY": ["BANKNIFTY", "BANK NIFTY"],
}


@router.get("/suggest")
def suggest(
    q: str = Query("", description="Search query, e.g. 'SILVER', 'RELIANCE', 'TCS'."),
    limit: int = Query(20, ge=1, le=50),
) -> dict[str, Any]:
    """Upstox-like suggestions across imported instruments.

    Uses both instrument_meta (tradingsymbol) and instrument_extra (name, underlying_symbol,
    segment/exchange/type) for richer matches.
    """

    qn = _normalize_query(q)
    if not qn:
        return {"ok": True, "query": qn, "results": []}

    toks = _tokens(qn)
    expanded_terms: set[str] = set([qn])
    for t in toks:
        if t in THEME_SYNONYMS:
            expanded_terms.update(THEME_SYNONYMS[t])

    like_terms = [f"%{t}%" for t in expanded_terms if t]
    if not like_terms:
        return {"ok": True, "query": qn, "results": []}

    # Prefer exact-ish matches by ordering.
    sql = """
    SELECT
        m.instrument_key as instrument_key,
        m.tradingsymbol as tradingsymbol,
        e.exchange as exchange,
        e.segment as segment,
        e.instrument_type as instrument_type,
        e.name as name,
        e.underlying_symbol as underlying_symbol,
        e.expiry as expiry,
        e.strike as strike,
        CASE
            WHEN UPPER(COALESCE(e.option_type,'')) IN ('CE','PE') THEN e.option_type
            WHEN UPPER(COALESCE(e.instrument_type,'')) IN ('CE','PE') THEN e.instrument_type
            ELSE NULL
        END as option_type
    FROM instrument_meta m
    LEFT JOIN instrument_extra e ON e.instrument_key = m.instrument_key
    WHERE (
        m.tradingsymbol LIKE ? OR
        m.instrument_key LIKE ? OR
        COALESCE(e.name,'') LIKE ? OR
        COALESCE(e.underlying_symbol,'') LIKE ?
    )
    ORDER BY
        CASE WHEN UPPER(m.tradingsymbol) = UPPER(?) THEN 0 ELSE 1 END,
        CASE WHEN UPPER(COALESCE(e.name,'')) LIKE UPPER(?) THEN 0 ELSE 1 END,
        LENGTH(COALESCE(m.tradingsymbol,'')) ASC
    LIMIT ?
    """

    # Run for each expanded term, merge distinct by instrument_key.
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    with db_conn() as conn:
        for term, like in zip(list(expanded_terms), like_terms):
            rows = conn.execute(
                sql,
                (
                    like,
                    like,
                    like,
                    like,
                    term,
                    like,
                    int(limit),
                ),
            ).fetchall()
            for r in rows:
                ik = r[0]
                if not ik or ik in seen:
                    continue
                seen.add(ik)
                out.append(
                    {
                        "instrument_key": r[0],
                        "tradingsymbol": r[1],
                        "exchange": r[2],
                        "segment": r[3],
                        "instrument_type": r[4],
                        "name": r[5],
                        "underlying_symbol": r[6],
                        "expiry": r[7],
                        "strike": r[8],
                        "option_type": r[9],
                    }
                )
                if len(out) >= int(limit):
                    break
            if len(out) >= int(limit):
                break

    # Light re-ranking: bring MCX commodities up when query hints commodity.
    if any(t in ("SILVER", "GOLD", "CRUDE") for t in toks):
        out.sort(
            key=lambda x: (
                0 if str(x.get("exchange") or "").upper() == "MCX" else 1,
                0 if str(x.get("instrument_type") or "").upper() in ("FUT", "COM" ,"COMMODITY") else 1,
                len(str(x.get("tradingsymbol") or "")),
            )
        )

    return {"ok": True, "query": qn, "expanded": sorted(expanded_terms), "results": out[: int(limit)]}
