from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.db import db_conn

router = APIRouter(prefix="/instruments", tags=["instruments"])


def _loaded_count() -> int:
    with db_conn() as conn:
        row = conn.execute("SELECT COUNT(1) AS n FROM instrument_meta").fetchone()
        return int(row["n"]) if row else 0


def _ensure_loaded() -> None:
    if _loaded_count() <= 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "Instrument master is not loaded. Ensure data/upstox_instruments.csv exists "
                "and restart the backend."
            ),
        )


@router.get("/status")
def instruments_status() -> dict:
    n = _loaded_count()
    return {"loaded": n > 0, "count": n}


def _parse_instrument_key(instrument_key: str) -> dict:
    ik = str(instrument_key or "").strip()
    left, _, right = ik.partition("|")

    exchange = ""
    segment = ""
    if left:
        parts = left.split("_")
        if len(parts) >= 2:
            exchange = parts[0]
            segment = parts[1]
        else:
            exchange = left

    isin = right.strip() if right else ""

    return {
        "exchange": exchange,
        "segment": segment,
        "isin": isin,
    }


def _to_instrument_info(*, instrument_key: str, tradingsymbol: str | None) -> dict:
    meta = _parse_instrument_key(instrument_key)
    token = instrument_key.split("|")[-1].strip() if "|" in instrument_key else instrument_key

    canonical = (tradingsymbol or "").strip() or token

    return {
        "canonical_symbol": canonical,
        "exchange": meta["exchange"],
        "segment": meta["segment"],
        "isin": meta["isin"],
        "instrument_key": instrument_key,
        "tradingsymbol": (tradingsymbol or "").strip() or canonical,
        "name": None,
    }


@router.get("/resolve")
def resolve(query: str) -> dict:
    q = str(query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    # BackendComplete behavior: if instrument master isn't loaded, return 503.
    _ensure_loaded()

    # If an instrument_key is provided, prefer direct match.
    with db_conn() as conn:
        row = None
        if "|" in q:
            row = conn.execute(
                "SELECT instrument_key, tradingsymbol FROM instrument_meta WHERE instrument_key=? LIMIT 1",
                (q,),
            ).fetchone()

        if row is None:
            # Match tradingsymbol exact first.
            row = conn.execute(
                "SELECT instrument_key, tradingsymbol FROM instrument_meta WHERE UPPER(tradingsymbol)=UPPER(?) LIMIT 1",
                (q,),
            ).fetchone()

        if row is None:
            # Fallback: prefix match (fast for suggestions pasted into resolve).
            like = f"{q.upper()}%"
            row = conn.execute(
                "SELECT instrument_key, tradingsymbol FROM instrument_meta WHERE UPPER(tradingsymbol) LIKE ? OR UPPER(instrument_key) LIKE ? ORDER BY updated_ts DESC LIMIT 1",
                (like, like),
            ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="instrument not found")

        return _to_instrument_info(instrument_key=str(row["instrument_key"]), tradingsymbol=row["tradingsymbol"])


@router.get("/suggest")
def suggest(q: str, limit: int = 5) -> dict:
    query = str(q or "").strip()
    lim = max(1, min(int(limit), 20))

    if not query:
        return {"results": [], "count": 0}

    _ensure_loaded()

    like = f"%{query.upper()}%"
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT instrument_key, tradingsymbol FROM instrument_meta WHERE UPPER(tradingsymbol) LIKE ? OR UPPER(instrument_key) LIKE ? ORDER BY updated_ts DESC LIMIT ?",
            (like, like, lim),
        ).fetchall()

    results = [_to_instrument_info(instrument_key=str(r["instrument_key"]), tradingsymbol=r["tradingsymbol"]) for r in rows]
    return {"results": results, "count": len(results)}


@router.get("/search")
def search(q: str, limit: int = 20) -> dict:
    query = str(q or "").strip()
    lim = max(1, min(int(limit), 100))
    if not query:
        raise HTTPException(status_code=400, detail="q is required")

    _ensure_loaded()

    like = f"%{query.upper()}%"
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT instrument_key, tradingsymbol FROM instrument_meta WHERE UPPER(tradingsymbol) LIKE ? OR UPPER(instrument_key) LIKE ? ORDER BY updated_ts DESC LIMIT ?",
            (like, like, lim),
        ).fetchall()

    results = [_to_instrument_info(instrument_key=str(r["instrument_key"]), tradingsymbol=r["tradingsymbol"]) for r in rows]
    return {"results": results, "count": len(results)}
