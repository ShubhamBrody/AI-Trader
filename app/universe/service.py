from __future__ import annotations

from datetime import datetime, timezone

from app.core.db import db_conn


class UniverseService:
    def upsert(self, *, instrument_key: str, tradingsymbol: str | None = None, cap_tier: str | None = None, upstox_token: str | None = None) -> dict:
        if not instrument_key:
            raise ValueError("instrument_key required")
        tier = (cap_tier or "unknown").lower()
        if tier not in {"large", "mid", "small", "unknown"}:
            tier = "unknown"

        ts = int(datetime.now(timezone.utc).timestamp())
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO instrument_meta (instrument_key, tradingsymbol, cap_tier, upstox_token, updated_ts) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(instrument_key) DO UPDATE SET tradingsymbol=excluded.tradingsymbol, cap_tier=excluded.cap_tier, upstox_token=excluded.upstox_token, updated_ts=excluded.updated_ts",
                (instrument_key, tradingsymbol, tier, upstox_token, ts),
            )
        return {"ok": True, "instrument_key": instrument_key, "cap_tier": tier}

    def list(self, *, cap_tier: str | None = None, limit: int = 500) -> list[dict]:
        limit = max(1, min(int(limit), 2000))
        tier = (cap_tier or "").lower().strip()
        with db_conn() as conn:
            if tier:
                cur = conn.execute(
                    "SELECT instrument_key, tradingsymbol, cap_tier, upstox_token, updated_ts FROM instrument_meta WHERE cap_tier=? ORDER BY updated_ts DESC LIMIT ?",
                    (tier, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT instrument_key, tradingsymbol, cap_tier, upstox_token, updated_ts FROM instrument_meta ORDER BY updated_ts DESC LIMIT ?",
                    (limit,),
                )
            return [
                {
                    "instrument_key": r["instrument_key"],
                    "tradingsymbol": r["tradingsymbol"],
                    "cap_tier": r["cap_tier"],
                    "upstox_token": r["upstox_token"],
                    "updated_ts": int(r["updated_ts"]),
                }
                for r in cur.fetchall()
            ]

    def list_keys_paged(
        self,
        *,
        prefix: str | None = None,
        cap_tier: str | None = None,
        limit: int = 500,
        after: str | None = None,
    ) -> dict:
        """Return a page of instrument_keys ordered by instrument_key.

        Useful for iterating the full NSE universe without fixed hard limits.

        Args:
            prefix: Optional prefix filter (e.g. "NSE_EQ|").
            cap_tier: Optional tier filter.
            limit: Page size (max 2000).
            after: Cursor; returns keys strictly greater than this value.

        Returns:
            {"keys": [...], "next_after": <last_key_or_None>}
        """

        limit = max(1, min(int(limit), 2000))
        tier = (cap_tier or "").lower().strip()
        pref = (prefix or "").strip()
        aft = (after or "").strip()

        where: list[str] = []
        params: list[object] = []

        if tier:
            where.append("cap_tier=?")
            params.append(tier)
        if pref:
            where.append("instrument_key LIKE ?")
            params.append(f"{pref}%")
        if aft:
            where.append("instrument_key > ?")
            params.append(aft)

        q = "SELECT instrument_key FROM instrument_meta"
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY instrument_key ASC LIMIT ?"
        params.append(int(limit))

        with db_conn() as conn:
            rows = conn.execute(q, tuple(params)).fetchall()
        keys = [str(r["instrument_key"]) for r in rows if r and r["instrument_key"]]
        next_after = keys[-1] if len(keys) == int(limit) else None
        return {"keys": keys, "next_after": next_after}

    def count(self, *, prefix: str | None = None, cap_tier: str | None = None) -> int:
        tier = (cap_tier or "").lower().strip()
        pref = (prefix or "").strip()

        where: list[str] = []
        params: list[object] = []

        if tier:
            where.append("cap_tier=?")
            params.append(tier)
        if pref:
            where.append("instrument_key LIKE ?")
            params.append(f"{pref}%")

        q = "SELECT COUNT(1) AS n FROM instrument_meta"
        if where:
            q += " WHERE " + " AND ".join(where)

        with db_conn() as conn:
            row = conn.execute(q, tuple(params)).fetchone()
        return int(row["n"] if row and row["n"] is not None else 0)

    def get_cap_tier(self, instrument_key: str) -> str:
        with db_conn() as conn:
            row = conn.execute(
                "SELECT cap_tier FROM instrument_meta WHERE instrument_key=?",
                (instrument_key,),
            ).fetchone()
            return (row["cap_tier"] if row and row["cap_tier"] else "unknown")

    def bulk_import(self, items: list[dict]) -> dict:
        ok = 0
        failed = 0
        for it in items or []:
            try:
                self.upsert(
                    instrument_key=str(it.get("instrument_key") or "").strip(),
                    tradingsymbol=(str(it.get("tradingsymbol")).strip() if it.get("tradingsymbol") is not None else None),
                    cap_tier=(str(it.get("cap_tier")).strip() if it.get("cap_tier") is not None else None),
                    upstox_token=(str(it.get("upstox_token")).strip() if it.get("upstox_token") is not None else None),
                )
                ok += 1
            except Exception:
                failed += 1
        return {"ok": True, "imported": ok, "failed": failed}
