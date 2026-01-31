from __future__ import annotations

import json
import time

from app.core.db import db_conn


def get_balance() -> float:
    with db_conn() as conn:
        row = conn.execute("SELECT balance FROM paper_account WHERE id=1").fetchone()
        return float(row["balance"]) if row else 0.0


def set_balance(balance: float) -> None:
    with db_conn() as conn:
        conn.execute("UPDATE paper_account SET balance=? WHERE id=1", (float(balance),))


def get_positions() -> list[dict]:
    with db_conn() as conn:
        cur = conn.execute("SELECT symbol, qty, avg_price FROM paper_positions ORDER BY symbol")
        return [
            {"symbol": r["symbol"], "qty": float(r["qty"]), "avg_price": float(r["avg_price"])}
            for r in cur.fetchall()
        ]


def upsert_position(symbol: str, qty: float, avg_price: float) -> None:
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO paper_positions(symbol, qty, avg_price)
            VALUES (?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, avg_price=excluded.avg_price
            """,
            (symbol, float(qty), float(avg_price)),
        )


def delete_position(symbol: str) -> None:
    with db_conn() as conn:
        conn.execute("DELETE FROM paper_positions WHERE symbol=?", (symbol,))


def add_journal(symbol: str, side: str, qty: float, price: float, fees: float, slippage: float, meta: dict | None) -> int:
    with db_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO paper_journal(ts, symbol, side, qty, price, fees, slippage, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(time.time()),
                symbol,
                side,
                float(qty),
                float(price),
                float(fees),
                float(slippage),
                json.dumps(meta or {}),
            ),
        )
        return int(cur.lastrowid)


def get_journal(limit: int = 100) -> list[dict]:
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT id, ts, symbol, side, qty, price, fees, slippage, meta FROM paper_journal ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        out = []
        for r in cur.fetchall():
            out.append(
                {
                    "id": int(r["id"]),
                    "ts": int(r["ts"]),
                    "symbol": r["symbol"],
                    "side": r["side"],
                    "qty": float(r["qty"]),
                    "price": float(r["price"]),
                    "fees": float(r["fees"]),
                    "slippage": float(r["slippage"]),
                    "meta": json.loads(r["meta"] or "{}"),
                }
            )
        return out
