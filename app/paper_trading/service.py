from __future__ import annotations

from dataclasses import dataclass

from app.paper_trading import persistence_sql as store


@dataclass(frozen=True)
class FeesModel:
    fee_rate: float = 0.0002  # 2 bps
    slippage_rate: float = 0.0005  # 5 bps


class PaperTradingService:
    def __init__(self) -> None:
        self._fees = FeesModel()

    def account(self) -> dict:
        return {"balance": store.get_balance()}

    def deposit(self, amount: float) -> dict:
        bal = store.get_balance()
        bal += float(amount)
        store.set_balance(bal)
        return {"balance": bal}

    def withdraw(self, amount: float) -> dict:
        bal = store.get_balance()
        if amount > bal:
            raise ValueError("Insufficient balance")
        bal -= float(amount)
        store.set_balance(bal)
        return {"balance": bal}

    def positions(self) -> dict:
        return {"positions": store.get_positions()}

    def execute(self, symbol: str, side: str, qty: float, price: float) -> dict:
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")

        qty = float(qty)
        price = float(price)

        fees = qty * price * self._fees.fee_rate
        slippage = qty * price * self._fees.slippage_rate

        bal = float(store.get_balance())
        positions = {p["symbol"]: p for p in store.get_positions()}
        pos = positions.get(symbol)
        old_qty = float(pos["qty"]) if pos is not None else 0.0
        old_avg = float(pos["avg_price"]) if pos is not None else 0.0

        if side == "BUY":
            total_cost = qty * price + fees + slippage
            if total_cost > bal:
                raise ValueError("Insufficient balance")
            bal -= total_cost

            if old_qty >= 0:
                # Add/increase long
                new_qty = old_qty + qty
                new_avg = price if old_qty <= 1e-9 else (old_qty * old_avg + qty * price) / max(new_qty, 1e-9)
            else:
                # Cover short
                cover_qty = min(qty, abs(old_qty))
                remaining_buy = qty - cover_qty
                new_qty = old_qty + qty  # closer to zero
                if new_qty < 0:
                    # still short, keep short avg
                    new_avg = old_avg
                elif new_qty <= 1e-9:
                    new_qty = 0.0
                    new_avg = 0.0
                else:
                    # flipped to long for the remaining_buy part
                    new_avg = price

            if abs(new_qty) <= 1e-9:
                store.delete_position(symbol)
            else:
                store.upsert_position(symbol, float(new_qty), float(new_avg))

        else:  # SELL
            proceeds = qty * price - fees - slippage
            bal += proceeds

            if old_qty <= 0:
                # Add/increase short
                new_qty = old_qty - qty
                add_qty = qty
                new_avg = price if abs(old_qty) <= 1e-9 else (abs(old_qty) * old_avg + add_qty * price) / max(abs(new_qty), 1e-9)
            else:
                # Reduce/close long (or flip to short)
                sell_from_long = min(qty, old_qty)
                remaining_sell = qty - sell_from_long
                new_qty = old_qty - qty
                if new_qty > 1e-9:
                    new_avg = old_avg
                elif abs(new_qty) <= 1e-9:
                    new_qty = 0.0
                    new_avg = 0.0
                else:
                    # flipped to short for remaining_sell part
                    new_avg = price

            if abs(new_qty) <= 1e-9:
                store.delete_position(symbol)
            else:
                store.upsert_position(symbol, float(new_qty), float(new_avg))

        store.set_balance(bal)
        journal_id = store.add_journal(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            fees=fees,
            slippage=slippage,
            meta={"mode": "paper"},
        )

        return {"journal_id": journal_id, "balance": bal, "positions": store.get_positions()}

    def journal(self, limit: int = 100) -> dict:
        return {"journal": store.get_journal(limit=limit)}

    def pnl(self) -> dict:
        # Realized PnL needs matching buys/sells; for demo return placeholders.
        return {"realized": 0.0, "unrealized": 0.0, "total": 0.0}
