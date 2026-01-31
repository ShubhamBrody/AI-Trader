from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.10
    max_trades_per_symbol_per_day: int = 3
    max_daily_drawdown_pct: float = 0.03
