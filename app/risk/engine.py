from __future__ import annotations

from app.risk.rules import RiskLimits


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def position_size(self, capital: float, entry: float, stop: float, confidence: float) -> dict:
        if entry <= 0 or stop <= 0 or capital <= 0:
            return {"qty": 0.0, "risk": 0.0}

        per_share_risk = abs(entry - stop)
        if per_share_risk <= 0:
            return {"qty": 0.0, "risk": 0.0}

        # Risk budget shrinks if confidence is low.
        conf = max(0.0, min(1.0, float(confidence)))
        risk_budget = capital * self.limits.max_risk_per_trade_pct * (0.5 + 0.5 * conf)
        qty = risk_budget / per_share_risk

        # Also cap by max position size.
        max_position_value = capital * self.limits.max_position_pct
        qty = min(qty, max_position_value / entry)

        return {"qty": float(max(0.0, qty)), "risk": float(qty * per_share_risk)}
