from app.ai.trend_confluence import confluence_policy


def test_policy_gates_on_unclear_when_confident() -> None:
    pol = confluence_policy(
        intended_action="BUY",
        confluence={"dir": "range", "confidence": 0.9},
    )
    assert pol.action == "HOLD"
    assert pol.risk_multiplier == 0.0


def test_policy_gates_on_strong_conflict() -> None:
    pol = confluence_policy(
        intended_action="BUY",
        confluence={"dir": "down", "confidence": 0.8},
    )
    assert pol.action == "HOLD"


def test_policy_allows_and_scales_up_when_aligned() -> None:
    pol = confluence_policy(
        intended_action="SELL",
        confluence={"dir": "down", "confidence": 0.9},
    )
    assert pol.action == "SELL"
    assert 0.9 <= pol.risk_multiplier <= 1.25


def test_policy_soft_disagree_scales_down() -> None:
    pol = confluence_policy(
        intended_action="SELL",
        confluence={"dir": "up", "confidence": 0.2},
    )
    assert pol.action == "SELL"
    assert 0.10 <= pol.risk_multiplier <= 0.75
