from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal, cast

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.ai.engine import AIEngine
from app.ai.indicators import compute_indicator_snapshot
from app.ai.intraday_overlays import analyze_intraday
from app.ai.trend_confluence import confluence_policy
from app.candles.service import CandleService
from app.core.settings import settings

router = APIRouter(prefix="/trader")

_candles = CandleService()
_ai = AIEngine()


class TraderAi(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    confidence: float = 0.0
    uncertainty: float = 1.0
    model: str | None = None


class TraderBollinger(BaseModel):
    mid: float = 0.0
    upper: float = 0.0
    lower: float = 0.0
    std: float = 0.0
    bandwidth: float = 0.0


class TraderMacd(BaseModel):
    macd: float = 0.0
    signal: float = 0.0
    hist: float = 0.0


class TraderStochastic(BaseModel):
    k: float = 0.0
    d: float = 0.0


class TraderIndicators(BaseModel):
    ema20: float = 0.0
    ema50: float = 0.0
    ema200: float = 0.0
    sma20: float = 0.0
    rsi14: float = 50.0
    bollinger: TraderBollinger = Field(default_factory=TraderBollinger)
    macd: TraderMacd = Field(default_factory=TraderMacd)
    stochastic: TraderStochastic = Field(default_factory=TraderStochastic)


class TraderOverlays(BaseModel):
    # Keep these flexible: they come from analyze_intraday() and may evolve.
    levels: dict[str, Any] | None = None
    candle_patterns: list[dict[str, Any]] | None = None
    trade: dict[str, Any] | None = None


class TraderDecisionPlan(BaseModel):
    side: Literal["buy", "sell"]
    entry: float
    stop: float
    target: float
    confidence: float | None = None
    source: str | None = None
    reason: str | None = None


class TraderDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    plan: TraderDecisionPlan | None = None
    risk_multiplier: float = 0.25
    risk_fraction_suggested: float = 0.001
    reasons: list[str] = Field(default_factory=list)


class TraderDecisionResponse(BaseModel):
    ok: bool = True
    instrument_key: str
    interval: str
    asof_ts: int | None = None
    n_candles: int = 0
    last_close: float = 0.0
    ai: TraderAi
    indicators: TraderIndicators
    overlays: TraderOverlays
    decision: TraderDecision
    trend_confluence: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


def _is_intraday_interval(interval: str) -> bool:
    s = str(interval or "")
    return s.endswith("m") or s.endswith("h")


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, float(v))))


@router.get("/decision", response_model=TraderDecisionResponse, response_model_exclude_none=True)
def trader_decision(
    instrument_key: str = Query(...),
    interval: str = Query("1m"),
    lookback_minutes: int = Query(240, ge=5, le=6 * 60),
    lookback_days: int | None = Query(None, ge=1, le=3650),
    horizon_steps: int = Query(12, ge=1, le=500),
    include_raw: bool = Query(False),
) -> dict[str, Any]:
    """AI Trader view.

    Goal: treat the AI as the "client" that consumes backend data.

    Returns a single object the frontend can render:
    - candles summary
    - indicator snapshot (EMA/SMA/RSI/BB/MACD/Stoch)
    - candlestick patterns + intraday trade plan (levels)
    - AIEngine prediction (direction/conf/uncertainty)
    - final decision + suggested risk sizing

    This endpoint is intentionally lightweight and works even without a trained deep model.
    """

    end = datetime.now(timezone.utc)

    if lookback_days is None:
        if _is_intraday_interval(interval):
            start = end - timedelta(minutes=int(lookback_minutes))
            lookback_days_eff = max(1, int(lookback_minutes // (60 * 24)) + 1)
        else:
            lookback_days_eff = 180
            start = end - timedelta(days=int(lookback_days_eff))
    else:
        lookback_days_eff = int(lookback_days)
        start = end - timedelta(days=int(lookback_days_eff))

    series = _candles.get_historical(str(instrument_key), str(interval), start, end, limit=None)
    cs = series.candles

    closes = [float(c.close) for c in cs]
    highs = [float(c.high) for c in cs]
    lows = [float(c.low) for c in cs]

    last_close = float(closes[-1]) if closes else 0.0
    asof_ts = int(cs[-1].ts) if cs else None

    indicators = compute_indicator_snapshot(closes=closes, highs=highs, lows=lows)

    overlays: dict[str, Any] = analyze_intraday(instrument_key=str(series.instrument_key), interval=str(interval), candles=cs)

    fam = "intraday" if _is_intraday_interval(interval) else "long"
    pred = cast(
        dict[str, Any],
        _ai.predict(
        instrument_key=str(series.instrument_key),
        interval=str(interval),
        lookback_days=int(lookback_days_eff),
        horizon_steps=int(horizon_steps),
        include_nifty=False,
        model_family=str(fam),
        ),
    )

    pred_p: dict[str, Any] = dict(pred.get("prediction") or {})
    ai_action = str(pred_p.get("signal") or "HOLD").upper()
    ai_conf = float(pred_p.get("confidence") or 0.0)
    ai_unc = float(pred_p.get("uncertainty") or 1.0)

    meta = dict(pred.get("meta") or {})
    ai_model = meta.get("model")
    ai_family = meta.get("model_family")

    # Pull predicted OHLC if present to build a lightweight plan.
    ohlc = dict(pred_p.get("next_hour_ohlc") or {}) if isinstance(pred_p.get("next_hour_ohlc"), dict) else {}
    pred_high = float(ohlc.get("high") or 0.0)
    pred_low = float(ohlc.get("low") or 0.0)
    pred_close = float(ohlc.get("close") or 0.0)

    reasons: list[str] = []
    reasons.append(
        "AI="
        + str(ai_action)
        + f" conf={ai_conf:.2f} unc={ai_unc:.2f}"
        + (f" model={ai_model}" if ai_model else "")
        + (f" family={ai_family}" if ai_family else "")
    )
    if pred_close > 0 and last_close > 0:
        reasons.append(f"PredClose={pred_close:.2f} vs Close={last_close:.2f}")

    # AI-first decision: action and plan come from the AI prediction only.
    chosen_action = "HOLD" if ai_action not in {"BUY", "SELL"} else ai_action
    chosen_plan: dict[str, Any] | None = None

    if chosen_action in {"BUY", "SELL"} and last_close > 0:
        # Derive simple SL/TP bands from uncertainty/confidence.
        base_sl = float(max(0.003, min(0.03, 0.004 + ai_unc * 0.02)))
        base_tp = float(max(0.003, min(0.05, 0.005 + ai_conf * 0.03)))

        if chosen_action == "BUY":
            stop = last_close * (1.0 - base_sl)
            target = last_close * (1.0 + base_tp)
            # Prefer AI-projected levels if they are sensible.
            if pred_low > 0:
                stop = min(stop, pred_low)
            if pred_high > 0:
                target = max(target, pred_high)
            side = "buy"
        else:
            stop = last_close * (1.0 + base_sl)
            target = last_close * (1.0 - base_tp)
            if pred_high > 0:
                stop = max(stop, pred_high)
            if pred_low > 0:
                target = min(target, pred_low)
            side = "sell"

        entry = float(last_close)
        chosen_plan = {
            "side": side,
            "entry": entry,
            "stop": float(stop),
            "target": float(target),
            "confidence": float(ai_conf),
            "source": "ai",
            "reason": "ai_prediction",
        }
        reasons.append("Plan=ai")

    # Risk sizing suggestion (AI-only)
    score = _clamp01(ai_conf * (1.0 - ai_unc))
    risk_multiplier = float(max(0.10, min(1.0, 0.25 + 0.75 * score)))
    risk_fraction_suggested = float(0.001 + 0.004 * score)  # 0.1% .. 0.5%

    trend_confluence: dict[str, Any] | None = None
    if bool(getattr(settings, "TRADER_TREND_CONFLUENCE_ENABLED", False)):
        try:
            trend_confluence = cast(dict[str, Any], _ai.analyze_trend_confluence(str(series.instrument_key)))
            tdir = str(trend_confluence.get("dir") or "unknown")
            tconf = float(trend_confluence.get("confidence") or 0.0)
            reasons.append(f"Confluence={tdir} conf={tconf:.2f}")

            # Let the confluence output decide whether to gate or just scale risk.
            if chosen_action in {"BUY", "SELL"}:
                pol = confluence_policy(intended_action=cast(Any, chosen_action), confluence=trend_confluence)
                if pol.action == "HOLD":
                    reasons.append(f"Gate=confluence_policy({pol.reason})")
                    chosen_action = "HOLD"
                    chosen_plan = None
                else:
                    risk_multiplier = float(max(0.05, min(1.25, float(risk_multiplier) * float(pol.risk_multiplier))))
                    reasons.append(f"Risk=confluence_mult x{float(pol.risk_multiplier):.2f}")

            if bool(getattr(settings, "TRADER_TREND_CONFLUENCE_GATE_ENABLED", False)) and chosen_action in {"BUY", "SELL"}:
                # Avoid choppy / range regime, and avoid conflicts.
                if tdir in {"range", "mixed", "unknown"}:
                    reasons.append("Gate=confluence_range_or_unclear")
                    chosen_action = "HOLD"
                    chosen_plan = None
                elif chosen_action == "BUY" and tdir != "up":
                    reasons.append("Gate=confluence_conflict")
                    chosen_action = "HOLD"
                    chosen_plan = None
                elif chosen_action == "SELL" and tdir != "down":
                    reasons.append("Gate=confluence_conflict")
                    chosen_action = "HOLD"
                    chosen_plan = None
        except Exception:
            # Confluence is best-effort; never fail the endpoint.
            trend_confluence = None

    payload: dict[str, Any] = {
        "ok": True,
        "instrument_key": str(series.instrument_key),
        "interval": str(interval),
        "asof_ts": asof_ts,
        "n_candles": int(len(cs)),
        "last_close": float(last_close),
        "ai": {
            "action": ai_action,
            "confidence": float(ai_conf),
            "uncertainty": float(ai_unc),
            "model": ai_model,
        },
        "indicators": {
            "ema20": float(indicators.ema20),
            "ema50": float(indicators.ema50),
            "ema200": float(indicators.ema200),
            "sma20": float(indicators.sma20),
            "rsi14": float(indicators.rsi14),
            "bollinger": dict(indicators.bollinger),
            "macd": dict(indicators.macd),
            "stochastic": dict(indicators.stochastic),
        },
        "overlays": {
            "levels": overlays.get("levels"),
            "candle_patterns": overlays.get("candle_patterns"),
            "trade": overlays.get("trade"),
        },
        "decision": {
            "action": str(chosen_action),
            "plan": chosen_plan,
            "risk_multiplier": float(risk_multiplier),
            "risk_fraction_suggested": float(risk_fraction_suggested),
            "reasons": reasons,
        },
        "trend_confluence": trend_confluence,
    }

    if include_raw:
        payload["raw"] = pred

    return payload
