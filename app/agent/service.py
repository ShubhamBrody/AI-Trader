from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from zoneinfo import ZoneInfo

from app.core.audit import log_event
from app.core.controls import get_controls
from app.core.guards import get_safety_status
from app.core.settings import settings
from app.markets.nse.calendar import load_market_session
from app.ai.engine import AIEngine
from app.ai.intraday_overlays import analyze_intraday
from app.candles.service import CandleService
from app.paper_trading.service import PaperTradingService
from app.portfolio.service import PortfolioService
from app.risk.engine import RiskEngine
from app.strategy.engine import StrategyEngine
from app.universe.service import UniverseService
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError
from app.integrations.upstox.order_builder import build_equity_intraday_market_order, build_equity_intraday_stop_order
from app.orders.state import create_order, get_order_by_client_order_id, update_order
from app.agent.state import (
    count_open_trades,
    count_trades_today,
    has_open_trade,
    has_traded_today,
    list_open_trades,
    mark_trade_closed,
    realized_pnl_today,
    record_trade_open,
    update_trade,
)
from app.realtime.bus import publish_sync
from app.portfolio.positions_state import broker_pnl_estimate, list_positions_state, reconcile_upstox_positions


@dataclass
class AgentStatus:
    running: bool
    started_ts: int | None
    last_cycle_ts: int | None
    last_error: str | None


class AutoTraderAgent:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._status = AgentStatus(running=False, started_ts=None, last_cycle_ts=None, last_error=None)
        self._uni = UniverseService()
        self._portfolio = PortfolioService()
        self._paper = PaperTradingService()
        self._risk = RiskEngine()
        self._ai = AIEngine()
        self._candles = CandleService()
        self._strategy = StrategyEngine()

    def status(self) -> dict[str, Any]:
        return {
            "running": self._status.running,
            "started_ts": self._status.started_ts,
            "last_cycle_ts": self._status.last_cycle_ts,
            "last_error": self._status.last_error,
            "broker": settings.AUTOTRADER_BROKER,
            "enabled": bool(settings.AUTOTRADER_ENABLED),
            "safe_mode": bool(settings.SAFE_MODE),
        }

    async def preview(self, instrument_key: str) -> dict[str, Any]:
        """Compute the full decision stack without placing any order."""
        instrument_key = str(instrument_key)

        # Capital source
        if settings.AUTOTRADER_BROKER == "paper":
            capital = float(self._paper.account()["balance"])
        else:
            bal = self._portfolio.balance(source="upstox")
            capital = float(bal.get("balance") or 0.0)

        long_pred = self._ai.predict(
            instrument_key,
            interval=settings.TRAIN_LONG_INTERVAL,
            lookback_days=settings.TRAIN_LONG_LOOKBACK_DAYS,
            horizon_steps=settings.TRAIN_LONG_HORIZON_STEPS,
            include_nifty=False,
        )
        intra_pred = self._ai.predict(
            instrument_key,
            interval=settings.TRAIN_INTRADAY_INTERVAL,
            lookback_days=settings.TRAIN_INTRADAY_LOOKBACK_DAYS,
            horizon_steps=settings.TRAIN_INTRADAY_HORIZON_STEPS,
            include_nifty=False,
        )
        long_sig = (long_pred.get("prediction") or {}).get("signal") or "HOLD"
        intra_sig = (intra_pred.get("prediction") or {}).get("signal") or "HOLD"
        long_conf = float((long_pred.get("prediction") or {}).get("confidence") or 0.0)
        intra_conf = float((intra_pred.get("prediction") or {}).get("confidence") or 0.0)
        conf = float(min(long_conf, intra_conf))

        if long_sig == "BUY" and intra_sig == "BUY":
            decision = "BUY"
        elif long_sig == "SELL" and intra_sig == "SELL":
            decision = "SELL"
        else:
            decision = "HOLD"

        # Strategy idea (intraday levels)
        series = self._candles.poll_intraday(instrument_key, interval="1m", lookback_minutes=60 * 6)
        if not series.candles:
            series = self._candles.poll_intraday(instrument_key, interval="5m", lookback_minutes=60 * 6)
        highs = [c.high for c in series.candles]
        lows = [c.low for c in series.candles]
        closes = [c.close for c in series.candles]
        idea = None
        sizing = None
        if closes:
            idea = self._strategy.build_idea(symbol=instrument_key, highs=highs, lows=lows, closes=closes)
            sizing = self._risk.position_size(capital=capital, entry=float(idea.entry), stop=float(idea.stop_loss), confidence=conf)

        out = {
            "ok": True,
            "instrument_key": instrument_key,
            "capital": capital,
            "signals": {
                "long": {"signal": long_sig, "confidence": long_conf, "raw": long_pred},
                "intraday": {"signal": intra_sig, "confidence": intra_conf, "raw": intra_pred},
            },
            "decision": decision,
            "confidence": conf,
            "strategy": None
            if idea is None
            else {
                "side": getattr(idea, "side", None),
                "entry": float(idea.entry),
                "stop": float(idea.stop_loss),
                "target": float(idea.target),
            },
            "risk": sizing,
        }
        publish_sync("agent", "agent.preview", {"instrument_key": instrument_key, "decision": decision, "confidence": conf})
        return out

    async def start(self) -> dict:
        if self._task and not self._task.done():
            return {"ok": True, "status": "already_running"}
        if not settings.AUTOTRADER_ENABLED:
            return {"ok": False, "detail": "AUTOTRADER_ENABLED=false"}
        if get_controls().agent_disabled:
            return {"ok": False, "detail": "agent_disabled=true (runtime override)"}

        if settings.AUTOTRADER_BROKER == "upstox" and settings.SAFE_MODE:
            return {"ok": False, "detail": "SAFE_MODE=true: cannot start live autotrader"}

        self._status.running = True
        self._status.started_ts = int(datetime.now(timezone.utc).timestamp())
        self._status.last_error = None

        # One-shot recovery step (restart-safety): reconcile any existing open trades before looping.
        try:
            rec = await self._reconcile_open_trades()
            log_event("agent.recover", {"reconciled": rec})
        except Exception as e:
            log_event("agent.recover_error", {"error": str(e)})

        self._task = asyncio.create_task(self._run_loop())
        log_event("agent.start", {"broker": settings.AUTOTRADER_BROKER})
        publish_sync("agent", "agent.start", {"broker": settings.AUTOTRADER_BROKER, "safe_mode": bool(settings.SAFE_MODE)})
        return {"ok": True}

    async def flatten(self) -> dict:
        # Always allow flatten in paper mode; in upstox mode require SAFE_MODE=false.
        if settings.AUTOTRADER_BROKER == "upstox" and settings.SAFE_MODE:
            return {"ok": False, "detail": "SAFE_MODE=true: cannot flatten live broker positions"}
        res = await self._square_off_open_trades(reason="manual_flatten")
        log_event("agent.flatten", res)
        return {"ok": True, **res}

    async def cancel_open_orders(self) -> dict:
        if settings.AUTOTRADER_BROKER != "upstox":
            return {"ok": False, "detail": "cancel-open-orders only supported for upstox broker"}
        if settings.SAFE_MODE:
            return {"ok": False, "detail": "SAFE_MODE=true: refusing to cancel live orders"}

        cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
        client: UpstoxClient | None = None
        cancelled = 0
        checked = 0
        errors: list[str] = []

        cancellable = {"OPEN", "PENDING", "TRIGGER_PENDING", "PLACED", "PUT_ORDER_REQ_RECEIVED"}
        try:
            client = UpstoxClient(cfg)
            book = client.order_book_v2()
            items = book.get("data") or []
            for it in items:
                oid = str(it.get("order_id") or "").strip()
                st = str(it.get("status") or it.get("order_status") or "").upper().strip()
                if not oid:
                    continue
                checked += 1
                if st and st not in cancellable:
                    continue
                try:
                    client.cancel_order_v3(oid)
                    cancelled += 1
                except Exception as e:
                    errors.append(f"{oid}:{e}")
        finally:
            if client is not None:
                client.close()

        return {"ok": True, "checked": checked, "cancelled": cancelled, "errors": errors[:20]}

    async def stop(self) -> dict:
        self._status.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        log_event("agent.stop", {})
        publish_sync("agent", "agent.stop", {})
        return {"ok": True}

    async def _run_loop(self) -> None:
        try:
            while self._status.running:
                try:
                    await self.run_once()
                    await asyncio.sleep(max(5, int(settings.AUTOTRADER_POLL_SECONDS)))
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    self._status.last_error = str(e)
                    log_event("agent.error", {"error": str(e)})
                    await asyncio.sleep(5)
        finally:
            self._status.running = False

    async def run_once(self) -> dict:
        controls = get_controls()
        if controls.agent_disabled:
            self._status.last_cycle_ts = int(datetime.now(timezone.utc).timestamp())
            return {"ok": True, "skipped": True, "reason": "agent_disabled=true"}

        # Market-hours guard.
        status = get_safety_status()
        if status.market_state != "LIVE":
            self._status.last_cycle_ts = int(datetime.now(timezone.utc).timestamp())
            log_event("agent.heartbeat", {"skipped": True, "reason": status.reason, "state": status.market_state})
            publish_sync("agent", "agent.heartbeat", {"skipped": True, "reason": status.reason, "state": status.market_state})
            return {"ok": True, "skipped": True, "reason": status.reason, "state": status.market_state}

        # End-of-day square-off while market is still LIVE.
        try:
            tz = ZoneInfo(settings.TIMEZONE)
            now_utc = datetime.now(timezone.utc)
            now_local = now_utc.astimezone(tz)
            session = load_market_session("app/config/market_hours.yaml")
            close_local = now_local.replace(hour=session.close_time.hour, minute=session.close_time.minute, second=0, microsecond=0)
            seconds_to_close = (close_local - now_local).total_seconds()
            if seconds_to_close <= max(0, int(settings.AUTOTRADER_SQUAREOFF_MINUTES_BEFORE_CLOSE)) * 60:
                sq = await self._square_off_open_trades(reason="eod_squareoff")
                self._status.last_cycle_ts = int(datetime.now(timezone.utc).timestamp())
                log_event("agent.squareoff", sq)
                publish_sync("agent", "agent.squareoff", sq)
                return {"ok": True, "placed": [], "squareoff": sq}
        except Exception as e:
            # Do not fail the agent cycle due to squareoff computation issues.
            log_event("agent.squareoff_error", {"error": str(e)})

        # First, manage open trades (targets, trailing stops, app-side stops).
        managed = await self._manage_open_trades()

        # Reconcile broker order state (best-effort, no network in SAFE_MODE).
        reconciled = await self._reconcile_open_trades()

        # Broker-truth snapshot (read-only). Used to prevent doubling into an existing
        # position after restarts/mismatches and to compute a best-effort PnL estimate.
        upstox_net_by_key: dict[str, float] = {}
        broker_pnl_today: float | None = None
        if settings.AUTOTRADER_BROKER == "upstox" and (not settings.SAFE_MODE) and settings.UPSTOX_ACCESS_TOKEN:
            try:
                reconcile_upstox_positions()
                for p in list_positions_state(broker="upstox", limit=2000):
                    ik = p.get("instrument_key")
                    if not ik:
                        continue
                    upstox_net_by_key[str(ik)] = float(p.get("net_qty") or 0.0)
                broker_pnl_today = broker_pnl_estimate(broker="upstox")
            except Exception:
                pass

        # Risk limits
        open_n = count_open_trades()
        if open_n >= int(settings.AUTOTRADER_MAX_OPEN_TRADES):
            self._status.last_cycle_ts = int(datetime.now(timezone.utc).timestamp())
            log_event("agent.risk_lock", {"reason": "max_open_trades", "open": open_n})
            publish_sync("agent", "agent.risk_lock", {"reason": "max_open_trades", "open": open_n})
            return {"ok": True, "placed": [], "managed": managed, "reconciled": reconciled, "risk_locked": "max_open_trades"}

        trades_today = count_trades_today(tz_name=settings.TIMEZONE)
        if trades_today >= int(settings.AUTOTRADER_MAX_TRADES_PER_DAY):
            self._status.last_cycle_ts = int(datetime.now(timezone.utc).timestamp())
            log_event("agent.risk_lock", {"reason": "max_trades_per_day", "trades_today": trades_today})
            publish_sync("agent", "agent.risk_lock", {"reason": "max_trades_per_day", "trades_today": trades_today})
            return {"ok": True, "placed": [], "managed": managed, "reconciled": reconciled, "risk_locked": "max_trades_per_day"}

        max_loss = float(settings.AUTOTRADER_MAX_DAILY_LOSS_INR or 0.0)
        if max_loss > 0:
            pnl_today = float(broker_pnl_today) if broker_pnl_today is not None else float(realized_pnl_today(tz_name=settings.TIMEZONE))
            if pnl_today <= -abs(max_loss):
                self._status.last_cycle_ts = int(datetime.now(timezone.utc).timestamp())
                log_event("agent.risk_lock", {"reason": "max_daily_loss", "pnl_today": pnl_today, "max_loss": max_loss})
                publish_sync("agent", "agent.risk_lock", {"reason": "max_daily_loss", "pnl_today": pnl_today, "max_loss": max_loss})
                return {"ok": True, "placed": [], "managed": managed, "reconciled": reconciled, "risk_locked": "max_daily_loss", "pnl_today": pnl_today}

        # Universe: use instrument_meta, capped per cycle.
        all_items = self._uni.list(limit=int(settings.AUTOTRADER_MAX_SYMBOLS_PER_CYCLE))
        keys = [i["instrument_key"] for i in all_items if i.get("instrument_key")]

        # Capital source
        if settings.AUTOTRADER_BROKER == "paper":
            capital = float(self._paper.account()["balance"])
        else:
            bal = self._portfolio.balance(source="upstox")
            capital = float(bal.get("balance") or 0.0)

        placed: list[dict[str, Any]] = []
        cycle_ts = int(datetime.now(timezone.utc).timestamp())
        publish_sync(
            "agent",
            "agent.cycle_start",
            {
                "cycle_ts": cycle_ts,
                "symbols": len(keys),
                "capital": float(capital),
                "broker": settings.AUTOTRADER_BROKER,
                "safe_mode": bool(settings.SAFE_MODE),
            },
        )

        for instrument_key in keys:
            # Avoid duplicate entries (one trade per symbol per day, and do not re-enter if already open).
            if has_open_trade(instrument_key=instrument_key):
                continue
            if has_traded_today(instrument_key=instrument_key, tz_name=settings.TIMEZONE):
                continue

            # Runtime freeze blocks NEW orders (but still allows management/reconcile).
            if controls.freeze_new_orders:
                continue

            # Broker-truth: do not place a new trade if broker already has net exposure.
            if settings.AUTOTRADER_BROKER == "upstox" and (not settings.SAFE_MODE):
                net_qty = float(upstox_net_by_key.get(instrument_key, 0.0) or 0.0)
                if abs(net_qty) > 1e-9:
                    continue

            # Dual timeframe signals:
            # 1) Long-term trend from 1d model (4y lookback)
            long_pred = self._ai.predict(
                instrument_key,
                interval=settings.TRAIN_LONG_INTERVAL,
                lookback_days=settings.TRAIN_LONG_LOOKBACK_DAYS,
                horizon_steps=settings.TRAIN_LONG_HORIZON_STEPS,
                include_nifty=False,
            )
            long_sig = (long_pred.get("prediction") or {}).get("signal") or "HOLD"
            long_conf = float((long_pred.get("prediction") or {}).get("confidence") or 0.0)

            # 2) Intraday next-hour from 1m model (3m lookback)
            intra_pred = self._ai.predict(
                instrument_key,
                interval=settings.TRAIN_INTRADAY_INTERVAL,
                lookback_days=settings.TRAIN_INTRADAY_LOOKBACK_DAYS,
                horizon_steps=settings.TRAIN_INTRADAY_HORIZON_STEPS,
                include_nifty=False,
            )
            intra_sig = (intra_pred.get("prediction") or {}).get("signal") or "HOLD"
            intra_conf = float((intra_pred.get("prediction") or {}).get("confidence") or 0.0)
            conf = float(min(long_conf, intra_conf))

            # Allow long + short when both models agree.
            if long_sig == "BUY" and intra_sig == "BUY":
                decision = "BUY"
            elif long_sig == "SELL" and intra_sig == "SELL":
                decision = "SELL"
            else:
                decision = "HOLD"
            if decision == "HOLD":
                publish_sync(
                    "agent",
                    "agent.decision",
                    {
                        "cycle_ts": cycle_ts,
                        "instrument_key": instrument_key,
                        "decision": "HOLD",
                        "confidence": conf,
                        "signals": {"long": {"signal": long_sig, "confidence": long_conf}, "intraday": {"signal": intra_sig, "confidence": intra_conf}},
                        "reason": "model_disagree_or_hold",
                    },
                )
                continue

            # Build support/resistance & stop/target from recent intraday candles
            # (Keep this rule-based; model handles direction.)
            series = self._candles.poll_intraday(instrument_key, interval="1m", lookback_minutes=60 * 6)
            if not series.candles:
                series = self._candles.poll_intraday(instrument_key, interval="5m", lookback_minutes=60 * 6)
            highs = [c.high for c in series.candles]
            lows = [c.low for c in series.candles]
            closes = [c.close for c in series.candles]
            if not closes:
                continue

            idea = self._strategy.build_idea(symbol=instrument_key, highs=highs, lows=lows, closes=closes)
            entry = float(idea.entry)
            stop = float(idea.stop_loss)
            target = float(idea.target)

            # Optional: overlays-based gating and plan override.
            overlay = None
            overlay_trade = None
            if settings.AUTOTRADER_OVERLAYS_ENTRY_ENABLED:
                ov_interval = str(settings.TRAIN_INTRADAY_INTERVAL or "1m")
                ov_lookback = int(settings.AUTOTRADER_OVERLAYS_LOOKBACK_MINUTES or 90)
                ov_min_n = int(settings.AUTOTRADER_OVERLAYS_MIN_CANDLES or 30)
                ov_min_conf = float(settings.AUTOTRADER_OVERLAYS_MIN_CONFIDENCE or 0.65)

                ov_series = self._candles.poll_intraday(instrument_key, interval=ov_interval, lookback_minutes=ov_lookback)
                ov_candles = list(ov_series.candles or [])
                overlay = analyze_intraday(instrument_key=instrument_key, interval=ov_interval, candles=ov_candles)

                if len(ov_candles) < ov_min_n:
                    publish_sync(
                        "agent",
                        "agent.decision",
                        {
                            "cycle_ts": cycle_ts,
                            "instrument_key": instrument_key,
                            "decision": "HOLD",
                            "confidence": conf,
                            "signals": {"long": {"signal": long_sig, "confidence": long_conf}, "intraday": {"signal": intra_sig, "confidence": intra_conf}},
                            "reason": "overlays_insufficient_candles",
                            "overlays": {"n": int(overlay.get("n") or 0), "min_n": int(ov_min_n)},
                        },
                    )
                    continue

                overlay_trade = (overlay or {}).get("trade") if isinstance(overlay, dict) else None
                if not isinstance(overlay_trade, dict):
                    publish_sync(
                        "agent",
                        "agent.decision",
                        {
                            "cycle_ts": cycle_ts,
                            "instrument_key": instrument_key,
                            "decision": "HOLD",
                            "confidence": conf,
                            "signals": {"long": {"signal": long_sig, "confidence": long_conf}, "intraday": {"signal": intra_sig, "confidence": intra_conf}},
                            "reason": "overlays_no_trade",
                        },
                    )
                    continue

                ov_side_raw = str(overlay_trade.get("side") or "").strip().lower()
                if ov_side_raw in {"buy", "long"}:
                    ov_side = "BUY"
                elif ov_side_raw in {"sell", "short"}:
                    ov_side = "SELL"
                else:
                    ov_side = ""
                if ov_side in {"BUY", "SELL"}:
                    if ov_side != decision:
                        publish_sync(
                            "agent",
                            "agent.decision",
                            {
                                "cycle_ts": cycle_ts,
                                "instrument_key": instrument_key,
                                "decision": "HOLD",
                                "confidence": conf,
                                "signals": {"long": {"signal": long_sig, "confidence": long_conf}, "intraday": {"signal": intra_sig, "confidence": intra_conf}},
                                "reason": "overlays_side_conflict",
                                "overlays": {"side": ov_side, "confidence": float(overlay_trade.get("confidence") or 0.0)},
                            },
                        )
                        continue

                ov_conf = float(overlay_trade.get("confidence") or 0.0)
                if ov_conf < ov_min_conf:
                    publish_sync(
                        "agent",
                        "agent.decision",
                        {
                            "cycle_ts": cycle_ts,
                            "instrument_key": instrument_key,
                            "decision": "HOLD",
                            "confidence": conf,
                            "signals": {"long": {"signal": long_sig, "confidence": long_conf}, "intraday": {"signal": intra_sig, "confidence": intra_conf}},
                            "reason": "overlays_low_confidence",
                            "overlays": {"confidence": float(ov_conf), "min_conf": float(ov_min_conf)},
                        },
                    )
                    continue

                # Overlays become the plan-of-record when gating is enabled.
                try:
                    entry = float(overlay_trade.get("entry") or entry)
                    stop = float(overlay_trade.get("stop") or stop)
                    target = float(overlay_trade.get("target") or target)
                    conf = float(min(conf, ov_conf))
                except Exception:
                    pass
            if entry <= 0 or stop <= 0:
                continue

            # Require strategy regime to not contradict model direction.
            if idea.side in {"BUY", "SELL"} and idea.side != decision:
                publish_sync(
                    "agent",
                    "agent.decision",
                    {
                        "cycle_ts": cycle_ts,
                        "instrument_key": instrument_key,
                        "decision": "HOLD",
                        "confidence": conf,
                        "signals": {"long": {"signal": long_sig, "confidence": long_conf}, "intraday": {"signal": intra_sig, "confidence": intra_conf}},
                        "reason": "strategy_side_conflict",
                        "strategy": {"side": getattr(idea, "side", None), "entry": float(entry), "stop": float(stop), "target": float(target)},
                    },
                )
                continue

            sizing = self._risk.position_size(capital=capital, entry=entry, stop=stop, confidence=conf)
            qty = float(sizing.get("qty") or 0.0)
            if qty <= 0:
                publish_sync(
                    "agent",
                    "agent.decision",
                    {
                        "cycle_ts": cycle_ts,
                        "instrument_key": instrument_key,
                        "decision": "HOLD",
                        "confidence": conf,
                        "signals": {"long": {"signal": long_sig, "confidence": long_conf}, "intraday": {"signal": intra_sig, "confidence": intra_conf}},
                        "reason": "risk_sizing_zero",
                        "strategy": {"entry": float(entry), "stop": float(stop), "target": float(target)},
                        "risk": sizing,
                    },
                )
                continue

            publish_sync(
                "agent",
                "agent.decision",
                {
                    "cycle_ts": cycle_ts,
                    "instrument_key": instrument_key,
                    "decision": decision,
                    "confidence": conf,
                    "signals": {"long": {"signal": long_sig, "confidence": long_conf}, "intraday": {"signal": intra_sig, "confidence": intra_conf}},
                    "strategy": {"entry": float(entry), "stop": float(stop), "target": float(target)},
                    "risk": sizing,
                    "capital": float(capital),
                },
            )

            # Execute
            if settings.AUTOTRADER_BROKER == "paper":
                res = self._paper.execute(symbol=instrument_key, side=decision, qty=qty, price=entry)
                trade_id = record_trade_open(
                    instrument_key=instrument_key,
                    side=decision,
                    qty=qty,
                    entry=entry,
                    stop=stop,
                    target=float(target),
                    entry_order_id=None,
                    sl_order_id=None,
                    monitor_app_stop=True,
                    meta={"mode": "paper", "overlays": (overlay if isinstance(overlay, dict) else None)},
                )
                order_state_id = create_order(
                    broker="paper",
                    instrument_key=instrument_key,
                    side=decision,
                    qty=float(qty),
                    order_kind="ENTRY",
                    order_type="PAPER",
                    price=float(entry),
                    status="FILLED",
                    meta={"mode": "paper", "trade_id": int(trade_id), "result": res},
                )
                update_trade(int(trade_id), meta_patch={"order_state": {"entry": order_state_id}})
            else:
                if settings.SAFE_MODE:
                    continue

                # Need instrument token mapping.
                row = next((x for x in all_items if x.get("instrument_key") == instrument_key), None)
                token = (row or {}).get("upstox_token")
                if not token:
                    continue

                cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
                client: UpstoxClient | None = None
                entry_order_id: str | None = None
                sl_order_id: str | None = None
                monitor_app_stop = False
                entry_state_id: str | None = None
                sl_state_id: str | None = None
                try:
                    client = UpstoxClient(cfg)
                    tag = f"agent:{instrument_key.replace('|', '_')}"
                    qty_i = max(1, int(qty))

                    # Crash-safe idempotency key (prevents duplicate live orders if the app
                    # restarts between placement and recording agent trade state).
                    tz = ZoneInfo(settings.TIMEZONE)
                    day_key = datetime.now(timezone.utc).astimezone(tz).date().isoformat()
                    key_prefix = f"agent:{instrument_key.replace('|', '_')}:{day_key}:{decision}"
                    entry_client_id = key_prefix + ":entry"
                    sl_client_id = key_prefix + ":sl"

                    # If a prior order state exists, do not place again.
                    existing_entry = get_order_by_client_order_id(entry_client_id)
                    if existing_entry is not None:
                        entry_state_id = existing_entry.id
                        entry_order_id = existing_entry.broker_order_id

                    existing_sl = get_order_by_client_order_id(sl_client_id)
                    if existing_sl is not None:
                        sl_state_id = existing_sl.id
                        sl_order_id = existing_sl.broker_order_id

                    entry_body = build_equity_intraday_market_order(
                        instrument_token=str(token),
                        side=decision,
                        qty=qty_i,
                        tag=tag,
                    )

                    entry_res = None
                    if entry_order_id is None:
                        entry_state_id = create_order(
                            broker="upstox",
                            instrument_key=instrument_key,
                            side=decision,
                            qty=float(qty_i),
                            order_kind="ENTRY",
                            order_type="MARKET",
                            client_order_id=entry_client_id,
                            status="NEW",
                            meta={"tag": tag, "body": entry_body, "idempotency": {"client_order_id": entry_client_id}},
                        )
                        entry_res = client.place_order_v3(entry_body)
                        entry_order_id = str((entry_res.get("data") or {}).get("order_id") or "") or None
                        if entry_state_id:
                            update_order(
                                entry_state_id,
                                status="SUBMITTED",
                                broker_order_id=entry_order_id,
                                meta_patch={"upstox": {"place": entry_res}},
                            )

                    # Try broker-native stop-loss if enabled.
                    if settings.UPSTOX_USE_BROKER_STOP:
                        exit_side = "SELL" if decision == "BUY" else "BUY"
                        sl_body = build_equity_intraday_stop_order(
                            instrument_token=str(token),
                            side=exit_side,
                            qty=qty_i,
                            trigger_price=float(stop),
                            tag=tag + ":sl",
                        )

                        sl_res = None
                        if sl_order_id is None:
                            sl_state_id = create_order(
                                broker="upstox",
                                instrument_key=instrument_key,
                                side=exit_side,
                                qty=float(qty_i),
                                order_kind="STOP",
                                order_type="SL",
                                client_order_id=sl_client_id,
                                status="NEW",
                                meta={
                                    "tag": tag + ":sl",
                                    "body": sl_body,
                                    "idempotency": {"client_order_id": sl_client_id},
                                    "parent": {"entry_order_id": entry_order_id, "entry_state_id": entry_state_id},
                                },
                            )
                            sl_res = client.place_order_v3(sl_body)
                            sl_order_id = str((sl_res.get("data") or {}).get("order_id") or "") or None
                            if sl_state_id:
                                update_order(
                                    sl_state_id,
                                    status="SUBMITTED",
                                    broker_order_id=sl_order_id,
                                    meta_patch={"upstox": {"place": sl_res}},
                                )

                    if sl_order_id is None and settings.UPSTOX_FALLBACK_APP_STOP:
                        monitor_app_stop = True

                    res = {
                        "status": "placed",
                        "entry": (entry_res if entry_res is not None else {"status": "skipped", "reason": "idempotent"}),
                        "stop": (None if sl_order_id is None else {"order_id": sl_order_id}),
                    }
                except UpstoxError as e:
                    res = {"status": "error", "error": str(e)}
                    monitor_app_stop = bool(settings.UPSTOX_FALLBACK_APP_STOP)

                    if entry_state_id:
                        update_order(entry_state_id, status="ERROR", last_error=str(e)[:500], meta_patch={"upstox": {"error": str(e)[:200]}})
                    if sl_state_id:
                        update_order(sl_state_id, status="ERROR", last_error=str(e)[:500], meta_patch={"upstox": {"error": str(e)[:200]}})
                finally:
                    if client is not None:
                        client.close()

                trade_id = record_trade_open(
                    instrument_key=instrument_key,
                    side=decision,
                    qty=float(max(1, int(qty))),
                    entry=entry,
                    stop=stop,
                    target=float(target),
                    entry_order_id=entry_order_id,
                    sl_order_id=sl_order_id,
                    monitor_app_stop=monitor_app_stop,
                    meta={"mode": "upstox", "signals": {"long": long_sig, "intraday": intra_sig}, "overlays": (overlay if isinstance(overlay, dict) else None)},
                )

                update_trade(
                    int(trade_id),
                    meta_patch={
                        "order_state": {
                            "entry": entry_state_id,
                            "sl": sl_state_id,
                        }
                    },
                )

            publish_sync(
                "agent",
                "trade.open",
                {
                    "cycle_ts": cycle_ts,
                    "trade_id": int(trade_id),
                    "instrument_key": instrument_key,
                    "side": decision,
                    "qty": float(qty),
                    "entry": float(entry),
                    "stop": float(stop),
                    "target": float(target),
                    "confidence": float(conf),
                    "broker": settings.AUTOTRADER_BROKER,
                    "result": res,
                },
            )

            placed.append(
                {
                    "instrument_key": instrument_key,
                    "decision": decision,
                    "qty": qty,
                    "entry": entry,
                    "stop": stop,
                    "target": float(target),
                    "confidence": conf,
                    "signals": {"long": long_sig, "intraday": intra_sig},
                    "trade_id": trade_id,
                    "result": res,
                }
            )

        self._status.last_cycle_ts = int(datetime.now(timezone.utc).timestamp())
        log_event(
            "agent.cycle",
            {
                "placed": len(placed),
                "managed": managed,
                "reconciled": reconciled.get("checked", 0),
            },
        )
        publish_sync(
            "agent",
            "agent.cycle_end",
            {
                "cycle_ts": cycle_ts,
                "placed": len(placed),
                "managed": managed,
                "reconciled": reconciled,
            },
        )
        return {"ok": True, "placed": placed, "managed": managed, "reconciled": reconciled}

    async def _manage_open_trades(self) -> dict[str, Any]:
        open_trades = list_open_trades(limit=300)
        if not open_trades:
            return {"checked": 0, "exited": 0, "trailed": 0}

        checked = 0
        exited = 0
        trailed = 0
        overlay_updates = 0

        # Token lookup for upstox exits.
        token_by_key: dict[str, Any] = {}
        if settings.AUTOTRADER_BROKER == "upstox" and not settings.SAFE_MODE:
            rows = self._uni.list(limit=5000)
            token_by_key = {str(r.get("instrument_key")): r.get("upstox_token") for r in rows}

        for t in open_trades:
            instrument_key = str(t["instrument_key"])
            side = str(t["side"]).upper()
            qty_i = max(1, int(float(t["qty"])))
            entry = float(t["entry"])
            stop = float(t["stop"])
            target = float(t["target"])

            series = self._candles.poll_intraday(instrument_key, interval="1m", lookback_minutes=5)
            if not series.candles:
                series = self._candles.poll_intraday(instrument_key, interval="5m", lookback_minutes=30)
            last = float(series.candles[-1].close) if series.candles else 0.0
            if last <= 0:
                continue
            checked += 1

            # Optional: overlay-driven management (tighten stop / update target).
            if settings.AUTOTRADER_OVERLAYS_MANAGEMENT_ENABLED:
                try:
                    ov_interval = str(settings.TRAIN_INTRADAY_INTERVAL or "1m")
                    ov_lookback = int(settings.AUTOTRADER_OVERLAYS_LOOKBACK_MINUTES or 90)
                    ov_min_n = int(settings.AUTOTRADER_OVERLAYS_MIN_CANDLES or 30)
                    ov_min_conf = float(settings.AUTOTRADER_OVERLAYS_MIN_CONFIDENCE or 0.65)
                    min_stop_move_pct = float(settings.AUTOTRADER_OVERLAYS_MIN_STOP_MOVE_PCT or 0.001)
                    min_target_move_pct = float(settings.AUTOTRADER_OVERLAYS_MIN_TARGET_MOVE_PCT or 0.002)

                    ov_series = self._candles.poll_intraday(instrument_key, interval=ov_interval, lookback_minutes=ov_lookback)
                    ov_candles = list(ov_series.candles or [])
                    overlay = analyze_intraday(instrument_key=instrument_key, interval=ov_interval, candles=ov_candles)
                    asof_ts = int((overlay or {}).get("asof_ts") or 0) if isinstance(overlay, dict) else 0

                    meta = t.get("meta") if isinstance(t, dict) else None
                    if not isinstance(meta, dict):
                        meta = {}
                    last_overlay_ts = int(meta.get("last_overlay_ts") or 0)

                    # Throttle: do not re-apply management if candle hasn't advanced.
                    if asof_ts > 0 and asof_ts == last_overlay_ts:
                        raise RuntimeError("overlay_throttled")

                    tr = (overlay or {}).get("trade") if isinstance(overlay, dict) else None
                    if isinstance(tr, dict) and len(ov_candles) >= ov_min_n:
                        tr_conf = float(tr.get("confidence") or 0.0)
                        tr_side_raw = str(tr.get("side") or "").strip().lower()
                        tr_side = ("BUY" if tr_side_raw in {"buy", "long"} else ("SELL" if tr_side_raw in {"sell", "short"} else ""))

                        if tr_side == side and tr_conf >= ov_min_conf:
                            cand_stop = float(tr.get("stop") or stop)
                            cand_target = float(tr.get("target") or target)

                            # Only move in the favorable direction.
                            if side == "BUY":
                                cand_stop = max(stop, cand_stop)
                                cand_target = max(target, cand_target)
                            elif side == "SELL":
                                cand_stop = min(stop, cand_stop)
                                cand_target = min(target, cand_target)

                            stop_changed = False
                            target_changed = False
                            if stop > 0 and cand_stop > 0:
                                if abs(cand_stop - stop) / max(1e-9, abs(stop)) >= min_stop_move_pct:
                                    stop = float(cand_stop)
                                    stop_changed = True
                            if target > 0 and cand_target > 0:
                                if abs(cand_target - target) / max(1e-9, abs(target)) >= min_target_move_pct:
                                    target = float(cand_target)
                                    target_changed = True

                            if stop_changed or target_changed:
                                overlay_updates += 1
                                update_trade(
                                    int(t["id"]),
                                    stop=(stop if stop_changed else None),
                                    target=(target if target_changed else None),
                                    monitor_app_stop=True,
                                    meta_patch={
                                        "last_overlay_ts": int(asof_ts) if asof_ts else None,
                                        "overlays_management": {
                                            "enabled": True,
                                            "interval": ov_interval,
                                            "confidence": float(tr_conf),
                                            "stop_changed": bool(stop_changed),
                                            "target_changed": bool(target_changed),
                                        },
                                    },
                                )
                                publish_sync(
                                    "agent",
                                    "trade.update",
                                    {
                                        "trade_id": int(t["id"]),
                                        "instrument_key": instrument_key,
                                        "stop": float(stop),
                                        "target": float(target),
                                        "reason": "overlays",
                                        "asof_ts": int(asof_ts) if asof_ts else None,
                                    },
                                )

                                # Best-effort: replace broker-side stop order if configured.
                                if (
                                    stop_changed
                                    and settings.AUTOTRADER_BROKER == "upstox"
                                    and (not settings.SAFE_MODE)
                                    and settings.UPSTOX_USE_BROKER_STOP
                                    and t.get("sl_order_id")
                                ):
                                    token = token_by_key.get(instrument_key)
                                    if token:
                                        cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
                                        client: UpstoxClient | None = None
                                        try:
                                            client = UpstoxClient(cfg)
                                            try:
                                                client.cancel_order_v3(str(t.get("sl_order_id")))
                                            except Exception:
                                                pass

                                            exit_side = "SELL" if side == "BUY" else "BUY"
                                            tag = f"agent:{instrument_key.replace('|', '_')}:sl:trail"
                                            sl_body = build_equity_intraday_stop_order(
                                                instrument_token=str(token),
                                                side=exit_side,
                                                qty=qty_i,
                                                trigger_price=float(stop),
                                                tag=tag,
                                            )
                                            sl_res = client.place_order_v3(sl_body)
                                            new_sl = str((sl_res.get("data") or {}).get("order_id") or "") or None
                                            if new_sl:
                                                update_trade(
                                                    int(t["id"]),
                                                    sl_order_id=new_sl,
                                                    monitor_app_stop=False,
                                                    meta_patch={"sl_replaced": True, "sl_replace_ts": int(datetime.now(timezone.utc).timestamp())},
                                                )
                                        except Exception as e:
                                            update_trade(int(t["id"]), monitor_app_stop=True, meta_patch={"sl_replace_error": str(e)[:200]})
                                        finally:
                                            if client is not None:
                                                client.close()
                except Exception:
                    # Never fail management loop due to overlay logic.
                    pass

            # Trailing stop
            if settings.AUTOTRADER_TRAILING_ENABLED:
                initial_r = abs(entry - stop)
                activation_r = float(settings.AUTOTRADER_TRAIL_ACTIVATION_R or 0.0)
                trail_pct = float(settings.AUTOTRADER_TRAIL_PCT or 0.0)
                if initial_r > 0 and trail_pct > 0:
                    if side == "BUY":
                        moved = last - entry
                        active = moved >= activation_r * initial_r
                        if active:
                            candidate = last * (1.0 - trail_pct)
                            if candidate > stop:
                                stop = float(candidate)
                                update_trade(int(t["id"]), stop=stop, monitor_app_stop=True, meta_patch={"trail": True})
                                trailed += 1
                                publish_sync("agent", "trade.update", {"trade_id": int(t["id"]), "instrument_key": instrument_key, "stop": float(stop), "reason": "trailing"})
                    elif side == "SELL":
                        moved = entry - last
                        active = moved >= activation_r * initial_r
                        if active:
                            candidate = last * (1.0 + trail_pct)
                            if candidate < stop:
                                stop = float(candidate)
                                update_trade(int(t["id"]), stop=stop, monitor_app_stop=True, meta_patch={"trail": True})
                                trailed += 1
                                publish_sync("agent", "trade.update", {"trade_id": int(t["id"]), "instrument_key": instrument_key, "stop": float(stop), "reason": "trailing"})

            # Exit conditions
            stop_breached = (side == "BUY" and last <= stop) or (side == "SELL" and last >= stop)
            target_hit = (side == "BUY" and last >= target) or (side == "SELL" and last <= target)

            # Apply app-side stop only when enabled on the trade (or when trailing enabled, we forced it).
            must_stop = bool(t.get("monitor_app_stop")) and stop_breached
            must_target = bool(settings.AUTOTRADER_TARGET_EXIT_ENABLED) and target_hit
            if not (must_stop or must_target):
                continue

            reason = "app_stop" if must_stop else "target_hit"
            exit_side = "SELL" if side == "BUY" else "BUY"

            if settings.AUTOTRADER_BROKER == "paper":
                try:
                    self._paper.execute(symbol=instrument_key, side=exit_side, qty=float(qty_i), price=float(last))
                    pnl = (last - entry) * qty_i if side == "BUY" else (entry - last) * qty_i
                    mark_trade_closed(int(t["id"]), reason=reason, close_price=float(last), pnl=float(pnl))
                    exited += 1
                    publish_sync(
                        "agent",
                        "trade.close",
                        {"trade_id": int(t["id"]), "instrument_key": instrument_key, "reason": reason, "close_price": float(last), "pnl": float(pnl)},
                    )
                except Exception as e:
                    update_trade(int(t["id"]), meta_patch={"exit_error": str(e)[:200]})
                continue

            # Upstox
            if settings.SAFE_MODE:
                update_trade(int(t["id"]), meta_patch={"exit_error": "SAFE_MODE"})
                continue

            token = token_by_key.get(instrument_key)
            if not token:
                update_trade(int(t["id"]), meta_patch={"exit_error": "missing_token"})
                continue

            cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
            client: UpstoxClient | None = None
            try:
                client = UpstoxClient(cfg)
                sl_order_id = t.get("sl_order_id")
                if sl_order_id:
                    try:
                        client.cancel_order_v3(str(sl_order_id))
                    except Exception:
                        pass

                tag = f"agent:{instrument_key.replace('|', '_')}:exit"
                body = build_equity_intraday_market_order(
                    instrument_token=str(token),
                    side=exit_side,
                    qty=qty_i,
                    tag=tag,
                )
                res = client.place_order_v3(body)
                exit_order_id = str((res.get("data") or {}).get("order_id") or "") or None
                mark_trade_closed(int(t["id"]), reason=reason, exit_order_id=exit_order_id, close_price=float(last))
                exited += 1
                publish_sync(
                    "agent",
                    "trade.close",
                    {"trade_id": int(t["id"]), "instrument_key": instrument_key, "reason": reason, "close_price": float(last), "exit_order_id": exit_order_id},
                )
            except Exception as e:
                update_trade(int(t["id"]), meta_patch={"exit_error": str(e)[:200]})
            finally:
                if client is not None:
                    client.close()

        return {"checked": checked, "exited": exited, "trailed": trailed, "overlay_updates": overlay_updates}

    async def _reconcile_open_trades(self) -> dict[str, Any]:
        if settings.AUTOTRADER_BROKER != "upstox":
            return {"checked": 0, "closed": 0, "updated": 0}
        if settings.SAFE_MODE:
            return {"checked": 0, "closed": 0, "updated": 0, "skipped": "SAFE_MODE"}

        open_trades = list_open_trades(limit=200)
        if not open_trades:
            return {"checked": 0, "closed": 0, "updated": 0}

        cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
        client: UpstoxClient | None = None
        checked = 0
        closed = 0
        updated = 0

        try:
            client = UpstoxClient(cfg)

            # Build token lookup once.
            rows = self._uni.list(limit=5000)
            token_by_key = {str(r.get("instrument_key")): r.get("upstox_token") for r in rows}

            for t in open_trades:
                entry_order_id = t.get("entry_order_id")
                if not entry_order_id:
                    continue
                checked += 1

                try:
                    details = client.order_details_v2(str(entry_order_id))
                except Exception as e:
                    update_trade(int(t["id"]), meta_patch={"reconcile_error": str(e)[:200]})
                    updated += 1
                    continue

                data = details.get("data") or {}
                status = str(data.get("status") or data.get("order_status") or "").upper()
                update_trade(int(t["id"]), meta_patch={"entry_status": status, "entry_details": data})
                updated += 1

                # Terminal failure statuses
                if status in {"REJECTED", "CANCELLED", "CANCELED", "ERROR"}:
                    mark_trade_closed(int(t["id"]), reason=f"entry_{status.lower()}")
                    closed += 1
                    continue

                # If entry is filled (or partially filled), ensure stop exists.
                if status in {"COMPLETE", "FILLED", "PARTIALLY_FILLED", "PARTIAL", "EXECUTED"}:
                    if t.get("sl_order_id"):
                        continue
                    if not settings.UPSTOX_USE_BROKER_STOP:
                        if settings.UPSTOX_FALLBACK_APP_STOP:
                            update_trade(int(t["id"]), monitor_app_stop=True)
                            updated += 1
                        continue

                    instrument_key = str(t["instrument_key"])
                    token = token_by_key.get(instrument_key)
                    if not token:
                        update_trade(int(t["id"]), monitor_app_stop=bool(settings.UPSTOX_FALLBACK_APP_STOP), meta_patch={"sl_error": "missing_token"})
                        updated += 1
                        continue

                    side = str(t["side"]).upper()
                    exit_side = "SELL" if side == "BUY" else "BUY"
                    qty = max(1, int(float(t["qty"])))
                    stop = float(t["stop"])
                    tag = f"agent:{instrument_key.replace('|', '_')}:sl"
                    try:
                        sl_body = build_equity_intraday_stop_order(
                            instrument_token=str(token),
                            side=exit_side,
                            qty=qty,
                            trigger_price=stop,
                            tag=tag,
                        )
                        sl_res = client.place_order_v3(sl_body)
                        sl_order_id = str((sl_res.get("data") or {}).get("order_id") or "") or None
                        update_trade(
                            int(t["id"]),
                            sl_order_id=sl_order_id,
                            monitor_app_stop=(False if sl_order_id else bool(settings.UPSTOX_FALLBACK_APP_STOP)),
                            meta_patch={"sl_placed": bool(sl_order_id)},
                        )
                        updated += 1
                    except Exception as e:
                        update_trade(
                            int(t["id"]),
                            monitor_app_stop=bool(settings.UPSTOX_FALLBACK_APP_STOP),
                            meta_patch={"sl_error": str(e)[:200]},
                        )
                        updated += 1
        finally:
            if client is not None:
                client.close()

        return {"checked": checked, "closed": closed, "updated": updated}

    async def _square_off_open_trades(self, *, reason: str) -> dict[str, Any]:
        open_trades = list_open_trades(limit=500)
        attempted = 0
        closed = 0
        errors: list[str] = []

        for t in open_trades:
            attempted += 1
            instrument_key = str(t["instrument_key"])
            side = str(t["side"]).upper()
            qty = max(1, int(float(t["qty"])))
            entry = float(t["entry"])

            series = self._candles.poll_intraday(instrument_key, interval="1m", lookback_minutes=5)
            if not series.candles:
                series = self._candles.poll_intraday(instrument_key, interval="5m", lookback_minutes=30)
            last = float(series.candles[-1].close) if series.candles else entry

            exit_side = "SELL" if side == "BUY" else "BUY"

            if settings.AUTOTRADER_BROKER == "paper":
                try:
                    self._paper.execute(symbol=instrument_key, side=exit_side, qty=float(qty), price=float(last))
                    pnl = (last - entry) * qty if side == "BUY" else (entry - last) * qty
                    mark_trade_closed(int(t["id"]), reason=reason, close_price=float(last), pnl=float(pnl))
                    closed += 1
                except Exception as e:
                    errors.append(f"paper:{instrument_key}:{e}")
                continue

            # Upstox
            if settings.SAFE_MODE:
                errors.append(f"upstox:{instrument_key}:SAFE_MODE")
                continue

            rows = self._uni.list(limit=2000)
            meta = next((x for x in rows if x.get("instrument_key") == instrument_key), None) or {}
            token = meta.get("upstox_token")
            if not token:
                errors.append(f"upstox:{instrument_key}:missing_token")
                continue

            cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
            client: UpstoxClient | None = None
            try:
                client = UpstoxClient(cfg)

                # Cancel SL if present (best-effort).
                sl_order_id = t.get("sl_order_id")
                if sl_order_id:
                    try:
                        client.cancel_order_v3(str(sl_order_id))
                    except Exception:
                        pass

                tag = f"agent:{instrument_key.replace('|', '_')}:squareoff"
                body = build_equity_intraday_market_order(
                    instrument_token=str(token),
                    side=exit_side,
                    qty=qty,
                    tag=tag,
                )
                res = client.place_order_v3(body)
                exit_order_id = str((res.get("data") or {}).get("order_id") or "") or None
                mark_trade_closed(int(t["id"]), reason=reason, exit_order_id=exit_order_id, close_price=float(last))
                closed += 1
            except Exception as e:
                errors.append(f"upstox:{instrument_key}:{e}")
            finally:
                if client is not None:
                    client.close()

        return {"attempted": attempted, "closed": closed, "errors": errors[:20]}

    async def _check_app_side_stops(self) -> dict:
        # Backward-compatible alias.
        return await self._manage_open_trades()


AGENT_SINGLETON = AutoTraderAgent()
