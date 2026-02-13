from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

from app.agent.state import list_open_trades, local_day_bounds_utc, mark_trade_closed, record_trade_open
from app.auth import token_store
from app.core.db import db_conn
from app.candles.service import CandleService
from app.core.audit import log_event
from app.core.controls import get_controls
from app.core.guards import get_safety_status
from app.core.settings import settings
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError
from app.integrations.upstox.order_builder import build_equity_intraday_market_order
from app.orders.state import create_order, update_order
from app.paper_trading.service import PaperTradingService
from app.portfolio.service import PortfolioService
from app.realtime.bus import publish_sync
from app.strategy.engine import StrategyEngine

from app.hft.index_options.calibration import snapshot as calib_snapshot
from app.hft.index_options.calibration import update_on_trade_close
from app.hft.index_options.selector import OptionContract, find_atm_option


logger = logging.getLogger(__name__)


BrokerName = Literal["paper", "upstox"]


@dataclass
class HftStatus:
    running: bool
    started_ts: int | None
    last_cycle_ts: int | None
    last_error: str | None
    last_decisions: list[dict[str, Any]]


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _safe_float(v: Any, default: float | None = None) -> float | None:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


class IndexOptionsHftService:
    """1m/5m index-options HFT loop.

    Safe-by-default:
    - disabled unless INDEX_OPTIONS_HFT_ENABLED=true
    - if broker=upstox, SAFE_MODE must be false
    """

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._status = HftStatus(running=False, started_ts=None, last_cycle_ts=None, last_error=None, last_decisions=[])
        self._candles = CandleService()
        self._strategy = StrategyEngine()
        self._paper = PaperTradingService()
        self._portfolio = PortfolioService()

        # Track last processed candle timestamp per (underlying, interval)
        self._last_ts: dict[tuple[str, str], int] = {}

    def _broker_override_key(self) -> str:
        return "index_options_hft_broker"

    def _load_broker_override(self) -> str | None:
        try:
            with db_conn() as conn:
                row = conn.execute(
                    "SELECT value FROM trade_controls WHERE key=? LIMIT 1",
                    (self._broker_override_key(),),
                ).fetchone()
                if row is None:
                    return None
                v = str(row["value"] or "").strip().lower()
                return v or None
        except Exception:
            return None

    def _save_broker_override(self, broker: str, *, actor: str = "api") -> None:
        ts = _now_ts()
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO trade_controls(key, value, ts_updated) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts_updated=excluded.ts_updated",
                (self._broker_override_key(), str(broker), int(ts)),
            )
            conn.execute(
                "INSERT INTO trade_controls(key, value, ts_updated) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts_updated=excluded.ts_updated",
                (
                    "last_change",
                    json.dumps({"actor": actor, "key": self._broker_override_key(), "broker": str(broker)}, separators=(",", ":")),
                    int(ts),
                ),
            )

    def broker(self) -> BrokerName:
        raw = self._load_broker_override() or str(getattr(settings, "INDEX_OPTIONS_HFT_BROKER", "paper"))
        broker: BrokerName = str(raw).lower().strip()  # type: ignore[assignment]
        if broker not in {"paper", "upstox"}:
            broker = "paper"  # type: ignore[assignment]
        return broker

    def set_broker(self, broker: str, *, actor: str = "api") -> dict[str, Any]:
        b = str(broker or "").strip().lower()
        if b not in {"paper", "upstox"}:
            return {"ok": False, "detail": {"detail": "unsupported broker", "broker": b, "supported": ["paper", "upstox"]}}

        if self._status.running:
            return {"ok": False, "detail": "stop the HFT loop before switching broker"}

        if b == "upstox":
            if bool(settings.SAFE_MODE):
                return {"ok": False, "detail": "SAFE_MODE=true: live trading disabled"}
            if not token_store.is_logged_in():
                return {"ok": False, "detail": "Upstox not authenticated. Open /api/auth/upstox/login"}

        self._save_broker_override(b, actor=str(actor))
        publish_sync("hft", "hft.broker", {"broker": b})
        log_event("hft.index_options.broker", {"broker": b, "actor": str(actor)})
        return {"ok": True, "broker": b}

    def status(self) -> dict[str, Any]:
        broker = str(self.broker())
        open_hft = self._count_open_hft_trades()
        today_bounds = self._today_bounds_utc()
        trades_today = self._count_trades_today_hft(bounds=today_bounds)
        pnl_today = self._realized_pnl_today_hft(bounds=today_bounds)
        return {
            "running": self._status.running,
            "started_ts": self._status.started_ts,
            "last_cycle_ts": self._status.last_cycle_ts,
            "last_error": self._status.last_error,
            "enabled": bool(getattr(settings, "INDEX_OPTIONS_HFT_ENABLED", False)),
            "broker": broker,
            "calibration": calib_snapshot(path=str(getattr(settings, "INDEX_OPTIONS_HFT_STATE_PATH", "data/hft_index_options_state.json"))),
            "open_trades": int(open_hft),
            "trades_today": int(trades_today),
            "pnl_today": float(pnl_today),
            "last_decisions": list(self._status.last_decisions or [])[-10:],
        }

    async def start(self) -> dict[str, Any]:
        if self._task and not self._task.done():
            return {"ok": True, "detail": "already running"}

        if not bool(getattr(settings, "INDEX_OPTIONS_HFT_ENABLED", False)):
            return {"ok": False, "detail": "INDEX_OPTIONS_HFT_ENABLED=false"}

        self._status.running = True
        self._status.started_ts = _now_ts()
        self._status.last_error = None
        self._task = asyncio.create_task(self._run_loop())
        publish_sync("hft", "hft.start", {"broker": str(self.broker()), "safe_mode": bool(settings.SAFE_MODE)})
        log_event("hft.index_options.start", {"broker": str(self.broker())})
        return {"ok": True}

    async def stop(self) -> dict[str, Any]:
        self._status.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        publish_sync("hft", "hft.stop", {})
        log_event("hft.index_options.stop", {})
        return {"ok": True}

    async def run_once(self, *, force_offmarket_paper: bool = False) -> dict[str, Any]:
        try:
            res = await self._run_cycle(allow_offmarket_paper=bool(force_offmarket_paper))
            self._status.last_cycle_ts = _now_ts()
            return res
        except Exception as e:
            self._status.last_error = str(e)
            publish_sync("hft", "hft.error", {"error": str(e)[:400]})
            log_event("hft.index_options.error", {"error": str(e)[:500]})
            raise

    async def flatten(self) -> dict[str, Any]:
        """Close all open HFT trades (best-effort)."""
        closed = 0
        errors: list[str] = []
        for t in list_open_trades(limit=1000):
            meta_raw = t.get("meta")
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            if str(meta.get("origin") or "") != "hft_index_options":
                continue
            try:
                await self._exit_trade_market(t, reason="manual_flatten")
                closed += 1
            except Exception as e:
                errors.append(str(e)[:200])
        publish_sync("hft", "hft.flatten", {"closed": closed, "errors": errors[:20]})
        return {"ok": True, "closed": closed, "errors": errors[:20]}

    async def _run_loop(self) -> None:
        try:
            while self._status.running:
                try:
                    await self._run_cycle()
                    poll = max(1, int(getattr(settings, "INDEX_OPTIONS_HFT_POLL_SECONDS", 5)))
                    await asyncio.sleep(poll)
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    self._status.last_error = str(e)
                    logger.exception("hft loop error: %s", e)
                    publish_sync("hft", "hft.error", {"error": str(e)[:400]})
                    await asyncio.sleep(2)
        finally:
            self._status.running = False

    async def _run_cycle(self, *, allow_offmarket_paper: bool = False) -> dict[str, Any]:
        if not bool(getattr(settings, "INDEX_OPTIONS_HFT_ENABLED", False)):
            return {"ok": True, "skipped": True, "reason": "disabled"}

        broker: BrokerName = self.broker()

        controls = get_controls()
        if getattr(controls, "agent_disabled", False):
            return {"ok": True, "skipped": True, "reason": "agent_disabled=true"}
        if getattr(controls, "freeze_new_orders", False):
            return {"ok": True, "skipped": True, "reason": "freeze_new_orders=true"}

        status = get_safety_status()
        ignore_market = bool(getattr(settings, "INDEX_OPTIONS_HFT_IGNORE_MARKET_STATE_WHEN_PAPER", False)) or bool(allow_offmarket_paper)
        if status.market_state != "LIVE":
            if not (broker == "paper" and ignore_market):
                return {"ok": True, "skipped": True, "reason": status.reason, "state": status.market_state}

        # Manage existing HFT trades (stop/target/time).
        managed = await self._manage_open_trades()

        # Hard risk limits.
        max_open = max(0, int(getattr(settings, "INDEX_OPTIONS_HFT_MAX_OPEN_TRADES", 2)))
        open_hft = self._count_open_hft_trades()
        if max_open > 0 and open_hft >= max_open:
            return {"ok": True, "managed": managed, "placed": [], "skipped": True, "reason": "max_open_trades", "open_trades": int(open_hft)}

        bounds = self._today_bounds_utc()
        max_today = max(0, int(getattr(settings, "INDEX_OPTIONS_HFT_MAX_TRADES_PER_DAY", 20)))
        trades_today = self._count_trades_today_hft(bounds=bounds)
        if max_today > 0 and trades_today >= max_today:
            return {"ok": True, "managed": managed, "placed": [], "skipped": True, "reason": "max_trades_per_day", "trades_today": int(trades_today)}

        max_loss = float(getattr(settings, "INDEX_OPTIONS_HFT_MAX_DAILY_LOSS_INR", 0.0))
        if max_loss > 0:
            pnl_today = self._realized_pnl_today_hft(bounds=bounds)
            if pnl_today <= -abs(max_loss):
                return {"ok": True, "managed": managed, "placed": [], "skipped": True, "reason": "daily_loss_stop", "pnl_today": float(pnl_today), "max_daily_loss": float(max_loss)}

        market = await self._market_alignment()
        require_align = bool(getattr(settings, "INDEX_OPTIONS_HFT_REQUIRE_MARKET_ALIGNMENT", False))
        if require_align and not market.get("aligned"):
            return {"ok": True, "managed": managed, "placed": [], "skipped": True, "reason": "market_not_aligned", "market": market}

        placed: list[dict[str, Any]] = []

        for sym in ("NIFTY", "SENSEX"):
            decision = await self._decision_for_underlying(sym, market=market, require_market_alignment=require_align)
            if decision is None:
                continue
            placed.append(decision)

        # Keep last decisions for UI
        if placed:
            self._status.last_decisions.extend(placed)
            self._status.last_decisions = self._status.last_decisions[-50:]

        return {"ok": True, "managed": managed, "placed": placed, "market": market}

    async def _market_alignment(self) -> dict[str, Any]:
        """Compute a coarse market regime using NIFTY+SENSEX 5m signals."""
        out: dict[str, Any] = {"aligned": False, "signals": {}}
        for sym in ("NIFTY", "SENSEX"):
            spot_q = getattr(settings, f"INDEX_OPTIONS_HFT_{sym}_SPOT_QUERY", sym)
            spot_key = CandleService._resolve_instrument_key(str(spot_q))
            sig5 = await self._spot_signal(spot_key, interval="5m", lookback_minutes=int(getattr(settings, "INDEX_OPTIONS_HFT_SPOT_LOOKBACK_MINUTES", 240)))
            out["signals"][sym] = {"instrument_key": spot_key, **sig5}

        s1 = str(out["signals"].get("NIFTY", {}).get("action") or "HOLD")
        s2 = str(out["signals"].get("SENSEX", {}).get("action") or "HOLD")
        out["aligned"] = (s1 in {"BUY", "SELL"} and s1 == s2)
        out["action"] = (s1 if out["aligned"] else "HOLD")
        return out

    async def _decision_for_underlying(
        self,
        underlying_symbol: str,
        *,
        market: dict[str, Any],
        require_market_alignment: bool,
    ) -> dict[str, Any] | None:
        sym = str(underlying_symbol).strip().upper()

        spot_q = getattr(settings, f"INDEX_OPTIONS_HFT_{sym}_SPOT_QUERY", sym)
        spot_key = CandleService._resolve_instrument_key(str(spot_q))

        sig1 = await self._spot_signal(spot_key, interval="1m", lookback_minutes=int(getattr(settings, "INDEX_OPTIONS_HFT_SPOT_LOOKBACK_MINUTES", 240)))
        sig5 = await self._spot_signal(spot_key, interval="5m", lookback_minutes=int(getattr(settings, "INDEX_OPTIONS_HFT_SPOT_LOOKBACK_MINUTES", 240)))

        # Require 1m and 5m to agree.
        if sig1["action"] != sig5["action"]:
            return None

        action = str(sig1["action"])
        if action not in {"BUY", "SELL"}:
            return None

        # Adaptive confidence threshold
        base_min = float(getattr(settings, "INDEX_OPTIONS_HFT_MIN_CONFIDENCE", 0.70))
        cal = calib_snapshot(path=str(getattr(settings, "INDEX_OPTIONS_HFT_STATE_PATH", "data/hft_index_options_state.json")))
        min_conf = _clamp(base_min + float(cal.get("confidence_delta") or 0.0), 0.55, 0.95)

        conf = min(float(sig1["confidence"]), float(sig5["confidence"]))
        if conf < min_conf:
            return None

        spot = float(sig1["spot"])

        # Optional: only trade in market direction (cross-index alignment).
        if require_market_alignment and str(market.get("action")) != action:
            return None

        opt_type = "CE" if action == "BUY" else "PE"
        contract = find_atm_option(
            underlying_symbol=sym,
            option_type=opt_type,  # type: ignore[arg-type]
            spot=spot,
            max_expiry_days=int(getattr(settings, "INDEX_OPTIONS_HFT_MAX_EXPIRY_DAYS", 14)),
        )
        if contract is None:
            return {
                "ok": False,
                "underlying": sym,
                "action": action,
                "confidence": conf,
                "reason": "no_option_contract_found",
                "spot_instrument_key": spot_key,
                "spot": spot,
                "signal_1m": sig1,
                "signal_5m": sig5,
            }

        # Debounce on candle timestamp: do at most one entry per new 1m candle.
        k = (sym, "1m")
        last_ts = self._last_ts.get(k)
        cur_ts = int(sig1.get("candle_ts") or 0)
        if last_ts is not None and cur_ts and cur_ts <= last_ts:
            return None

        # Derive entry price from option 1m last close.
        opt_price = await self._option_last_price(contract.instrument_key)
        if opt_price is None or opt_price <= 0:
            return {
                "ok": False,
                "underlying": sym,
                "action": action,
                "confidence": conf,
                "reason": "no_option_price",
                "option": contract.__dict__,
            }

        lot_size = int(getattr(settings, f"INDEX_OPTIONS_HFT_{sym}_LOT_SIZE", 1))
        lot_size = max(1, lot_size)

        # Risk sizing: long options => max loss ~= premium.
        bal = self._portfolio.balance(source="auto")
        balance = _safe_float(bal.get("balance"), 0.0) or 0.0
        risk_frac = float(getattr(settings, "INDEX_OPTIONS_HFT_RISK_FRACTION", 0.002))
        risk_budget = max(0.0, balance * risk_frac)

        max_cap = float(getattr(settings, "INDEX_OPTIONS_HFT_MAX_CAPITAL_PER_TRADE_INR", 5000.0))
        risk_budget = min(risk_budget, max_cap)

        per_lot_cost = float(opt_price) * float(lot_size)
        if per_lot_cost <= 0:
            return None

        lots = int(risk_budget // per_lot_cost)
        lots = max(0, min(lots, int(getattr(settings, "INDEX_OPTIONS_HFT_MAX_LOTS", 1))))
        if lots <= 0:
            return {
                "ok": False,
                "underlying": sym,
                "action": action,
                "confidence": conf,
                "reason": "risk_budget_too_small",
                "risk_budget": risk_budget,
                "per_lot_cost": per_lot_cost,
                "balance": balance,
                "option": contract.__dict__,
            }

        qty = int(lots * lot_size)

        # Stops/targets in option premium space.
        sl_pct = float(getattr(settings, "INDEX_OPTIONS_HFT_OPTION_STOP_PCT", 0.12))
        tp_pct = float(getattr(settings, "INDEX_OPTIONS_HFT_OPTION_TARGET_PCT", 0.18))
        stop = float(opt_price) * (1.0 - _clamp(sl_pct, 0.01, 0.95))
        target = float(opt_price) * (1.0 + _clamp(tp_pct, 0.01, 2.0))

        broker: BrokerName = self.broker()

        # Place entry.
        placed = await self._enter_long_option(
            broker=broker,
            contract=contract,
            qty=qty,
            entry_price=float(opt_price),
            stop=float(stop),
            target=float(target),
            meta={
                "origin": "hft_index_options",
                "underlying": sym,
                "spot_instrument_key": spot_key,
                "spot": spot,
                "signal_1m": sig1,
                "signal_5m": sig5,
                "market": market,
                "confidence": conf,
                "min_confidence": min_conf,
            },
        )

        if placed.get("ok"):
            self._last_ts[k] = cur_ts

        return {
            "underlying": sym,
            "action": action,
            "confidence": conf,
            "min_confidence": min_conf,
            "spot": spot,
            "spot_instrument_key": spot_key,
            "option": contract.__dict__,
            "option_entry": float(opt_price),
            "qty": int(qty),
            "stop": float(stop),
            "target": float(target),
            "placed": placed,
        }

    def _today_bounds_utc(self) -> tuple[int, int]:
        tz = str(getattr(settings, "TIMEZONE", "Asia/Kolkata"))
        try:
            day_local = datetime.now(ZoneInfo(tz)).date()
        except Exception:
            day_local = date.today()
        return local_day_bounds_utc(day_local=day_local, tz_name=tz)

    def _count_open_hft_trades(self) -> int:
        n = 0
        for t in list_open_trades(limit=2000):
            meta = t.get("meta") if isinstance(t, dict) else None
            if isinstance(meta, dict) and str(meta.get("origin") or "") == "hft_index_options":
                n += 1
        return int(n)

    def _count_trades_today_hft(self, *, bounds: tuple[int, int]) -> int:
        start_ts, end_ts = bounds
        like = '%"origin":"hft_index_options"%'
        with db_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(1) AS c FROM agent_trades "
                "WHERE ts_open >= ? AND ts_open < ? AND meta_json LIKE ?",
                (int(start_ts), int(end_ts), like),
            ).fetchone()
            return int(row["c"] if row else 0)

    def _realized_pnl_today_hft(self, *, bounds: tuple[int, int]) -> float:
        start_ts, end_ts = bounds
        like = '%"origin":"hft_index_options"%'
        total = 0.0
        with db_conn() as conn:
            rows = conn.execute(
                "SELECT meta_json FROM agent_trades "
                "WHERE status='CLOSED' AND ts_close IS NOT NULL AND ts_close >= ? AND ts_close < ? AND meta_json LIKE ? "
                "ORDER BY id DESC LIMIT 5000",
                (int(start_ts), int(end_ts), like),
            ).fetchall()
        for r in rows:
            try:
                meta = json.loads(r["meta_json"] or "{}")
            except Exception:
                meta = {}
            try:
                total += float(meta.get("pnl") or 0.0)
            except Exception:
                continue
        return float(total)

    async def _spot_signal(self, instrument_key: str, *, interval: str, lookback_minutes: int) -> dict[str, Any]:
        series = await asyncio.to_thread(self._candles.poll_intraday, instrument_key, interval, lookback_minutes)
        candles = list(series.candles or [])
        if len(candles) < 60:
            return {"action": "HOLD", "confidence": 0.10, "spot": (float(candles[-1].close) if candles else 0.0), "candle_ts": (int(candles[-1].ts) if candles else None), "reason": "not_enough_candles"}

        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        closes = [float(c.close) for c in candles]
        idea = self._strategy.build_idea(symbol=instrument_key, highs=highs, lows=lows, closes=closes)
        action = str(idea.side).upper()

        if action == "HOLD":
            confidence = 0.30
        else:
            rr = float(idea.rr or 0.0)
            confidence = 0.55 + _clamp((rr - 1.0) / 4.0, 0.0, 0.40)

        last_close = float(closes[-1]) if closes else 0.0
        last_ts = int(getattr(candles[-1], "ts")) if candles else None

        return {
            "action": action,
            "confidence": _clamp(float(confidence), 0.0, 0.99),
            "spot": float(last_close),
            "candle_ts": last_ts,
            "reason": str(idea.reason),
            "entry": float(idea.entry),
            "stop_loss": float(idea.stop_loss),
            "target": float(idea.target),
            "rr": float(idea.rr),
            "interval": str(interval),
        }

    async def _option_last_price(self, option_instrument_key: str) -> float | None:
        series = await asyncio.to_thread(self._candles.poll_intraday, option_instrument_key, "1m", 120)
        candles = list(series.candles or [])
        if not candles:
            return None
        try:
            return float(candles[-1].close)
        except Exception:
            return None

    async def _enter_long_option(
        self,
        *,
        broker: BrokerName,
        contract: OptionContract,
        qty: int,
        entry_price: float,
        stop: float,
        target: float,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        symbol = str(contract.instrument_key)
        qty_i = int(qty)
        if qty_i <= 0:
            return {"ok": False, "detail": "qty<=0"}

        tag = f"hft:index_options:{symbol}"

        if broker == "upstox":
            if settings.SAFE_MODE:
                return {"ok": False, "detail": "SAFE_MODE=true: live orders disabled"}
            token = str(contract.upstox_token or "").strip()
            if not token:
                return {"ok": False, "detail": "missing upstox_token for option contract"}

            cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
            client = UpstoxClient(cfg)
            entry_client_id = f"hft-io-entry-{symbol}-{_now_ts()}"
            body = build_equity_intraday_market_order(instrument_token=token, side="BUY", qty=qty_i, tag=tag)

            state_id = create_order(
                broker="upstox",
                instrument_key=symbol,
                side="BUY",
                qty=float(qty_i),
                order_kind="ENTRY",
                order_type="MARKET",
                client_order_id=entry_client_id,
                status="NEW",
                meta={"tag": tag, "body": body, "idempotency": {"client_order_id": entry_client_id}},
            )

            try:
                res = client.place_order_v3(body)
                order_id = str((res.get("data") or {}).get("order_id") or "") or None
                update_order(state_id, status="SUBMITTED", broker_order_id=order_id, meta_patch={"upstox": {"place": res}})
            except UpstoxError as e:
                update_order(state_id, status="ERROR", last_error=str(e)[:500], meta_patch={"error": str(e)})
                raise
            finally:
                client.close()

            trade_id = record_trade_open(
                instrument_key=symbol,
                side="BUY",
                qty=float(qty_i),
                entry=float(entry_price),
                stop=float(stop),
                target=float(target),
                entry_order_id=None,
                sl_order_id=None,
                monitor_app_stop=False,
                meta=meta,
            )

            publish_sync("hft", "hft.entry", {"trade_id": int(trade_id), "instrument_key": symbol, "qty": qty_i, "entry": float(entry_price)})
            log_event("hft.index_options.entry", {"trade_id": int(trade_id), "instrument_key": symbol, "qty": qty_i, "entry": float(entry_price)})
            return {"ok": True, "trade_id": int(trade_id), "broker": "upstox", "order_state_id": state_id}

        # Paper
        paper_res = self._paper.execute(symbol=symbol, side="BUY", qty=float(qty_i), price=float(entry_price))
        state_id = create_order(
            broker="paper",
            instrument_key=symbol,
            side="BUY",
            qty=float(qty_i),
            order_kind="ENTRY",
            order_type="PAPER",
            price=float(entry_price),
            status="FILLED",
            meta={"tag": tag, "result": paper_res, "meta": meta},
        )
        trade_id = record_trade_open(
            instrument_key=symbol,
            side="BUY",
            qty=float(qty_i),
            entry=float(entry_price),
            stop=float(stop),
            target=float(target),
            entry_order_id=str(state_id),
            sl_order_id=None,
            monitor_app_stop=False,
            meta=meta,
        )
        publish_sync("hft", "hft.entry", {"trade_id": int(trade_id), "instrument_key": symbol, "qty": qty_i, "entry": float(entry_price)})
        return {"ok": True, "trade_id": int(trade_id), "broker": "paper", "order_state_id": state_id}

    async def _manage_open_trades(self) -> dict[str, Any]:
        closed = 0
        checked = 0
        max_age_min = int(getattr(settings, "INDEX_OPTIONS_HFT_MAX_HOLD_MINUTES", 30))
        now = _now_ts()

        for t in list_open_trades(limit=500):
            meta_raw = t.get("meta")
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            if str(meta.get("origin") or "") != "hft_index_options":
                continue

            checked += 1
            try:
                instrument_key = str(t.get("instrument_key") or "")
                if not instrument_key:
                    continue

                price = await self._option_last_price(instrument_key)
                if price is None:
                    continue

                stop = float(t.get("stop") or 0.0)
                target = float(t.get("target") or 0.0)

                age_min = int((now - int(t.get("ts_open") or now)) // 60)
                if max_age_min > 0 and age_min >= max_age_min:
                    await self._exit_trade_market(t, reason="time_exit")
                    closed += 1
                    continue

                if price <= stop and stop > 0:
                    await self._exit_trade_market(t, reason="stop_hit")
                    closed += 1
                    continue

                if price >= target and target > 0:
                    await self._exit_trade_market(t, reason="target_hit")
                    closed += 1
                    continue

                # no action
            except Exception as e:
                publish_sync("hft", "hft.manage_error", {"error": str(e)[:200]})

        return {"checked": checked, "closed": closed}

    async def _exit_trade_market(self, trade: dict[str, Any], *, reason: str) -> None:
        instrument_key = str(trade.get("instrument_key") or "")
        if not instrument_key:
            return

        qty = float(trade.get("qty") or 0.0)
        if qty <= 0:
            return

        price = await self._option_last_price(instrument_key)
        if price is None or price <= 0:
            return

        broker: BrokerName = self.broker()
        if broker == "upstox":
            # For now, we only record close; live exit wiring can be expanded similarly to entry.
            # Guarded by SAFE_MODE.
            if settings.SAFE_MODE:
                return

        # Paper exit
        self._paper.execute(symbol=instrument_key, side="SELL", qty=float(qty), price=float(price))
        create_order(
            broker="paper",
            instrument_key=instrument_key,
            side="SELL",
            qty=float(qty),
            order_kind="EXIT",
            order_type="PAPER",
            price=float(price),
            status="FILLED",
            meta={"reason": reason, "origin": "hft_index_options"},
        )

        # Update trade + calibration.
        entry = float(trade.get("entry") or 0.0)
        pnl = (float(price) - entry) * float(qty)

        trade_id_raw = trade.get("id")
        if trade_id_raw is None:
            return
        trade_id = int(trade_id_raw)
        mark_trade_closed(trade_id, reason=str(reason), close_price=float(price), pnl=float(pnl))
        update_on_trade_close(path=str(getattr(settings, "INDEX_OPTIONS_HFT_STATE_PATH", "data/hft_index_options_state.json")), pnl=float(pnl))
        publish_sync("hft", "hft.exit", {"trade_id": trade_id, "instrument_key": instrument_key, "price": float(price), "pnl": float(pnl), "reason": reason})


HFT_INDEX_OPTIONS = IndexOptionsHftService()
