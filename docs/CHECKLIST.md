# Backend Completion Checklist (AI Trader)

This checklist is based on the original multi-phase backend request plus the later expansion into a “true autonomous intraday equity trading platform”.

## Status key
- [x] Done
- [ ] Not done / in progress
- [~] Partially done

---

## 1) Platform foundation
- [x] FastAPI app scaffold + lifespan init
- [x] Config via env (`Settings`) + safe defaults
- [x] SQLite persistence + schema init
- [x] Basic tests (pytest smoke suite)

## 2) Market safety & compliance
- [x] Market-hours guard (NSE calendar + sessions)
- [x] `SAFE_MODE=true` default; blocks live trading
- [x] Audit event logging + recent retrieval
- [x] Kill-switch endpoints: flatten (square-off) + cancel-open-orders

## 3) Market data
- [x] Candle storage in DB
- [x] Candle ingestion endpoints
- [x] Upstox V3 candles integration (when token configured)
- [x] “Auto” candles endpoint (live intraday vs EOD based on market state)
- [~] Bulk backfill orchestration (4y `1d` + ~3m `1m`) across many symbols
- [ ] Data quality checks (missing sessions, outliers, adjustments)

## 4) Learning / models
- [x] Train/load model pipeline with persistence
- [x] Model keys extended to support model families + cap tiers
- [x] Preset training endpoint (long + intraday presets)
- [~] Scheduled retraining (cron/job runner)
- [ ] Walk-forward validation / leakage checks
- [ ] Advanced models (deep forecasting / causal inference)

## 5) Recommendations
- [x] Top-N recommendations endpoint
- [x] Composite confidence + breakdown
- [~] Portfolio-aware allocation + constraints (recommended next)

## 6) Portfolio & broker integration
- [x] Paper portfolio/account/positions + paper order placement
- [x] Paper shorting support (long/short position accounting)
- [x] Upstox positions/holdings endpoints
- [x] Upstox funds/balance endpoint
- [~] Broker reconciliation loops (orders/positions) for live trading

## 7) News & sentiment
- [x] RSS ingestion + storage
- [x] Simple sentiment/impact heuristics
- [ ] Entity linking (ticker-level attribution)
- [ ] Market-impact modelling (beyond heuristics)

## 8) Alerts & realtime
- [x] Low-balance alerting with cooldown persistence
- [x] WebSocket alerts feed
- [~] Trade lifecycle alerts (placed/filled/rejected/stop/target)

## 9) Universe / cap tiers
- [x] Universe metadata store (cap_tier + upstox_token)
- [x] CRUD/list/import endpoints
- [~] Automated universe population (from broker instruments list)

## 10) Autonomous trading agent (intraday equity)
- [x] Agent API endpoints: status/start/stop/run-once
- [x] Dual timeframe direction check (long + intraday)
- [x] Uses StrategyEngine for entry/SL/target
- [x] Live entry placement via Upstox (guarded by SAFE_MODE)
- [x] Broker stop-loss attempt + app-side stop fallback
- [x] **Risk limits** (max trades/day, max open trades, max daily loss)
- [x] **End-of-day square-off** (close open trades before close)
- [~] Broker reconciliation (detect rejected/cancelled; ensure SL after fill)
- [x] Target exits + trailing stops (app-side management; long + short)
- [ ] Partial fills handling + retries/idempotency
- [~] Restart recovery (reconcile open trades from broker order status; broker position sync still pending)

## 11) Security & production hardening
- [~] Optional API key auth and rate limiting present
- [ ] Enforce auth for all trading/agent endpoints in prod
- [ ] Secrets handling guidance (vault/rotation)
- [ ] Observability: metrics + correlation IDs + dashboards

---

## Recommended “next 3” implementation steps
1) Agent risk limits + EOD square-off + tests (safety ROI)
2) Broker reconciliation (order status + position sync) + tests (live robustness)
3) Target exits + trailing stops (strategy completeness)
