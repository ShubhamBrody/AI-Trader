# Improvements Checklist

Last updated: 2026-01-27

## Completed
- [x] Added persistent prediction storage and online calibration tables (SQLite).
- [x] Implemented live next-hour predictor loop (creates predictions, resolves outcomes, updates EMA calibration).
- [x] Added prediction HTTP APIs and prediction WebSocket stream.
- [x] Added `automation_runs` persistence + list/get endpoints and automation websocket stream.
- [x] Scheduler now records auto cycles into `automation_runs` with progress + success/fail.
- [x] Implemented deep walk-forward evaluation path (torch-optional) and wired into scheduler.
- [x] Added predictor autostart options (settings + lifespan wiring).
- [x] Added durable per-order state persistence (`order_states`, `order_events`) with list/get/events endpoints.
- [x] Paper orders now write an `order_state_id` for crash-safe auditability.
- [x] Autotrader now records entry/SL order intents and updates them with broker order IDs.

## In progress
- [ ] Add richer prediction analytics endpoints (rollups by instrument, calibration trends, hit-rate).
- [ ] Add retention/cleanup for old prediction events to keep DB small.
- [ ] Implement Upstox reconciliation for missing broker_order_id via order book/tag matching.
- [ ] Add manual override APIs that operate on persisted order/trade state (cancel/close/pause).

## Planned (next)
- [ ] Add UI-friendly event schemas and frontend examples.
