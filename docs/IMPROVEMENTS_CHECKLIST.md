# Improvements Checklist

Last updated: 2026-02-14

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
- [x] Added AI gate + risk scaling for HFT Index Options (configurable via `INDEX_OPTIONS_HFT_AI_*`).
- [x] Added quick lightweight training script for ridge models (`scripts/train_quick_light.py`) to validate end-to-end AI signals without torch.
- [x] Added indicator snapshot compute module (EMA/SMA/RSI/Bollinger/MACD/Stochastic).
- [x] Added AI Trader decision API (`GET /api/trader/decision`) returning candles summary + indicators + overlays + AI prediction + explainable reasons.
- [x] Wired AI Trader observability into frontend-v2 (Candles/Intraday/HFT show an “AI Trader” panel; auto-refresh enabled).
- [x] Standardized the “AI decision” contract across backend+frontend.
  - [x] Added Pydantic response model for `/api/trader/decision` (avoid ad-hoc dicts).
  - [x] Added a matching TS type in frontend-v2 (single source of truth pattern).
  - [x] Added minimal tests for `/api/trader/decision` response shape.

## In progress
- [ ] Add richer prediction analytics endpoints.
  - [x] Rollups by instrument/interval/horizon (hit-rate, avg error, confusion matrix for direction).
  - [x] Calibration trend endpoints (EMA tables, reliability curves, drift metrics).
  - [x] Basic UI/JSON examples for charting these in frontend.
- [ ] Add retention/cleanup for old prediction events to keep DB small.
  - [x] Policy knobs: max age (days) and max rows per table.
  - [x] Scheduled cleanup job + admin API to run cleanup on-demand.
- [ ] Implement Upstox reconciliation for missing `broker_order_id` via order book/tag matching.
- [ ] Add manual override APIs that operate on persisted order/trade state.
  - [ ] Cancel/close trade by `order_state_id`.
  - [ ] Pause/resume autotrader and/or freeze new entries.
  - [ ] Audit trail via `order_events` (who/when/why).
- [ ] Add UI-friendly event schemas (stable payloads) for websockets + examples.
  - [x] Define Pydantic models for WS messages (prediction/automation/training/hft).
  - [x] Add TypeScript types + parsing examples in frontend-v2.
  - [x] Add stable WS envelope (id/channel/type/payload/ts) and keep legacy keys for compatibility.
  - [x] Add frontend normalizers for candle + intraday overlays streams.

## Planned (next)
- [ ] Unify HFT entry pipeline with the same explainable decision object used by `/api/trader/decision`.
  - [ ] Ensure HFT stores the final decision + reasons in trade meta consistently.
  - [ ] Use one “decision builder” for overlays + AI gate + risk sizing.
- [ ] Expand training features to optionally include the richer indicator set.
  - [ ] Keep backward compatibility for existing ridge models (feature versioning).
  - [ ] Add a “feature set” label to trained models and expose it in learning status.
- [ ] Add documentation pages for the AI observability flow.
  - [ ] How `/api/trader/decision` is built (candles → indicators → overlays → AI → decision).
  - [ ] How to run `scripts/train_quick_light.py` and validate via API/UI.
