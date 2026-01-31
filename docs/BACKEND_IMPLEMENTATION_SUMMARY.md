# Backend Implementation Summary

This repo is a runnable FastAPI backend scaffold for an AI Trading System.

## Implemented (Runnable)

- Market clock + safety status endpoints
- Candle ingestion (deterministic stub) + SQLite persistence
- Intraday polling endpoint
- AI prediction endpoint (statistical stub)
- Recommendations endpoint (demo universe)
- Paper trading account + journal
- WebSocket demo stream for candles

## Notes

- Upstox V3 candles are used automatically when `UPSTOX_ACCESS_TOKEN` is configured (fallback to synthetic candles if not).
- Upstox order/portfolio endpoints are wired (V2/V3); live order actions are blocked by default via `SAFE_MODE=true` and market-hours guard.

Additional components:

- Post-market learning pipeline: trains a simple ridge regression model on stored candles, persists model + metrics in SQLite, and AI predictions prefer the trained model when available.
- Audit logging: order actions and learning events are recorded in SQLite and viewable via `/api/audit/recent`.
- Optional hardening: API-key auth and rate limiting middleware (disabled by default).
- SQLite DB defaults to `./data/app.db`.
- See [README.md](README.md) for running instructions.
