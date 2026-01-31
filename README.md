# AI Trading Backend (FastAPI)

Backend scaffold for an AI Trading System using **FastAPI**.

## Quickstart (Windows / PowerShell)

1) Create venv + install deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Create `.env`

```powershell
Copy-Item .env.example .env
```

3) Run API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Tests

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -q
```

Open:
- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Key Endpoints

- Market/safety: `GET /api/safety/status`
- Candles:
	- `POST /api/candles/historical/load`
	- `POST /api/intraday/poll`
- AI:
	- `GET /api/ai/predict` (supports `interval`, `lookback_days`, `horizon_steps`)
- Orders:
	- Paper: `POST /api/orders/place` with `broker="paper"`
	- Upstox (read): `GET /api/orders/upstox/book`, `GET /api/orders/upstox/details`, `GET /api/orders/upstox/positions`, `GET /api/orders/upstox/holdings`
	- Upstox (write, requires `SAFE_MODE=false` and market LIVE): `POST /api/orders/upstox/place-v3`, `PUT /api/orders/upstox/modify-v3`, `DELETE /api/orders/upstox/cancel-v3`
- Learning:
	- `POST /api/learning/train` (trains + persists model in SQLite)
	- `GET /api/learning/status`
	- `GET /api/learning/predict`
	- Deep (optional / GPU-capable):
		- `POST /api/learning/train-deep`
		- `GET /api/learning/status-deep`
		- `GET /api/learning/predict-deep`
- Audit: `GET /api/audit/recent`
 - Realtime:
	- WebSockets: `GET /api/ws/agent`, `GET /api/ws/training` (JSON event stream)
	- HTTP (recent): `GET /api/realtime/agent-events`, `GET /api/realtime/training-events`

## Notes
- Candles use Upstox V3 when `UPSTOX_ACCESS_TOKEN` is set; otherwise it falls back to deterministic synthetic candles.
- Live trading is blocked by default via `SAFE_MODE=true`.
- SQLite DB defaults to `./data/app.db`.

## Deep Learning (Optional)

- Install GPU PyTorch (Windows/NVIDIA): see [docs/DEEP_GPU_WINDOWS.md](docs/DEEP_GPU_WINDOWS.md)
- Extra deps file: `requirements-deep.txt`

## Training Jobs (Async)

- Classic ridge (threaded): `POST /api/learning/train-async`
- Deep (threaded): `POST /api/learning/train-deep-async`
- Job status: `GET /api/learning/jobs/{job_id}` and `GET /api/learning/jobs`

## Data Pipeline (Async)

- Backfill job: `POST /api/data/backfill-async`
- Job status: `GET /api/data/jobs/{job_id}` and `GET /api/data/jobs`
- Live events: `GET /api/ws/data`

## Auto Retrain / Promote (Scheduler)

- Status: `GET /api/automation/status`
- Start/stop loop: `POST /api/automation/start` / `POST /api/automation/stop`
- Trigger now: `POST /api/automation/run-once?force=true`

When enabled, the scheduler runs a daily pipeline: refresh data → train (ridge + optional deep) → evaluate (walk-forward for ridge) → drift check → promote to production registry.

## Agent Visibility

- Trades: `GET /api/agent/trades/open`, `GET /api/agent/trades/recent`, `GET /api/agent/trades/{trade_id}`
- Preview decision stack (no order): `GET /api/agent/preview?instrument_key=...`
- Today P&L: `GET /api/agent/performance/today`

## Optional Security

- API key auth (disabled by default): set `REQUIRE_API_KEY=true` and `API_KEY=...`, then send header `X-API-Key` on protected routes (`/api/orders/*`, `/api/learning/*`).
- Rate limiting (disabled by default): set `RATE_LIMIT_ENABLED=true` and tune `RATE_LIMIT_PER_MINUTE`.
