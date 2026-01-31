# AITrader Frontend

Professional UI for the FastAPI backend in this repo.

## Prereqs
- Node.js 18+ (recommended)

## Run
From the repo root:

1) Start backend (in one terminal)

- `uvicorn app.main:app --reload`

2) Start frontend (in another terminal)

- `cd frontend`
- `npm install`
- `npm run dev`

The dev server proxies `/api` and `/health` to `http://localhost:8000`.

## Live candles (WebSocket)
The backend WS endpoint is:
- `ws://localhost:8000/api/ws/candles`

In dev, the UI connects via same-origin proxy:
- `ws://localhost:5173/api/ws/candles`

## Configure backend URL
If your backend is not on 8000, set:
- `VITE_BACKEND_TARGET=http://localhost:8000`

Example (PowerShell):
- `$env:VITE_BACKEND_TARGET='http://localhost:8000'; npm run dev`
