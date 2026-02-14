# WebSocket Contracts

This project uses a stable envelope for WebSocket messages.

## Stable envelope

All EventBus-backed streams (and some legacy streams) send:

```json
{
  "id": 123,
  "channel": "prediction",
  "type": "prediction.created",
  "payload": {},
  "ts": 1739570000
}
```

- `id`: monotonic event id (EventBus streams only)
- `channel`: stream name
- `type`: event type within that channel
- `payload`: event-specific object
- `ts`: UTC epoch seconds

## Channels

### prediction (`/api/ws/prediction`)
- `predictor.start` payload: `{ instruments: string[], cfg: object }`
- `predictor.stop` payload: `{}`
- `predictor.error` payload: `{ error: string }`
- `prediction.created` payload: `{ id, instrument_key, interval, horizon_steps, ts_pred, ts_target, pred_ret, pred_ret_adj, calib_bias, model_kind }`
- `prediction.resolved` payload: `{ id, instrument_key, interval, horizon_steps, ts_pred, ts_target, actual_ret, pred_ret_adj, error, calibration }`
- `predictor.cycle` payload: `{ created: number, resolved: number, anchor: number }`

### automation (`/api/ws/automation`)
- `automation_run.started` payload: `{ job_id: string, kind: string, params: object }`
- `automation_run.updated` payload: `{ job_id: string, progress?, message?, status?, stats?, error? }`

### training (`/api/ws/training`)
- `training.started` payload: `{ job_id: string, kind: string, instrument_key: string, interval: string }`
- `training.progress` payload: `{ job_id: string, progress: number, message: string, metrics: object }`
- `training.succeeded` payload: `{ job_id: string, result: object }`
- `training.failed` payload: `{ job_id: string, result?: object, error?: string }`

### hft (`/api/ws/hft`)
- `hft.start` payload: `{ broker: string, safe_mode: boolean, live_trading_enabled: boolean }`
- `hft.stop` payload: `{}`
- `hft.broker` payload: `{ broker: string }`
- `hft.error` payload: `{ error: string }`
- `hft.flatten` payload: `{ closed: number, errors: string[] }`
- `hft.entry` payload: `{ trade_id: number, instrument_key: string, qty: number, entry: number }`
- `hft.exit` payload: `{ trade_id: number, instrument_key: string, price: number, pnl: number, reason: string }`
- `hft.trade_risk` payload: `{ trade_id: number, stop: number, target: number }`
- `hft.manage_error` payload: `{ error: string }`

## Source of truth

Backend Pydantic models live in `app/api/ws_channel_models.py`.
Frontend TypeScript types + parsers live in `frontend-v2/src/lib/ws.ts`.
