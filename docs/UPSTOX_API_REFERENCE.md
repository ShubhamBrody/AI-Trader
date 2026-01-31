# Upstox API Reference

This backend supports:

- Upstox **V3 candles** for market data when `UPSTOX_ACCESS_TOKEN` is configured.
- Upstox **V2/V3 orders + portfolio** endpoints, with live order actions blocked by default via `SAFE_MODE=true`.

## Environment Variables

- `UPSTOX_CLIENT_ID`
- `UPSTOX_CLIENT_SECRET`
- `UPSTOX_ACCESS_TOKEN`

Optional:

- `UPSTOX_STRICT` (when true, candle fetch errors do not fall back to synthetic candles)
- `UPSTOX_BASE_URL` (default `https://api.upstox.com`)
- `UPSTOX_HFT_BASE_URL` (default `https://api-hft.upstox.com`)

## Implemented Backend Endpoints

Diagnostics:

- `GET /api/upstox/status`

Orders / Portfolio (proxy-style):

- Read-only:
	- `GET /api/orders/upstox/book` → Upstox V2 `GET /v2/order/retrieve-all`
	- `GET /api/orders/upstox/details?order_id=...` → Upstox V2 `GET /v2/order/details`
	- `GET /api/orders/upstox/positions` → Upstox V2 `GET /v2/portfolio/short-term-positions`
	- `GET /api/orders/upstox/holdings` → Upstox V2 `GET /v2/portfolio/long-term-holdings`
	- `GET /api/orders/upstox/funds?segment=EQ` → Upstox V2 `GET /v2/user/get-funds-and-margin`
	- `POST /api/orders/upstox/margin` → Upstox V2 `POST /v2/charges/margin`

Portfolio (backend abstraction):

	- `GET /api/portfolio/balance?source=upstox` → uses Upstox V2 `GET /v2/user/get-funds-and-margin` to compute `balance`.

- Live order actions (require `SAFE_MODE=false` and NSE LIVE session):
	- `POST /api/orders/upstox/place-v3` → Upstox V3 `POST https://api-hft.upstox.com/v3/order/place`
	- `PUT /api/orders/upstox/modify-v3` → Upstox V3 `PUT https://api-hft.upstox.com/v3/order/modify`
	- `DELETE /api/orders/upstox/cancel-v3?order_id=...` → Upstox V3 `DELETE https://api-hft.upstox.com/v3/order/cancel`

## Safety

- Keep `SAFE_MODE=true` until you’re ready to place live orders.
- Live order routes are also guarded by the NSE session lock (LIVE only).
