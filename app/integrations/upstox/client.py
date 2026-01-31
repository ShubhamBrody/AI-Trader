from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from urllib.parse import quote
import time

import httpx
from loguru import logger

from app.core.settings import settings
from app.auth import token_store


@dataclass(frozen=True)
class UpstoxConfig:
    base_url: str = "https://api.upstox.com"
    hft_base_url: str = "https://api-hft.upstox.com"
    access_token: str | None = None


class UpstoxError(RuntimeError):
    pass


class UpstoxClient:
    # Module-level circuit breaker state shared across client instances.
    _cb_failures: int = 0
    _cb_open_until: float = 0.0
    _cb_last_failure_ts: float = 0.0

    def __init__(self, cfg: UpstoxConfig | None = None) -> None:
        self.cfg = cfg or UpstoxConfig()

        access_token = (self.cfg.access_token or token_store.get_access_token() or settings.UPSTOX_ACCESS_TOKEN or "").strip()
        if not access_token:
            raise UpstoxError("Upstox is not authenticated. Open /api/auth/upstox/login")

        self._client = httpx.Client(
            timeout=httpx.Timeout(30.0),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )

    def close(self) -> None:
        self._client.close()

    @classmethod
    def _cb_is_open(cls) -> bool:
        return float(cls._cb_open_until) > time.time()

    @classmethod
    def _cb_on_success(cls) -> None:
        cls._cb_failures = 0
        cls._cb_last_failure_ts = 0.0
        cls._cb_open_until = 0.0

    @classmethod
    def _cb_on_failure(cls) -> None:
        now = time.time()
        # reset failure count if last failure was long ago
        if cls._cb_last_failure_ts and (now - float(cls._cb_last_failure_ts)) > 60:
            cls._cb_failures = 0
        cls._cb_last_failure_ts = now
        cls._cb_failures += 1
        # Open circuit after N consecutive failures for a short cooldown.
        if cls._cb_failures >= int(getattr(settings, "UPSTOX_CB_FAILURES", 5) or 5):
            cooldown = float(getattr(settings, "UPSTOX_CB_COOLDOWN_SECONDS", 20) or 20)
            cls._cb_open_until = now + max(5.0, cooldown)

    def _request_with_resilience(self, method: str, url: str, *, params: dict[str, Any] | None = None, body: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._cb_is_open():
            raise UpstoxError("Upstox circuit breaker open; refusing request")

        max_attempts = max(1, int(getattr(settings, "UPSTOX_RETRY_MAX_ATTEMPTS", 3) or 3))
        backoff_base = float(getattr(settings, "UPSTOX_RETRY_BACKOFF_SECONDS", 0.5) or 0.5)

        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                if method == "GET":
                    r = self._client.get(url, params=params)
                elif method == "DELETE":
                    r = self._client.delete(url, params=params)
                elif method == "POST":
                    r = self._client.post(url, json=body or {})
                elif method == "PUT":
                    r = self._client.put(url, json=body or {})
                else:
                    raise UpstoxError(f"Unsupported method: {method}")

                # Retry on transient server errors / rate limit.
                if r.status_code in (429, 500, 502, 503, 504):
                    raise UpstoxError(f"Upstox HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    # non-retryable client error
                    raise UpstoxError(f"Upstox HTTP {r.status_code}: {r.text[:500]}")

                data = r.json()
                if isinstance(data, dict) and data.get("status") not in (None, "success"):
                    raise UpstoxError(f"Upstox response status not success: {data}")

                self._cb_on_success()
                return data
            except (httpx.TimeoutException, httpx.NetworkError, UpstoxError) as e:
                last_err = e
                self._cb_on_failure()
                if attempt >= max_attempts:
                    break
                # exponential backoff
                sleep_s = backoff_base * (2 ** (attempt - 1))
                time.sleep(min(5.0, max(0.05, sleep_s)))
            except Exception as e:
                last_err = e
                self._cb_on_failure()
                break

        raise UpstoxError(str(last_err) if last_err is not None else "Upstox request failed")

    def _get(self, url: str) -> dict[str, Any]:
        return self._request_with_resilience("GET", url)

    def _delete(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request_with_resilience("DELETE", url, params=params)

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._request_with_resilience("POST", url, body=body)

    def _put(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._request_with_resilience("PUT", url, body=body)

    @staticmethod
    def _iso_to_epoch_seconds(ts: str) -> int:
        # Example: 2025-01-12T15:15:00+05:30
        dt = datetime.fromisoformat(ts)
        return int(dt.timestamp())

    def historical_candles_v3(
        self,
        instrument_key: str,
        unit: str,
        interval: int,
        to_date: date,
        from_date: date | None = None,
    ) -> list[list[Any]]:
        ik = quote(instrument_key, safe="")
        base = self.cfg.base_url.rstrip("/")
        path = f"/v3/historical-candle/{ik}/{unit}/{int(interval)}/{to_date.isoformat()}"
        if from_date is not None:
            path += f"/{from_date.isoformat()}"
        url = base + path
        payload = self._get(url)
        return payload.get("data", {}).get("candles", [])

    def intraday_candles_v3(self, instrument_key: str, unit: str, interval: int) -> list[list[Any]]:
        ik = quote(instrument_key, safe="")
        base = self.cfg.base_url.rstrip("/")
        url = base + f"/v3/historical-candle/intraday/{ik}/{unit}/{int(interval)}"
        payload = self._get(url)
        return payload.get("data", {}).get("candles", [])

    def place_order_v2(self, body: dict[str, Any]) -> dict[str, Any]:
        # Guarded outside by SAFE_MODE in broker, but keep extra hard stop here.
        if settings.SAFE_MODE:
            raise UpstoxError("SAFE_MODE=true: refusing to place live orders")

        base = self.cfg.hft_base_url.rstrip("/")
        url = base + "/v2/order/place"
        return self._post(url, body)

    def place_order_v3(self, body: dict[str, Any]) -> dict[str, Any]:
        if settings.SAFE_MODE:
            raise UpstoxError("SAFE_MODE=true: refusing to place live orders")
        base = self.cfg.hft_base_url.rstrip("/")
        url = base + "/v3/order/place"
        return self._post(url, body)

    def modify_order_v3(self, body: dict[str, Any]) -> dict[str, Any]:
        if settings.SAFE_MODE:
            raise UpstoxError("SAFE_MODE=true: refusing to modify live orders")
        base = self.cfg.hft_base_url.rstrip("/")
        url = base + "/v3/order/modify"
        return self._put(url, body)

    def cancel_order_v3(self, order_id: str) -> dict[str, Any]:
        if settings.SAFE_MODE:
            raise UpstoxError("SAFE_MODE=true: refusing to cancel live orders")
        base = self.cfg.hft_base_url.rstrip("/")
        url = base + "/v3/order/cancel"
        return self._delete(url, params={"order_id": order_id})

    def order_book_v2(self) -> dict[str, Any]:
        base = self.cfg.base_url.rstrip("/")
        url = base + "/v2/order/retrieve-all"
        return self._get(url)

    def order_details_v2(self, order_id: str) -> dict[str, Any]:
        base = self.cfg.base_url.rstrip("/")
        url = base + "/v2/order/details"
        return self._request_with_resilience("GET", url, params={"order_id": order_id})

    def positions_v2(self) -> dict[str, Any]:
        base = self.cfg.base_url.rstrip("/")
        url = base + "/v2/portfolio/short-term-positions"
        return self._get(url)

    def holdings_v2(self) -> dict[str, Any]:
        base = self.cfg.base_url.rstrip("/")
        url = base + "/v2/portfolio/long-term-holdings"
        return self._get(url)

    def margin_v2(self, body: dict[str, Any]) -> dict[str, Any]:
        base = self.cfg.base_url.rstrip("/")
        url = base + "/v2/charges/margin"
        return self._post(url, body)

    def funds_and_margin_v2(self, segment: str | None = "EQ") -> dict[str, Any]:
        """Get available funds & margin.

        Upstox endpoint: GET /v2/user/get-funds-and-margin
        """

        base = self.cfg.base_url.rstrip("/")
        url = base + "/v2/user/get-funds-and-margin"
        params: dict[str, Any] = {}
        if segment:
            params["segment"] = segment

        r = self._client.get(url, params=params or None)
        if r.status_code >= 400:
            raise UpstoxError(f"Upstox HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        if isinstance(data, dict) and data.get("status") not in (None, "success"):
            raise UpstoxError(f"Upstox response status not success: {data}")
        return data


def parse_upstox_candles(candles: list[list[Any]]) -> list[dict[str, Any]]:
    """Convert Upstox candle arrays into dicts.

    Upstox format: [timestamp, open, high, low, close, volume, open_interest]
    """

    out: list[dict[str, Any]] = []
    for row in candles:
        if not row or len(row) < 6:
            continue
        ts_s = str(row[0])
        try:
            ts = UpstoxClient._iso_to_epoch_seconds(ts_s)
        except Exception:
            logger.warning("Failed parsing candle timestamp from Upstox: {}", ts_s)
            continue

        out.append(
            {
                "ts": ts,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )

    # Upstox generally returns newest-first; we want ascending.
    out.sort(key=lambda x: x["ts"])
    return out
