from __future__ import annotations

import re

from app.core.settings import settings
from app.integrations.upstox.client import UpstoxClient, UpstoxConfig, UpstoxError
from app.paper_trading.service import PaperTradingService
from app.auth import token_store


class PortfolioService:
    """Portfolio API backed by paper trading for now.

    Replace this with a live broker adapter when enabling Upstox.
    """

    def __init__(self) -> None:
        self._paper = PaperTradingService()

    def _source(self, source: str | None) -> str:
        s = (source or "auto").lower()
        if s not in {"auto", "paper", "upstox"}:
            return "auto"
        if s == "auto":
            return "upstox" if token_store.is_logged_in() or bool(settings.UPSTOX_ACCESS_TOKEN) else "paper"
        return s

    @staticmethod
    def _to_float(v: object) -> float | None:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                s = v.strip()
                s = s.replace(",", "")
                s = re.sub(r"[^0-9eE+\-\.]", "", s)
                return float(s)
            except Exception:
                return None
        return None

    def _scan_for_number(self, obj: object, *, key_hints: tuple[str, ...]) -> float | None:
        """Best-effort recursive scan for a numeric-ish field.

        Prefers keys containing any of key_hints.
        """

        def walk(x: object, depth: int = 0) -> list[tuple[str, float]]:
            if depth > 4:
                return []
            out: list[tuple[str, float]] = []
            if isinstance(x, dict):
                for k, v in x.items():
                    key = str(k)
                    val = self._to_float(v)
                    if val is not None:
                        out.append((key, val))
                    out.extend(walk(v, depth + 1))
            elif isinstance(x, list):
                for it in x[:50]:
                    out.extend(walk(it, depth + 1))
            return out

        found = walk(obj)
        if not found:
            return None

        def score(item: tuple[str, float]) -> tuple[int, float]:
            k, v = item
            kl = k.lower()
            hit = 0
            for h in key_hints:
                if h in kl:
                    hit += 1
            return (hit, abs(float(v)))

        found.sort(key=score, reverse=True)
        return float(found[0][1])

    def _extract_upstox_balance(self, payload: dict, segment: str) -> float | None:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return None

        # Some Upstox responses are grouped by segment (e.g., equity/commodity).
        seg_block: dict | None = None
        if segment.upper() == "EQ":
            seg_block = data.get("equity") if isinstance(data.get("equity"), dict) else None
        elif segment.upper() == "FO":
            seg_block = data.get("derivatives") if isinstance(data.get("derivatives"), dict) else None
        elif segment.upper() == "CM":
            seg_block = data.get("commodity") if isinstance(data.get("commodity"), dict) else None

        block = seg_block or data

        # Try common fields first.
        for key in (
            "available_margin",
            "available_funds",
            "available_cash",
            "cash_available",
            "net",
            "opening_balance",
            "cash",
        ):
            if isinstance(block, dict) and key in block:
                val = self._to_float(block.get(key))
                if val is not None:
                    return val

        # Fallback: recursive scan.
        return self._scan_for_number(block, key_hints=("available", "avail", "margin", "cash", "net"))

    def balance(self, source: str | None = None) -> dict:
        s = self._source(source)
        if s == "paper":
            bal = float(self._paper.account()["balance"])
            out = {"source": "paper", "balance": bal}
            try:
                from app.alerts.service import AlertService

                alert = AlertService().maybe_emit_low_balance(source="paper", balance=bal)
                if alert is not None:
                    out["alert"] = {"type": alert.alert_type, "ts": alert.ts, "payload": alert.payload}
            except Exception:
                pass
            return out

        cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
        try:
            client = UpstoxClient(cfg)
        except UpstoxError as e:
            return {"source": "upstox", "balance": None, "error": str(e)}

        # Upstox uses segment codes like SEC/COM/FO. Many accounts reject EQ.
        segment = "SEC"
        try:
            try:
                payload = client.funds_and_margin_v2(segment=segment)
            except UpstoxError:
                # Some deployments reject the segment parameter entirely.
                payload = client.funds_and_margin_v2(segment=None)
            bal = self._extract_upstox_balance(payload, segment=segment)
            out = {"source": "upstox", "segment": segment, "balance": bal, "funds": payload.get("data", payload)}
            try:
                from app.alerts.service import AlertService

                alert = AlertService().maybe_emit_low_balance(source="upstox", balance=bal, segment=segment)
                if alert is not None:
                    out["alert"] = {"type": alert.alert_type, "ts": alert.ts, "payload": alert.payload}
            except Exception:
                pass
            return out
        except UpstoxError as e:
            return {"source": "upstox", "segment": segment, "balance": None, "error": str(e)}
        finally:
            client.close()

    def holdings(self, source: str | None = None) -> dict:
        s = self._source(source)
        if s == "paper":
            # For demo, treat positions as holdings.
            out = self._paper.positions()
            out["source"] = "paper"
            return out

        cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
        try:
            client = UpstoxClient(cfg)
        except UpstoxError as e:
            return {"source": "upstox", "error": str(e)}
        try:
            data = client.holdings_v2()
            return {"source": "upstox", **data}
        except UpstoxError as e:
            return {"source": "upstox", "error": str(e)}
        finally:
            client.close()

    def positions(self, source: str | None = None) -> dict:
        s = self._source(source)
        if s == "paper":
            out = self._paper.positions()
            out["source"] = "paper"
            return out

        cfg = UpstoxConfig(base_url=settings.UPSTOX_BASE_URL, hft_base_url=settings.UPSTOX_HFT_BASE_URL)
        try:
            client = UpstoxClient(cfg)
        except UpstoxError as e:
            return {"source": "upstox", "error": str(e)}
        try:
            data = client.positions_v2()
            return {"source": "upstox", **data}
        except UpstoxError as e:
            return {"source": "upstox", "error": str(e)}
        finally:
            client.close()

    def positions_state(self, *, broker: str = "upstox", limit: int = 500) -> dict:
        from app.portfolio.positions_state import list_positions_state

        return {"ok": True, "broker": broker, "items": list_positions_state(broker=broker, limit=limit)}

    def reconcile_positions(self, *, broker: str = "upstox") -> dict:
        if broker != "upstox":
            return {"ok": False, "detail": f"unsupported broker={broker}"}
        from app.portfolio.positions_state import reconcile_upstox_positions

        return reconcile_upstox_positions()

    def pnl(self, source: str | None = None) -> dict:
        s = self._source(source)
        if s == "paper":
            out = self._paper.pnl()
            out["source"] = "paper"
            return out
        return {"source": "upstox", "detail": "Use /api/orders/upstox/positions for broker P&L"}
