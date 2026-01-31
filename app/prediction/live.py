from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from app.ai.engine import AIEngine
from app.candles.persistence_sql import get_candles
from app.prediction.calibration import get_calibration, update_ema
from app.prediction.persistence import fetch_due_pending, insert_prediction, resolve_prediction
from app.realtime.bus import publish_sync


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _interval_seconds(interval: str) -> int:
    interval = str(interval)
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    if interval.endswith("h"):
        return int(interval[:-1]) * 3600
    if interval.endswith("d"):
        return int(interval[:-1]) * 86400
    return 60


def _round_down(ts: int, step: int) -> int:
    if step <= 0:
        return int(ts)
    return int(ts - (ts % step))


def _already_predicted(instrument_key: str, interval: str, *, ts_pred: int, horizon_steps: int) -> bool:
    # Idempotency gate per anchor.
    from app.core.db import db_conn

    with db_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM prediction_events WHERE instrument_key=? AND interval=? AND horizon_steps=? AND ts_pred=? LIMIT 1",
            (str(instrument_key), str(interval), int(horizon_steps), int(ts_pred)),
        ).fetchone()
        return row is not None


def _pick_close_at_or_before(candles: list[Any], ts: int) -> float | None:
    # candles are sorted ASC.
    close: float | None = None
    for c in candles:
        if int(getattr(c, "ts", 0)) <= int(ts):
            close = float(getattr(c, "close", 0.0))
        else:
            break
    if close is None or close <= 0:
        return None
    return float(close)


@dataclass
class LivePredictorConfig:
    interval: str = "1m"
    horizon_steps: int = 60  # 60 * 1m = next hour
    lookback_minutes: int = 240
    poll_seconds: int = 10
    calibration_alpha: float = 0.05


class LiveNextHourPredictor:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_error: str | None = None
        self._last_cycle_ts: int | None = None

        self._ai = AIEngine()

    def status(self) -> dict[str, Any]:
        return {
            "running": bool(self._running and self._task and not self._task.done()),
            "last_cycle_ts": self._last_cycle_ts,
            "last_error": self._last_error,
        }

    async def start(self, *, instrument_keys: list[str], cfg: LivePredictorConfig | None = None) -> dict[str, Any]:
        if self._task and not self._task.done():
            return {"ok": True, "status": "already_running"}
        cfg = cfg or LivePredictorConfig()
        self._running = True
        self._last_error = None
        self._task = asyncio.create_task(self._run(instrument_keys=[str(k) for k in instrument_keys], cfg=cfg))
        publish_sync("prediction", "predictor.start", {"instruments": list(instrument_keys), "cfg": cfg.__dict__})
        return {"ok": True}

    async def stop(self) -> dict[str, Any]:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        publish_sync("prediction", "predictor.stop", {})
        return {"ok": True}

    async def _run(self, *, instrument_keys: list[str], cfg: LivePredictorConfig) -> None:
        try:
            while self._running:
                try:
                    await self.run_once(instrument_keys=instrument_keys, cfg=cfg)
                    await asyncio.sleep(max(2, int(cfg.poll_seconds)))
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    self._last_error = str(e)
                    publish_sync("prediction", "predictor.error", {"error": str(e)})
                    await asyncio.sleep(3)
        finally:
            self._running = False

    async def run_once(self, *, instrument_keys: list[str], cfg: LivePredictorConfig) -> dict[str, Any]:
        now = _now_ts()
        step = _interval_seconds(cfg.interval)
        # Sliding window anchor: the latest completed candle boundary.
        anchor = _round_down(now - step, step)

        # 1) Generate/update predictions for the current anchor.
        created = 0
        for k in instrument_keys:
            ts_pred = int(anchor)
            if _already_predicted(str(k), str(cfg.interval), ts_pred=ts_pred, horizon_steps=int(cfg.horizon_steps)):
                continue

            out = self._ai.predict(
                instrument_key=str(k),
                interval=str(cfg.interval),
                lookback_days=max(1, int(cfg.lookback_minutes // (60 * 24)) + 1),
                horizon_steps=int(cfg.horizon_steps),
                include_nifty=False,
                model_family="intraday",
            )

            if out.get("error"):
                continue

            pred = (out.get("prediction") or {})
            meta = (out.get("meta") or {})
            feats = (out.get("features") or {})

            model_name = str(meta.get("model") or "")
            if model_name.startswith("torch"):
                model_kind = "deep"
            elif model_name.startswith("ridge"):
                model_kind = "ridge"
            else:
                model_kind = "stub"

            # Convert predicted next_close to return.
            ohlc = (pred.get("next_hour_ohlc") or {})
            last_close = float(feats.get("last_close") or 0.0)
            next_close = float(ohlc.get("close") or 0.0)
            if last_close <= 0 or next_close <= 0:
                continue
            pred_ret = float((next_close - last_close) / last_close)

            cal_key = f"{k}::{cfg.interval}::h{int(cfg.horizon_steps)}"
            cal = get_calibration(cal_key) or {"bias": 0.0, "mae": 0.0, "n": 0}
            bias = float(cal.get("bias") or 0.0)
            pred_adj = float(pred_ret - bias)

            ts_target = int(anchor + step * int(cfg.horizon_steps))
            pred_id = insert_prediction(
                ts_pred=ts_pred,
                instrument_key=str(k),
                interval=str(cfg.interval),
                horizon_steps=int(cfg.horizon_steps),
                model_kind=str(model_kind),
                model_key=None,
                pred_ret=float(pred_ret),
                pred_ret_adj=float(pred_adj),
                calib_bias=float(bias),
                ts_target=ts_target,
                meta={
                    "signal": pred.get("signal"),
                    "confidence": pred.get("confidence"),
                    "uncertainty": pred.get("uncertainty"),
                    "engine_model": model_name,
                    "anchor": int(anchor),
                },
            )

            created += 1
            publish_sync(
                "prediction",
                "prediction.created",
                {
                    "id": int(pred_id),
                    "instrument_key": str(k),
                    "interval": str(cfg.interval),
                    "horizon_steps": int(cfg.horizon_steps),
                    "ts_pred": int(ts_pred),
                    "ts_target": int(ts_target),
                    "pred_ret": float(pred_ret),
                    "pred_ret_adj": float(pred_adj),
                    "calib_bias": float(bias),
                    "model_kind": str(model_kind),
                },
            )

        # 2) Resolve any predictions whose target time has arrived.
        due = fetch_due_pending(now_ts=int(now), limit=500)
        resolved = 0
        for r in due:
            try:
                inst = str(r["instrument_key"])
                interval = str(r["interval"])
                horizon = int(r["horizon_steps"])
                ts_pred2 = int(r["ts_pred"])
                ts_target = int(r["ts_target"])

                # Actual return: close(at/before ts_target) vs close(at/before ts_pred)
                w = step * 5
                candles_pred = get_candles(inst, interval, ts_pred2 - w, ts_pred2 + w)
                candles_tgt = get_candles(inst, interval, ts_target - w, ts_target + w)
                c0 = _pick_close_at_or_before(candles_pred, ts_pred2)
                c1 = _pick_close_at_or_before(candles_tgt, ts_target)
                if c0 is None or c1 is None or c0 <= 0:
                    continue

                actual_ret = float((c1 - c0) / c0)
                resolve_prediction(pred_id=int(r["id"]), actual_ret=float(actual_ret))

                residual = float(float(r["pred_ret_adj"]) - float(actual_ret))
                cal_key = f"{inst}::{interval}::h{horizon}"
                cal = update_ema(
                    cal_key,
                    residual=float(residual),
                    abs_error=float(abs(residual)),
                    alpha=float(cfg.calibration_alpha),
                )

                resolved += 1
                publish_sync(
                    "prediction",
                    "prediction.resolved",
                    {
                        "id": int(r["id"]),
                        "instrument_key": inst,
                        "interval": interval,
                        "horizon_steps": horizon,
                        "ts_pred": int(ts_pred2),
                        "ts_target": int(ts_target),
                        "actual_ret": float(actual_ret),
                        "pred_ret_adj": float(r["pred_ret_adj"]),
                        "error": float(residual),
                        "calibration": cal,
                    },
                )
            except Exception as e:
                logger.debug("prediction resolve failed: {}", str(e))

        self._last_cycle_ts = int(now)
        publish_sync(
            "prediction",
            "predictor.cycle",
            {"created": int(created), "resolved": int(resolved), "anchor": int(anchor)},
        )
        return {"ok": True, "created": int(created), "resolved": int(resolved), "anchor": int(anchor)}


PREDICTOR = LiveNextHourPredictor()
