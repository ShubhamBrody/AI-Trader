from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from threading import Lock

from fastapi import APIRouter, Query
from fastapi import BackgroundTasks, HTTPException

from app.ai.candlestick_patterns import detect_patterns, to_candles
from app.ai.engine import AIEngine
from app.ai.pattern_memory import ensure_all_catalog_rows, get_weights_cached
from app.ai.recommendation_engine import RecommendationEngine
from app.ai.recommendations_cache_files import (
    backup_path,
    delete_backup,
    ensure_backup_from_primary_or_payload,
    primary_path,
    read_json,
    write_json_atomic,
)
from app.core.db import db_conn
from app.realtime.bus import publish_sync

router = APIRouter(prefix="/recommendations")
engine = RecommendationEngine()

_ai = AIEngine()


_jobs_lock = Lock()
_jobs: dict[str, dict] = {}
_current_job_id: str | None = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_running_job() -> dict | None:
    with _jobs_lock:
        # Prefer pinned current job id.
        if _current_job_id and _current_job_id in _jobs and _jobs[_current_job_id].get("status") == "running":
            return dict(_jobs[_current_job_id])
        # Otherwise, pick the most recently started running job.
        running = [j for j in _jobs.values() if j.get("status") == "running"]
        running.sort(key=lambda x: x.get("started_at_utc") or "", reverse=True)
        return dict(running[0]) if running else None


@router.get("/top")
def top(
    n: int = Query(10, ge=1, le=50),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
    max_risk: float = Query(0.7, ge=0.0, le=1.0),
    use_cache: bool = Query(True),
    force_refresh: bool = Query(False),
    universe_limit: int = Query(200, ge=10, le=5000),
    universe_since_days: int = Query(7, ge=1, le=365),
) -> dict:
    cache_meta: dict = {}
    raw_recs: list[dict]

    running = _get_running_job()
    if running is not None:
        # While refresh is running, serve the backup cache file (stable UX).
        backup = read_json(backup_path())
        backup_recs = (backup or {}).get("recommendations") if isinstance(backup, dict) else None
        if isinstance(backup_recs, list) and backup_recs:
            raw_recs = list(backup_recs)
            cache_meta = {
                **(backup.get("meta") or {}),
                "cache": "backup_file",
                "is_stale": True,
                "served_from_backup": True,
                "refresh_running": True,
                "refresh_job_id": running.get("job_id"),
                "refresh_progress": running.get("progress"),
            }
        else:
            # Fallback to DB cache if backup file is missing.
            raw_recs, cache_meta = engine.get_cached_top(
                n=n,
                min_confidence=min_confidence,
                max_risk=max_risk,
                universe_limit=universe_limit,
                universe_since_days=universe_since_days,
                allow_stale=True,
            )
            cache_meta = {
                **(cache_meta or {}),
                "served_from_backup": False,
                "refresh_running": True,
                "refresh_job_id": running.get("job_id"),
                "refresh_progress": running.get("progress"),
            }
    else:
        if force_refresh or not use_cache:
            raw_recs, cache_meta = engine.refresh_top(
                n=n,
                min_confidence=min_confidence,
                max_risk=max_risk,
                universe_limit=universe_limit,
                universe_since_days=universe_since_days,
            )
        else:
            raw_recs, cache_meta = engine.get_cached_top(
                n=n,
                min_confidence=min_confidence,
                max_risk=max_risk,
                universe_limit=universe_limit,
                universe_since_days=universe_since_days,
                allow_stale=True,
            )

    # If cache is stale and we're NOT refreshing, create a backup cache file so
    # the UI has a stable fallback while user decides to refresh.
    try:
        if running is None and isinstance(cache_meta, dict) and bool(cache_meta.get("is_stale")):
            ensure_backup_from_primary_or_payload(
                {
                    "created_at_utc": _now_utc_iso(),
                    "recommendations": list(raw_recs or []),
                    "meta": dict(cache_meta or {}),
                }
            )
    except Exception:
        pass

    def _symbol_for(instrument_key: str) -> str:
        token = instrument_key.split("|")[-1].strip() if "|" in instrument_key else instrument_key
        with db_conn() as conn:
            row = conn.execute(
                "SELECT tradingsymbol FROM instrument_meta WHERE instrument_key=? LIMIT 1",
                (instrument_key,),
            ).fetchone()
            if row and row["tradingsymbol"]:
                return str(row["tradingsymbol"]) 
        return token

    now = datetime.now(timezone.utc).isoformat()
    recommendations = []
    for r in raw_recs:
        ik = str(r.get("instrument_key") or "")
        expected_return = float(r.get("expected_return") or 0.0)
        recommendations.append(
            {
                "instrument_key": ik,
                "symbol": _symbol_for(ik) if ik else None,
                # UI expects a human-friendly score; expose expected return in %.
                "score": float(expected_return * 100.0),
                "confidence": r.get("confidence"),
                "predicted_action": r.get("signal"),
                "predicted_return_pct": (expected_return * 100.0) if ik else None,
                "predicted_close": r.get("predicted_close"),
                "risk_score": r.get("uncertainty"),
                "risk_reward_ratio": None,
                "trend_strength": None,
                "momentum_score": None,
                "reasons": r.get("reasons") or [],
                "timestamp": now,
            }
        )

    return {
        # BackendComplete contract
        "recommendations": recommendations,
        "count": len(recommendations),
        "meta": {
            "use_cache": use_cache,
            "force_refresh": force_refresh,
            "universe_limit": universe_limit,
            "universe_since_days": universe_since_days,
            **(cache_meta or {}),
            "refresh_suggested": bool(cache_meta.get("is_stale")) if isinstance(cache_meta, dict) else False,
        },
        # Back-compat keys used by older tests/clients
        "n": int(n),
        "results": list(raw_recs),
    }


@router.get("/cache")
def cache_status() -> dict:
    return engine.cache_status()


@router.post("/refresh")
def refresh(
    background: BackgroundTasks,
    consent: bool = Query(False, description="User consent required to compute fresh recommendations"),
    n: int = Query(10, ge=1, le=50),
    min_confidence: float = Query(0.6, ge=0.0, le=1.0),
    max_risk: float = Query(0.7, ge=0.0, le=1.0),
    universe_limit: int = Query(200, ge=10, le=5000),
    universe_since_days: int = Query(7, ge=1, le=365),
) -> dict:
    global _current_job_id

    """Consent-based refresh.

    The frontend should call this only after explicit user approval.
    A realtime notification is published to channel 'recommendations' once done.
    """

    if not consent:
        raise HTTPException(status_code=400, detail="consent=true is required to refresh recommendations")

    # If a refresh is already running, return that job id (idempotent UX).
    existing = _get_running_job()
    if existing is not None:
        return {"ok": True, "job_id": existing.get("job_id"), "status": "running"}

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _current_job_id = job_id
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "started_at_utc": _now_utc_iso(),
            "finished_at_utc": None,
            "error": None,
            "progress": {
                "pct": 0,
                "processed": 0,
                "total": 0,
                "message": "queued",
                "updated_at_utc": _now_utc_iso(),
            },
            "meta": {
                "n": int(n),
                "min_confidence": float(min_confidence),
                "max_risk": float(max_risk),
                "universe_limit": int(universe_limit),
                "universe_since_days": int(universe_since_days),
            },
        }

    # Create a backup cache file now (either from primary cache file or from DB cache).
    try:
        recs, meta = engine.get_cached_top(
            n=n,
            min_confidence=min_confidence,
            max_risk=max_risk,
            universe_limit=universe_limit,
            universe_since_days=universe_since_days,
            allow_stale=True,
        )
        ensure_backup_from_primary_or_payload(
            {
                "created_at_utc": _now_utc_iso(),
                "recommendations": list(recs or []),
                "meta": dict(meta or {}),
            }
        )
    except Exception:
        pass

    def _run() -> None:
        global _current_job_id
        try:
            last_publish = 0.0

            def _progress_cb(processed: int, total: int, phase: str, message: str) -> None:
                nonlocal last_publish
                total_i = max(1, int(total))
                processed_i = max(0, min(int(processed), total_i))
                pct = int(round((processed_i / total_i) * 100.0))
                payload = {
                    "pct": pct,
                    "processed": processed_i,
                    "total": total_i,
                    "phase": str(phase or ""),
                    "message": str(message or ""),
                    "updated_at_utc": _now_utc_iso(),
                }
                with _jobs_lock:
                    if job_id in _jobs and _jobs[job_id].get("status") == "running":
                        _jobs[job_id]["progress"] = payload

                now = time.time()
                if now - last_publish >= 1.0:
                    last_publish = now
                    publish_sync("recommendations", "progress", {"job_id": job_id, **payload})

            _recs, meta = engine.refresh_top(
                n=n,
                min_confidence=min_confidence,
                max_risk=max_risk,
                universe_limit=universe_limit,
                universe_since_days=universe_since_days,
                progress_cb=_progress_cb,
            )

            # Persist a primary cache file (latest computed) and delete backup.
            try:
                write_json_atomic(
                    primary_path(),
                    {
                        "created_at_utc": _now_utc_iso(),
                        "recommendations": list(_recs or []),
                        "meta": dict(meta or {}),
                    },
                )
            except Exception:
                pass
            delete_backup()

            with _jobs_lock:
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["finished_at_utc"] = _now_utc_iso()
                _jobs[job_id]["result_meta"] = meta
                if _current_job_id == job_id:
                    _current_job_id = None
            publish_sync(
                "recommendations",
                "refreshed",
                {
                    "job_id": job_id,
                    "meta": meta,
                    "count": len(_recs),
                },
            )
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["finished_at_utc"] = _now_utc_iso()
                _jobs[job_id]["error"] = str(e)
                if _current_job_id == job_id:
                    _current_job_id = None
            publish_sync(
                "recommendations",
                "refresh_failed",
                {"job_id": job_id, "error": str(e)},
            )

    background.add_task(_run)
    return {
        "ok": True,
        "job_id": job_id,
        "status": "started",
        "backup_cache": {
            "path": str(backup_path()),
            "exists": bool(backup_path().exists()),
        },
    }


@router.get("/refresh/status")
def refresh_status(job_id: str | None = Query(None)) -> dict:
    if job_id is None:
        job = _get_running_job()
        if not job:
            return {"ok": False, "detail": "no active refresh"}
        return {"ok": True, **job}

    with _jobs_lock:
        job = _jobs.get(str(job_id))
    if not job:
        return {"ok": False, "detail": "unknown job_id"}
    return {"ok": True, **job}


@router.get("/refresh/current")
def refresh_current() -> dict:
    job = _get_running_job()
    if not job:
        return {"ok": False, "refreshing": False}
    return {"ok": True, "refreshing": True, **job}


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v != v:  # NaN
        return 0.0
    return float(max(0.0, min(1.0, v)))


def _pattern_stats(names: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    uniq = [str(n).strip() for n in (names or []) if str(n).strip()]
    if not uniq:
        return out
    # Ensure the catalog is present in DB; safe to call repeatedly.
    try:
        ensure_all_catalog_rows()
    except Exception:
        pass

    # SQLite parameter limit is typically 999; keep this small.
    uniq = uniq[:200]
    qs = ",".join(["?"] * len(uniq))
    with db_conn() as conn:
        rows = conn.execute(
            f"SELECT pattern, weight, seen, wins, losses, ema_winrate FROM candle_pattern_stats WHERE pattern IN ({qs})",
            tuple(uniq),
        ).fetchall()
    for r in rows:
        name = str(r["pattern"])
        out[name] = {
            "weight": float(r["weight"] or 1.0),
            "seen": int(r["seen"] or 0),
            "wins": int(r["wins"] or 0),
            "losses": int(r["losses"] or 0),
            "ema_winrate": float(r["ema_winrate"] or 0.5),
        }
    return out


def _statistical_recommendation_for(
    *,
    instrument_key: str,
    interval: str,
    lookback_days: int,
    horizon_steps: int,
    max_patterns: int,
) -> dict:
    ik = str(instrument_key or "").strip()
    if not ik:
        return {"ok": False, "detail": "instrument_key is required"}

    interval_s = str(interval or "1m")
    lookback_days_i = max(1, min(int(lookback_days), 365))
    horizon_i = max(1, min(int(horizon_steps), 500))
    # Kept for backward-compatible API params; patterns are no longer fused here.
    max_patterns_i = max(0, min(int(max_patterns), 8))

    # 1) Model prediction (direction + uncertainty)
    pred_out = _ai.predict(
        instrument_key=ik,
        interval=interval_s,
        lookback_days=lookback_days_i,
        horizon_steps=horizon_i,
        include_nifty=False,
        model_family=("intraday" if interval_s.endswith("m") else "long"),
    )

    pred = (pred_out.get("prediction") or {}) if isinstance(pred_out, dict) else {}
    feats = (pred_out.get("features") or {}) if isinstance(pred_out, dict) else {}
    meta = (pred_out.get("meta") or {}) if isinstance(pred_out, dict) else {}
    model_name = str(meta.get("model") or "")

    model_signal = str(pred.get("signal") or "HOLD").upper()
    model_conf = _clamp01(float(pred.get("confidence") or 0.0))
    model_unc = _clamp01(float(pred.get("uncertainty") or 1.0))

    last_close = float(feats.get("last_close") or 0.0)
    next_close = float(((pred.get("next_hour_ohlc") or {}) if isinstance(pred.get("next_hour_ohlc"), dict) else {}).get("close") or 0.0)
    expected_return = 0.0
    if last_close > 0 and next_close > 0:
        expected_return = float((next_close - last_close) / last_close)

    # AI-only recommendation: use the model's own direction and confidence.
    signal = model_signal if model_signal in {"BUY", "SELL", "HOLD"} else "HOLD"
    confidence = float(model_conf)

    # Signed score aligns with direction but remains comparable across instruments.
    signed_return = float(expected_return)
    if signal == "SELL":
        signed_return = -abs(float(expected_return))
    if signal == "HOLD":
        signed_return = 0.0

    reasons: list[str] = []
    mf = str(meta.get("model_family") or "")
    reasons.append(f"ai:{model_name or 'model'} family={mf or '-'} {signal} conf={model_conf:.2f} unc={model_unc:.2f}")
    if max_patterns_i:
        reasons.append("patterns:disabled")

    return {
        "ok": True,
        "instrument_key": ik,
        "interval": interval_s,
        "horizon_steps": int(horizon_i),
        "signal": signal,
        "confidence": float(confidence),
        "uncertainty": float(model_unc),
        "p_buy": None,
        "expected_return": float(signed_return),
        "expected_return_raw": float(expected_return),
        "reasons": reasons,
    }


@router.get("/statistical")
def statistical(
    instrument_key: str,
    interval: str = Query("5m"),
    lookback_days: int = Query(30, ge=1, le=365),
    horizon_steps: int = Query(12, ge=1, le=500),
    max_patterns: int = Query(3, ge=0, le=8),
) -> dict:
    """Generate an AI-only BUY/SELL/HOLD recommendation (no order placement).

    Uses AIEngine.predict() direction/confidence/uncertainty.
    (Historical pattern fusion is disabled to keep recommendations AI-driven.)
    """

    return _statistical_recommendation_for(
        instrument_key=instrument_key,
        interval=interval,
        lookback_days=int(lookback_days),
        horizon_steps=int(horizon_steps),
        max_patterns=int(max_patterns),
    )


@router.get("/statistical/top")
def statistical_top(
    n: int = Query(10, ge=1, le=50),
    min_confidence: float = Query(0.6, ge=0.0, le=1.0),
    max_risk: float = Query(0.7, ge=0.0, le=1.0),
    universe_limit: int = Query(50, ge=10, le=500),
    interval: str = Query("5m"),
    lookback_days: int = Query(30, ge=1, le=365),
    horizon_steps: int = Query(12, ge=1, le=500),
    max_patterns: int = Query(3, ge=0, le=8),
) -> dict:
    """Scan a small universe and return top statistical recommendations.

    Notes:
    - Conservative default universe_limit=50 to keep latency reasonable.
    - Does not place orders.
    """

    keys = []
    try:
        # Reuse the same stable universe selection as the main recommendation engine.
        keys = list(engine._universe_keys(limit=int(universe_limit)))  # type: ignore[attr-defined]
    except Exception:
        keys = []

    out: list[dict] = []
    analyzed = 0
    failed = 0

    for k in keys:
        analyzed += 1
        try:
            rec = _statistical_recommendation_for(
                instrument_key=str(k),
                interval=str(interval),
                lookback_days=int(lookback_days),
                horizon_steps=int(horizon_steps),
                max_patterns=int(max_patterns),
            )
            if not rec.get("ok"):
                failed += 1
                continue
            conf = float(rec.get("confidence") or 0.0)
            risk = float(rec.get("uncertainty") or 1.0)
            if conf < float(min_confidence):
                continue
            if risk > float(max_risk):
                continue
            out.append(rec)
        except Exception:
            failed += 1

    # Sort by absolute expected return weighted by confidence.
    out.sort(key=lambda r: abs(float(r.get("expected_return") or 0.0)) * float(r.get("confidence") or 0.0), reverse=True)
    out = out[: int(n)]

    # Map into the same frontend-friendly shape as /top.
    now = datetime.now(timezone.utc).isoformat()
    recs_ui: list[dict] = []
    for r in out:
        ik = str(r.get("instrument_key") or "")
        expected_return = float(r.get("expected_return") or 0.0)
        symbol = None
        try:
            with db_conn() as conn:
                row = conn.execute(
                    "SELECT tradingsymbol FROM instrument_meta WHERE instrument_key=? LIMIT 1",
                    (ik,),
                ).fetchone()
            if row and row["tradingsymbol"]:
                symbol = str(row["tradingsymbol"])
        except Exception:
            symbol = None

        recs_ui.append(
            {
                "instrument_key": ik,
                "symbol": symbol,
                "score": float(expected_return * 100.0),
                "confidence": float(r.get("confidence") or 0.0),
                "predicted_action": str(r.get("signal") or "HOLD"),
                "predicted_return_pct": float(expected_return * 100.0),
                "risk_score": float(r.get("uncertainty") or 1.0),
                "reasons": list(r.get("reasons") or []),
                "timestamp": now,
            }
        )

    return {
        "recommendations": recs_ui,
        "count": len(recs_ui),
        "meta": {
            "mode": "statistical",
            "interval": str(interval),
            "lookback_days": int(lookback_days),
            "horizon_steps": int(horizon_steps),
            "universe_limit": int(universe_limit),
            "analyzed": int(analyzed),
            "failed": int(failed),
        },
    }
