from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Callable
from zoneinfo import ZoneInfo

from app.ai.engine import AIEngine
from app.core.db import db_conn
from app.core.settings import settings
from app.news.service import NewsService
from app.watchlist.service import WatchlistService


@dataclass(frozen=True)
class _CacheEntry:
    trading_day: str
    cache_key: str
    created_at_utc: datetime
    recommendations: list[dict]
    analyzed: int
    qualified: int


class RecommendationEngine:
    def __init__(self) -> None:
        self._ai = AIEngine()
        self._watchlist = WatchlistService()
        self._news = NewsService()

        self._cache_lock = Lock()
        self._cache: dict[str, _CacheEntry] = {}

    def _trading_day_key(self) -> str:
        # Prefer local market day (IST by default).
        try:
            tz = ZoneInfo(str(settings.TIMEZONE))
        except Exception:
            tz = timezone.utc
        return datetime.now(tz).date().isoformat()

    @staticmethod
    def _cache_row_to_entry(row) -> _CacheEntry | None:
        try:
            created = datetime.fromtimestamp(int(row["created_ts"]), tz=timezone.utc)
            payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
            recs = list(payload.get("recommendations") or [])
            return _CacheEntry(
                trading_day=str(row["trading_day"]),
                cache_key=str(row["cache_key"]),
                created_at_utc=created,
                recommendations=recs,
                analyzed=int(row["analyzed"]),
                qualified=int(row["qualified"]),
            )
        except Exception:
            return None

    def _load_cache_exact(self, cache_key: str) -> _CacheEntry | None:
        with db_conn() as conn:
            row = conn.execute(
                "SELECT cache_key,trading_day,created_ts,analyzed,qualified,params_json,payload_json "
                "FROM recommendations_cache WHERE cache_key=? LIMIT 1",
                (cache_key,),
            ).fetchone()
        return self._cache_row_to_entry(row) if row is not None else None

    def _load_cache_latest_for_params(self, params_key: str) -> _CacheEntry | None:
        with db_conn() as conn:
            row = conn.execute(
                "SELECT cache_key,trading_day,created_ts,analyzed,qualified,params_json,payload_json "
                "FROM recommendations_cache WHERE params_json=? ORDER BY created_ts DESC LIMIT 1",
                (params_key,),
            ).fetchone()
        return self._cache_row_to_entry(row) if row is not None else None

    def _save_cache_entry(self, entry: _CacheEntry, *, params_key: str) -> None:
        payload = {
            "recommendations": list(entry.recommendations or []),
        }
        now_ts = int(entry.created_at_utc.timestamp())
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO recommendations_cache (cache_key,trading_day,created_ts,analyzed,qualified,params_json,payload_json) "
                "VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(cache_key) DO UPDATE SET "
                "trading_day=excluded.trading_day,created_ts=excluded.created_ts,analyzed=excluded.analyzed,qualified=excluded.qualified,params_json=excluded.params_json,payload_json=excluded.payload_json",
                (
                    entry.cache_key,
                    entry.trading_day,
                    now_ts,
                    int(entry.analyzed),
                    int(entry.qualified),
                    str(params_key),
                    json.dumps(payload),
                ),
            )

    def cache_status(self) -> dict:
        entries: list[dict] = []

        # DB-backed cache (authoritative across restarts)
        with db_conn() as conn:
            rows = conn.execute(
                "SELECT cache_key,trading_day,created_ts,analyzed,qualified,params_json,payload_json "
                "FROM recommendations_cache ORDER BY created_ts DESC LIMIT 200"
            ).fetchall()
        for row in rows:
            e = self._cache_row_to_entry(row)
            if e is None:
                continue
            entries.append(
                {
                    "trading_day": e.trading_day,
                    "cache_key": e.cache_key,
                    "created_at_utc": e.created_at_utc.isoformat(),
                    "count": len(e.recommendations),
                    "analyzed": e.analyzed,
                    "qualified": e.qualified,
                }
            )

        # In-memory cache (best-effort; useful during a single process lifetime)
        with self._cache_lock:
            for e in self._cache.values():
                entries.append(
                    {
                        "trading_day": e.trading_day,
                        "cache_key": e.cache_key,
                        "created_at_utc": e.created_at_utc.isoformat(),
                        "count": len(e.recommendations),
                        "analyzed": e.analyzed,
                        "qualified": e.qualified,
                    }
                )
        entries.sort(key=lambda x: x["created_at_utc"], reverse=True)
        return {"entries": entries}

    def get_cached_top(
        self,
        *,
        n: int,
        min_confidence: float,
        max_risk: float,
        universe_limit: int,
        universe_since_days: int,
        allow_stale: bool = True,
    ) -> tuple[list[dict], dict]:
        """Return cached recs + a meta block.

        If today's cache is missing and allow_stale=True, returns the most recent cache for
        the same params and marks meta.is_stale=True.
        """

        n = max(1, min(int(n), 50))
        universe_limit = max(10, min(int(universe_limit), 5000))
        universe_since_days = max(1, min(int(universe_since_days), 365))

        trading_day = self._trading_day_key()
        params_key = json.dumps(
            {
                "n": int(n),
                "min_confidence": float(min_confidence),
                "max_risk": float(max_risk),
                "universe_limit": int(universe_limit),
                "universe_since_days": int(universe_since_days),
            },
            sort_keys=True,
        )

        cache_key = (
            f"{trading_day}|n={n}|minc={min_confidence:.3f}|maxr={max_risk:.3f}"
            f"|univ={universe_limit}|since={universe_since_days}d"
        )

        # 1) memory exact
        with self._cache_lock:
            entry = self._cache.get(cache_key)
            if entry is not None and entry.trading_day == trading_day:
                return list(entry.recommendations), {
                    "cache": "memory",
                    "cache_key": entry.cache_key,
                    "cache_trading_day": entry.trading_day,
                    "created_at_utc": entry.created_at_utc.isoformat(),
                    "is_stale": False,
                }

        # 2) DB exact for today
        exact = self._load_cache_exact(cache_key)
        if exact is not None and exact.trading_day == trading_day:
            with self._cache_lock:
                self._cache[cache_key] = exact
            return list(exact.recommendations), {
                "cache": "db",
                "cache_key": exact.cache_key,
                "cache_trading_day": exact.trading_day,
                "created_at_utc": exact.created_at_utc.isoformat(),
                "is_stale": False,
            }

        # 3) Latest cache for same params (stale)
        if allow_stale:
            latest = self._load_cache_latest_for_params(params_key)
            if latest is not None:
                return list(latest.recommendations), {
                    "cache": "db",
                    "cache_key": latest.cache_key,
                    "cache_trading_day": latest.trading_day,
                    "created_at_utc": latest.created_at_utc.isoformat(),
                    "is_stale": latest.trading_day != trading_day,
                }

        return [], {
            "cache": None,
            "cache_key": cache_key,
            "cache_trading_day": None,
            "created_at_utc": None,
            "is_stale": True,
        }

    def refresh_top(
        self,
        *,
        n: int,
        min_confidence: float,
        max_risk: float,
        universe_limit: int,
        universe_since_days: int,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
    ) -> tuple[list[dict], dict]:
        """Compute + persist today's cache and return recs + meta."""

        n = max(1, min(int(n), 50))
        universe_limit = max(10, min(int(universe_limit), 5000))
        universe_since_days = max(1, min(int(universe_since_days), 365))

        trading_day = self._trading_day_key()
        params_key = json.dumps(
            {
                "n": int(n),
                "min_confidence": float(min_confidence),
                "max_risk": float(max_risk),
                "universe_limit": int(universe_limit),
                "universe_since_days": int(universe_since_days),
            },
            sort_keys=True,
        )
        cache_key = (
            f"{trading_day}|n={n}|minc={min_confidence:.3f}|maxr={max_risk:.3f}"
            f"|univ={universe_limit}|since={universe_since_days}d"
        )

        # Try strict filters; if nothing qualifies, relax slightly so UI isn't empty.
        scored = self.top(
            n=universe_limit,
            min_confidence=min_confidence,
            max_risk=max_risk,
            progress_cb=progress_cb,
        )
        results = list((scored.get("results") or [])) if isinstance(scored, dict) else []

        relaxed = False
        if not results:
            relaxed = True
            scored = self.top(
                n=universe_limit,
                min_confidence=max(0.0, float(min_confidence) - 0.15),
                max_risk=min(1.0, float(max_risk) + 0.15),
                progress_cb=progress_cb,
            )
            results = list((scored.get("results") or [])) if isinstance(scored, dict) else []

        analyzed = int(scored.get("analyzed") or universe_limit) if isinstance(scored, dict) else universe_limit
        qualified = len(results)
        recs = results[:n]

        entry = _CacheEntry(
            trading_day=trading_day,
            cache_key=cache_key,
            created_at_utc=datetime.now(timezone.utc),
            recommendations=recs,
            analyzed=analyzed,
            qualified=qualified,
        )
        with self._cache_lock:
            self._cache[cache_key] = entry
        self._save_cache_entry(entry, params_key=params_key)

        return list(recs), {
            "cache": "computed",
            "cache_key": cache_key,
            "cache_trading_day": trading_day,
            "created_at_utc": entry.created_at_utc.isoformat(),
            "is_stale": False,
            "relaxed": relaxed,
        }

    def get_top_recommendations(
        self,
        *,
        n: int = 10,
        min_confidence: float = 0.5,
        max_risk: float = 0.7,
        use_cache: bool = True,
        force_refresh: bool = False,
        universe_limit: int = 200,
        universe_since_days: int = 7,
    ) -> list[dict]:
        if use_cache and not force_refresh:
            recs, _meta = self.get_cached_top(
                n=n,
                min_confidence=min_confidence,
                max_risk=max_risk,
                universe_limit=universe_limit,
                universe_since_days=universe_since_days,
                allow_stale=True,
            )
            if recs:
                return recs

        recs, _meta = self.refresh_top(
            n=n,
            min_confidence=min_confidence,
            max_risk=max_risk,
            universe_limit=universe_limit,
            universe_since_days=universe_since_days,
        )
        return recs

    def _universe_keys(self, *, limit: int) -> list[str]:
        """Return a stable universe for recommendations.

        Goal: do NOT restrict to current holdings/watchlist. Prefer instrument master
        (instrument_meta) + DEFAULT_UNIVERSE, then union in watchlist keys.
        """

        limit = max(10, min(int(limit), 5000))

        out: list[str] = []
        seen: set[str] = set()

        def add_many(items: list[str]) -> None:
            for k in items:
                kk = str(k or "").strip()
                if not kk or kk in seen:
                    continue
                seen.add(kk)
                out.append(kk)
                if len(out) >= limit:
                    return

        # 1) Settings default universe first (deterministic, good for cold start)
        defaults = [k.strip() for k in str(settings.DEFAULT_UNIVERSE or "").split(",") if k.strip()]
        add_many(defaults)

        # 2) Instrument master (prefer large/mid caps; fall back to any)
        try:
            with db_conn() as conn:
                rows = conn.execute(
                    "SELECT instrument_key, cap_tier FROM instrument_meta "
                    "ORDER BY CASE cap_tier WHEN 'large' THEN 0 WHEN 'mid' THEN 1 WHEN 'small' THEN 2 ELSE 3 END, updated_ts DESC "
                    "LIMIT ?",
                    (max(limit * 2, 200),),
                ).fetchall()
            keys = [str(r["instrument_key"]) for r in rows if r and r["instrument_key"]]
            add_many(keys)
        except Exception:
            pass

        # 3) Watchlist keys (optional; user-curated)
        try:
            items = self._watchlist.list()
            keys = [str(i.get("instrument_key")) for i in (items or []) if i and i.get("instrument_key")]
            add_many(keys)
        except Exception:
            pass

        return out[:limit]

    def top(
        self,
        n: int,
        min_confidence: float,
        max_risk: float,
        *,
        progress_cb: Callable[[int, int, str, str], None] | None = None,
    ) -> dict:
        return self._top(n=n, min_confidence=min_confidence, max_risk=max_risk, progress_cb=progress_cb)

    def _top(
        self,
        *,
        n: int,
        min_confidence: float,
        max_risk: float,
        progress_cb: Callable[[int, int, str, str], None] | None,
    ) -> dict:
        scored = []
        market_sent = float(self._news.market_sentiment().get("sentiment", 0.0))

        keys = self._universe_keys(limit=n)
        total = len(keys)
        if progress_cb is not None:
            try:
                progress_cb(0, max(1, total), "starting", "building universe")
            except Exception:
                pass

        processed = 0
        for key in keys:
            processed += 1
            if progress_cb is not None:
                try:
                    progress_cb(processed, max(1, total), "predicting", f"analyzing {key}")
                except Exception:
                    pass
            p = self._ai.predict(key, interval="5m", lookback_days=30, horizon_steps=12, include_nifty=False)
            pred = p.get("prediction", {})
            model_conf = float(pred.get("confidence", 0.0))
            unc = float(pred.get("uncertainty", 1.0))
            sig = pred.get("signal", "HOLD")

            meta = p.get("meta", {}) or {}
            model_name = str(meta.get("model") or "")
            agreement = float(pred.get("ensemble_agreement", 0.0))
            data_quality = float(meta.get("data_quality", 0.0))

            # Composite confidence: reward data quality + agreement + learned models,
            # penalize high uncertainty. Keep in [0,1].
            model_bonus = 0.10 if model_name.startswith("ridge_regression") else -0.05
            conf = (model_conf * 0.60) + (agreement * 0.20) + (data_quality * 0.20) + model_bonus
            conf = max(0.0, min(1.0, conf * (1.0 - 0.50 * unc)))

            if conf < min_confidence:
                continue
            if unc > max_risk:
                continue

            ohlc = pred.get("next_hour_ohlc", {})
            if not ohlc:
                continue

            last_close = float(p.get("features", {}).get("last_close", 0.0))
            next_close = float(ohlc.get("close", 0.0))
            if last_close <= 0:
                continue

            expected_ret = (next_close - last_close) / last_close
            # Blend in news sentiment as a small prior; do not let it dominate.
            news_bonus = float(max(-0.02, min(0.02, market_sent * 0.02)))
            score = (expected_ret + news_bonus) * conf * (1.0 - unc)

            scored.append(
                {
                    "instrument_key": key,
                    "score": float(score),
                    "signal": sig,
                    "confidence": float(conf),
                    "model_confidence": float(model_conf),
                    "uncertainty": unc,
                    "expected_return": float(expected_ret),
                    "market_sentiment": market_sent,
                    "confidence_breakdown": {
                        "model": model_name,
                        "model_confidence": float(model_conf),
                        "ensemble_agreement": float(agreement),
                        "data_quality": float(data_quality),
                        "uncertainty": float(unc),
                    },
                    "reasons": [
                        f"Signal={sig}",
                        f"Confidence={conf:.2f} (model={model_name})",
                        f"Uncertainty={unc:.2f}",
                        f"MarketSentiment={market_sent:.2f}",
                    ],
                }
            )

        if progress_cb is not None:
            try:
                progress_cb(max(1, total), max(1, total), "ranking", "sorting results")
            except Exception:
                pass
        scored.sort(key=lambda x: x["score"], reverse=True)
        if progress_cb is not None:
            try:
                progress_cb(max(1, total), max(1, total), "done", "finalizing")
            except Exception:
                pass

        return {"n": n, "analyzed": int(total), "results": scored[:n]}
