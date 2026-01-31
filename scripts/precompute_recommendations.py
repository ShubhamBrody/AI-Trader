from __future__ import annotations

import argparse

from datetime import datetime, timezone

from app.ai.recommendation_engine import RecommendationEngine
from app.ai.recommendations_cache_files import delete_backup, primary_path, write_json_atomic
from app.core.db import init_db


def main() -> int:
    p = argparse.ArgumentParser(description="Precompute and persist today's recommendations cache")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.add_argument("--max-risk", type=float, default=0.7)
    p.add_argument("--universe-limit", type=int, default=200)
    p.add_argument("--universe-since-days", type=int, default=7)
    args = p.parse_args()

    init_db()
    engine = RecommendationEngine()
    recs, meta = engine.refresh_top(
        n=args.n,
        min_confidence=args.min_confidence,
        max_risk=args.max_risk,
        universe_limit=args.universe_limit,
        universe_since_days=args.universe_since_days,
    )

    try:
        write_json_atomic(
            primary_path(),
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "recommendations": list(recs or []),
                "meta": dict(meta or {}),
            },
        )
        delete_backup()
    except Exception:
        pass

    print("ok")
    print(f"count={len(recs)}")
    print(f"cache_key={meta.get('cache_key')}")
    print(f"trading_day={meta.get('cache_trading_day')}")
    print(f"created_at_utc={meta.get('created_at_utc')}")
    if meta.get("relaxed"):
        print("note=filters relaxed (no strict-qualified results)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
