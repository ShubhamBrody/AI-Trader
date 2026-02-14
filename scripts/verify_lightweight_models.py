from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _pick_order_column(columns: list[str]) -> str:
    for col in ("created_at", "updated_at", "trained_at", "timestamp"):
        if col in columns:
            return col
    if "trained_ts" in columns:
        return "trained_ts"
    return "rowid"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify trained lightweight models exist in the SQLite registry.")
    parser.add_argument(
        "--db",
        default=None,
        help="Path to app.db (defaults to settings.DATABASE_PATH or ./data/app.db)",
    )
    parser.add_argument(
        "--suffix",
        default="lightweight",
        help="Suffix used for lightweight families (default: lightweight). Matches long_<suffix>, intraday_<suffix>.",
    )
    parser.add_argument("--limit", type=int, default=12, help="How many latest rows to display (default: 12).")
    args = parser.parse_args()

    # Import settings late so this script works even if run outside uvicorn.
    db_path: str
    if args.db:
        db_path = args.db
    else:
        try:
            from app.core.settings import settings

            db_path = str(getattr(settings, "DATABASE_PATH", "./data/app.db") or "./data/app.db")
        except Exception:
            db_path = "./data/app.db"

    db_file = Path(db_path)
    if not db_file.exists():
        print(f"DB not found: {db_file.resolve()}")
        return 2

    suffix = str(args.suffix).strip()
    if not suffix:
        print("Empty --suffix is not allowed")
        return 2
    if suffix.startswith("_"):
        suffix = suffix[1:]
    # In this DB schema, model family is encoded in `model_key` as:
    #   <family>::<cap_tier>::<instrument_key>::<interval>::h<horizon>
    # So we filter on `model_key` prefix patterns.
    model_key_like = f"%_{suffix}::%"

    con = sqlite3.connect(str(db_file))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trained_models'")
    if not cur.fetchone():
        print("No trained_models table found in DB")
        return 3

    cur.execute("PRAGMA table_info(trained_models)")
    columns = [r[1] for r in cur.fetchall()]
    order_col = _pick_order_column(columns)

    # Extract family from model_key (substring before first '::').
    cur.execute(
        """
        SELECT
          substr(model_key, 1, instr(model_key, '::') - 1) AS family,
          COUNT(*) AS n
        FROM trained_models
        WHERE model_key LIKE ?
        GROUP BY family
        ORDER BY n DESC
        """.strip(),
        (model_key_like,),
    )
    rows = cur.fetchall()

    print("DB:", db_file.resolve())
    print("Suffix:", suffix)
    print("Model key LIKE:", model_key_like)
    print("Order column:", order_col)

    if not rows:
        print("\nNo lightweight model rows found.")
        return 1

    print("\nCounts by family:")
    for r in rows:
        print(f" - {r['family']}: {r['n']}")

    query = f"""
SELECT
    instrument_key,
    interval,
    horizon_steps,
    substr(model_key, 1, instr(model_key, '::') - 1) AS family,
    {order_col} AS ts,
    model_key
FROM trained_models
WHERE model_key LIKE ?
ORDER BY {order_col} DESC
LIMIT ?
""".strip()

    cur.execute(query, (model_key_like, int(args.limit)))
    latest = cur.fetchall()

    print("\nLatest entries:")
    for r in latest:
        print(
            f" - {str(r['family']):18s} {str(r['instrument_key'])} {str(r['interval'])} "
            f"h={int(r['horizon_steps'])} ts={r['ts']}"
        )

    families = {r["family"] for r in rows}
    print("\nSanity:")
    print(" long_... present:", any(f.startswith("long_") for f in families))
    print(" intraday_... present:", any(f.startswith("intraday_") for f in families))

    con.close()
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
