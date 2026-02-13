from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.core.db import db_conn, init_db


def main() -> None:
    init_db()
    with db_conn() as conn:
        print("Top instrument_type (overall):")
        rows = conn.execute(
            "SELECT COALESCE(instrument_type,'') AS t, COUNT(1) AS c "
            "FROM instrument_extra GROUP BY t ORDER BY c DESC LIMIT 50"
        ).fetchall()
        for r in rows:
            print(f"{r['t']}: {int(r['c'])}")

        print("\nTop segment (overall):")
        rows = conn.execute(
            "SELECT COALESCE(segment,'') AS s, COUNT(1) AS c "
            "FROM instrument_extra GROUP BY s ORDER BY c DESC LIMIT 50"
        ).fetchall()
        for r in rows:
            print(f"{r['s']}: {int(r['c'])}")

        print("\nCount where option_type is populated:")
        n_opt = int(
            conn.execute(
                "SELECT COUNT(1) AS c FROM instrument_extra WHERE option_type IS NOT NULL AND TRIM(option_type) != ''"
            ).fetchone()["c"]
        )
        print(n_opt)

        print("\nSample likely-derivatives rows (instrument_type contains 'OPT' or 'FUT' or segment contains 'FO'):")
        rows = conn.execute(
            "SELECT instrument_key, exchange, segment, instrument_type, name, underlying_symbol, expiry, strike, option_type "
            "FROM instrument_extra "
            "WHERE (UPPER(COALESCE(instrument_type,'')) LIKE '%OPT%' OR UPPER(COALESCE(instrument_type,'')) LIKE '%FUT%' "
            "   OR UPPER(COALESCE(segment,'')) LIKE '%FO%') "
            "ORDER BY updated_ts DESC LIMIT 30"
        ).fetchall()
        for r in rows:
            print(
                f"{r['instrument_key']} | ex={r['exchange']} | seg={r['segment']} | type={r['instrument_type']} | name={r['name']} | "
                f"und={r['underlying_symbol']} | exp={r['expiry']} | strike={r['strike']} | opt={r['option_type']}"
            )


if __name__ == "__main__":
    main()
