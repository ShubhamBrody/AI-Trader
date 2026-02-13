from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.core.db import db_conn, init_db
from app.core.settings import settings


def main() -> None:
    init_db()
    print(f"DB_PATH={settings.DATABASE_PATH}")
    with db_conn() as conn:
        meta_n = int(conn.execute("SELECT COUNT(1) FROM instrument_meta").fetchone()[0])
        extra_n = int(conn.execute("SELECT COUNT(1) FROM instrument_extra").fetchone()[0])
        print(f"instrument_meta={meta_n}")
        print(f"instrument_extra={extra_n}")

        q = (
            "SELECT COUNT(1) FROM instrument_extra "
            "WHERE UPPER(COALESCE(underlying_symbol,''))=? "
            "AND UPPER(COALESCE(option_type, instrument_type,'')) IN ('CE','PE')"
        )
        for u in ("NIFTY", "SENSEX"):
            n = int(conn.execute(q, (u,)).fetchone()[0])
            print(f"options_{u}={n}")


if __name__ == "__main__":
    main()
