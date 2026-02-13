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
        cur = conn.execute(
            "UPDATE instrument_extra "
            "SET option_type = instrument_type "
            "WHERE (option_type IS NULL OR TRIM(option_type) = '') "
            "AND UPPER(COALESCE(instrument_type,'')) IN ('CE','PE')"
        )
        print(f"updated_rows={cur.rowcount}")


if __name__ == "__main__":
    main()
