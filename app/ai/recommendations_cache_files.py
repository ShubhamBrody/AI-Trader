from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.core.settings import settings


def _cache_dir() -> Path:
    # Co-locate with SQLite DB by default (DATABASE_PATH defaults to ./data/app.db)
    try:
        db_path = Path(str(settings.DATABASE_PATH))
        return (db_path.parent if db_path.parent else Path("./data")).resolve()
    except Exception:
        return Path("./data").resolve()


def primary_path() -> Path:
    return _cache_dir() / "recommendations_cache.json"


def backup_path() -> Path:
    return _cache_dir() / "recommendations_cache.backup.json"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_name, str(path))
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except Exception:
            pass


def ensure_backup_from_primary_or_payload(payload: dict[str, Any]) -> None:
    bp = backup_path()
    if bp.exists():
        return

    pp = primary_path()
    if pp.exists():
        try:
            write_json_atomic(bp, read_json(pp) or payload)
            return
        except Exception:
            pass

    write_json_atomic(bp, payload)


def delete_backup() -> None:
    bp = backup_path()
    try:
        if bp.exists():
            bp.unlink()
    except Exception:
        pass
