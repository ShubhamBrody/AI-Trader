from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from app.core.settings import settings


@dataclass(frozen=True)
class UpstoxToken:
    access_token: str
    token_type: str | None = None
    expires_at_utc: datetime | None = None
    raw: dict[str, Any] | None = None


def _token_path() -> Path:
    # Kept configurable for dev/prod; tests intentionally ignore disk.
    path = getattr(settings, "UPSTOX_TOKEN_PATH", None) or "data/upstox_token.json"
    return Path(path)


_cached: UpstoxToken | None = None


def _parse_expires_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def load_from_disk() -> UpstoxToken | None:
    global _cached

    p = _token_path()
    if not p.exists():
        return None

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        token = data.get("access_token")
        if not token:
            return None

        parsed = UpstoxToken(
            access_token=str(token),
            token_type=data.get("token_type"),
            expires_at_utc=_parse_expires_at(data.get("expires_at_utc")),
            raw=data.get("raw"),
        )
        _cached = parsed
        return parsed
    except Exception as e:
        logger.warning(f"Failed to read Upstox token cache at {p}: {e}")
        return None


def save_to_disk(token: UpstoxToken) -> None:
    p = _token_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "access_token": token.access_token,
        "token_type": token.token_type,
        "expires_at_utc": token.expires_at_utc.isoformat() if token.expires_at_utc else None,
        "raw": token.raw,
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_token(
    access_token: str,
    token_type: str | None = None,
    expires_in: int | None = None,
    raw: dict[str, Any] | None = None,
) -> UpstoxToken:
    global _cached

    expires_at = None
    if isinstance(expires_in, int) and expires_in > 0:
        expires_at = datetime.now(tz=timezone.utc) + timedelta(seconds=expires_in)

    tok = UpstoxToken(
        access_token=access_token,
        token_type=token_type,
        expires_at_utc=expires_at,
        raw=raw,
    )
    _cached = tok
    save_to_disk(tok)
    return tok


def clear_token() -> None:
    global _cached
    _cached = None

    p = _token_path()
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def get_token() -> UpstoxToken | None:
    # Tests should be fully offline/deterministic.
    if getattr(settings, "APP_ENV", "") == "test":
        return None

    # 1) in-memory
    if _cached is not None:
        return _cached

    # 2) disk
    tok = load_from_disk()
    if tok is not None:
        return tok

    # 3) env/config fallback
    env_token = (settings.UPSTOX_ACCESS_TOKEN or "").strip()
    if env_token:
        return UpstoxToken(access_token=env_token)

    return None


def get_access_token() -> str | None:
    tok = get_token()
    return tok.access_token if tok else None


def is_logged_in() -> bool:
    return bool(get_access_token())
