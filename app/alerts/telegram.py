from __future__ import annotations

from typing import Any

import httpx

from app.core.settings import settings


def send_telegram_message(text: str) -> dict[str, Any]:
    if not settings.ALERT_TELEGRAM_ENABLED:
        return {"ok": False, "detail": "ALERT_TELEGRAM_ENABLED=false"}
    if not settings.ALERT_TELEGRAM_BOT_TOKEN or not settings.ALERT_TELEGRAM_CHAT_ID:
        return {"ok": False, "detail": "telegram not configured"}

    token = str(settings.ALERT_TELEGRAM_BOT_TOKEN)
    chat_id = str(settings.ALERT_TELEGRAM_CHAT_ID)

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": str(text), "disable_web_page_preview": True}

    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return {"ok": True, "result": data}
    except Exception as e:
        return {"ok": False, "detail": str(e)}
