import os
import sys

import pytest

# Ensure repository root is on sys.path so `import app` works when running pytest.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture(autouse=True)
def _isolate_sqlite_db(tmp_path, monkeypatch):
    """Ensure tests don't share a persistent SQLite database.

    Several endpoints now persist runtime controls (e.g. freeze/agent disable). If
    we use the default on-disk DB (./data/app.db), those values can leak across
    tests and make outcomes order-dependent.
    """

    from app.core.settings import settings as app_settings

    db_path = tmp_path / "test.db"
    monkeypatch.setattr(app_settings, "DATABASE_PATH", str(db_path), raising=False)
    monkeypatch.setattr(app_settings, "APP_ENV", "test", raising=False)
    # Ensure tests are deterministic/offline even if developer machine has live creds in env.
    monkeypatch.setattr(app_settings, "UPSTOX_ACCESS_TOKEN", None, raising=False)
    monkeypatch.setattr(app_settings, "UPSTOX_CLIENT_ID", None, raising=False)
    monkeypatch.setattr(app_settings, "UPSTOX_CLIENT_SECRET", None, raising=False)
    monkeypatch.setattr(app_settings, "UPSTOX_STRICT", False, raising=False)
    yield
