from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from threading import Lock

from app.core.settings import settings


_DB_INIT_LOCK = Lock()
_DB_INITIALIZED_FOR_PATH: str | None = None


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def init_db() -> None:
    _ensure_parent_dir(settings.DATABASE_PATH)
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                instrument_key TEXT NOT NULL,
                interval TEXT NOT NULL,
                ts INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (instrument_key, interval, ts)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                balance REAL NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_positions (
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                avg_price REAL NOT NULL,
                PRIMARY KEY (symbol)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                fees REAL NOT NULL,
                slippage REAL NOT NULL,
                meta TEXT
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT,
                request_id TEXT,
                payload_json TEXT
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trained_models (
                model_key TEXT PRIMARY KEY,
                instrument_key TEXT NOT NULL,
                interval TEXT NOT NULL,
                horizon_steps INTEGER NOT NULL,
                trained_ts INTEGER NOT NULL,
                n_samples INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                model_json TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                instrument_key TEXT PRIMARY KEY,
                label TEXT,
                created_ts INTEGER NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                summary TEXT,
                sentiment REAL NOT NULL,
                impact REAL NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news_url TEXT NOT NULL,
                instrument_key TEXT NOT NULL,
                relevance REAL NOT NULL,
                reason TEXT,
                created_ts INTEGER NOT NULL,
                UNIQUE (news_url, instrument_key)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_mentions_instrument ON news_mentions(instrument_key, created_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_mentions_url ON news_mentions(news_url)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_state (
                key TEXT PRIMARY KEY,
                last_ts INTEGER NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS instrument_meta (
                instrument_key TEXT PRIMARY KEY,
                tradingsymbol TEXT,
                cap_tier TEXT NOT NULL,
                upstox_token TEXT,
                updated_ts INTEGER NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS instrument_extra (
                instrument_key TEXT PRIMARY KEY,
                exchange TEXT,
                segment TEXT,
                instrument_type TEXT,
                name TEXT,
                underlying_symbol TEXT,
                expiry TEXT,
                strike REAL,
                option_type TEXT,
                raw_json TEXT,
                updated_ts INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_instrument_extra_segment ON instrument_extra(segment, exchange)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations_cache (
                cache_key TEXT PRIMARY KEY,
                trading_day TEXT NOT NULL,
                created_ts INTEGER NOT NULL,
                analyzed INTEGER NOT NULL,
                qualified INTEGER NOT NULL,
                params_json TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_cache_day ON recommendations_cache(trading_day, created_ts)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intraday_overlay_models (
                instrument_key TEXT NOT NULL,
                interval TEXT NOT NULL,
                created_ts INTEGER NOT NULL,
                model_json TEXT NOT NULL,
                PRIMARY KEY (instrument_key, interval)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_overlay_models_ts ON intraday_overlay_models(created_ts)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_open INTEGER NOT NULL,
                ts_close INTEGER,
                instrument_key TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                entry REAL NOT NULL,
                stop REAL NOT NULL,
                target REAL NOT NULL,
                status TEXT NOT NULL,
                entry_order_id TEXT,
                sl_order_id TEXT,
                exit_order_id TEXT,
                close_reason TEXT,
                monitor_app_stop INTEGER NOT NULL,
                meta_json TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS order_states (
                id TEXT PRIMARY KEY,
                ts_created INTEGER NOT NULL,
                ts_updated INTEGER NOT NULL,
                broker TEXT NOT NULL,
                instrument_key TEXT,
                symbol TEXT,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                order_kind TEXT NOT NULL,
                order_type TEXT NOT NULL,
                price REAL,
                trigger_price REAL,
                client_order_id TEXT,
                broker_order_id TEXT,
                status TEXT NOT NULL,
                last_error TEXT,
                meta_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_order_states_client_order_id "
            "ON order_states(client_order_id) "
            "WHERE client_order_id IS NOT NULL AND client_order_id <> ''"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_order_states_broker_order_id ON order_states(broker_order_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_order_states_status ON order_states(broker, status, ts_created)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_order_states_instrument ON order_states(instrument_key, ts_created)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS order_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                order_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_order_events_order_id ON order_events(order_id, ts)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_controls (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                ts_updated INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_controls_ts ON trade_controls(ts_updated)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                broker TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_snapshots_ts ON positions_snapshots(broker, ts)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions_state (
                broker TEXT NOT NULL,
                instrument_key TEXT,
                instrument_token TEXT NOT NULL,
                net_qty REAL NOT NULL,
                avg_price REAL,
                last_price REAL,
                pnl REAL,
                ts_updated INTEGER NOT NULL,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (broker, instrument_token)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_state_key ON positions_state(broker, instrument_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_state_qty ON positions_state(broker, net_qty, ts_updated)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                job_id TEXT PRIMARY KEY,
                ts_start INTEGER NOT NULL,
                ts_end INTEGER,
                status TEXT NOT NULL,
                kind TEXT NOT NULL,
                instrument_key TEXT,
                interval TEXT,
                horizon_steps INTEGER,
                model_family TEXT,
                cap_tier TEXT,
                progress REAL NOT NULL,
                message TEXT,
                metrics_json TEXT,
                error TEXT
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                job_id TEXT PRIMARY KEY,
                ts_start INTEGER NOT NULL,
                ts_end INTEGER,
                status TEXT NOT NULL,
                kind TEXT NOT NULL,
                progress REAL NOT NULL,
                message TEXT,
                params_json TEXT,
                stats_json TEXT,
                error TEXT
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                slot_key TEXT PRIMARY KEY,
                model_key TEXT NOT NULL,
                kind TEXT NOT NULL,
                stage TEXT NOT NULL,
                updated_ts INTEGER NOT NULL,
                metrics_json TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                slot_key TEXT NOT NULL,
                model_key TEXT NOT NULL,
                kind TEXT NOT NULL,
                eval_kind TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS automation_runs (
                job_id TEXT PRIMARY KEY,
                ts_start INTEGER NOT NULL,
                ts_end INTEGER,
                status TEXT NOT NULL,
                kind TEXT NOT NULL,
                progress REAL NOT NULL,
                message TEXT,
                params_json TEXT,
                stats_json TEXT,
                error TEXT
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_pred INTEGER NOT NULL,
                instrument_key TEXT NOT NULL,
                interval TEXT NOT NULL,
                horizon_steps INTEGER NOT NULL,
                model_kind TEXT NOT NULL,
                model_key TEXT,
                pred_ret REAL NOT NULL,
                pred_ret_adj REAL NOT NULL,
                calib_bias REAL NOT NULL,
                ts_target INTEGER NOT NULL,
                status TEXT NOT NULL,
                ts_resolved INTEGER,
                actual_ret REAL,
                error REAL,
                meta_json TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_events_due ON prediction_events(status, ts_target);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_events_anchor ON prediction_events(instrument_key, interval, horizon_steps, ts_pred);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS online_calibration (
                key TEXT PRIMARY KEY,
                updated_ts INTEGER NOT NULL,
                n INTEGER NOT NULL,
                bias REAL NOT NULL,
                mae REAL NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candle_pattern_stats (
                pattern TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                side TEXT NOT NULL,
                base_reliability REAL NOT NULL,
                weight REAL NOT NULL,
                seen INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                ema_winrate REAL NOT NULL,
                updated_ts INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candle_pattern_stats_updated ON candle_pattern_stats(updated_ts)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candle_pattern_params (
                pattern TEXT PRIMARY KEY,
                params_json TEXT NOT NULL,
                updated_ts INTEGER NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candle_pattern_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                pattern TEXT NOT NULL,
                good INTEGER NOT NULL,
                magnitude REAL NOT NULL,
                instrument_key TEXT,
                interval TEXT,
                asof_ts INTEGER,
                note TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candle_pattern_feedback_ts ON candle_pattern_feedback(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candle_pattern_feedback_pattern ON candle_pattern_feedback(pattern, ts)")

        # seed paper account
        cur = conn.execute("SELECT balance FROM paper_account WHERE id=1")
        row = cur.fetchone()
        if row is None:
            conn.execute("INSERT INTO paper_account (id, balance) VALUES (1, ?)", (100000.0,))

    global _DB_INITIALIZED_FOR_PATH
    _DB_INITIALIZED_FOR_PATH = str(settings.DATABASE_PATH)


def ensure_db_initialized() -> None:
    """Initialize DB schema once per process.

    Calling init_db() is relatively expensive (many DDL statements). Under high request rates,
    doing it per connection can dominate latency.
    """

    global _DB_INITIALIZED_FOR_PATH
    db_path = str(settings.DATABASE_PATH)
    if _DB_INITIALIZED_FOR_PATH == db_path:
        return
    with _DB_INIT_LOCK:
        if _DB_INITIALIZED_FOR_PATH == db_path:
            return
        init_db()


@contextmanager
def db_conn():
    ensure_db_initialized()
    conn = sqlite3.connect(settings.DATABASE_PATH, timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        # Per-connection pragmas: keep SQLite responsive under concurrent reads/writes.
        # WAL mode is enabled at DB init; set additional performance-oriented knobs here.
        try:
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            # Negative cache_size means KiB. ~20 MiB cache helps repeated lookups.
            conn.execute("PRAGMA cache_size=-20000;")
        except Exception:
            pass
        yield conn
        conn.commit()
    finally:
        conn.close()
