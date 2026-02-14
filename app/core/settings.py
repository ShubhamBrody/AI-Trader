from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    # Performance logging (console)
    # Logs request durations in ms. Useful for diagnosing sluggishness.
    PERF_LOG_ENABLED: bool = True
    # Log slow operations (requests / WS iterations) at WARNING when >= this threshold.
    PERF_LOG_SLOW_MS: int = 250
    # Internal (non-request) spans: DB ops, upstream calls, analysis functions.
    PERF_LOG_INNER_ENABLED: bool = True
    # If true, logs all internal spans (can be noisy). If false, logs only slow spans.
    PERF_LOG_INNER_ALWAYS: bool = False

    # Trading / safety
    SAFE_MODE: bool = True
    # Second, explicit guard for real-money order placement.
    # SAFE_MODE should remain the default local/dev posture; enabling live trading should require
    # *both* SAFE_MODE=false and LIVE_TRADING_ENABLED=true.
    LIVE_TRADING_ENABLED: bool = False
    TIMEZONE: str = "Asia/Kolkata"

    # Storage
    DATABASE_PATH: str = "./data/app.db"

    # Candles
    # When multiple clients request the same intraday candles at the same time (e.g., multiple WS connections),
    # this short de-dup window avoids redundant upstream calls without changing candle values.
    CANDLES_POLL_DEDUP_MS: int = 250

    # Trend confluence (multi-timeframe, multi-indicator) (OFF by default)
    # Uses SMA(50/200) + MACD(12,26,9) + RSI(14) + ADX(14) + volume EMA(10)
    # across 1d/4h/1h to produce a high-confidence trend direction.
    TRADER_TREND_CONFLUENCE_ENABLED: bool = False
    # If true, /api/trader/decision will gate BUY/SELL when confluence is "range" or conflicts.
    TRADER_TREND_CONFLUENCE_GATE_ENABLED: bool = False
    # Lookback windows for each timeframe.
    TRADER_TREND_CONFLUENCE_DAILY_DAYS: int = 420
    TRADER_TREND_CONFLUENCE_H4_DAYS: int = 90
    TRADER_TREND_CONFLUENCE_H1_DAYS: int = 30

    # Optional sequence-based candlestick pattern model (OFF by default)
    PATTERN_SEQ_MODEL_ENABLED: bool = False
    PATTERN_SEQ_MODEL_PATH: str = "data/models/pattern_seq.pt"
    PATTERN_SEQ_MODEL_SEQ_LEN: int = 64

    # Model selection variant
    # When set (e.g. "_lightweight"), the AI engine will append this suffix to the computed
    # model_family (e.g. long -> long_lightweight) for both ridge and deep model lookups.
    MODEL_FAMILY_SUFFIX: str = ""

    # CORS
    CORS_ALLOW_ORIGINS: list[str] = ["*"]

    # API security (optional)
    REQUIRE_API_KEY: bool = False
    API_KEY: str | None = None

    # Optional split keys (recommended for real money)
    # - WRITE: required for any order placement / agent controls / emergency actions.
    # - READ: optional; if unset falls back to API_KEY.
    API_KEY_WRITE: str | None = None
    API_KEY_READ: str | None = None

    # Optional admin key for dangerous actions (recommended).
    # If set, it is required for /api/controls/emergency/* and select agent actions.
    API_KEY_ADMIN: str | None = None

    # Optional allowlist for API writes (comma-separated CIDRs).
    # Example: "127.0.0.1/32,10.0.0.0/8".
    API_WRITE_ALLOWLIST_CIDRS: str = ""

    # WebSockets are unauthenticated by default to ease local frontend dev.
    # For production, enable and pass `x-api-key` header or `?api_key=...`.
    REQUIRE_API_KEY_WEBSOCKETS: bool = False

    # Metrics (Prometheus) (OFF by default)
    METRICS_ENABLED: bool = False

    # Basic rate limiting (optional)
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_PER_MINUTE: int = 120

    # News (RSS) + sentiment
    NEWS_RSS_URLS: str = "https://www.moneycontrol.com/rss/business.xml,https://economictimes.indiatimes.com/rssfeedsdefault.cms"

    # News (LLM summaries via Ollama) (OFF by default)
    # When enabled, the backend will attempt to fetch the article HTML for each news URL,
    # extract text, and generate a short summary using a local Ollama server.
    NEWS_LLM_SUMMARY_ENABLED: bool = False
    # Cache duration for generated summaries.
    NEWS_LLM_SUMMARY_TTL_DAYS: int = 7
    # Ollama connection + model
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:latest"

    # Recommendation universe (fallback if watchlist is empty)
    DEFAULT_UNIVERSE: str = "NSE_EQ|INE002A01018,NSE_EQ|INE009A01021,NSE_EQ|INE467B01029,NSE_EQ|INE040A01034"

    # Alerts
    # Set to 0 to disable low-balance alerts.
    LOW_BALANCE_THRESHOLD_INR: float = 0.0
    LOW_BALANCE_COOLDOWN_SECONDS: int = 600

    # Universe tiers (comma-separated instrument_keys). Optional: populate via /api/universe and watchlist.
    LARGE_CAP_UNIVERSE: str = ""
    MID_CAP_UNIVERSE: str = ""
    SMALL_CAP_UNIVERSE: str = ""

    # Autotrader (OFF by default)
    AUTOTRADER_ENABLED: bool = False
    AUTOTRADER_BROKER: str = "paper"  # paper|upstox
    AUTOTRADER_POLL_SECONDS: int = 30
    AUTOTRADER_MAX_SYMBOLS_PER_CYCLE: int = 30

    # Autotrader risk limits (safety defaults)
    AUTOTRADER_MAX_OPEN_TRADES: int = 3
    AUTOTRADER_MAX_TRADES_PER_DAY: int = 10
    # Set to 0 to disable daily loss stop.
    AUTOTRADER_MAX_DAILY_LOSS_INR: float = 0.0

    # Autotrader EOD handling
    # Close open trades this many minutes before market close (while still LIVE).
    AUTOTRADER_SQUAREOFF_MINUTES_BEFORE_CLOSE: int = 10

    # Autotrader exits
    AUTOTRADER_TARGET_EXIT_ENABLED: bool = True
    AUTOTRADER_TRAILING_ENABLED: bool = True

    # If enabled, the agent will compute stop/target from the intraday AI prediction
    # (projected high/low + confidence/uncertainty) instead of using only rule-based levels.
    AUTOTRADER_AI_RISK_PLAN_ENABLED: bool = False
    # Activate trailing once price has moved this many R in favor (R = initial risk per share).
    AUTOTRADER_TRAIL_ACTIVATION_R: float = 1.0
    # Trailing distance as fraction of price. Example 0.005 => 0.5%
    AUTOTRADER_TRAIL_PCT: float = 0.005

    # Intraday overlays integration (OFF by default to preserve legacy behavior)
    # When enabled, the agent requires overlays confidence + enough candles before entering.
    AUTOTRADER_OVERLAYS_ENTRY_ENABLED: bool = False
    AUTOTRADER_OVERLAYS_MANAGEMENT_ENABLED: bool = False
    AUTOTRADER_OVERLAYS_LOOKBACK_MINUTES: int = 90
    AUTOTRADER_OVERLAYS_MIN_CANDLES: int = 30
    AUTOTRADER_OVERLAYS_MIN_CONFIDENCE: float = 0.65
    # Only apply stop/target updates if the change is meaningful.
    AUTOTRADER_OVERLAYS_MIN_STOP_MOVE_PCT: float = 0.001
    AUTOTRADER_OVERLAYS_MIN_TARGET_MOVE_PCT: float = 0.002

    # Optional: multi-timeframe confluence gate + risk scaling (OFF by default)
    AUTOTRADER_TREND_CONFLUENCE_ENABLED: bool = False

    # Index Options HFT (OFF by default)
    # Trades long CALL/PUT options on NIFTY and SENSEX using 1m/5m signals.
    INDEX_OPTIONS_HFT_ENABLED: bool = False
    INDEX_OPTIONS_HFT_AUTOSTART: bool = False
    INDEX_OPTIONS_HFT_BROKER: str = "paper"  # paper|upstox
    INDEX_OPTIONS_HFT_POLL_SECONDS: int = 5

    # Behavior toggles
    # If true, only trade when NIFTY and SENSEX agree on 5m direction.
    INDEX_OPTIONS_HFT_REQUIRE_MARKET_ALIGNMENT: bool = False
    # If true, allow running paper HFT even when market_state != LIVE (useful for testing).
    INDEX_OPTIONS_HFT_IGNORE_MARKET_STATE_WHEN_PAPER: bool = False

    # Instrument queries used to resolve spot instrument_keys via instrument_meta.
    # These should match entries in your imported Upstox exchange instruments.
    INDEX_OPTIONS_HFT_NIFTY_SPOT_QUERY: str = "NIFTY"
    INDEX_OPTIONS_HFT_SENSEX_SPOT_QUERY: str = "SENSEX"

    # Lot sizes (set to your broker's current lot sizes)
    INDEX_OPTIONS_HFT_NIFTY_LOT_SIZE: int = 50
    INDEX_OPTIONS_HFT_SENSEX_LOT_SIZE: int = 10

    # Selection
    INDEX_OPTIONS_HFT_MAX_EXPIRY_DAYS: int = 14

    # Signal/risk controls
    INDEX_OPTIONS_HFT_SPOT_LOOKBACK_MINUTES: int = 240
    INDEX_OPTIONS_HFT_MIN_CONFIDENCE: float = 0.70
    INDEX_OPTIONS_HFT_RISK_FRACTION: float = 0.002
    INDEX_OPTIONS_HFT_MAX_CAPITAL_PER_TRADE_INR: float = 5000.0
    INDEX_OPTIONS_HFT_MAX_LOTS: int = 1

    # Additional safety caps
    INDEX_OPTIONS_HFT_MAX_OPEN_TRADES: int = 2
    INDEX_OPTIONS_HFT_MAX_TRADES_PER_DAY: int = 20
    # Set to 0 to disable daily loss stop.
    INDEX_OPTIONS_HFT_MAX_DAILY_LOSS_INR: float = 0.0

    # Exits (premium-based)
    INDEX_OPTIONS_HFT_OPTION_STOP_PCT: float = 0.12
    INDEX_OPTIONS_HFT_OPTION_TARGET_PCT: float = 0.18
    INDEX_OPTIONS_HFT_MAX_HOLD_MINUTES: int = 30

    # AI gate (optional): require AIEngine BUY/SELL approval before entering.
    # Note: Even without trained models, AIEngine falls back to a statistical stub.
    INDEX_OPTIONS_HFT_AI_GATE_ENABLED: bool = False
    INDEX_OPTIONS_HFT_AI_MIN_CONFIDENCE: float = 0.60
    INDEX_OPTIONS_HFT_AI_INTERVAL: str = "1m"
    INDEX_OPTIONS_HFT_AI_LOOKBACK_DAYS: int = 7
    INDEX_OPTIONS_HFT_AI_HORIZON_STEPS: int = 12

    # Optional: multi-timeframe confluence gate + risk scaling (OFF by default)
    INDEX_OPTIONS_HFT_TREND_CONFLUENCE_ENABLED: bool = False

    # Adaptive learning persistence
    INDEX_OPTIONS_HFT_STATE_PATH: str = "data/hft_index_options_state.json"

    # Training presets
    TRAIN_LONG_INTERVAL: str = "1d"
    TRAIN_LONG_LOOKBACK_DAYS: int = 1460  # ~4 years
    TRAIN_LONG_HORIZON_STEPS: int = 20

    TRAIN_INTRADAY_INTERVAL: str = "1m"
    TRAIN_INTRADAY_LOOKBACK_DAYS: int = 95  # ~3 months
    TRAIN_INTRADAY_HORIZON_STEPS: int = 60  # 60 minutes

    # Data pipeline (OFF by default)
    DATA_PIPELINE_ENABLED: bool = False
    DATA_PIPELINE_POLL_SECONDS: int = 30
    DATA_PIPELINE_MAX_SYMBOLS_PER_CYCLE: int = 50
    DATA_PIPELINE_INTRADAY_INTERVAL: str = "1m"
    DATA_PIPELINE_LOOKBACK_MINUTES: int = 30
    DATA_PIPELINE_DAILY_INTERVAL: str = "1d"
    # Daily refresh window as a timedelta (keep light).
    # Note: pydantic-settings doesn't parse timedeltas from env by default; keep as minutes.
    DATA_PIPELINE_DAILY_LOOKBACK_DAYS: int = 30

    # Auto retrain / eval / promotion (OFF by default)
    AUTO_RETRAIN_ENABLED: bool = False
    AUTO_RETRAIN_POLL_SECONDS: int = 60
    # Retrain long models once per day after market close.
    AUTO_RETRAIN_LONG_HOUR_LOCAL: int = 16
    AUTO_RETRAIN_LONG_MINUTE_LOCAL: int = 0
    # Retrain intraday models once per day after close (safer than during market).
    AUTO_RETRAIN_INTRADAY_HOUR_LOCAL: int = 16
    AUTO_RETRAIN_INTRADAY_MINUTE_LOCAL: int = 15
    AUTO_RETRAIN_MAX_SYMBOLS: int = 100

    # Promotion rules (to reduce overtraining)
    PROMOTION_MIN_DIRECTION_ACC: float = 0.52
    PROMOTION_MAX_RMSE: float = 0.05
    DRIFT_Z_THRESHOLD: float = 3.0

    # Live next-hour predictor (OFF by default)
    PREDICTOR_AUTOSTART: bool = False
    PREDICTOR_INSTRUMENT_KEYS: str = ""  # comma-separated instrument keys
    PREDICTOR_INTERVAL: str = "1m"
    PREDICTOR_HORIZON_STEPS: int = 60
    PREDICTOR_LOOKBACK_MINUTES: int = 240
    PREDICTOR_POLL_SECONDS: int = 10
    PREDICTOR_CALIBRATION_ALPHA: float = 0.05

    # Prediction retention/cleanup (OFF by default)
    # When enabled, old rows in prediction_events are deleted to keep DB size bounded.
    PREDICTION_RETENTION_ENABLED: bool = False
    # If > 0: delete rows where ts_pred < now - max_age_days.
    PREDICTION_RETENTION_MAX_AGE_DAYS: int = 0
    # If > 0: keep only the newest max_rows rows.
    PREDICTION_RETENTION_MAX_ROWS: int = 0
    # How often to run retention in the background.
    PREDICTION_RETENTION_POLL_SECONDS: int = 60 * 60  # 1 hour

    # Upstox (optional)
    UPSTOX_CLIENT_ID: str | None = None
    UPSTOX_CLIENT_SECRET: str | None = None
    UPSTOX_ACCESS_TOKEN: str | None = None

    # Upstox OAuth helpers (optional)
    FRONTEND_URL: str = "http://localhost:5173/"
    UPSTOX_REDIRECT_URI: str = "http://localhost:8000/api/auth/upstox/callback"
    UPSTOX_TOKEN_PATH: str = "data/upstox_token.json"

    # Upstox behavior
    UPSTOX_STRICT: bool = False  # if true, do not fall back to stub on Upstox failures
    UPSTOX_BASE_URL: str = "https://api.upstox.com"
    UPSTOX_HFT_BASE_URL: str = "https://api-hft.upstox.com"

    # Upstox resilience
    UPSTOX_RETRY_MAX_ATTEMPTS: int = 3
    UPSTOX_RETRY_BACKOFF_SECONDS: float = 0.5
    UPSTOX_CB_FAILURES: int = 5
    UPSTOX_CB_COOLDOWN_SECONDS: int = 20

    # Upstox order defaults (Equity intraday)
    UPSTOX_EQ_PRODUCT: str = "I"  # Intraday
    UPSTOX_EQ_VALIDITY: str = "DAY"
    UPSTOX_ENTRY_ORDER_TYPE: str = "MARKET"  # MARKET|LIMIT
    UPSTOX_USE_BROKER_STOP: bool = True
    UPSTOX_STOP_ORDER_TYPE: str = "SL-M"  # SL-M preferred for intraday stop
    UPSTOX_FALLBACK_APP_STOP: bool = True

    # Startup crash recovery (OFF by default)
    # If enabled, the app will reconcile recent Upstox orders on startup.
    ORDER_RECOVERY_ON_STARTUP: bool = False
    ORDER_RECOVERY_LOOKBACK_HOURS: int = 24
    ORDER_RECOVERY_STARTUP_LIMIT: int = 200

    # Continuous live reconciliation (OFF by default)
    LIVE_RECONCILE_ENABLED: bool = False
    LIVE_RECONCILE_POLL_SECONDS: int = 30

    # Live failsafes (when reconciliation is enabled)
    FAILSAFE_AUTOFREEZE_ON_UNKNOWN_BROKER_ORDERS: bool = True
    FAILSAFE_DISABLE_AGENT_ON_UNKNOWN_BROKER_ORDERS: bool = True
    FAILSAFE_AUTOFREEZE_ON_DAILY_LOSS: bool = True
    FAILSAFE_DISABLE_AGENT_ON_DAILY_LOSS: bool = True

    # Alert delivery (optional)
    ALERT_TELEGRAM_ENABLED: bool = False
    ALERT_TELEGRAM_BOT_TOKEN: str | None = None
    ALERT_TELEGRAM_CHAT_ID: str | None = None
    ALERT_TELEGRAM_COOLDOWN_SECONDS: int = 60


settings = Settings()
