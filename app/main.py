from contextlib import asynccontextmanager

from fastapi import FastAPI

import subprocess
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.db import init_db
from app.core.middleware import ApiKeyAuthMiddleware, PerformanceLogMiddleware, RateLimitMiddleware, RequestContextMiddleware
from app.core.settings import settings
from app.data.continuous import INGESTION
from app.automation.scheduler import AUTORETRAIN
from app.utils.logger import configure_logging
from app.prediction.live import LivePredictorConfig, PREDICTOR
from app.orders.startup_recovery import startup_order_recovery
from app.reconcile.loop import RECONCILER
from app.hft.index_options.service import HFT_INDEX_OPTIONS


def create_app() -> FastAPI:
    configure_logging(settings.LOG_LEVEL)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        init_db()

        # Best-effort crash recovery (read-only broker calls)
        try:
            await startup_order_recovery()
        except Exception:
            pass

        # Background loops (opt-in)
        if settings.DATA_PIPELINE_ENABLED:
            await INGESTION.start()
        if settings.AUTO_RETRAIN_ENABLED:
            await AUTORETRAIN.start()
        if settings.LIVE_RECONCILE_ENABLED:
            await RECONCILER.start()
        if settings.PREDICTOR_AUTOSTART:
            keys = [k.strip() for k in str(settings.PREDICTOR_INSTRUMENT_KEYS or "").split(",") if k.strip()]
            if keys:
                cfg = LivePredictorConfig(
                    interval=str(settings.PREDICTOR_INTERVAL),
                    horizon_steps=int(settings.PREDICTOR_HORIZON_STEPS),
                    lookback_minutes=int(settings.PREDICTOR_LOOKBACK_MINUTES),
                    poll_seconds=int(settings.PREDICTOR_POLL_SECONDS),
                    calibration_alpha=float(settings.PREDICTOR_CALIBRATION_ALPHA),
                )
                await PREDICTOR.start(instrument_keys=keys, cfg=cfg)

        if getattr(settings, "INDEX_OPTIONS_HFT_AUTOSTART", False):
            # Safe-by-default: also requires INDEX_OPTIONS_HFT_ENABLED=true.
            try:
                await HFT_INDEX_OPTIONS.start()
            except Exception:
                pass
        yield
        # best-effort shutdown
        try:
            await PREDICTOR.stop()
        except Exception:
            pass
        try:
            await HFT_INDEX_OPTIONS.stop()
        except Exception:
            pass
        try:
            await RECONCILER.stop()
        except Exception:
            pass
        try:
            await AUTORETRAIN.stop()
        except Exception:
            pass
        try:
            await INGESTION.stop()
        except Exception:
            pass

    app = FastAPI(
        title="AI Trading Backend",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS: permissive by default for local frontend integration.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request IDs + optional auth + optional rate limiting
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(ApiKeyAuthMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(PerformanceLogMiddleware)

    app.include_router(api_router)

    @lru_cache(maxsize=1)
    def _gpu_probe() -> dict:
        # Keep this endpoint usable without heavyweight deps.
        # Prefer torch CUDA availability when installed; otherwise best-effort driver detection.
        try:
            import torch  # type: ignore

            available = bool(torch.cuda.is_available())
            name = None
            if available:
                try:
                    name = str(torch.cuda.get_device_name(0))
                except Exception:
                    name = None
            return {
                "gpu_available": available,
                "gpu_name": name,
                "gpu_provider": "torch-cuda",
            }
        except Exception:
            # Best-effort: detect an NVIDIA GPU via nvidia-smi if present.
            try:
                import shutil

                if shutil.which("nvidia-smi") is None:
                    return {"gpu_available": False, "gpu_name": None, "gpu_provider": "none"}
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                    check=False,
                )
                out = (proc.stdout or "").strip()
                name = out.splitlines()[0].strip() if out else None
                return {
                    "gpu_available": False,
                    "gpu_name": name,
                    "gpu_provider": "nvidia-smi" if name else "none",
                }
            except Exception:
                return {"gpu_available": False, "gpu_name": None, "gpu_provider": "none"}

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "env": settings.APP_ENV,
            **_gpu_probe(),
        }

    return app


app = create_app()
