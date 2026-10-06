import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.configs.settings import settings
from app.errors.error_handler import register_error_handlers
from app.guardrails.station_validator import StationValidator
from app.pipelines.stt_pipeline import stt_pipeline
from app.routes.briefing_route import router as briefing_router
from app.routes.health_route import router as health_router
from app.routes.nlu_route import router as nlu_router
from app.routes.preference_route import router as preference_router
from app.routes.stt_route import router as stt_router
from app.routes.voice_process_route import router as voice_process_router
from app.workers.voice_worker import VoiceWorker

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sub1s-ai")

background_worker: Optional[VoiceWorker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global background_worker
    logger.info("Initializing Sub1S AI Service...")
    # Whisper 모델 및 지하철 역명 사전 워밍업
    try:
        StationValidator.get_instance()
        logger.info("StationValidator dictionary warmed up.")
    except Exception as e:
        logger.warning(f"StationValidator warmup failed: {e}")

    try:
        stt_pipeline.load_model()
    except Exception as e:
        logger.warning(f"Whisper warmup deferred: {e}")

    # Valkey BullMQ 백그라운드 워커 기동 (설정 시)
    if settings.enable_bullmq_worker:
        try:
            background_worker = VoiceWorker()
            await background_worker.start()
        except Exception as e:
            logger.warning(f"BullMQ background worker auto-start failed: {e}")

    yield

    if background_worker:
        await background_worker.stop()
    logger.info("Shutting down Sub1S AI Service...")


app = FastAPI(
    title="Sub1S AI Worker API",
    description="Sub1S faster-whisper STT, Gemini NLU, and Edge-TTS Pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 글로벌 에러 핸들러 등록
register_error_handlers(app)

# 라우터 등록
app.include_router(health_router)
app.include_router(stt_router)
app.include_router(nlu_router)
app.include_router(briefing_router)
app.include_router(preference_router)
app.include_router(voice_process_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.ai_host,
        port=settings.ai_port,
        reload=settings.debug,
    )
