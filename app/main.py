import logging
from contextlib import asynccontextmanager
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

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sub1s-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    yield
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.ai_host,
        port=settings.ai_port,
        reload=settings.debug,
    )
