from app.routes.briefing_route import router as briefing_router
from app.routes.health_route import router as health_router
from app.routes.nlu_route import router as nlu_router
from app.routes.preference_route import router as preference_router
from app.routes.stt_route import router as stt_router

__all__ = [
    "health_router",
    "stt_router",
    "nlu_router",
    "briefing_router",
    "preference_router",
]
