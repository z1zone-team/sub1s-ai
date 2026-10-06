from fastapi import APIRouter
from app.configs.settings import settings
from app.dtos.response_dto import HealthResponseDto

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponseDto)
async def check_health():
    return HealthResponseDto(
        status="ok",
        service="sub1s-ai",
        whisper_model=f"{settings.whisper_model_size}-{settings.whisper_compute_type}",
    )
