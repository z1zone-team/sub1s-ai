import redis.asyncio as aioredis
from fastapi import APIRouter

from app.configs.settings import settings
from app.dtos.response_dto import HealthResponseDto
from app.guardrails.station_validator import StationValidator

router = APIRouter(tags=["Health"])


async def check_valkey_status() -> str:
    try:
        client = aioredis.Redis(
            host=settings.valkey_host,
            port=settings.valkey_port,
            password=settings.valkey_password,
            socket_connect_timeout=0.1,
            socket_timeout=0.1,
        )
        await client.ping()
        await client.aclose()
        return "connected"
    except Exception:
        return "disconnected"


@router.get("/health", response_model=HealthResponseDto)
async def check_health():
    validator = StationValidator.get_instance()
    station_count = len(validator.valid_names)
    valkey_status = await check_valkey_status()

    return HealthResponseDto(
        status="ok",
        service="sub1s-ai",
        whisper_model=f"{settings.whisper_model_size}-{settings.whisper_compute_type}",
        station_count=station_count,
        valkey_status=valkey_status,
    )
