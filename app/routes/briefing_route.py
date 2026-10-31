import logging
from fastapi import APIRouter, Depends, status

from app.dtos.request_dto import (
    BriefingRequestDto,
)
from app.dtos.response_dto import (
    BriefingResponseDto,
)
from app.pipelines.briefing_pipeline import BriefingPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["Briefing"])


def get_briefing_pipeline() -> BriefingPipeline:
    return BriefingPipeline.get_instance()


@router.post(
    "/briefing",
    response_model=BriefingResponseDto,
    status_code=status.HTTP_200_OK,
    summary="다익스트라 경로 추천 기반 자연어 음성 브리핑 생성",
)
async def create_briefing(
    payload: BriefingRequestDto,
    pipeline: BriefingPipeline = Depends(get_briefing_pipeline),
) -> BriefingResponseDto:
    logger.info(
        f"[POST /api/v1/ai/briefing] 요청: {payload.origin_station_name} -> {payload.dest_station_name} ({payload.preferred_mode})"
    )
    return await pipeline.generate_briefing(
        origin_station_name=payload.origin_station_name,
        dest_station_name=payload.dest_station_name,
        preferred_mode=payload.preferred_mode,
        route=payload.route,
    )
