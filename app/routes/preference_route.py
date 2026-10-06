import logging
from fastapi import APIRouter, Depends, status

from app.dtos.preference_dto import (
    PreferenceVectorRequestDto,
    PreferenceVectorResponseDto,
)
from app.pipelines.embedding_pipeline import EmbeddingPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["Preference Vector"])


def get_embedding_pipeline() -> EmbeddingPipeline:
    return EmbeddingPipeline.get_instance()


@router.post(
    "/preference-vector",
    response_model=PreferenceVectorResponseDto,
    status_code=status.HTTP_200_OK,
    summary="이동 이력 기반 6차원 도메인 행동 피처 벡터 산출",
)
async def compute_preference_vector(
    payload: PreferenceVectorRequestDto,
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline),
) -> PreferenceVectorResponseDto:
    logger.info(f"[POST /api/v1/ai/preference-vector] trips={len(payload.trips)}")
    return pipeline.compute_vector(payload.trips)
