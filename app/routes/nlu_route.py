import logging
from fastapi import APIRouter, Depends, status

from app.dtos.request_dto import NluRequestDto
from app.dtos.response_dto import NluResponseDto
from app.pipelines.nlu_pipeline import NluPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["NLU"])


def get_nlu_pipeline() -> NluPipeline:
    return NluPipeline.get_instance()


@router.post(
    "/nlu",
    response_model=NluResponseDto,
    status_code=status.HTTP_200_OK,
    summary="음성 전사 텍스트 기반 NLU 의도 분석 및 슬롯 필링",
    description="""
    자연어 텍스트로부터 출발역, 도착역, 추천 모드(OPTIMAL, COMFORTABLE, PERSONAL, CHEAPEST)를 단일 호출(Single-Call) 구조로 분석합니다.
    - devine_ai 스타일의 Zero-Token Station Guardrail을 통해 역명 환각을 0%로 차단합니다.
    - Gemini 지연(2초 초과) 또는 장애 발생 시 결정론적 Fallback 안전망이 즉시 가동되어 무중단 응답을 보장합니다.
    """,
)
async def analyze_nlu(
    payload: NluRequestDto,
    pipeline: NluPipeline = Depends(get_nlu_pipeline),
) -> NluResponseDto:
    logger.info(f"[POST /api/v1/ai/nlu] 요청 수신: text='{payload.text}'")
    return await pipeline.parse_intent(
        text=payload.text,
        preferred_mode=payload.preferred_mode,
    )
