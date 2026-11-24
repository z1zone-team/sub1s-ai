from fastapi import APIRouter

from app.pipelines.preference_embedding import (
    PreferenceEmbeddingRequest, PreferenceEmbeddingResponse, compute_preference_embedding,
)
from app.pipelines.preference_v2 import (
    PreferenceRequestV2, PreferenceResponseV2, compute_preference_v2,
)

router = APIRouter(prefix="/api/v1/ai", tags=["Preference Vector"])


@router.post(
    '/preference-vector/v2',
    response_model=PreferenceResponseV2,
    summary='선택 대안 비교 기반 10축 선호와 관측 신뢰도',
)
async def compute_versioned_preference(payload: PreferenceRequestV2):
    return compute_preference_v2(payload)


@router.post(
    '/preference-embedding',
    response_model=PreferenceEmbeddingResponse,
    summary='이동 목적, 경로 취향, 자주 고른 경로로 만든 취향 문장의 e5 임베딩(384차원, 정규화)',
)
def compute_preference_text_embedding(payload: PreferenceEmbeddingRequest):
    # 모델 추론은 CPU를 쓰므로 동기 함수로 둬서 FastAPI가 스레드 풀에서 실행하게 한다
    return compute_preference_embedding(payload)
