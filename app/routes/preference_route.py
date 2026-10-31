from fastapi import APIRouter

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
