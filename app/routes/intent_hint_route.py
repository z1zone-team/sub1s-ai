from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.dtos.response_dto import IntentHintDto, IntentHintResponseDto
from app.pipelines.speech_pipeline import shared_gemini_provider

router = APIRouter(prefix="/api/v1/ai", tags=["Intent"])


class IntentHintRequestDto(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="받아 적은 문장(앱 기기 인식이나 서버 STT 결과)")


@router.post("/intent-hint", response_model=IntentHintResponseDto)
async def intent_hint(body: IntentHintRequestDto) -> IntentHintResponseDto:
    """
    받아 적은 문장에서 출발지, 도착지, 경로 조건 힌트를 Gemini로 받는다.
    서버는 규칙과 역 사전으로 풀지 못했을 때만 부르고, 결과는 다시 역 사전으로 확인한다.
    Gemini가 쿼터 초과로 쉬는 중이거나 키가 없으면 `available: false`를 바로 돌려준다(음성 인식과 같은 쉬는 시간을 쓴다).
    """
    hint = await shared_gemini_provider().intent_hint(body.text)
    if hint is None:
        return IntentHintResponseDto(available=False)
    return IntentHintResponseDto(available=True, hint=IntentHintDto(**hint.to_dict()))
