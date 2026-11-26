from unittest.mock import AsyncMock, MagicMock
import pytest

from app.clients.gemini_client import GeminiClient
from app.dtos.request_dto import RouteMode
from app.pipelines.nlu_pipeline import NluPipeline


@pytest.fixture
def mock_gemini_client():
    client = MagicMock(spec=GeminiClient)
    client.generate_nlu = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
async def test_nlu_pipeline_deterministic_fallback(mock_gemini_client):
    # Gemini 장애/타임아웃 시 0-토큰 결정론적 Fallback 검증
    pipeline = NluPipeline(gemini_client=mock_gemini_client)

    result = await pipeline.parse_intent(
        "강남역에서 을지로입구역까지 덜 붐비는 경로로 가고 싶어"
    )

    assert result.origin_station_name == "강남"
    assert result.dest_station_name == "을지로입구"
    assert result.preferred_mode == RouteMode.COMFORTABLE
    assert result.is_fallback is True
    assert result.confidence == 0.90


@pytest.mark.asyncio
async def test_nlu_pipeline_preferred_mode_override(mock_gemini_client):
    # 사용자 지정 모드가 우선 적용되는지 검증
    pipeline = NluPipeline(gemini_client=mock_gemini_client)

    result = await pipeline.parse_intent(
        "사당역에서 판교역까지",
        preferred_mode=RouteMode.CHEAPEST,
    )

    assert result.origin_station_name == "사당"
    assert result.dest_station_name == "판교"
    assert result.preferred_mode == RouteMode.CHEAPEST


@pytest.mark.asyncio
async def test_nlu_pipeline_llm_success_and_guardrail_pass():
    # Gemini LLM 응답 성공 및 StationValidator 통과 검증
    mock_client = MagicMock(spec=GeminiClient)
    mock_client.generate_nlu = AsyncMock(
        return_value={
            "origin": "동대문역사문화공원",
            "dest": "사당",
            "mode": "OPTIMAL",
            "reason": "소요 시간이 가장 빠른 최적 경로를 추천합니다.",
            "confidence": 0.98,
        }
    )

    pipeline = NluPipeline(gemini_client=mock_client)
    result = await pipeline.parse_intent("동대문역사문화공원에서 사당역으로 갈래")

    assert result.origin_station_name == "동대문역사문화공원"
    assert result.dest_station_name == "사당"
    assert result.preferred_mode == RouteMode.OPTIMAL
    assert result.is_fallback is False
    assert result.confidence == 0.98


@pytest.mark.asyncio
async def test_nlu_pipeline_llm_hallucination_triggers_fallback():
    # Gemini가 환각 역명("안드로메다")을 출력했을 때 가드레일이 감지하여 Fallback으로 안전 전환
    mock_client = MagicMock(spec=GeminiClient)
    mock_client.generate_nlu = AsyncMock(
        return_value={
            "origin": "안드로메다",
            "dest": "사당",
            "mode": "OPTIMAL",
            "confidence": 0.5,
        }
    )

    pipeline = NluPipeline(gemini_client=mock_client)
    # 실제 원본 문장에는 "강남역에서 사당역까지"가 들어있다고 가정
    result = await pipeline.parse_intent("강남역에서 사당역까지 가고 싶어")

    # LLM의 '안드로메다' 환각이 차단되고 문장 기반 '강남'으로 안전 복구
    assert result.origin_station_name == "강남"
    assert result.dest_station_name == "사당"
    assert result.is_fallback is True


@pytest.mark.asyncio
async def test_empty_text_is_not_turned_into_a_made_up_route(mock_gemini_client):
    """예전에는 빈 문장에 강남 -> 을지로입구를 돌려줬다. 말하지 않은 경로를 만들지 않는다"""
    from app.errors.exceptions import IntentNotUnderstoodException

    pipeline = NluPipeline(gemini_client=mock_gemini_client)
    with pytest.raises(IntentNotUnderstoodException) as error:
        await pipeline.parse_intent("   ")
    assert error.value.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "어디든 가고 싶어",  # 역이 없음
        "사당역까지 가고 싶어",  # 출발역 없음(예전에는 강남으로 채움)
    ],
)
async def test_missing_station_is_reported_instead_of_filled_in(mock_gemini_client, text):
    from app.errors.exceptions import IntentNotUnderstoodException

    pipeline = NluPipeline(gemini_client=mock_gemini_client)
    with pytest.raises(IntentNotUnderstoodException):
        await pipeline.parse_intent(text)
