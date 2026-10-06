from unittest.mock import AsyncMock, MagicMock
import pytest

from app.clients.gemini_client import GeminiClient
from app.dtos.request_dto import PathSegmentDto, RouteMode, SubwayRouteDto
from app.pipelines.briefing_pipeline import BriefingPipeline


@pytest.fixture
def sample_route():
    return SubwayRouteDto(
        mode=RouteMode.OPTIMAL,
        totalDurationSec=1500,  # 25분
        totalDistanceKm=14.5,
        totalFare=1400,
        transferCount=1,
        segments=[
            PathSegmentDto(
                lineId="02",
                fromStationName="강남",
                toStationName="사당",
                stationCount=4,
                durationSec=600,
                recommendedCarNo=4,
            ),
            PathSegmentDto(
                lineId="04",
                fromStationName="사당",
                toStationName="을지로입구",
                stationCount=8,
                durationSec=900,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_briefing_pipeline_template_generation(sample_route):
    # Gemini가 None을 반환(SLA 타임아웃/쿼터 소진) 시 규칙 기반 템플릿 검증
    mock_gemini = MagicMock(spec=GeminiClient)
    mock_gemini.generate_briefing = AsyncMock(return_value=None)

    pipeline = BriefingPipeline(gemini_client=mock_gemini)
    res = await pipeline.generate_briefing(
        origin_station_name="강남",
        dest_station_name="을지로입구",
        preferred_mode=RouteMode.OPTIMAL,
        route=sample_route,
    )

    assert "강남역에서 을지로입구역까지 최적 경로로 안내해 드릴게요." in res.briefing_text
    assert "총 약 25분 소요되며 환승은 1회 필요해요." in res.briefing_text
    assert "4번 칸에 탑승하시면 빠른 환승이 가능해요." in res.briefing_text
    assert res.audio_tts_url is not None
    assert "/api/v1/ai/tts/stream?text=" in res.audio_tts_url
    assert res.is_llm_generated is False


@pytest.mark.asyncio
async def test_briefing_pipeline_direct_route():
    # 환승 0회 직통 경로 안내 검증
    mock_gemini = MagicMock(spec=GeminiClient)
    mock_gemini.generate_briefing = AsyncMock(return_value=None)

    direct_route = SubwayRouteDto(
        mode=RouteMode.COMFORTABLE,
        totalDurationSec=600,  # 10분
        transferCount=0,
        segments=[],
    )

    pipeline = BriefingPipeline(gemini_client=mock_gemini)
    res = await pipeline.generate_briefing(
        origin_station_name="강남",
        dest_station_name="역삼",
        preferred_mode=RouteMode.COMFORTABLE,
        route=direct_route,
    )

    assert "여유 좌석 경로" in res.briefing_text
    assert "환승 없이 직통으로 이동해요." in res.briefing_text


@pytest.mark.asyncio
async def test_briefing_pipeline_llm_success(sample_route):
    # Gemini LLM 응답 성공 케이스 검증
    mock_gemini = MagicMock(spec=GeminiClient)
    mock_gemini.generate_briefing = AsyncMock(
        return_value="강남역에서 을지로입구역까지 최적 경로로 이동합니다. 4번 칸을 이용하시면 빠른 환승이 가능해요."
    )

    pipeline = BriefingPipeline(gemini_client=mock_gemini)
    res = await pipeline.generate_briefing(
        origin_station_name="강남",
        dest_station_name="을지로입구",
        preferred_mode=RouteMode.OPTIMAL,
        route=sample_route,
    )

    assert res.is_llm_generated is True
    assert "빠른 환승" in res.briefing_text
