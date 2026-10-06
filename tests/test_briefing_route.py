import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_briefing_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ai/briefing",
            json={
                "originStationName": "강남",
                "destStationName": "을지로입구",
                "preferredMode": "OPTIMAL",
                "route": {
                    "totalDurationSec": 1800,
                    "transferCount": 1,
                    "segments": [],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "briefingText" in data
        assert "강남역에서 을지로입구역까지" in data["briefingText"]
        assert "audioTtsUrl" in data


@pytest.mark.asyncio
async def test_tts_streaming_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ai/tts",
            json={
                "text": "경로 안내를 시작합니다.",
            },
        )
        assert response.status_code == 200
        assert "audio/mpeg" in response.headers["content-type"]
        assert len(response.content) > 1000


@pytest.mark.asyncio
async def test_tts_get_stream_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/ai/tts/stream?text=안전운행되세요",
        )
        assert response.status_code == 200
        assert "audio/mpeg" in response.headers["content-type"]
        assert len(response.content) > 1000


@pytest.mark.asyncio
async def test_voice_summary_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ai/voice-summary",
            json={
                "originStationName": "사당",
                "destStationName": "판교",
                "preferredMode": "COMFORTABLE",
                "includeBase64": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "briefingText" in data
        assert data["audioBase64"] is not None
        assert len(data["audioBase64"]) > 500
