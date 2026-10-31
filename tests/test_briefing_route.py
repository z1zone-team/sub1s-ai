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
