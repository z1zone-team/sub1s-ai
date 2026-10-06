import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_nlu_endpoint_success():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ai/nlu",
            json={
                "text": "강남역에서 을지로입구역까지 편하게 가고 싶어",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("originStationName") == "강남" or data.get("origin_station_name") == "강남"
        assert data.get("destStationName") == "을지로입구" or data.get("dest_station_name") == "을지로입구"
        assert data.get("preferredMode") == "COMFORTABLE" or data.get("preferred_mode") == "COMFORTABLE"
        assert "reason" in data


@pytest.mark.asyncio
async def test_nlu_endpoint_with_mode_override():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ai/nlu",
            json={
                "text": "사당역에서 판교역까지",
                "preferredMode": "CHEAPEST",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("originStationName") == "사당" or data.get("origin_station_name") == "사당"
        assert data.get("destStationName") == "판교" or data.get("dest_station_name") == "판교"
        assert data.get("preferredMode") == "CHEAPEST" or data.get("preferred_mode") == "CHEAPEST"


@pytest.mark.asyncio
async def test_nlu_endpoint_validation_error():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 빈 텍스트 전송 시 유효성 검사 에러 (422)
        response = await client.post(
            "/api/v1/ai/nlu",
            json={
                "text": "",
            },
        )
        assert response.status_code == 422
