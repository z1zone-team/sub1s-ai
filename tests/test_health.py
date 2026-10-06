from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "sub1s-ai"
    assert "whisper_model" in data
    assert "stationCount" in data
    assert data["stationCount"] > 600
    assert "valkeyStatus" in data
    assert data["valkeyStatus"] in ("connected", "disconnected")
