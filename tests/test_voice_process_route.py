import io
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_voice_process_with_transcript_hint():
    # 텍스트 힌트를 포함한 동기 통합 음성 처리 검증
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ai/voice-process",
            data={
                "transcriptHint": "강남역에서 을지로입구역까지 편하게 가고 싶어",
                "preferredMode": "COMFORTABLE",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "강남역에서 을지로입구역까지 편하게 가고 싶어"
        assert data["originStationName"] == "강남"
        assert data["destStationName"] == "을지로입구"
        assert data["preferredMode"] == "COMFORTABLE"
        assert data["processingTimeSec"] >= 0.0


@pytest.mark.asyncio
async def test_voice_process_with_audio_file():
    # 실제 WAV 오디오 파일 업로드 통합 처리 검증
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 빈 WAV 헤더 생성 (44 bytes)
        wav_header = (
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
            b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        files = {"file": ("test.wav", io.BytesIO(wav_header), "audio/wav")}
        data = {"preferredMode": "OPTIMAL"}

        response = await client.post(
            "/api/v1/ai/voice-process",
            files=files,
            data=data,
        )
        assert response.status_code == 200
        result = response.json()
        assert "transcript" in result
        assert "originStationName" in result
        assert "destStationName" in result
        assert result["preferredMode"] == "OPTIMAL"


@pytest.mark.asyncio
async def test_voice_process_invalid_extension():
    # 미허용 확장자(.exe) 차단 400 검증
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("malicious.exe", io.BytesIO(b"fake"), "application/octet-stream")}
        response = await client.post(
            "/api/v1/ai/voice-process",
            files=files,
        )
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_voice_process_missing_both_file_and_hint():
    # 파일과 힌트가 모두 누락된 경우 400 검증
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ai/voice-process",
            data={},
        )
        assert response.status_code == 400
