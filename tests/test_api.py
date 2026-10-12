import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_stt_endpoint_unsupported_extension_returns_400():
    fake_file = io.BytesIO(b"malicious content")
    response = client.post(
        "/api/v1/ai/stt",
        files={"file": ("virus.exe", fake_file, "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "지원하지 않는 오디오 확장자" in data["error"]["message"]


def test_stt_endpoint_valid_audio_file_returns_transcript(fake_whisper):
    fake_whisper.texts = ["강남역에서 사당역까지"]
    fake_wav = io.BytesIO(b"RIFF....WAVEfmt ....data....")
    response = client.post(
        "/api/v1/ai/stt",
        files={"file": ("sample.wav", fake_wav, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["transcript"] == "강남역에서 사당역까지"
    assert data["language"] == "ko"


def test_stt_endpoint_without_speech_returns_empty_transcript(fake_whisper):
    fake_whisper.texts = []
    response = client.post(
        "/api/v1/ai/stt",
        files={"file": ("silence.wav", io.BytesIO(b"RIFF....WAVEfmt ....data...."), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["transcript"] == ""
