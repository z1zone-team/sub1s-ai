from unittest.mock import MagicMock
import pytest
from bullmq import Job

from app.workers.voice_worker import VoiceWorker


@pytest.fixture
def worker():
    return VoiceWorker(
        valkey_host="127.0.0.1",
        valkey_port=6399,  # 존재하지 않는 포트로 안전하게 격리
    )


@pytest.mark.asyncio
async def test_worker_valkey_unavailable_graceful(worker):
    # Valkey가 미실행 상태여도 에러 없이 False를 반환하는 Fault Tolerance 검증
    available = await worker.is_valkey_available()
    assert available is False

    started = await worker.start()
    assert started is False
    assert worker._is_running is False

    # stop() 호출 시에도 크래시 없음 검증
    await worker.stop()


@pytest.mark.asyncio
async def test_worker_process_voice_job(worker):
    # BullMQ 잡 프로세서 voice_process 액션 검증
    mock_job = MagicMock(spec=Job)
    mock_job.id = "job-1"
    mock_job.name = "voice_process"
    mock_job.data = {
        "action": "voice_process",
        "transcript": "강남역에서 사당역까지",
        "preferredMode": "OPTIMAL",
    }

    result = await worker.process_job(mock_job)
    assert result["originStationName"] == "강남"
    assert result["destStationName"] == "사당"
    assert result["preferredMode"] == "OPTIMAL"


@pytest.mark.asyncio
async def test_worker_process_tts_job(worker):
    # BullMQ 잡 프로세서 tts 액션 검증
    mock_job = MagicMock(spec=Job)
    mock_job.id = "job-2"
    mock_job.name = "tts"
    mock_job.data = {
        "action": "tts",
        "text": "안내를 시작합니다.",
    }

    result = await worker.process_job(mock_job)
    assert "audioBase64" in result
    assert result["bytesLength"] > 1000


@pytest.mark.asyncio
async def test_worker_process_unsupported_job(worker):
    # 미지원 액션 안전 처리 검증
    mock_job = MagicMock(spec=Job)
    mock_job.id = "job-3"
    mock_job.name = "unknown"
    mock_job.data = {"action": "unsupported_action"}

    result = await worker.process_job(mock_job)
    assert result["status"] == "unsupported_action"
