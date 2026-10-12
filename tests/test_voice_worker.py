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


class _BlockingStt:
    """전사에 시간이 걸리는 동기 STT를 흉내 낸다 (Whisper처럼 CPU를 오래 쓰는 호출)"""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.thread_ids = []

    def transcribe(self, audio_bytes: bytes, **_kwargs):
        import threading
        import time
        from types import SimpleNamespace

        self.thread_ids.append(threading.get_ident())
        time.sleep(self.seconds)
        return SimpleNamespace(transcript="강남역에서 사당역까지", duration_sec=1.5)


@pytest.mark.asyncio
async def test_worker_stt_does_not_block_event_loop():
    """STT가 도는 동안 같은 프로세스의 다른 코루틴(헬스체크 등)이 계속 실행돼야 한다"""
    import asyncio
    import base64
    import time

    stt = _BlockingStt(seconds=0.4)
    worker = VoiceWorker(valkey_host="127.0.0.1", valkey_port=6399, stt=stt)
    job = MagicMock(spec=Job)
    job.id = "job-stt"
    job.name = "stt"
    job.data = {"action": "stt", "audioBase64": base64.b64encode(b"audio").decode()}

    ticks = []

    async def heartbeat():
        # 이벤트 루프가 막히면 0.4초 동안 한 번도 못 돈다
        end = time.monotonic() + 0.4
        while time.monotonic() < end:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.02)

    result, _ = await asyncio.gather(worker.process_job(job), heartbeat())

    assert result == {"transcript": "강남역에서 사당역까지", "durationSec": 1.5}
    assert len(ticks) >= 10, f"이벤트 루프가 막혔습니다 (heartbeat {len(ticks)}회)"


@pytest.mark.asyncio
async def test_worker_stt_runs_off_the_main_thread():
    import base64
    import threading

    stt = _BlockingStt(seconds=0.0)
    worker = VoiceWorker(valkey_host="127.0.0.1", valkey_port=6399, stt=stt)
    job = MagicMock(spec=Job)
    job.id = "job-stt"
    job.name = "stt"
    job.data = {"action": "stt", "audio_base64": base64.b64encode(b"audio").decode()}

    await worker.process_job(job)

    assert stt.thread_ids and stt.thread_ids[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_worker_stt_without_audio_returns_error():
    worker = VoiceWorker(valkey_host="127.0.0.1", valkey_port=6399)
    job = MagicMock(spec=Job)
    job.id = "job-stt"
    job.name = "stt"
    job.data = {"action": "stt"}

    assert await worker.process_job(job) == {"error": "Missing audio_base64 in payload"}


def test_worker_concurrency_defaults_to_one():
    worker = VoiceWorker(valkey_host="127.0.0.1", valkey_port=6399)

    assert worker.concurrency == 1


def test_worker_concurrency_can_be_set_and_never_below_one():
    assert VoiceWorker(valkey_host="127.0.0.1", valkey_port=6399, concurrency=3).concurrency == 3
    assert VoiceWorker(valkey_host="127.0.0.1", valkey_port=6399, concurrency=-2).concurrency == 1
