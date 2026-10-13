import pytest

from app.configs.settings import settings
from app.workers import worker_main


@pytest.mark.asyncio
async def test_preload_loads_the_model_off_the_event_loop(monkeypatch):
    import threading

    called_in = []
    monkeypatch.setattr(worker_main.stt_pipeline, "load_model", lambda: called_in.append(threading.get_ident()))
    monkeypatch.setattr(settings, "whisper_preload", True)

    assert await worker_main.preload_model() is True

    # 메인(이벤트 루프) 스레드가 아니라 스레드풀에서 불렸다
    assert called_in and called_in[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_preload_is_skipped_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(worker_main.stt_pipeline, "load_model", lambda: calls.append(1))
    monkeypatch.setattr(settings, "whisper_preload", False)

    assert await worker_main.preload_model() is False
    assert calls == []


@pytest.mark.asyncio
async def test_preload_failure_does_not_stop_the_worker(monkeypatch):
    def boom():
        raise RuntimeError("model download failed")

    monkeypatch.setattr(worker_main.stt_pipeline, "load_model", boom)
    monkeypatch.setattr(settings, "whisper_preload", True)

    # 예외를 던지지 않고 False를 돌려준다. 워커는 그대로 뜨고 첫 작업에서 다시 시도한다
    assert await worker_main.preload_model() is False


def test_preload_is_on_by_default():
    assert settings.whisper_preload is True
