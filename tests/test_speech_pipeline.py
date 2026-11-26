import io
import wave
from types import SimpleNamespace

import pytest

from app.errors.exceptions import SttProcessingException
from app.pipelines.speech_pipeline import (
    GeminiSpeechProvider,
    SpeechPipeline,
    WhisperSpeechProvider,
    cooldown_seconds_for,
    parse_intent_hint,
    to_wav_16k_mono,
)


def _wav(seconds: float = 1.0, rate: int = 44100) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * int(seconds * rate))
    return buffer.getvalue()


class FakeGeminiClient:
    def __init__(self, result=None, error=None, configured=True):
        self.result = result
        self.error = error
        self.configured = configured
        self.calls = 0
        self.hint_calls = 0
        self.models = []

    def is_configured(self):
        return self.configured

    def transcribe_voice_sync(self, audio, mime_type, language="ko", model=None):
        self.calls += 1
        self.models.append(model)
        assert mime_type == "audio/wav"
        if self.error:
            raise self.error
        return self.result

    def intent_hint_sync(self, text, model=None):
        self.hint_calls += 1
        if self.error:
            raise self.error
        return self.result


class FakeStt:
    def __init__(self, text="강남역에서 사당역까지", error=None):
        self.text = text
        self.error = error
        self.calls = 0

    def transcribe(self, audio_bytes, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(transcript=self.text, duration_sec=1.5, processing_time_sec=11.6)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _pipeline(client, stt=None, clock=None, models=("m1",)):
    gemini = GeminiSpeechProvider(client=client, clock=clock or Clock(), models=list(models))
    stt = stt or FakeStt()
    return SpeechPipeline([gemini, WhisperSpeechProvider(stt)]), gemini, stt


GEMINI_OK = {
    "transcript": "신도림역에서 왕십리역까지 덜 붐비는 길",
    "origin": "신도림",
    "destination": "왕십리역",
    "mode": "COMFORTABLE",
}


@pytest.mark.asyncio
async def test_gemini_result_is_used_with_intent_hint():
    pipeline, _, stt = _pipeline(FakeGeminiClient(result=GEMINI_OK))
    result = await pipeline.transcribe(_wav())
    assert result.provider == "gemini"
    assert result.transcript == "신도림역에서 왕십리역까지 덜 붐비는 길"
    assert result.intent.to_dict() == {"origin": "신도림", "destination": "왕십리", "mode": "COMFORTABLE"}
    assert result.fallback_from == []
    assert result.duration_sec == pytest.approx(1.0, abs=0.05)
    assert stt.calls == 0


@pytest.mark.asyncio
async def test_quota_error_falls_back_to_whisper_and_pauses_gemini_for_retry_delay():
    clock = Clock()
    client = FakeGeminiClient(
        error=Exception("429 RESOURCE_EXHAUSTED. {'retryDelay': '19218s'}")
    )
    pipeline, gemini, stt = _pipeline(client, clock=clock)

    first = await pipeline.transcribe(_wav())
    assert first.provider == "whisper"
    assert first.fallback_from == ["gemini:quota"]
    assert gemini.paused_until["m1"] == pytest.approx(1000.0 + 19218)

    # 쉬는 동안에는 Gemini를 부르지 않고 바로 Whisper로 간다
    second = await pipeline.transcribe(_wav())
    assert second.fallback_from == ["gemini:paused"]
    assert client.calls == 1
    assert stt.calls == 2

    # 쉬는 시간이 지나면 다시 부른다
    clock.now += 19219
    client.error = None
    client.result = GEMINI_OK
    third = await pipeline.transcribe(_wav())
    assert third.provider == "gemini"


@pytest.mark.asyncio
async def test_unavailable_error_pauses_briefly():
    clock = Clock()
    pipeline, gemini, _ = _pipeline(
        FakeGeminiClient(error=Exception("503 UNAVAILABLE. high demand")), clock=clock
    )
    result = await pipeline.transcribe(_wav())
    assert result.fallback_from == ["gemini:unavailable"]
    assert gemini.paused_until["m1"] == pytest.approx(1000.0 + 30.0)


@pytest.mark.asyncio
async def test_other_gemini_errors_fall_back_without_pausing():
    pipeline, gemini, _ = _pipeline(FakeGeminiClient(error=ValueError("boom")))
    result = await pipeline.transcribe(_wav())
    assert result.provider == "whisper"
    assert result.fallback_from == ["gemini:error"]
    assert gemini.paused_until == {}


@pytest.mark.asyncio
async def test_gemini_timeout_falls_back(monkeypatch):
    import time as time_module

    from app.configs.settings import settings

    monkeypatch.setattr(settings, "gemini_stt_timeout_sec", 0.05)

    class SlowClient(FakeGeminiClient):
        def transcribe_voice_sync(self, audio, mime_type, language="ko", model=None):
            time_module.sleep(0.3)
            return GEMINI_OK

    pipeline, gemini, _ = _pipeline(SlowClient())
    result = await pipeline.transcribe(_wav())
    assert result.provider == "whisper"
    assert result.fallback_from == ["gemini:timeout"]
    assert gemini.paused_until == {}


@pytest.mark.asyncio
async def test_missing_key_goes_straight_to_whisper():
    client = FakeGeminiClient(configured=False)
    pipeline, _, _ = _pipeline(client)
    result = await pipeline.transcribe(_wav())
    assert result.provider == "whisper"
    assert result.fallback_from == ["gemini:no_key"]
    assert client.calls == 0


@pytest.mark.asyncio
async def test_bad_gemini_response_falls_back():
    pipeline, _, _ = _pipeline(FakeGeminiClient(result={"text": "형식이 다름"}))
    result = await pipeline.transcribe(_wav())
    assert result.fallback_from == ["gemini:bad_response"]


@pytest.mark.asyncio
async def test_whisper_failure_as_last_provider_is_raised():
    pipeline, _, _ = _pipeline(
        FakeGeminiClient(configured=False),
        stt=FakeStt(error=SttProcessingException("모델 없음")),
    )
    with pytest.raises(SttProcessingException):
        await pipeline.transcribe(_wav())


def test_from_settings_always_ends_with_whisper(monkeypatch):
    from app.configs.settings import settings

    monkeypatch.setattr(settings, "stt_providers", "gemini")
    gemini = GeminiSpeechProvider(client=FakeGeminiClient(), models=["m1"])
    pipeline = SpeechPipeline.from_settings(stt=FakeStt(), gemini=gemini)
    assert [p.name for p in pipeline.providers] == ["gemini", "whisper"]

    monkeypatch.setattr(settings, "stt_providers", "whisper")
    pipeline = SpeechPipeline.from_settings(stt=FakeStt(), gemini=gemini)
    assert [p.name for p in pipeline.providers] == ["whisper"]


def test_parse_intent_hint_cleans_station_names_and_rejects_unknown_modes():
    hint = parse_intent_hint({"origin": "강남역", "destination": " null ", "mode": "SUPER_FAST"})
    assert hint.to_dict() == {"origin": "강남", "destination": None, "mode": None}


def test_cooldown_without_retry_delay_uses_default():
    from app.configs.settings import settings

    assert cooldown_seconds_for(Exception("429 RESOURCE_EXHAUSTED")) == settings.gemini_quota_cooldown_sec
    assert cooldown_seconds_for(Exception("400 INVALID_ARGUMENT")) is None


def test_audio_is_converted_to_16k_mono_wav():
    wav, duration = to_wav_16k_mono(_wav(seconds=2.0, rate=44100))
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, 16000, 2)
    assert duration == pytest.approx(2.0, abs=0.05)


@pytest.mark.asyncio
async def test_intent_hint_shares_the_pause():
    clock = Clock()
    client = FakeGeminiClient(error=Exception("429 RESOURCE_EXHAUSTED {'retryDelay': '60s'}"))
    gemini = GeminiSpeechProvider(client=client, clock=clock, models=["m1"])
    assert await gemini.intent_hint("잠시력 가자") is None
    assert gemini.paused_until["m1"] == pytest.approx(1060.0)
    assert await gemini.intent_hint("잠시력 가자") is None
    assert client.hint_calls == 1

    clock.now += 61
    client.error = None
    client.result = {"origin": None, "destination": "잠실", "mode": "OPTIMAL"}
    hint = await gemini.intent_hint("잠시력 가자")
    assert hint.to_dict() == {"origin": None, "destination": "잠실", "mode": "OPTIMAL"}


def test_intent_hint_route_reports_unavailable(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.routes import intent_hint_route

    gemini = GeminiSpeechProvider(client=FakeGeminiClient(configured=False), models=["m1"])
    monkeypatch.setattr(intent_hint_route, "shared_gemini_provider", lambda: gemini)
    res = TestClient(app).post("/api/v1/ai/intent-hint", json={"text": "강남역 가자"})
    assert res.status_code == 200
    assert res.json() == {"available": False, "hint": None}


class PerModelClient(FakeGeminiClient):
    """모델마다 다른 결과를 낸다"""

    def __init__(self, outcomes):
        super().__init__()
        self.outcomes = outcomes

    def transcribe_voice_sync(self, audio, mime_type, language="ko", model=None):
        self.models.append(model)
        outcome = self.outcomes[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_quota_on_first_model_moves_to_next_model():
    client = PerModelClient(
        {
            "gemini-2.5-flash": Exception("429 RESOURCE_EXHAUSTED {'retryDelay': '100s'}"),
            "gemini-3.5-flash-lite": GEMINI_OK,
        }
    )
    pipeline, _, stt = _pipeline(client, models=("gemini-2.5-flash", "gemini-3.5-flash-lite"))
    result = await pipeline.transcribe(_wav())
    assert result.provider == "gemini"
    assert client.models == ["gemini-2.5-flash", "gemini-3.5-flash-lite"]
    assert stt.calls == 0
    # 다음 요청은 쉬는 모델을 건너뛰고 바로 두 번째 모델을 쓴다
    await pipeline.transcribe(_wav())
    assert client.models[-1] == "gemini-3.5-flash-lite"
    assert client.models.count("gemini-2.5-flash") == 1


@pytest.mark.asyncio
async def test_error_other_than_quota_does_not_try_next_model():
    client = PerModelClient(
        {"gemini-2.5-flash": ValueError("boom"), "gemini-3.5-flash-lite": GEMINI_OK}
    )
    pipeline, _, _ = _pipeline(client, models=("gemini-2.5-flash", "gemini-3.5-flash-lite"))
    result = await pipeline.transcribe(_wav())
    assert result.provider == "whisper"
    assert client.models == ["gemini-2.5-flash"]


@pytest.mark.asyncio
async def test_all_models_quota_falls_back_to_whisper():
    quota = Exception("429 RESOURCE_EXHAUSTED {'retryDelay': '100s'}")
    client = PerModelClient({"a": quota, "b": quota})
    pipeline, gemini, _ = _pipeline(client, models=("a", "b"))
    result = await pipeline.transcribe(_wav())
    assert result.fallback_from == ["gemini:quota"]
    assert set(gemini.paused_until) == {"a", "b"}
    second = await pipeline.transcribe(_wav())
    assert second.fallback_from == ["gemini:paused"]


def test_json_config_turns_thinking_off_per_model_generation():
    from app.clients.gemini_client import GeminiClient

    assert GeminiClient._json_config("gemini-2.5-flash").thinking_config.thinking_budget == 0
    lite = GeminiClient._json_config("gemini-3.5-flash-lite").thinking_config
    assert lite.thinking_budget is None and str(lite.thinking_level).endswith("MINIMAL")


def test_first_object_unwraps_a_single_object_array():
    from app.clients.gemini_client import first_object

    payload = {"transcript": "강남역 가자", "origin": None, "destination": "강남", "mode": "OPTIMAL"}
    assert first_object(payload) == payload
    assert first_object([payload]) == payload
    assert first_object([1, payload]) == payload
    assert first_object([]) is None
    assert first_object("강남역 가자") is None


def test_transcribe_voice_sync_accepts_array_wrapped_json():
    from types import SimpleNamespace

    from app.clients.gemini_client import GeminiClient

    client = GeminiClient(api_key="test")

    class Models:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text='[\n {"transcript": "서울역에서 잠실역까지", "origin": "서울", "destination": "잠실", "mode": "FASTEST"}\n]')

    client._client = SimpleNamespace(models=Models())
    result = client.transcribe_voice_sync(b"wav", "audio/wav", model="gemini-3.5-flash-lite")
    assert result["transcript"] == "서울역에서 잠실역까지"
    assert result["mode"] == "FASTEST"
