import sys
from types import SimpleNamespace

import pytest

from app.configs.settings import Settings
from app.errors.exceptions import InvalidAudioFormatException, SttProcessingException
from app.pipelines.stt_pipeline import SttPipeline, clean_transcript

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def test_stt_pipeline_empty_bytes_raises_exception():
    pipeline = SttPipeline()
    with pytest.raises(InvalidAudioFormatException):
        pipeline.transcribe(b"", filename="empty.wav")


def test_stt_pipeline_oversized_file_raises_exception():
    pipeline = SttPipeline()
    oversized = b"x" * (5 * 1024 * 1024 + 1)
    with pytest.raises(InvalidAudioFormatException):
        pipeline.transcribe(oversized, filename="large.wav")


def test_stt_pipeline_transcribes_valid_payload(fake_whisper):
    fake_whisper.texts = ["강남역에서 사당역까지"]

    res = SttPipeline().transcribe(WAV, filename="test.wav")

    assert res.transcript == "강남역에서 사당역까지"
    assert res.language == "ko"
    assert res.duration_sec == 2.5


def test_stt_pipeline_joins_segments(fake_whisper):
    fake_whisper.texts = ["아, 강남역 가는 길.", "강남역으로 가는 길."]

    res = SttPipeline().transcribe(WAV, filename="test.wav")

    assert res.transcript == "아, 강남역 가는 길. 강남역으로 가는 길."


def test_stt_pipeline_asks_for_korean_with_hallucination_guards(fake_whisper):
    SttPipeline().transcribe(WAV, filename="test.wav")

    kwargs = fake_whisper.calls[0]
    assert kwargs["language"] == "ko"
    assert kwargs["vad_filter"] is True
    # 환각 루프로 워커가 오래 붙잡히지 않게 하는 옵션
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_new_tokens"] == 128
    assert kwargs["no_repeat_ngram_size"] == 3


def test_no_speech_returns_empty_transcript_not_a_made_up_sentence(fake_whisper):
    fake_whisper.texts = []

    res = SttPipeline().transcribe(WAV, filename="silence.wav")

    assert res.transcript == ""
    assert "강남" not in res.transcript


def test_punctuation_only_result_is_treated_as_no_speech(fake_whisper):
    fake_whisper.texts = [",,,,", " . "]

    assert SttPipeline().transcribe(WAV, filename="noise.wav").transcript == ""


def test_repeated_punctuation_from_a_hallucination_is_collapsed(fake_whisper):
    fake_whisper.texts = ["경기도 공유시 문화,,,,,,,,,,,,,,"]

    assert SttPipeline().transcribe(WAV, filename="loop.wav").transcript == "경기도 공유시 문화,"


def test_missing_model_raises_instead_of_returning_a_fake_transcript(monkeypatch):
    pipeline = SttPipeline()
    monkeypatch.setattr(SttPipeline, "_model", None)
    monkeypatch.setattr(pipeline, "load_model", lambda: None)

    with pytest.raises(SttProcessingException):
        pipeline.transcribe(WAV, filename="test.wav")


def test_model_is_loaded_with_the_configured_size_and_one_cpu_thread(monkeypatch):
    created = {}

    class RecordingModel:
        def __init__(self, size, **kwargs):
            created["size"] = size
            created.update(kwargs)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=RecordingModel))
    monkeypatch.setattr(SttPipeline, "_model", None)

    SttPipeline().load_model()

    assert created["size"] == "small"
    assert created["compute_type"] == "int8"
    # 컨테이너 CPU 한도(0.6코어)보다 스레드가 많으면 스로틀링으로 느려진다
    assert created["cpu_threads"] == 1


def test_settings_defaults():
    settings = Settings()

    assert settings.whisper_model_size == "small"
    assert settings.whisper_cpu_threads == 1
    assert settings.whisper_max_new_tokens == 128
    assert settings.whisper_no_repeat_ngram_size == 3


class TestCleanTranscript:
    def test_collapses_repeated_punctuation(self):
        assert clean_transcript("가나다,,,,라마") == "가나다,라마"
        assert clean_transcript("정말???") == "정말?"

    def test_keeps_single_punctuation(self):
        assert clean_transcript("아, 강남역 가는 길.") == "아, 강남역 가는 길."

    def test_punctuation_and_spaces_only_is_empty(self):
        assert clean_transcript(" , . ") == ""
        assert clean_transcript("...") == ""
        assert clean_transcript("") == ""

    def test_strips_surrounding_spaces(self):
        assert clean_transcript("  강남역  ") == "강남역"
