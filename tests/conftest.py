from types import SimpleNamespace

import pytest

from app.pipelines.stt_pipeline import SttPipeline


class FakeWhisperModel:
    """실제 Whisper 모델 대신 쓰는 가짜. 테스트가 모델을 내려받거나 CPU를 쓰지 않게 하고 호출 인자를 기록한다"""

    def __init__(self, texts=("강남역에서 사당역까지",)):
        self.texts = list(texts)
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        segments = [SimpleNamespace(text=text) for text in self.texts]
        info = SimpleNamespace(language="ko", duration=2.5, language_probability=0.97)
        return iter(segments), info


def _forget_loaded_model():
    """싱글톤 인스턴스에 `load_model`이 남긴 `_model`을 지운다. 지우지 않으면 뒤 테스트가 클래스에 둔 가짜 모델 대신 이것을 쓴다"""
    SttPipeline().__dict__.pop("_model", None)


@pytest.fixture
def fake_whisper(monkeypatch):
    """STT 파이프라인이 가짜 모델을 쓰게 한다. 테스트에서 `fake_whisper.texts`를 바꿔 전사 결과를 정한다"""
    model = FakeWhisperModel()
    _forget_loaded_model()
    monkeypatch.setattr(SttPipeline, "_model", model)
    yield model
    _forget_loaded_model()


@pytest.fixture(autouse=True)
def _default_fake_whisper(request, monkeypatch):
    """모든 테스트의 기본값. 모델이 필요 없는 테스트도 실수로 실제 모델을 내려받지 않게 한다"""
    if "fake_whisper" in request.fixturenames:
        yield
        return
    _forget_loaded_model()
    monkeypatch.setattr(SttPipeline, "_model", FakeWhisperModel())
    yield
    _forget_loaded_model()
