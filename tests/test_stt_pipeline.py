import pytest
from app.errors.exceptions import InvalidAudioFormatException
from app.pipelines.stt_pipeline import SttPipeline


def test_stt_pipeline_empty_bytes_raises_exception():
    pipeline = SttPipeline()
    with pytest.raises(InvalidAudioFormatException):
        pipeline.transcribe(b"", filename="empty.wav")


def test_stt_pipeline_oversized_file_raises_exception():
    pipeline = SttPipeline()
    oversized = b"x" * (5 * 1024 * 1024 + 1)
    with pytest.raises(InvalidAudioFormatException):
        pipeline.transcribe(oversized, filename="large.wav")


def test_stt_pipeline_transcribes_valid_payload():
    pipeline = SttPipeline()
    fake_audio = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    res = pipeline.transcribe(fake_audio, filename="test.wav")
    assert res.transcript is not None
    assert len(res.transcript) > 0
    assert res.language == "ko"
