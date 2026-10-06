import pytest
from app.pipelines.tts_pipeline import TtsPipeline


@pytest.fixture
def tts_pipeline():
    return TtsPipeline.get_instance()


@pytest.mark.asyncio
async def test_tts_pipeline_synthesize_mp3(tts_pipeline):
    # 실제 Edge-TTS 비동기 MP3 바이너리 합성 검증
    audio_bytes = await tts_pipeline.synthesize_mp3_bytes("안전한 이동 되세요.")

    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 1000  # MP3 바이너리가 생성되어 크기 확보


@pytest.mark.asyncio
async def test_tts_pipeline_stream_audio_chunks(tts_pipeline):
    # 스트리밍 제너레이터 검증
    chunks = []
    async for chunk in tts_pipeline.stream_audio_chunks("안녕하세요."):
        chunks.append(chunk)

    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 1000


@pytest.mark.asyncio
async def test_tts_pipeline_empty_text(tts_pipeline):
    # 빈 텍스트 요청 시 빈 바이트 반환
    assert await tts_pipeline.synthesize_mp3_bytes("") == b""
