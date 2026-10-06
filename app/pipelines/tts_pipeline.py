import logging
from typing import AsyncGenerator, Optional

import edge_tts

from app.configs.settings import settings

logger = logging.getLogger(__name__)


class TtsPipeline:
    """
    Edge-TTS 기반 고품질 신경망 음성 합성 파이프라인.
    Microsoft Edge 음성 서비스(ko-KR-SunHiNeural)를 비동기 스트리밍 호출하여
    0.3~0.5초 이내 고속 MP3 바이너리 합성 및 청크 스트리밍을 제공합니다.
    """

    _instance: Optional["TtsPipeline"] = None

    def __init__(self, default_voice: Optional[str] = None):
        self.default_voice = default_voice or settings.tts_voice

    @classmethod
    def get_instance(cls) -> "TtsPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def synthesize_mp3_bytes(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> bytes:
        """
        주어진 텍스트를 전체 MP3 바이너리(bytes)로 합성하여 반환합니다.
        """
        cleaned = text.strip()
        if not cleaned:
            return b""

        target_voice = voice or self.default_voice
        communicate = edge_tts.Communicate(
            text=cleaned,
            voice=target_voice,
            rate=settings.tts_rate,
            volume=settings.tts_volume,
            pitch=settings.tts_pitch,
        )

        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        total_bytes = b"".join(audio_chunks)
        logger.info(
            f"[TtsPipeline] 음성 합성 완료: {len(cleaned)}자 -> {len(total_bytes)} 바이트"
        )
        return total_bytes

    async def stream_audio_chunks(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        주어진 텍스트를 스트리밍 가능한 오디오 바이너리 청크 단위로 비동기 제너레이트합니다.
        FastAPI StreamingResponse에 직접 전달됩니다.
        """
        cleaned = text.strip()
        if not cleaned:
            return

        target_voice = voice or self.default_voice
        communicate = edge_tts.Communicate(
            text=cleaned,
            voice=target_voice,
            rate=settings.tts_rate,
            volume=settings.tts_volume,
            pitch=settings.tts_pitch,
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
