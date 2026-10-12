import io
import re
import time
import logging
from typing import Optional, Tuple
from app.configs.settings import settings
from app.dtos.response_dto import SttResponseDto
from app.errors.exceptions import InvalidAudioFormatException, SttProcessingException

logger = logging.getLogger(__name__)

_REPEATED_PUNCTUATION = re.compile(r"([,.?!、。])\1+")
_ONLY_PUNCTUATION = re.compile(r"^[\s,.?!、。…~·\-]*$")


def clean_transcript(text: str) -> str:
    """
    전사 결과에서 같은 문장부호가 연달아 반복된 것(`,,,,`)을 하나로 줄이고,
    문장부호와 공백만 남았으면 빈 문자열로 본다 (말소리가 없었다는 뜻)
    """
    collapsed = _REPEATED_PUNCTUATION.sub(r"\1", text).strip()
    return "" if _ONLY_PUNCTUATION.match(collapsed) else collapsed


class SttPipeline:
    """
    faster-whisper 기반 로컬 CPU 고속 음성 인식(STT) 파이프라인.
    1 vCPU 환경에 최적화된 int8 양자화 및 싱글톤 모델 캐싱을 지원합니다.
    """
    _instance: Optional["SttPipeline"] = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SttPipeline, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # 싱글톤 초기화는 1회만 수행
        pass

    def load_model(self):
        """Whisper 모델을 인메모리에 로드합니다."""
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel

            started = time.time()
            logger.info(
                f"Loading faster-whisper model: {settings.whisper_model_size} "
                f"(device={settings.whisper_device}, compute_type={settings.whisper_compute_type})"
            )
            self._model = WhisperModel(
                settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                cpu_threads=settings.whisper_cpu_threads,
            )
            logger.info(f"faster-whisper model loaded in {time.time() - started:.2f}s")
            return self._model
        except Exception as e:
            logger.warning(
                f"faster-whisper loading failed or library not available: {e}. "
                "Fallback mode will be used if needed."
            )
            return None

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "",
        language: str = "ko",
    ) -> SttResponseDto:
        """
        오디오 바이트 스트림을 전사하여 텍스트 및 메타데이터를 반환합니다.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise InvalidAudioFormatException("오디오 데이터가 비어 있습니다.")

        # 5MB 초과 제한 검사
        if len(audio_bytes) > 5 * 1024 * 1024:
            raise InvalidAudioFormatException("오디오 파일 크기는 최대 5MB를 초과할 수 없습니다.")

        model = self.load_model()
        started_at = time.time()

        if model is None:
            # 모델이 없을 때 가짜 문장을 돌려주면 말을 하지 않아도 경로가 나와서 오류를 숨긴다. 오류로 알린다
            logger.error("Whisper model is not available. Refusing to return a made-up transcript.")
            raise SttProcessingException("음성 인식 모델을 사용할 수 없습니다.")

        try:
            audio_stream = io.BytesIO(audio_bytes)
            segments, info = model.transcribe(
                audio_stream,
                language=language,
                beam_size=1,  # 고속 처리를 위한 Greedy search
                vad_filter=True,  # 무음 구간 제거
                # 환각 루프 방지: 온도를 올려 다시 해독하는 재시도를 끄고(재시도가 구간마다 반복돼 수십 초가 걸렸다),
                # 만들 수 있는 토큰 수와 반복을 제한한다
                temperature=0.0,
                max_new_tokens=settings.whisper_max_new_tokens,
                no_repeat_ngram_size=settings.whisper_no_repeat_ngram_size,
            )

            collected_texts = []
            for segment in segments:
                collected_texts.append(segment.text.strip())

            # 말소리가 없으면 빈 전사다. 가짜 문장으로 채우지 않는다
            transcript = clean_transcript(" ".join(collected_texts))

            processing_time = time.time() - started_at
            return SttResponseDto(
                transcript=transcript,
                language=info.language if hasattr(info, "language") else language,
                duration_sec=round(getattr(info, "duration", 0.0), 2),
                processing_time_sec=round(processing_time, 3),
                confidence=round(getattr(info, "language_probability", 1.0), 2),
            )
        except Exception as e:
            logger.error(f"STT transcription failed: {e}", exc_info=True)
            raise SttProcessingException(f"음성 전사 처리 실패: {str(e)}")


# 전역 싱글톤 인스턴스
stt_pipeline = SttPipeline()
