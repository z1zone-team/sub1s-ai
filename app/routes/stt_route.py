from fastapi import APIRouter, File, UploadFile, Query
from app.dtos.response_dto import IntentHintDto, SttResponseDto
from app.errors.exceptions import InvalidAudioFormatException
from app.pipelines.speech_pipeline import SpeechPipeline

router = APIRouter(prefix="/api/v1/ai", tags=["STT"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}


@router.post("/stt", response_model=SttResponseDto)
async def transcribe_audio(
    file: UploadFile = File(..., description="변환할 오디오 파일 (wav, mp3, m4a, webm)"),
    language: str = Query(default="ko", description="음성 언어 코드"),
):
    filename = file.filename or "audio.wav"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

    if ext and ext not in ALLOWED_EXTENSIONS:
        raise InvalidAudioFormatException(
            f"지원하지 않는 오디오 확장자입니다: {ext} (허용: {', '.join(ALLOWED_EXTENSIONS)})"
        )

    content = await file.read()
    if not content:
        raise InvalidAudioFormatException("업로드된 오디오 파일이 비어 있습니다.")

    # 큐 워커와 같은 제공자 순서를 쓴다(기본 gemini, whisper). Whisper는 스레드에서 돌아 이벤트 루프를 막지 않는다
    result = await _speech().transcribe(content, language=language)
    return SttResponseDto(
        transcript=result.transcript,
        language=language,
        duration_sec=result.duration_sec,
        processing_time_sec=result.processing_time_sec,
        provider=result.provider,
        fallback_from=result.fallback_from,
        intent=IntentHintDto(**result.intent.to_dict()) if result.intent else None,
    )


_pipeline = None


def _speech() -> SpeechPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SpeechPipeline.from_settings()
    return _pipeline
