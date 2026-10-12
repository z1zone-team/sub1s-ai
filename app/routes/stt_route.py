from fastapi import APIRouter, File, UploadFile, Query
from fastapi.concurrency import run_in_threadpool
from app.dtos.response_dto import SttResponseDto
from app.errors.exceptions import InvalidAudioFormatException
from app.pipelines.stt_pipeline import stt_pipeline

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

    # 전사는 동기이고 CPU를 오래 써서 이벤트 루프를 막지 않도록 스레드풀에서 돌린다
    return await run_in_threadpool(
        stt_pipeline.transcribe,
        audio_bytes=content,
        filename=filename,
        language=language,
    )
