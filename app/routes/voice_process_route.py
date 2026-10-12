import io
import logging
import os
import time
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.dtos.request_dto import RouteMode
from app.dtos.response_dto import VoiceProcessResponseDto
from app.errors.exceptions import InvalidAudioFormatException
from app.pipelines.nlu_pipeline import NluPipeline
from app.pipelines.stt_pipeline import SttPipeline, stt_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["Voice Processing"])

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm"}
MAX_AUDIO_SIZE_BYTES = 5 * 1024 * 1024  # 5MB


def get_nlu_pipeline() -> NluPipeline:
    return NluPipeline.get_instance()


def get_stt_pipeline() -> SttPipeline:
    return stt_pipeline


@router.post(
    "/voice-process",
    response_model=VoiceProcessResponseDto,
    status_code=status.HTTP_200_OK,
    summary="음성 오디오 파일 수신 및 STT + NLU 원스톱 동기 처리",
    description="""
    음성 파일(Multipart)을 수신받아 로컬 faster-whisper STT 전사와
    Zero-Token Guardrail 기반 NLU 의도 분석(출발역, 도착역, 추천 모드 슬롯 필링)을 단일 요청으로 원스톱 처리합니다.
    - 최대 5MB 크기 제한 및 wav/mp3/m4a/webm 지원
    - transcript_hint 입력 시 전사문 힌트 우선 반영
    - 1.5초 이내 고속 응답 및 장애 격리(Fault Tolerance)
    """,
)
async def process_voice(
    file: Optional[UploadFile] = File(None, description="음성 녹음 파일 (최대 5MB)"),
    transcript_hint: Optional[str] = Form(
        None, alias="transcriptHint", description="클라이언트 음성 텍스트 힌트 (선택)"
    ),
    preferred_mode: Optional[RouteMode] = Form(
        None, alias="preferredMode", description="사용자 지정 우선 탐색 모드 (선택)"
    ),
    stt: SttPipeline = Depends(get_stt_pipeline),
    nlu: NluPipeline = Depends(get_nlu_pipeline),
) -> VoiceProcessResponseDto:
    start_time = time.time()
    final_transcript = ""

    # 1. 오디오 파일 유효성 검사 및 STT 처리
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise InvalidAudioFormatException(
                f"지원하지 않는 오디오 확장자입니다: {ext} (허용: {list(ALLOWED_AUDIO_EXTENSIONS)})"
            )

        content = await file.read()
        if len(content) == 0:
            raise InvalidAudioFormatException("업로드된 오디오 파일이 비어 있습니다.")
        if len(content) > MAX_AUDIO_SIZE_BYTES:
            raise InvalidAudioFormatException("오디오 파일 크기는 최대 5MB를 초과할 수 없습니다.")

        # STT 음성 전사
        # 전사는 동기이고 CPU를 오래 써서 이벤트 루프를 막지 않도록 스레드풀에서 돌린다
        stt_result = await run_in_threadpool(stt.transcribe, audio_bytes=content, filename=file.filename)
        final_transcript = stt_result.transcript.strip()

    # 2. transcript_hint가 제공된 경우 보정 또는 힌트 우선 적용
    if transcript_hint and transcript_hint.strip():
        final_transcript = transcript_hint.strip()

    if not final_transcript:
        raise InvalidAudioFormatException(
            "오디오 파일(file) 또는 텍스트 힌트(transcriptHint) 중 하나는 반드시 제공되어야 합니다."
        )

    # 3. NLU 의도 분석 및 슬롯 필링 (Zero-Token Station Guardrail 적용)
    nlu_result = await nlu.parse_intent(
        text=final_transcript,
        preferred_mode=preferred_mode,
    )

    elapsed = round(time.time() - start_time, 4)
    logger.info(
        f"[POST /api/v1/ai/voice-process] 처리 완료: transcript='{final_transcript}', time={elapsed}s"
    )

    return VoiceProcessResponseDto(
        transcript=final_transcript,
        origin_station_name=nlu_result.origin_station_name,
        dest_station_name=nlu_result.dest_station_name,
        preferred_mode=nlu_result.preferred_mode,
        reason=nlu_result.reason,
        confidence=nlu_result.confidence,
        is_fallback=nlu_result.is_fallback,
        processing_time_sec=elapsed,
    )
