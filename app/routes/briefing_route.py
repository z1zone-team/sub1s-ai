import base64
import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse

from app.dtos.request_dto import (
    BriefingRequestDto,
    TtsRequestDto,
    VoiceSummaryRequestDto,
)
from app.dtos.response_dto import (
    BriefingResponseDto,
    VoiceSummaryResponseDto,
)
from app.pipelines.briefing_pipeline import BriefingPipeline
from app.pipelines.tts_pipeline import TtsPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["Briefing & TTS"])


def get_briefing_pipeline() -> BriefingPipeline:
    return BriefingPipeline.get_instance()


def get_tts_pipeline() -> TtsPipeline:
    return TtsPipeline.get_instance()


@router.post(
    "/briefing",
    response_model=BriefingResponseDto,
    status_code=status.HTTP_200_OK,
    summary="다익스트라 경로 추천 기반 자연어 음성 브리핑 생성",
)
async def create_briefing(
    payload: BriefingRequestDto,
    pipeline: BriefingPipeline = Depends(get_briefing_pipeline),
) -> BriefingResponseDto:
    logger.info(
        f"[POST /api/v1/ai/briefing] 요청: {payload.origin_station_name} -> {payload.dest_station_name} ({payload.preferred_mode})"
    )
    return await pipeline.generate_briefing(
        origin_station_name=payload.origin_station_name,
        dest_station_name=payload.dest_station_name,
        preferred_mode=payload.preferred_mode,
        route=payload.route,
    )


@router.post(
    "/tts",
    summary="텍스트 기반 Edge-TTS MP3 오디오 스트리밍",
    response_class=StreamingResponse,
)
async def synthesize_tts(
    payload: TtsRequestDto,
    pipeline: TtsPipeline = Depends(get_tts_pipeline),
) -> StreamingResponse:
    logger.info(f"[POST /api/v1/ai/tts] 요청: text='{payload.text[:30]}...'")
    stream = pipeline.stream_audio_chunks(text=payload.text, voice=payload.voice)
    return StreamingResponse(
        stream,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )


@router.get(
    "/tts/stream",
    summary="GET 쿼리 스트리밍용 Edge-TTS 엔드포인트",
    response_class=StreamingResponse,
)
async def stream_tts(
    text: str = Query(..., min_length=1, max_length=1000, description="합성할 텍스트"),
    voice: Optional[str] = Query(None, description="음성 모델"),
    pipeline: TtsPipeline = Depends(get_tts_pipeline),
) -> StreamingResponse:
    logger.info(f"[GET /api/v1/ai/tts/stream] 요청: text='{text[:30]}...'")
    stream = pipeline.stream_audio_chunks(text=text, voice=voice)
    return StreamingResponse(
        stream,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )


@router.post(
    "/voice-summary",
    response_model=VoiceSummaryResponseDto,
    status_code=status.HTTP_200_OK,
    summary="브리핑 텍스트 생성 및 MP3 음성 원스톱 합성",
)
async def create_voice_summary(
    payload: VoiceSummaryRequestDto,
    briefing_pipeline: BriefingPipeline = Depends(get_briefing_pipeline),
    tts_pipeline: TtsPipeline = Depends(get_tts_pipeline),
) -> VoiceSummaryResponseDto:
    logger.info(
        f"[POST /api/v1/ai/voice-summary] 요청: {payload.origin_station_name} -> {payload.dest_station_name}"
    )
    briefing = await briefing_pipeline.generate_briefing(
        origin_station_name=payload.origin_station_name,
        dest_station_name=payload.dest_station_name,
        preferred_mode=payload.preferred_mode,
        route=payload.route,
    )

    audio_base64 = None
    if payload.include_base64:
        mp3_bytes = await tts_pipeline.synthesize_mp3_bytes(
            text=briefing.briefing_text,
            voice=payload.voice,
        )
        audio_base64 = base64.b64encode(mp3_bytes).decode("utf-8")

    return VoiceSummaryResponseDto(
        briefing_text=briefing.briefing_text,
        audio_base64=audio_base64,
        audio_tts_url=briefing.audio_tts_url,
    )
