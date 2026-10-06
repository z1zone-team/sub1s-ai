from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from app.dtos.request_dto import RouteMode


class HealthResponseDto(BaseModel):
    status: str = Field(default="ok", description="서버 상태")
    service: str = Field(default="sub1s-ai", description="서비스명")
    whisper_model: str = Field(description="로드된 Whisper 모델")


class SttResponseDto(BaseModel):
    transcript: str = Field(description="STT 전사 텍스트")
    language: str = Field(default="ko", description="감지된 언어 코드")
    duration_sec: float = Field(default=0.0, description="오디오 길이(초)")
    processing_time_sec: float = Field(default=0.0, description="STT 처리 소요 시간(초)")
    confidence: Optional[float] = Field(default=None, description="인식 신뢰도")


class NluResponseDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    origin_station_name: str = Field(
        ...,
        alias="originStationName",
        description="출발역 이름 (정규화된 공식 역명)",
    )
    dest_station_name: str = Field(
        ...,
        alias="destStationName",
        description="도착역 이름 (정규화된 공식 역명)",
    )
    preferred_mode: RouteMode = Field(
        default=RouteMode.OPTIMAL,
        alias="preferredMode",
        description="판별된 추천 경로 모드",
    )
    reason: str = Field(
        default="소요 시간과 환승 횟수를 종합 고려한 최적 경로를 추천합니다.",
        description="모드 및 경로 선정 사유",
    )
    confidence: float = Field(default=0.95, description="추론 신뢰도")
    is_fallback: bool = Field(
        default=False,
        alias="isFallback",
        description="Gemini 장애/타임아웃으로 인한 결정론적 Fallback 적용 여부",
    )


class BriefingResponseDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    briefing_text: str = Field(
        ...,
        alias="briefingText",
        description="자연어 경로 요약 브리핑 멘트",
    )
    audio_tts_url: Optional[str] = Field(
        None,
        alias="audioTtsUrl",
        description="Edge-TTS 오디오 스트리밍 엔드포인트 URL",
    )
    is_llm_generated: bool = Field(
        default=False,
        alias="isLlmGenerated",
        description="Gemini LLM 생성 여부 (실패 시 False/규칙 기반 템플릿 생성)",
    )


class VoiceSummaryResponseDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    briefing_text: str = Field(
        ...,
        alias="briefingText",
        description="자연어 경로 요약 브리핑 멘트",
    )
    audio_duration_sec: Optional[float] = Field(
        None,
        alias="audioDurationSec",
        description="합성된 음성 길이(초)",
    )
    audio_base64: Optional[str] = Field(
        None,
        alias="audioBase64",
        description="합성된 MP3 바이너리의 Base64 인코딩 문자열",
    )
    audio_tts_url: Optional[str] = Field(
        None,
        alias="audioTtsUrl",
        description="Edge-TTS 오디오 스트리밍 엔드포인트 URL",
    )
