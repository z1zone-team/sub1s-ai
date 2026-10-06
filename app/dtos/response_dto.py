from typing import Optional
from pydantic import BaseModel, Field


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
