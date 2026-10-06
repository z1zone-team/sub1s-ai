from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class RouteMode(str, Enum):
    OPTIMAL = "OPTIMAL"
    COMFORTABLE = "COMFORTABLE"
    PERSONAL = "PERSONAL"
    CHEAPEST = "CHEAPEST"


class NluRequestDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(..., min_length=1, max_length=500, description="사용자 자연어 음성 전사 텍스트")
    preferred_mode: Optional[RouteMode] = Field(
        None,
        alias="preferredMode",
        description="사용자 지정 우선 탐색 모드 (선택)",
    )
