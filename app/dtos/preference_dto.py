from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TripRecordDto(BaseModel):
    """사용자 이동 이력 1건 (NestJS TravelLog에서 추출한 정량 피처)"""

    model_config = ConfigDict(populate_by_name=True)

    duration_sec: int = Field(..., ge=1, alias="durationSec", description="실제 소요 시간(초)")
    fastest_duration_sec: Optional[int] = Field(
        None, ge=1, alias="fastestDurationSec", description="해당 구간 최단 소요 시간(초), 알 수 없으면 생략"
    )
    transfer_count: int = Field(0, ge=0, alias="transferCount", description="환승 횟수")
    avg_congestion_rate: Optional[float] = Field(
        None, ge=0, alias="avgCongestionRate", description="탑승 구간 평균 혼잡도 (0~1, 1 초과 시 백분율로 간주)"
    )
    transfer_walk_m: Optional[float] = Field(
        None, ge=0, alias="transferWalkM", description="환승 도보 거리 합계(m)"
    )
    used_weak_ac: bool = Field(False, alias="usedWeakAc", description="약냉방칸 탑승 여부")
    route_signature: Optional[str] = Field(
        None, alias="routeSignature", description="경로 식별 문자열 (예: 강남>사당>판교)"
    )


class PreferenceVectorRequestDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    trips: List[TripRecordDto] = Field(default_factory=list, max_length=500, description="이동 이력 (최대 500건)")


class RoutePreferenceWeightsDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    wc: float = Field(..., description="혼잡도 회피 가중치 (1.0~3.0)")
    wt: float = Field(..., description="환승 페널티 가중치 (120~600초)")
    prefer_weak_air_conditioning: bool = Field(
        ..., alias="preferWeakAirConditioning", description="약냉방칸 우선 여부"
    )


class PreferenceVectorResponseDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    vector: List[float] = Field(
        ..., description="[v_time, v_transfer, v_crowd, v_walk, v_ac, v_regular] (각 0.0~1.0)"
    )
    pgvector_literal: str = Field(
        ..., alias="pgvectorLiteral", description="pgvector 저장/질의용 리터럴 (예: [0.5,0.5,...])"
    )
    weights: RoutePreferenceWeightsDto = Field(..., description="다익스트라 가중치 변환 결과")
    trip_count: int = Field(..., alias="tripCount", description="산출에 사용된 이동 이력 수")
    is_cold_start: bool = Field(
        ..., alias="isColdStart", description="이력 부족으로 중립값 보정이 적용되었는지 여부"
    )
