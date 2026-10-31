from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RouteMode(str, Enum):
    OPTIMAL = "OPTIMAL"
    COMFORTABLE = "COMFORTABLE"
    PERSONAL = "PERSONAL"
    CHEAPEST = "CHEAPEST"
    FASTEST = "FASTEST"
    LEAST_WALK = "LEAST_WALK"
    FEWEST_TRANSFERS = "FEWEST_TRANSFERS"


class NluRequestDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(..., min_length=1, max_length=500, description="사용자 자연어 음성 전사 텍스트")
    preferred_mode: Optional[RouteMode] = Field(
        None,
        alias="preferredMode",
        description="사용자 지정 우선 탐색 모드 (선택)",
    )


class PathSegmentDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    line_id: str = Field(..., alias="lineId", description="호선 ID")
    direction: Optional[str] = Field(None, description="운행 방향 (UP/DOWN/INNER/OUTER)")
    is_express: bool = Field(False, alias="isExpress", description="급행 여부")
    from_station_name: str = Field(..., alias="fromStationName", description="승차 역명")
    to_station_name: str = Field(..., alias="toStationName", description="하차 역명")
    station_count: int = Field(1, alias="stationCount", description="통과 역 개수")
    duration_sec: int = Field(0, alias="durationSec", description="구간 소요 시간(초)")
    distance_km: float = Field(0.0, alias="distanceKm", description="구간 이동 거리(km)")
    recommended_car_no: Optional[int] = Field(
        None, alias="recommendedCarNo", description="추천 탑승 칸 번호 (빠른 환승)"
    )
    is_weak_air_conditioned: bool = Field(
        False, alias="isWeakAirConditioned", description="약냉방칸 탑승 권장 여부"
    )
    congestion_level: Optional[str] = Field(
        None, alias="congestionLevel", description="혼잡도 등급 (LOW/MEDIUM/HIGH/VERY_HIGH)"
    )


class SubwayRouteDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    mode: RouteMode = Field(default=RouteMode.OPTIMAL, description="경로 탐색 모드")
    total_duration_sec: int = Field(..., alias="totalDurationSec", description="총 소요 시간(초)")
    total_distance_km: float = Field(0.0, alias="totalDistanceKm", description="총 이동 거리(km)")
    total_fare: int = Field(1400, alias="totalFare", description="예상 요금(원)")
    transfer_count: int = Field(0, alias="transferCount", description="환승 횟수")
    estimated_arrival_timestamp: Optional[str] = Field(
        None, alias="estimatedArrivalTimestamp", description="예상 도착 시각 (ISO-8601)"
    )
    segments: List[PathSegmentDto] = Field(default_factory=list, description="상세 구간 목록")


class BriefingRequestDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    origin_station_name: str = Field(
        ..., alias="originStationName", description="출발역 이름"
    )
    dest_station_name: str = Field(
        ..., alias="destStationName", description="도착역 이름"
    )
    preferred_mode: RouteMode = Field(
        default=RouteMode.OPTIMAL, alias="preferredMode", description="추천 모드"
    )
    route: Optional[SubwayRouteDto] = Field(None, description="다익스트라 추천 경로 세부 데이터")
