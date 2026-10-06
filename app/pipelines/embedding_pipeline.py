import logging
import math
from collections import Counter
from typing import List, Optional, Sequence

from app.dtos.preference_dto import (
    PreferenceVectorResponseDto,
    RoutePreferenceWeightsDto,
    TripRecordDto,
)

logger = logging.getLogger(__name__)

NEUTRAL = 0.5
MIN_TRIPS_FOR_FULL_CONFIDENCE = 5
TRANSFER_SATURATION = 3.0  # 평균 환승 3회 이상이면 환승 기피 성향 0
WALK_SATURATION_M = 300.0  # 환승 도보 평균 300m 이상이면 도보 기피 성향 0
VECTOR_DIM = 6


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


class EmbeddingPipeline:
    """
    6차원 도메인 행동 피처 벡터 파이프라인.
    이동 이력으로 [v_time, v_transfer, v_crowd, v_walk, v_ac, v_regular]를 0.0~1.0으로 산출한다.
    DB(pgvector) 접근은 NestJS 서버가 담당하며, 이 파이프라인은 계산과 변환만 수행한다.
    """

    _instance: Optional["EmbeddingPipeline"] = None

    @classmethod
    def get_instance(cls) -> "EmbeddingPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def compute_vector(self, trips: List[TripRecordDto]) -> PreferenceVectorResponseDto:
        trip_count = len(trips)
        raw = [
            self._v_time(trips),
            self._v_transfer(trips),
            self._v_crowd(trips),
            self._v_walk(trips),
            self._v_ac(trips),
            self._v_regular(trips),
        ]

        # 콜드 스타트 보정: 이력이 적을수록 중립값(0.5)으로 수렴
        confidence = _clamp01(trip_count / MIN_TRIPS_FOR_FULL_CONFIDENCE)
        vector = [
            round(_clamp01(NEUTRAL + (value - NEUTRAL) * confidence), 4) for value in raw
        ]

        logger.info(
            f"[EmbeddingPipeline] 벡터 산출: trips={trip_count}, confidence={confidence:.2f}, vector={vector}"
        )
        return PreferenceVectorResponseDto(
            vector=vector,
            pgvector_literal=self.to_pgvector_literal(vector),
            weights=self.to_route_weights(vector),
            trip_count=trip_count,
            is_cold_start=trip_count < MIN_TRIPS_FOR_FULL_CONFIDENCE,
        )

    # ---- 차원별 산출 ----

    def _v_time(self, trips: List[TripRecordDto]) -> float:
        """최단 시간 성향: 실제 소요가 최단 경로에 가까울수록 1.0"""
        ratios = [
            _clamp01(t.fastest_duration_sec / t.duration_sec)
            for t in trips
            if t.fastest_duration_sec
        ]
        mean = _mean(ratios)
        return NEUTRAL if mean is None else mean

    def _v_transfer(self, trips: List[TripRecordDto]) -> float:
        """환승 기피 성향: 평균 환승 횟수가 적을수록 1.0"""
        mean = _mean([float(t.transfer_count) for t in trips])
        return NEUTRAL if mean is None else _clamp01(1.0 - mean / TRANSFER_SATURATION)

    def _v_crowd(self, trips: List[TripRecordDto]) -> float:
        """혼잡도 회피 성향: 평균 탑승 혼잡도가 낮을수록 1.0 (1 초과 값은 백분율로 간주)"""
        rates = [
            t.avg_congestion_rate / 100.0 if t.avg_congestion_rate > 1.0 else t.avg_congestion_rate
            for t in trips
            if t.avg_congestion_rate is not None
        ]
        mean = _mean(rates)
        return NEUTRAL if mean is None else _clamp01(1.0 - mean)

    def _v_walk(self, trips: List[TripRecordDto]) -> float:
        """도보 기피 성향: 환승 도보 거리가 짧을수록 1.0"""
        walks = [t.transfer_walk_m for t in trips if t.transfer_walk_m is not None]
        mean = _mean(walks)
        return NEUTRAL if mean is None else _clamp01(1.0 - mean / WALK_SATURATION_M)

    def _v_ac(self, trips: List[TripRecordDto]) -> float:
        """약냉방칸 선호: 약냉방칸 탑승 비율"""
        if not trips:
            return NEUTRAL
        return sum(1 for t in trips if t.used_weak_ac) / len(trips)

    def _v_regular(self, trips: List[TripRecordDto]) -> float:
        """고정 경로 선호: 가장 자주 이용한 경로의 비중 (경로 식별 이력이 2건 미만이면 중립)"""
        signatures = [t.route_signature for t in trips if t.route_signature]
        if len(signatures) < 2:
            return NEUTRAL
        most_common = Counter(signatures).most_common(1)[0][1]
        return most_common / len(signatures)

    # ---- 변환 헬퍼 ----

    @staticmethod
    def to_pgvector_literal(vector: Sequence[float]) -> str:
        """pgvector 저장/질의용 리터럴 (예: [0.5,0.5,0.5,0.5,0.5,0.5])"""
        if len(vector) != VECTOR_DIM:
            raise ValueError(f"벡터 차원은 {VECTOR_DIM}이어야 합니다: {len(vector)}")
        return "[" + ",".join(f"{v:g}" for v in vector) + "]"

    @staticmethod
    def to_route_weights(vector: Sequence[float]) -> RoutePreferenceWeightsDto:
        """SCHEMA_DESIGN 1.1절 공식: w_c = 1 + 2*v_crowd, w_t = 120 + 480*v_transfer, v_ac >= 0.5면 약냉방 우선"""
        if len(vector) != VECTOR_DIM:
            raise ValueError(f"벡터 차원은 {VECTOR_DIM}이어야 합니다: {len(vector)}")
        v_crowd = vector[2]
        v_transfer = vector[1]
        v_ac = vector[4]
        return RoutePreferenceWeightsDto(
            wc=round(1.0 + 2.0 * v_crowd, 4),
            wt=round(120.0 + 480.0 * v_transfer, 4),
            prefer_weak_air_conditioning=v_ac >= 0.5,
        )

    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        """두 벡터의 코사인 유사도 (pgvector `<=>` 거리 = 1 - 유사도). 영벡터는 0.0"""
        if len(a) != len(b):
            raise ValueError("벡터 차원이 일치해야 합니다")
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)
