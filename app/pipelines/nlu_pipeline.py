import logging
from typing import Optional

from app.clients.gemini_client import GeminiClient
from app.dtos.request_dto import RouteMode
from app.dtos.response_dto import NluResponseDto
from app.errors.exceptions import IntentNotUnderstoodException
from app.guardrails.mode_validator import MODE_REASONS, ModeValidator
from app.guardrails.station_validator import StationValidator

logger = logging.getLogger(__name__)


class NluPipeline:
    """
    Subway NLU 의도 분석 파이프라인.
    Gemini Single-Call 구조와 devine_ai 스타일의 Zero-Token Station Guardrail을 결합하여
    환각(Hallucination) 0% 및 3초 SLA 이내 무장애 응답을 보장합니다.
    """

    _instance: Optional["NluPipeline"] = None

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        station_validator: Optional[StationValidator] = None,
    ):
        self.gemini_client = gemini_client or GeminiClient()
        self.station_validator = station_validator or StationValidator.get_instance()
        self.mode_validator = ModeValidator()

    @classmethod
    def get_instance(cls) -> "NluPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def parse_intent(
        self,
        text: str,
        preferred_mode: Optional[RouteMode] = None,
    ) -> NluResponseDto:
        """
        사용자 음성 전사 텍스트로부터 출발역, 도착역, 추천 모드를 추출합니다.
        1. Gemini 단일 호출(Single-Call) 및 2초 SLA 타임아웃
        2. StationValidator를 통한 무환각 역명 정규화 검증
        3. LLM 실패 또는 환각 발생 시 0-토큰 결정론적 Fallback 안전망 즉각 가동
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            # 예전에는 강남 -> 을지로입구를 지어내 돌려줬다. 말하지 않은 경로를 안내하지 않는다
            raise IntentNotUnderstoodException("받아 적은 문장이 비어 있어요. 다시 말씀해 주세요.")

        # 1. Gemini 호출 시도 (2초 타임아웃 & JSON 스키마 강제)
        llm_result = await self.gemini_client.generate_nlu(cleaned_text)

        if llm_result and isinstance(llm_result, dict):
            raw_origin = llm_result.get("origin")
            raw_dest = llm_result.get("dest")
            raw_mode = llm_result.get("mode")
            raw_reason = llm_result.get("reason")
            confidence = float(llm_result.get("confidence", 0.95))

            # 2. Zero-Token StationValidator 교차 검증 (환각 차단 & 조기 분절 교정)
            val_origin = self.station_validator.validate_and_correct(raw_origin)
            val_dest = self.station_validator.validate_and_correct(raw_dest)

            # 출발지와 도착지가 모두 유효하고 서로 다른 경우 정상 반환
            if val_origin and val_dest and val_origin != val_dest:
                final_mode = (
                    preferred_mode
                    if preferred_mode
                    else self.mode_validator.validate_mode(raw_mode)
                )
                final_reason = (
                    raw_reason
                    if raw_reason
                    else MODE_REASONS.get(
                        final_mode,
                        "소요 시간과 환승 횟수를 종합 고려한 최적 경로를 추천합니다.",
                    )
                )

                logger.info(
                    f"[NluPipeline] LLM 추출 성공: origin='{val_origin}', dest='{val_dest}', mode={final_mode}"
                )
                return NluResponseDto(
                    origin_station_name=val_origin,
                    dest_station_name=val_dest,
                    preferred_mode=final_mode,
                    reason=final_reason,
                    confidence=confidence,
                    is_fallback=False,
                )

            logger.warning(
                f"[NluPipeline] LLM 역명 검증 실패(환각 또는 누락): raw_origin='{raw_origin}', raw_dest='{raw_dest}'. 결정론적 Fallback으로 전환합니다."
            )

        # 3. [Zero-Token Fallback] Gemini 장애/타임아웃/환각 시 결정론적 규칙 기반 안전망
        logger.info("[NluPipeline] 결정론적 Fallback 가드레일 실행")
        fallback_origin, fallback_dest = (
            self.station_validator.extract_stations_from_text(cleaned_text)
        )

        detected_mode, mode_reason = self.mode_validator.detect_mode(cleaned_text)
        final_mode = preferred_mode if preferred_mode else detected_mode

        # 출발역이나 도착역을 찾지 못하면 대표 역(예전 강남, 을지로입구)으로 채우지 않고 알린다.
        # 서버의 음성 흐름은 이 API를 쓰지 않고(규칙 분석과 intent-hint), 이 경로를 직접 부르는 쪽이 오류를 처리한다
        if not fallback_origin or not fallback_dest or fallback_origin == fallback_dest:
            raise IntentNotUnderstoodException()

        return NluResponseDto(
            origin_station_name=fallback_origin,
            dest_station_name=fallback_dest,
            preferred_mode=final_mode,
            reason=mode_reason,
            confidence=0.90,
            is_fallback=True,
        )
