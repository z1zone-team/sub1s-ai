import logging
from typing import Optional
from urllib.parse import quote

from app.clients.gemini_client import GeminiClient
from app.dtos.request_dto import RouteMode, SubwayRouteDto
from app.dtos.response_dto import BriefingResponseDto

logger = logging.getLogger(__name__)

MODE_LABELS = {
    RouteMode.OPTIMAL: "최적",
    RouteMode.COMFORTABLE: "여유 좌석",
    RouteMode.PERSONAL: "개인 맞춤",
    RouteMode.CHEAPEST: "최소 비용",
    RouteMode.FASTEST: "최소 시간",
    RouteMode.LEAST_WALK: "최소 도보",
    RouteMode.FEWEST_TRANSFERS: "최소 환승",
}


class BriefingPipeline:
    """
    자연어 경로 요약 브리핑 생성 파이프라인.
    다익스트라 알고리즘으로 산출된 추천 경로 데이터로부터
    규칙 기반 템플릿(결정론적 0ms)과 Gemini 생성 하이브리드로 음성 브리핑을 합성합니다.
    """

    _instance: Optional["BriefingPipeline"] = None

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.gemini_client = gemini_client or GeminiClient()

    @classmethod
    def get_instance(cls) -> "BriefingPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def generate_briefing(
        self,
        origin_station_name: str,
        dest_station_name: str,
        preferred_mode: RouteMode = RouteMode.OPTIMAL,
        route: Optional[SubwayRouteDto] = None,
    ) -> BriefingResponseDto:
        """
        출발역, 도착역, 추천 모드 및 상세 경로를 바탕으로 음성 브리핑 멘트를 생성합니다.
        """
        mode_label = MODE_LABELS.get(preferred_mode, "추천")
        duration_min = round(route.total_duration_sec / 60) if route else 25
        transfer_count = route.transfer_count if route else 0

        tips_summary = self._extract_tips(route)

        # 1. Gemini LLM 브리핑 시도 (2초 타임아웃)
        llm_text = await self.gemini_client.generate_briefing(
            origin=origin_station_name,
            dest=dest_station_name,
            mode_label=mode_label,
            duration_min=duration_min,
            transfer_count=transfer_count,
            tips=tips_summary,
        )

        if llm_text and len(llm_text.strip()) > 10:
            final_text = llm_text.strip()
            is_llm = True
            logger.info("[BriefingPipeline] Gemini LLM 브리핑 생성 완료")
        else:
            final_text = self._build_template_briefing(
                origin_station_name,
                dest_station_name,
                mode_label,
                duration_min,
                transfer_count,
                tips_summary,
            )
            is_llm = False
            logger.info("[BriefingPipeline] 규칙 템플릿 기반 브리핑 생성 완료")

        audio_tts_url = f"/api/v1/ai/tts/stream?text={quote(final_text)}"

        return BriefingResponseDto(
            briefing_text=final_text,
            audio_tts_url=audio_tts_url,
            is_llm_generated=is_llm,
        )

    def _extract_tips(self, route: Optional[SubwayRouteDto]) -> str:
        if not route or not route.segments:
            return "빠른 환승 정보를 확인하세요."

        tips = []
        for seg in route.segments:
            if seg.recommended_car_no:
                tips.append(
                    f"{seg.from_station_name}역에서 {seg.recommended_car_no}번 칸에 탑승하시면 빠른 환승이 가능해요."
                )
                break
            if seg.is_weak_air_conditioned:
                tips.append("약냉방칸이 준비되어 있어 쾌적하게 이동하실 수 있어요.")
                break

        return " ".join(tips) if tips else "안전하고 편안한 이동 되세요."

    def _build_template_briefing(
        self,
        origin: str,
        dest: str,
        mode_label: str,
        duration_min: int,
        transfer_count: int,
        tips: str,
    ) -> str:
        if transfer_count == 0:
            transfer_text = "환승 없이 직통으로 이동해요."
        else:
            transfer_text = f"환승은 {transfer_count}회 필요해요."

        first_sentence = (
            f"{origin}역에서 {dest}역까지 {mode_label} 경로로 안내해 드릴게요. "
            f"총 약 {duration_min}분 소요되며 {transfer_text}"
        )

        second_sentence = f"{tips}" if tips else "안전하고 편안한 이동 되세요."

        return f"{first_sentence} {second_sentence}"
