import re
from typing import Optional, Tuple
from app.dtos.request_dto import RouteMode


MODE_REASONS = {
    RouteMode.OPTIMAL: "소요 시간과 환승 횟수를 종합 고려한 최적 경로를 추천합니다.",
    RouteMode.COMFORTABLE: "혼잡도가 낮고 여유로운 좌석 우선 경로를 감지했습니다.",
    RouteMode.CHEAPEST: "환승 및 요금 기준 가장 경제적인 최소 비용 경로를 감지했습니다.",
    RouteMode.PERSONAL: "사용자 통근 성향 및 개인 맞춤형 경로를 감지했습니다.",
}


class ModeValidator:
    """
    Subway Route Mode Validator & Classifier.
    4대 경로 모드(OPTIMAL, COMFORTABLE, PERSONAL, CHEAPEST)의 유효성을 검증하고,
    자연어 의도 분석 시 결정론적 키워드 기반으로 안전하게 분류합니다.
    """

    @staticmethod
    def validate_mode(raw_mode: Optional[str]) -> RouteMode:
        """
        LLM 출력 또는 입력 문자열을 RouteMode 열거형으로 안전하게 변환합니다.
        알 수 없거나 유효하지 않은 경우 OPTIMAL로 기본 설정됩니다.
        """
        if not raw_mode:
            return RouteMode.OPTIMAL

        normalized = raw_mode.strip().upper()
        for mode in RouteMode:
            if mode.value == normalized:
                return mode

        return RouteMode.OPTIMAL

    @staticmethod
    def detect_mode(text: str) -> Tuple[RouteMode, str]:
        """
        [Zero-Token Classification] 문장 내 키워드 패턴 기반 4대 모드 판별 및 추천 사유 반환.
        """
        cleaned = text.strip()

        # 1. COMFORTABLE (여유/쾌적 좌석 우선)
        if re.search(r"(여유|편한|편하게|자리|앉아|쾌적|덜\s*붐비)", cleaned):
            return RouteMode.COMFORTABLE, MODE_REASONS[RouteMode.COMFORTABLE]

        # 2. CHEAPEST (최소 비용)
        if re.search(r"(최소\s*비용|저렴|제일\s*싼|가장\s*싼|알뜰|요금\s*절약|싼\s*길|싸게)", cleaned):
            return RouteMode.CHEAPEST, MODE_REASONS[RouteMode.CHEAPEST]

        # 3. PERSONAL (개인 맞춤)
        if re.search(r"(취향|맞춤|내\s*스타일|평소|자주\s*가던|내가\s*타던)", cleaned):
            return RouteMode.PERSONAL, MODE_REASONS[RouteMode.PERSONAL]

        # 4. OPTIMAL (기본 최적 경로)
        return RouteMode.OPTIMAL, MODE_REASONS[RouteMode.OPTIMAL]
