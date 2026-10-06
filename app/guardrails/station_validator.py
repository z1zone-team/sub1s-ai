import difflib
import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from app.configs.settings import settings

logger = logging.getLogger(__name__)

# 공통 지하철 약칭 및 별칭 사전
COMMON_ALIASES: Dict[str, str] = {
    # 구 명칭 / 병기역명 분리
    "지제": "평택지제",
    "평택지제": "평택지제",
    "자양": "뚝섬유원지",
    "뚝섬유원지": "뚝섬유원지",
    "화전": "한국항공대",
    "한국항공대": "한국항공대",
    "원곡": "시우",
    "시우": "시우",
    "서해구청": "서구청",
    "서구청": "서구청",
    "이수": "총신대입구",
    "총신대입구": "총신대입구",
    "총신대입구(이수)": "총신대입구",
    "광교중앙(아주대)": "광교중앙",
    "아주대": "광교중앙",
    "양재(서초구청)": "양재",
    "온수(성공회대입구)": "온수",
    "성공회대입구": "온수",
    "대림(구로구청)": "대림",
    "구로구청": "대림",
    # 일상 구어 축약어
    "고터": "고속터미널",
    "설대입구": "서울대입구",
    "설대": "서울대입구",
    "건대": "건대입구",
    "홍대": "홍대입구",
    "이대": "이화여대",
    "숙대": "숙대입구",
    "외대": "외대앞",
    "교대": "교대",
    "중대": "흑석",
    "DMC": "디지털미디어시티",
    "dmc": "디지털미디어시티",
    "디엠씨": "디지털미디어시티",
    "과기대": "공릉",
}

# 한국어 역명 뒤에 결합되는 조사 및 분절 접미사
SUFFIX_PARTICLES: List[str] = [
    "역에서",
    "역부터",
    "역까지",
    "역으로",
    "역로",
    "역행",
    "에서",
    "부터",
    "까지",
    "으로",
    "로",
    "행",
    "역",
]


class StationValidator:
    """
    Zero-Token Deterministic Subway Station Validator (devine_ai 패턴 실증 구현).
    수도권 700여 개 지하철 공식 역명 사전을 메모리에 인덱싱하여,
    토큰 소비 없이 0ms 이내에 LLM 환각 차단, 한국어 조기 분절 교정, 약칭 정규화를 수행합니다.
    """

    _instance: Optional["StationValidator"] = None

    def __init__(self, stations_json_path: Optional[str] = None):
        self.stations_path = stations_json_path or settings.subway_stations_file
        self.valid_names: Set[str] = set()
        self.alias_map: Dict[str, str] = dict(COMMON_ALIASES)
        self.sorted_search_terms: List[str] = []
        self._load_stations()

    @classmethod
    def get_instance(cls, stations_json_path: Optional[str] = None) -> "StationValidator":
        if cls._instance is None:
            cls._instance = cls(stations_json_path)
        return cls._instance

    def _load_stations(self) -> None:
        if not os.path.exists(self.stations_path):
            logger.warning(
                f"[StationValidator] Station data not found at {self.stations_path}. Using fallback list."
            )
            fallback_names = [
                "강남", "역삼", "선릉", "삼성", "을지로입구", "시청", "서울역",
                "동대문역사문화공원", "신도림", "사당", "교대", "잠실", "홍대입구",
                "판교", "수원", "고속터미널", "용산", "여의도", "합정"
            ]
            for name in fallback_names:
                norm = self._strip_station_suffix(name)
                self.valid_names.add(norm)
        else:
            try:
                with open(self.stations_path, "r", encoding="utf-8") as f:
                    stations = json.load(f)

                for item in stations:
                    raw_name = item.get("stationName", "").strip()
                    if not raw_name:
                        continue

                    # 1. 원본 역명 등록
                    norm = self._strip_station_suffix(raw_name)
                    self.valid_names.add(norm)

                    # 2. 괄호 병기역명 분리 (예: "총신대입구(이수)" -> "총신대입구", "이수")
                    paren_match = re.match(r"^([^\(]+)\(([^\)]+)\)$", raw_name)
                    if paren_match:
                        main_name = paren_match.group(1).strip()
                        sub_name = paren_match.group(2).strip()
                        main_norm = self._strip_station_suffix(main_name)
                        sub_norm = self._strip_station_suffix(sub_name)

                        self.valid_names.add(main_norm)
                        self.valid_names.add(sub_norm)
                        self.alias_map[raw_name] = main_norm
                        self.alias_map[main_name] = main_norm
                        self.alias_map[sub_name] = main_norm

                logger.info(
                    f"[StationValidator] Successfully loaded {len(self.valid_names)} unique stations."
                )
            except Exception as e:
                logger.error(f"[StationValidator] Failed to load stations: {e}", exc_info=True)

        # 검색용 단어 목록: 유효 역명 및 별칭 전체를 글자 수 내림차순 정렬 (긴 복합명칭 우선 매칭)
        all_terms = set(self.valid_names) | set(self.alias_map.keys())
        self.sorted_search_terms = sorted(
            [t for t in all_terms if len(t) >= 2],
            key=lambda x: len(x),
            reverse=True,
        )

    def _strip_station_suffix(self, name: str) -> str:
        """역명 끝의 '역' 접미사를 정규화 (단, 2글자 이상일 때만)"""
        trimmed = name.strip()
        if trimmed.endswith("역") and len(trimmed) > 1:
            return trimmed[:-1]
        return trimmed

    def normalize_station_name(self, raw_name: Optional[str]) -> Optional[str]:
        """
        입력된 문자열에서 조사, 접미사, 공백을 제거하고 공식 역명으로 정규화합니다.
        """
        if not raw_name:
            return None

        cleaned = raw_name.strip()
        if not cleaned:
            return None

        # 1. 별칭 사전에 등록된 경우 최우선 정규화 (예: '이수' -> '총신대입구', '고터' -> '고속터미널')
        if cleaned in self.alias_map:
            return self.alias_map[cleaned]

        # 2. 이미 정확한 유효 역명인 경우 즉시 반환 (예: '구로', '종로' 등 '로'로 끝나는 역명 방어)
        if cleaned in self.valid_names:
            return cleaned

        # 3. 끝의 '역' 제거 시 유효하거나 별칭에 있는 경우
        if cleaned.endswith("역") and len(cleaned) > 1:
            cand = cleaned[:-1]
            if cand in self.alias_map:
                return self.alias_map[cand]
            if cand in self.valid_names:
                return cand

        # 4. 한국어 조사 및 접미사 순차 제거 검사
        for suffix in SUFFIX_PARTICLES:
            if cleaned.endswith(suffix):
                cand = cleaned[: -len(suffix)].strip()
                if cand:
                    if cand in self.alias_map:
                        return self.alias_map[cand]
                    if cand in self.valid_names:
                        return cand
                    cand_no_yeok = self._strip_station_suffix(cand)
                    if cand_no_yeok in self.alias_map:
                        return self.alias_map[cand_no_yeok]
                    if cand_no_yeok in self.valid_names:
                        return cand_no_yeok

        cand = self._strip_station_suffix(cleaned)
        if cand in self.alias_map:
            return self.alias_map[cand]
        if cand in self.valid_names:
            return cand

        return cand

    def validate_and_correct(self, name: Optional[str]) -> Optional[str]:
        """
        단일 역명 유효성 검증 및 교정
        1. 별칭 및 정규화 매칭
        2. 정확 매칭
        3. 경미한 오탈자/음성 잡음 fuzzy 유사도 교정 (예: '신도림녁' -> '신도림')
        4. 일치하지 않는 가상의 역은 환각으로 판정하여 None 반환
        """
        if not name:
            return None

        norm = self.normalize_station_name(name)
        if not norm:
            return None

        # 1. 별칭 매핑 우선
        if norm in self.alias_map:
            return self.alias_map[norm]

        # 2. 유효 역명 정확 매치
        if norm in self.valid_names:
            return norm

        # 3. Fuzzy 매칭 (음성 인식 받침 오탈자 보정, 유사도 임계치 0.75)
        close_matches = difflib.get_close_matches(
            norm, self.valid_names, n=1, cutoff=0.75
        )
        if close_matches:
            corrected = close_matches[0]
            logger.info(
                f"[StationValidator] Fuzzy 교정: '{name}' (정규화: '{norm}') -> '{corrected}'"
            )
            return self.alias_map.get(corrected, corrected)

        # 4. 환각 (존재하지 않는 가상의 역명)
        logger.warning(
            f"[StationValidator] 유효하지 않은 역명 감지 (환각 차단): '{name}'"
        )
        return None

    def extract_stations_from_text(
        self, text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        [Zero-Token Fallback] 문장 패턴 및 역명 사전을 활용한 결정론적 출발지/도착지 추출.
        Gemini API가 장애 또는 2초 타임아웃 발생 시 0-토큰으로 즉각 복구합니다.
        """
        if not text:
            return None, None

        cleaned = text.strip()

        # 방법 1: 사전 기반 긴 단어 우선 탐색 (가장 정확하고 조기 분절 없음)
        matches: List[Tuple[int, int, str]] = []  # (start, end, station_name)

        for term in self.sorted_search_terms:
            search_pos = 0
            while search_pos < len(cleaned):
                idx = cleaned.find(term, search_pos)
                if idx == -1:
                    break

                end = idx + len(term)
                # 이미 매칭된 구간과 겹치지 않는지 확인
                overlaps = any(
                    not (end <= m[0] or idx >= m[1]) for m in matches
                )
                if not overlaps:
                    canonical = self.alias_map.get(term, term)
                    matches.append((idx, end, canonical))
                search_pos = idx + 1

        matches.sort(key=lambda m: m[0])

        if len(matches) >= 2:
            orig = matches[0][2]
            dest = matches[1][2]
            if orig != dest:
                return orig, dest

        if len(matches) == 1:
            return matches[0][2], None

        # 방법 2: 화살표 패턴 ([출발] -> [도착])
        arrow_match = re.search(
            r"([가-힣0-9a-zA-Z·]+?)\s*(?:->|~)\s*([가-힣0-9a-zA-Z·]+)", cleaned
        )
        if arrow_match:
            orig_candidate = self.validate_and_correct(arrow_match.group(1))
            dest_candidate = self.validate_and_correct(arrow_match.group(2))
            if orig_candidate and dest_candidate and orig_candidate != dest_candidate:
                return orig_candidate, dest_candidate

        # 방법 3: 구문 분리 정규식
        regex_from_to = re.compile(
            r"([가-힣0-9a-zA-Z·]{2,})(?:역에서|역부터|에서|부터)\s+([가-힣0-9a-zA-Z·]{2,})(?:역까지|역으로|역로|까지|으로|로)"
        )
        match_from_to = regex_from_to.search(cleaned)
        if match_from_to:
            orig_candidate = self.validate_and_correct(match_from_to.group(1))
            dest_candidate = self.validate_and_correct(match_from_to.group(2))
            if orig_candidate and dest_candidate and orig_candidate != dest_candidate:
                return orig_candidate, dest_candidate

        return None, None
