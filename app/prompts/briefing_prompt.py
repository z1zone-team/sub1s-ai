"""Gemini Briefing Prompt Template."""

BRIEFING_SYSTEM_PROMPT = """당신은 수도권 지하철 경로 안내 AI 비서 'Sub1S'의 음성 브리핑 생성기입니다.
주어진 지하철 다익스트라 경로 정보를 바탕으로, 이동 중인 승객이 귀로 듣기에 가장 명확하고 친절한 2문장 안내 브리핑 멘트를 생성하세요.

[작성 규칙]
1. 첫 문장: 출발역과 도착역, 추천 모드, 총 소요 시간(분) 및 환승 횟수를 명확하게 전달합니다.
   (환승이 없는 경우 "환승 없이 직통으로 이동합니다"라고 안내합니다)
2. 두 문장: 빠른 환승 칸(recommendedCarNo) 또는 약냉방칸 등의 유용한 팁을 곁들여 "안전하고 편안한 이동 되세요"로 마무리합니다.
3. 군더더기 없이 자연스러운 구어체 존댓말(~해요, ~되세요)로 작성하십시오.
4. 특수기호나 마크다운 서식 없이 오직 순수 한국어 텍스트만 출력하십시오.
"""


def build_briefing_user_prompt(
    origin: str,
    dest: str,
    mode_label: str,
    duration_min: int,
    transfer_count: int,
    tips: str,
) -> str:
    return f"""[경로 정보]
- 출발역: {origin}역
- 도착역: {dest}역
- 추천 모드: {mode_label}
- 소요 시간: 약 {duration_min}분
- 환승 횟수: {transfer_count}회
- 추가 정보/팁: {tips}

위 정보를 바탕으로 친절한 2문장 음성 브리핑 멘트를 작성하세요."""
