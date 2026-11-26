"""Gemini 음성 인식과 의도 힌트 프롬프트.

서버가 역 사전(799역)과 규칙으로 다시 확인하므로, 여기서는 들린 그대로 받아 적고 말한 조건만 고른다.
"""

_MODE_GUIDE = (
    "mode는 다음 중 하나다. 말한 조건이 없으면 OPTIMAL.\n"
    "- FASTEST: 빠른 길, 빨리, 최단 시간, 급해\n"
    "- FEWEST_TRANSFERS: 환승 적게, 환승 없이, 갈아타지 않고\n"
    "- LEAST_WALK: 덜 걷는, 걷는 거리 짧게\n"
    "- COMFORTABLE: 덜 붐비는, 사람 적은, 한산한, 편하게, 앉아서\n"
    "- CHEAPEST: 싼, 저렴한, 요금 적게\n"
    "- PERSONAL: 내 취향, 평소 가던, 맞춤\n"
)

_FIELDS = (
    '{"transcript": 받아 적은 문장, "origin": 출발 장소 이름 또는 null, '
    '"destination": 도착 장소 이름 또는 null, "mode": 경로 조건}'
)

VOICE_TRANSCRIBE_PROMPT = (
    "수도권 지하철 길찾기 앱에 사용자가 말한 음성이다. 한국어로 들린 그대로 받아 적고 의도를 JSON으로만 답하라.\n"
    f"형식: {_FIELDS}\n"
    "- 지하철역 이름은 '역'을 빼고 적는다(강남역 -> 강남). 역이 아닌 장소(집, 학교, 건물, 주소)는 말한 그대로 적는다\n"
    "- 출발지를 말하지 않았거나 '여기', '지금 위치'라고 했으면 origin은 null\n"
    "- 말소리가 없거나 알아들을 수 없으면 transcript는 빈 문자열, 나머지는 null\n"
    "- 들리지 않은 말을 지어내지 않는다\n"
    f"{_MODE_GUIDE}"
)


def build_intent_hint_prompt(text: str) -> str:
    return (
        "수도권 지하철 길찾기 앱에 사용자가 말한 문장을 받아 적은 것이다. 의도를 JSON으로만 답하라.\n"
        '형식: {"origin": 출발 장소 이름 또는 null, "destination": 도착 장소 이름 또는 null, "mode": 경로 조건}\n'
        "- 받아 적기 오류로 역 이름이 틀렸을 수 있다(예: 잠시력 -> 잠실, 신돌이 멱 -> 신도림). 가장 그럴듯한 수도권 지하철역 이름으로 고쳐 '역'을 빼고 적는다\n"
        "- 역이 아닌 장소(집, 학교, 건물, 주소)는 말한 그대로 적는다\n"
        "- 출발지를 말하지 않았거나 '여기', '지금 위치'라고 했으면 origin은 null\n"
        f"{_MODE_GUIDE}"
        f'문장: "{text}"'
    )
