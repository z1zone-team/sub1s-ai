import pytest
from app.guardrails.station_validator import StationValidator


@pytest.fixture
def validator():
    return StationValidator.get_instance()


def test_station_validator_exact_match(validator):
    assert validator.validate_and_correct("강남") == "강남"
    assert validator.validate_and_correct("을지로입구") == "을지로입구"
    assert validator.validate_and_correct("사당") == "사당"


def test_station_validator_particle_stripping(validator):
    # 한국어 조사 및 '역' 접미사 자동 제거 검증
    assert validator.validate_and_correct("강남역에서") == "강남"
    assert validator.validate_and_correct("을지로입구역까지") == "을지로입구"
    assert validator.validate_and_correct("판교역으로") == "판교"
    assert validator.validate_and_correct("시청부터") == "시청"
    assert validator.validate_and_correct("여의도역") == "여의도"


def test_station_validator_compound_station_no_early_segmentation(validator):
    # 긴 복합 역명이 조기 분절되지 않고 온전히 보존되는지 검증
    assert validator.validate_and_correct("동대문역사문화공원") == "동대문역사문화공원"
    assert validator.validate_and_correct("동대문역사문화공원역에서") == "동대문역사문화공원"
    assert validator.validate_and_correct("디지털미디어시티") == "디지털미디어시티"


def test_station_validator_aliases_and_abbreviations(validator):
    # 일상 축약어 및 병기역명 별칭 정규화 검증
    assert validator.validate_and_correct("고터") == "고속터미널"
    assert validator.validate_and_correct("홍대") == "홍대입구"
    assert validator.validate_and_correct("이수") == "총신대입구"
    assert validator.validate_and_correct("총신대입구(이수)") == "총신대입구"
    assert validator.validate_and_correct("지제") == "평택지제"
    assert validator.validate_and_correct("DMC") == "디지털미디어시티"


def test_station_validator_fuzzy_correction(validator):
    # 음성 잡음/받침 오인식 fuzzy 보정 검증
    assert validator.validate_and_correct("신도림녁") == "신도림"


def test_station_validator_hallucination_rejection(validator):
    # 수도권 지하철 노선망에 존재하지 않는 가상의 역명 100% 차단 검증
    assert validator.validate_and_correct("안드로메다") is None
    assert validator.validate_and_correct("화성역") is None
    assert validator.validate_and_correct("우리집") is None


def test_extract_stations_from_text(validator):
    # 자연어 문장에서 결정론적 출발지/도착지 추출 검증
    orig, dest = validator.extract_stations_from_text(
        "강남역에서 을지로입구역까지 편하게 가고 싶어"
    )
    assert orig == "강남"
    assert dest == "을지로입구"

    orig, dest = validator.extract_stations_from_text("사당 -> 판교")
    assert orig == "사당"
    assert dest == "판교"

    orig, dest = validator.extract_stations_from_text(
        "홍대에서 고터까지 덜 붐비는 길 알려줘"
    )
    assert orig == "홍대입구"
    assert dest == "고속터미널"
