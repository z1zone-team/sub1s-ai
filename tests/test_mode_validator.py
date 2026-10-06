from app.dtos.request_dto import RouteMode
from app.guardrails.mode_validator import ModeValidator


def test_mode_validator_validate_mode():
    assert ModeValidator.validate_mode("OPTIMAL") == RouteMode.OPTIMAL
    assert ModeValidator.validate_mode("comfortable") == RouteMode.COMFORTABLE
    assert ModeValidator.validate_mode("cheapest") == RouteMode.CHEAPEST
    assert ModeValidator.validate_mode("personal") == RouteMode.PERSONAL
    assert ModeValidator.validate_mode("INVALID_MODE") == RouteMode.OPTIMAL
    assert ModeValidator.validate_mode(None) == RouteMode.OPTIMAL


def test_mode_validator_detect_mode():
    mode, _ = ModeValidator.detect_mode("덜 붐비고 편하게 앉아갈 수 있는 길")
    assert mode == RouteMode.COMFORTABLE

    mode, _ = ModeValidator.detect_mode("제일 싼 최소 비용 경로로 가자")
    assert mode == RouteMode.CHEAPEST

    mode, _ = ModeValidator.detect_mode("내 스타일에 맞춘 자주 가던 경로")
    assert mode == RouteMode.PERSONAL

    mode, _ = ModeValidator.detect_mode("가장 빠른 길로 안내해줘")
    assert mode == RouteMode.OPTIMAL
