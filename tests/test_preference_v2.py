import math
import pytest
from pydantic import ValidationError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.pipelines.preference_v2 import AXES, PreferenceRequestV2, compute_preference_v2
from app.routes.preference_route import router

def test_ten_axes_and_survey_cold_start():
    model = compute_preference_v2(PreferenceRequestV2(routePrefs=['LOW_FARE', 'ELEVATOR']))
    assert model.axisKeys == AXES
    assert len(model.vector) == 10
    assert model.vector[6] == model.vector[7] == 1
    assert model.observationCounts == [0] * 10
    assert model.confidence == [0] * 10

def test_missing_and_unchanging_axes_do_not_inflate_confidence():
    model = compute_preference_v2(PreferenceRequestV2(choices=[{'selected': {'fare': 1000, 'time': 100}, 'alternatives': [{'fare': 2000, 'time': 100}]}]))
    assert model.vector[6] > 0.5
    assert model.observationCounts[6] == 1
    assert model.observationCounts[0] == model.observationCounts[4] == 0

def test_unknown_axes_and_nonfinite_features_are_rejected():
    for selected in [{'seatChance': 0.9}, {'fare': math.nan}, {'walk': -1}]:
        with pytest.raises(ValidationError):
            PreferenceRequestV2(choices=[{'selected': selected, 'alternatives': []}])

def test_versioned_http_contract_keeps_legacy_endpoint():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        result = client.post('/api/v1/ai/preference-vector/v2', json={'routePrefs': ['LOW_FARE'], 'choices': []})
        assert result.status_code == 200
        assert result.json()['modelVersion'] == 'route-preference-v2'
        assert len(result.json()['vector']) == 10
        legacy = client.post('/api/v1/ai/preference-vector', json={'trips': []})
        assert legacy.status_code == 200
        assert len(legacy.json()['vector']) == 6


def test_route_dto_accepts_new_server_criteria():
    from app.dtos.request_dto import RouteMode, SubwayRouteDto
    for mode in ['FASTEST', 'LEAST_WALK', 'FEWEST_TRANSFERS']:
        assert RouteMode(mode).value == mode
        route = SubwayRouteDto(mode=mode, totalDurationSec=600, totalDistanceKm=2, totalFare=1500, transferCount=0, segments=[])
        assert route.mode.value == mode

def test_duplicate_alternatives_do_not_change_the_model():
    # 서버는 같은 경로가 여러 모드로 나오면 같은 값을 alternatives에 여러 번 넣는다. 축마다 최솟값과 최댓값만 쓰므로 결과가 같아야 한다
    once = {'selected': {'fare': 1000, 'time': 500}, 'alternatives': [{'fare': 2000, 'time': 300}, {'fare': 1000, 'time': 500}]}
    twice = {'selected': {'fare': 1000, 'time': 500}, 'alternatives': [{'fare': 2000, 'time': 300}, {'fare': 2000, 'time': 300}, {'fare': 1000, 'time': 500}, {'fare': 1000, 'time': 500}]}
    assert compute_preference_v2(PreferenceRequestV2(choices=[once])) == compute_preference_v2(PreferenceRequestV2(choices=[twice]))
