import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.pipelines import preference_embedding as pe
from app.pipelines.preference_embedding import (
    E5Embedder,
    PreferenceEmbeddingRequest,
    build_preference_text,
    compute_preference_embedding,
)
from app.routes.preference_route import router


def test_text_uses_labels_in_fixed_order_regardless_of_input_order():
    a = build_preference_text(PreferenceEmbeddingRequest(
        purposes=['TRAVEL', 'COMMUTE'], routePrefs=['FEW_TRANSFERS', 'FAST']))
    b = build_preference_text(PreferenceEmbeddingRequest(
        purposes=['COMMUTE', 'TRAVEL'], routePrefs=['FAST', 'FEW_TRANSFERS']))
    assert a == b
    assert a == '이동 목적: 출퇴근, 여행. 경로 취향: 빠르게, 환승 적게. 자주 고른 경로: 없음.'


def test_empty_and_unknown_codes_become_none():
    text = build_preference_text(PreferenceEmbeddingRequest(
        purposes=['UNKNOWN'], routePrefs=[], recentModes=['NOPE']))
    assert text == '이동 목적: 없음. 경로 취향: 없음. 자주 고른 경로: 없음.'


def test_frequent_modes_rank_by_count_then_recency_and_keep_top_three():
    text = build_preference_text(PreferenceEmbeddingRequest(recentModes=[
        'COMFORTABLE', 'FEWEST_TRANSFERS', 'FEWEST_TRANSFERS', 'CHEAPEST', 'OPTIMAL', 'COMFORTABLE', 'FASTEST',
    ]))
    # 최소 환승 2회, 여유 2회(더 최근), 그다음 1회 중 가장 최근인 최소 비용
    assert text.endswith('자주 고른 경로: 여유, 최소 환승, 최소 비용.')


def test_text_has_no_leading_trailing_or_double_spaces():
    text = build_preference_text(PreferenceEmbeddingRequest(purposes=['MEETING']))
    assert text == ' '.join(text.split())


class FakeSentencePiece:
    def __init__(self, pieces):
        self.pieces = pieces

    def encode(self, text):
        return list(self.pieces)


def test_token_ids_follow_xlm_roberta_rule():
    model = E5Embedder(model_dir='unused')
    model._session = object()
    model._tokenizer = FakeSentencePiece([5, 0, 9])
    # <s>, spm+1, 알 수 없는 조각(0)은 <unk>=3, </s>
    assert model.token_ids('아무 문장') == [0, 6, 3, 10, 2]


def test_token_ids_are_truncated_to_the_model_limit():
    model = E5Embedder(model_dir='unused')
    model._session = object()
    model._tokenizer = FakeSentencePiece(range(1, 1000))
    ids = model.token_ids('긴 문장')
    assert len(ids) == pe.MAX_TOKENS
    assert ids[0] == 0 and ids[-1] == 2


def test_missing_model_files_raise_service_unavailable(tmp_path):
    with pytest.raises(pe.EmbeddingModelUnavailableException) as error:
        E5Embedder(model_dir=str(tmp_path)).embed('문장')
    assert error.value.status_code == 503


class FakeEmbedder:
    def __init__(self):
        self.texts = []

    def embed(self, text):
        self.texts.append(text)
        return [1.0] + [0.0] * (pe.DIMENSIONS - 1)


def test_http_contract_with_fake_model(monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr(pe, 'embedder', fake)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        result = client.post('/api/v1/ai/preference-embedding', json={
            'purposes': ['COMMUTE'], 'routePrefs': ['FAST'], 'recentModes': ['FEWEST_TRANSFERS'],
        })
    assert result.status_code == 200
    body = result.json()
    assert body['modelVersion'] == pe.MODEL_VERSION
    assert body['dimensions'] == 384
    assert len(body['embedding']) == 384
    assert body['text'] == '이동 목적: 출퇴근. 경로 취향: 빠르게. 자주 고른 경로: 최소 환승.'
    assert fake.texts == [body['text']]


def test_http_rejects_oversized_lists():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        result = client.post('/api/v1/ai/preference-embedding', json={'recentModes': ['OPTIMAL'] * 51})
    assert result.status_code == 422


# 실제 모델 확인. 모델 폴더(SUB1S_E5_MODEL_DIR)가 있을 때만 돈다. CI에는 모델이 없다
MODEL_DIR = os.environ.get('SUB1S_E5_MODEL_DIR')


@pytest.mark.skipif(not MODEL_DIR, reason='SUB1S_E5_MODEL_DIR not set')
def test_real_model_returns_unit_vectors_and_ranks_similar_preferences_higher():
    model = E5Embedder(model_dir=MODEL_DIR)

    def vec(**kwargs):
        return compute_preference_embedding(PreferenceEmbeddingRequest(**kwargs), model).embedding

    commute_fast = vec(purposes=['COMMUTE'], routePrefs=['FAST', 'FEW_TRANSFERS'], recentModes=['FASTEST'])
    school_fast = vec(purposes=['SCHOOL'], routePrefs=['FAST', 'FEW_TRANSFERS'], recentModes=['FASTEST'])
    travel_cheap = vec(purposes=['TRAVEL'], routePrefs=['LOW_FARE', 'LESS_WALK'], recentModes=['CHEAPEST'])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert len(commute_fast) == 384
    assert abs(dot(commute_fast, commute_fast) - 1) < 1e-4
    assert dot(commute_fast, school_fast) > dot(commute_fast, travel_cheap)
