import pytest
from httpx import ASGITransport, AsyncClient

from app.dtos.preference_dto import TripRecordDto
from app.main import app
from app.pipelines.embedding_pipeline import EmbeddingPipeline


@pytest.fixture
def pipeline():
    return EmbeddingPipeline()


def make_trip(**kwargs) -> TripRecordDto:
    defaults = dict(durationSec=1000)
    defaults.update(kwargs)
    return TripRecordDto(**defaults)


def test_empty_history_is_neutral_cold_start(pipeline):
    res = pipeline.compute_vector([])
    assert res.vector == [0.5] * 6
    assert res.is_cold_start is True
    assert res.trip_count == 0
    assert res.weights.wc == pytest.approx(2.0)
    assert res.weights.wt == pytest.approx(360.0)
    assert res.weights.prefer_weak_air_conditioning is True  # v_ac 0.5 >= 0.5


def test_full_history_extracts_each_dimension(pipeline):
    trips = [
        make_trip(
            durationSec=1000,
            fastestDurationSec=900,
            transferCount=0,
            avgCongestionRate=0.2,
            transferWalkM=30,
            usedWeakAc=True,
            routeSignature="강남>사당",
        )
        for _ in range(5)
    ]
    res = pipeline.compute_vector(trips)
    v_time, v_transfer, v_crowd, v_walk, v_ac, v_regular = res.vector

    assert res.is_cold_start is False
    assert v_time == pytest.approx(0.9)
    assert v_transfer == pytest.approx(1.0)
    assert v_crowd == pytest.approx(0.8)
    assert v_walk == pytest.approx(0.9)
    assert v_ac == pytest.approx(1.0)
    assert v_regular == pytest.approx(1.0)


def test_many_transfers_and_crowd_lower_scores(pipeline):
    trips = [
        make_trip(transferCount=3, avgCongestionRate=90, transferWalkM=400)
        for _ in range(5)
    ]
    res = pipeline.compute_vector(trips)
    _, v_transfer, v_crowd, v_walk, v_ac, _ = res.vector

    assert v_transfer == 0.0
    assert v_crowd == pytest.approx(0.1)  # 90은 백분율로 간주
    assert v_walk == 0.0
    assert v_ac == 0.0


def test_cold_start_blends_toward_neutral(pipeline):
    # 이력 1건이면 신뢰도 0.2: 원값 1.0이 0.5 + 0.5*0.2 = 0.6으로 보정
    res = pipeline.compute_vector([make_trip(transferCount=0)])
    assert res.is_cold_start is True
    assert res.vector[1] == pytest.approx(0.6)


def test_regular_route_share(pipeline):
    trips = [
        make_trip(routeSignature="A"),
        make_trip(routeSignature="A"),
        make_trip(routeSignature="A"),
        make_trip(routeSignature="B"),
        make_trip(routeSignature="C"),
    ]
    res = pipeline.compute_vector(trips)
    assert res.vector[5] == pytest.approx(0.6)


def test_missing_features_stay_neutral(pipeline):
    trips = [make_trip() for _ in range(5)]
    res = pipeline.compute_vector(trips)
    assert res.vector[0] == 0.5  # fastest 없음
    assert res.vector[2] == 0.5  # 혼잡도 없음
    assert res.vector[3] == 0.5  # 도보 없음
    assert res.vector[5] == 0.5  # 경로 식별 없음


def test_vector_values_always_in_unit_range(pipeline):
    trips = [
        make_trip(durationSec=10, fastestDurationSec=9999, avgCongestionRate=500, transferWalkM=0)
        for _ in range(10)
    ]
    assert all(0.0 <= v <= 1.0 for v in pipeline.compute_vector(trips).vector)


def test_pgvector_literal_and_dimension_guard():
    assert EmbeddingPipeline.to_pgvector_literal([0.5, 0.5, 1.0, 0.0, 0.25, 0.5]) == "[0.5,0.5,1,0,0.25,0.5]"
    with pytest.raises(ValueError):
        EmbeddingPipeline.to_pgvector_literal([0.1, 0.2])
    with pytest.raises(ValueError):
        EmbeddingPipeline.to_route_weights([0.1])


def test_route_weights_follow_schema_formula():
    weights = EmbeddingPipeline.to_route_weights([0.0, 1.0, 1.0, 0.0, 0.4, 0.0])
    assert weights.wc == pytest.approx(3.0)
    assert weights.wt == pytest.approx(600.0)
    assert weights.prefer_weak_air_conditioning is False


def test_cosine_similarity():
    assert EmbeddingPipeline.cosine_similarity([1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]) == pytest.approx(1.0)
    assert EmbeddingPipeline.cosine_similarity([1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]) == pytest.approx(0.0)
    assert EmbeddingPipeline.cosine_similarity([0] * 6, [1] * 6) == 0.0
    with pytest.raises(ValueError):
        EmbeddingPipeline.cosine_similarity([1, 2], [1, 2, 3])


@pytest.mark.asyncio
async def test_preference_vector_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ai/preference-vector",
            json={
                "trips": [
                    {"durationSec": 1200, "fastestDurationSec": 1200, "transferCount": 0, "usedWeakAc": True}
                    for _ in range(5)
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["vector"]) == 6
        assert data["tripCount"] == 5
        assert data["isColdStart"] is False
        assert data["pgvectorLiteral"].startswith("[") and data["pgvectorLiteral"].endswith("]")
        assert data["weights"]["preferWeakAirConditioning"] is True


@pytest.mark.asyncio
async def test_preference_vector_endpoint_empty_and_invalid():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        empty = await client.post("/api/v1/ai/preference-vector", json={"trips": []})
        assert empty.status_code == 200
        assert empty.json()["isColdStart"] is True

        invalid = await client.post(
            "/api/v1/ai/preference-vector", json={"trips": [{"durationSec": 0}]}
        )
        assert invalid.status_code == 422
