"""취향 문장 임베딩.

이동 목적, 경로 취향, 최근 고른 경로 모드를 한국어 문장으로 만들고 공개 문장 임베딩 모델
multilingual-e5-small(intfloat, MIT)로 384차원 벡터를 만든다. 서버는 이 벡터의 코사인 유사도로
10축 L2 거리로 고른 이웃 후보를 다시 정렬한다. 10축 벡터 계산에는 쓰지 않는다.

메모리를 줄이려고 PyTorch 대신 ONNX 양자화본과 onnxruntime을 쓰고, 토크나이저는 tokenizer.json(약 254MB)
대신 원본 SentencePiece 모델(약 36MB)로 같은 토큰 ID를 만든다(XLM-RoBERTa 규칙).
"""
import os
import threading
from collections import Counter
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field

from app.errors.exceptions import EmbeddingModelUnavailableException

# numpy, onnxruntime, sentencepiece, 설정은 쓸 때 불러온다. 선호 계약 CI는 pydantic과 fastapi만 설치하고
# 이 모듈을 라우터와 함께 불러오기 때문이다

MODEL_VERSION = 'multilingual-e5-small-q8'
DIMENSIONS = 384
MAX_TOKENS = 512

# 코드와 라벨은 앱 온보딩 선택지(API_SPEC 2.1.2)와 같다. 목록 순서로 문장에 넣어 같은 선택이면 같은 문장이 된다
PURPOSE_LABELS = {
    'COMMUTE': '출퇴근', 'SCHOOL': '통학', 'MEETING': '약속·모임', 'TRAVEL': '여행',
    'EXERCISE': '운동', 'SHOPPING': '쇼핑', 'HOSPITAL': '병원', 'DATE': '데이트',
}
ROUTE_PREF_LABELS = {
    'FAST': '빠르게', 'FEW_TRANSFERS': '환승 적게', 'WEAK_AC': '약냉방칸', 'UNCROWDED': '덜 붐비게',
    'LESS_WALK': '덜 걷게', 'SEAT': '앉아서 가기', 'ELEVATOR': '엘리베이터 있는 역', 'LOW_FARE': '요금 아끼기',
    'AVOID_STAIRS': '계단 피하기', 'EXPRESS': '급행 선호', 'FAMILIAR_ROUTE': '익숙한 경로',
}
ROUTE_MODE_LABELS = {
    'OPTIMAL': '종합 최적', 'COMFORTABLE': '여유', 'PERSONAL': '맞춤 추천', 'CHEAPEST': '최소 비용',
    'ONLY_WALK': '도보', 'FASTEST': '최소 시간', 'LEAST_WALK': '최소 도보', 'FEWEST_TRANSFERS': '최소 환승',
}
# 자주 고른 경로 모드는 많이 고른 순으로 이만큼만 문장에 넣는다
TOP_MODES = 3


class PreferenceEmbeddingRequest(BaseModel):
    purposes: List[str] = Field(default_factory=list, max_length=16)
    routePrefs: List[str] = Field(default_factory=list, max_length=32)
    # 최근 이동 기록에서 고른 경로 모드, 최신순
    recentModes: List[str] = Field(default_factory=list, max_length=50)


class PreferenceEmbeddingResponse(BaseModel):
    modelVersion: str = MODEL_VERSION
    dimensions: int = DIMENSIONS
    text: str
    embedding: List[float]


def _labels(codes: Sequence[str], table: dict) -> List[str]:
    """아는 코드만 표의 순서대로. 모르는 코드는 문장에 넣지 않는다"""
    chosen = set(codes)
    return [label for code, label in table.items() if code in chosen]


def _frequent_modes(modes: Sequence[str]) -> List[str]:
    """많이 고른 순, 같으면 최근에 고른 순"""
    known = [mode for mode in modes if mode in ROUTE_MODE_LABELS]
    counts = Counter(known)
    first_seen = {mode: index for index, mode in reversed(list(enumerate(known)))}
    ranked = sorted(counts, key=lambda mode: (-counts[mode], first_seen[mode]))
    return [ROUTE_MODE_LABELS[mode] for mode in ranked[:TOP_MODES]]


def build_preference_text(request: PreferenceEmbeddingRequest) -> str:
    """취향 문장. 선택이 없는 항목은 '없음'으로 둬서 항목 구조는 사용자마다 같다"""
    purposes = _labels(request.purposes, PURPOSE_LABELS)
    prefs = _labels(request.routePrefs, ROUTE_PREF_LABELS)
    modes = _frequent_modes(request.recentModes)
    parts = [
        f"이동 목적: {', '.join(purposes) if purposes else '없음'}.",
        f"경로 취향: {', '.join(prefs) if prefs else '없음'}.",
        f"자주 고른 경로: {', '.join(modes) if modes else '없음'}.",
    ]
    # 앞뒤와 연속 공백이 있으면 SentencePiece와 tokenizer.json의 토큰이 달라진다
    return ' '.join(' '.join(parts).split())


class E5Embedder:
    """처음 호출할 때 모델을 불러온다. 여러 요청이 동시에 와도 한 번만 불러온다"""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            if self.model_dir:
                directory = self.model_dir
            else:
                from app.configs.settings import settings

                directory = settings.preference_embedding_model_dir
            model_path = os.path.join(directory, 'model_quantized.onnx')
            spm_path = os.path.join(directory, 'sentencepiece.bpe.model')
            try:
                import numpy  # noqa: F401  embed에서 쓴다. 없으면 여기서 모델 사용 불가로 알린다
                import onnxruntime as ort
                import sentencepiece as spm

                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.enable_cpu_mem_arena = False
                session = ort.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
                tokenizer = spm.SentencePieceProcessor(model_file=spm_path)
            except Exception as error:  # 파일 없음, 패키지 없음
                raise EmbeddingModelUnavailableException(
                    f"취향 임베딩 모델을 불러오지 못했습니다: {type(error).__name__}"
                ) from error
            self._tokenizer = tokenizer
            self._session = session

    def token_ids(self, text: str) -> List[int]:
        """XLM-RoBERTa(fairseq) 규칙: <s>=0, </s>=2, <unk>=3, SentencePiece ID + 1. e5는 'query: ' 접두어를 쓴다"""
        self._load()
        pieces = self._tokenizer.encode('query: ' + text)
        ids = [piece + 1 if piece != 0 else 3 for piece in pieces][: MAX_TOKENS - 2]
        return [0, *ids, 2]

    def embed(self, text: str) -> List[float]:
        token_ids = self.token_ids(text)
        import numpy as np

        ids = np.array([token_ids], dtype=np.int64)
        mask = np.ones_like(ids)
        output = self._session.run(
            None, {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': np.zeros_like(ids)}
        )[0]
        # 평균 풀링 후 L2 정규화. 정규화했으므로 내적이 곧 코사인 유사도다
        pooled = (output * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
        vector = pooled[0] / np.linalg.norm(pooled[0])
        return [round(float(x), 6) for x in vector]


embedder = E5Embedder()


def compute_preference_embedding(
    request: PreferenceEmbeddingRequest, model: Optional[E5Embedder] = None
) -> PreferenceEmbeddingResponse:
    text = build_preference_text(request)
    vector = (model or embedder).embed(text)
    return PreferenceEmbeddingResponse(text=text, embedding=vector)
