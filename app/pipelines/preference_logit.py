"""선택 모델(conditional logit)로 경로 추천 가중치를 학습한다.

사용자가 제시된 경로 중 하나를 고르는 확률을 P(k) = exp(-score_k) / sum(exp(-score_j))로 두고,
score = sum(weight_axis * value_axis)의 가중치를 최대 사후 확률(MAP)로 추정한다.
설문에서 얻은 기본 가중치를 사전 분포의 중심으로 삼아 이력이 적으면 설문 쪽에 머문다.

value_axis는 서버 `chooseRecommendation`과 같은 방식으로 후보 집합 안에서 정규화한 값이라,
학습한 가중치를 그대로 서버 추천 점수에 넣을 수 있다.
"""
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

# 서버 recommendation-policy.ts의 BASE, SPANS, 방향과 같아야 한다.
BASE = {'time': 1.0, 'fare': 0.3, 'transfer': 0.4, 'crowd': 0.4, 'walk': 0.2,
        'express': 0.0, 'familiarRoute': 0.0, 'weakAc': 0.0, 'elevator': 0.0}
SPANS = {'time': 180.0, 'fare': 200.0, 'transfer': 1.0, 'crowd': 10.0, 'walk': 216.0,
         'express': 1.0, 'familiarRoute': 1.0, 'weakAc': 1.0, 'elevator': 1.0}
HIGHER_IS_BETTER = {'express', 'familiarRoute', 'weakAc', 'elevator'}
AXES = list(BASE)

# 사전 분포의 세기. 값이 클수록 설문 가중치에서 덜 벗어난다.
DEFAULT_PRIOR_STRENGTH = 8.0
ITERATIONS = 150


def prior_weights(survey_importance: Mapping[str, float]) -> Dict[str, float]:
    """설문 중요도(0~1, 중립 0.5)를 서버 가중치 식으로 바꾼다."""
    weights = {}
    for axis in AXES:
        importance = survey_importance.get(axis, 0.5)
        if axis in HIGHER_IS_BETTER:
            weights[axis] = max(0.0, importance - 0.5)
        else:
            weights[axis] = BASE[axis] * 2 * importance
    return weights


def normalized_matrix(options: Sequence[Mapping[str, float]], axes: Sequence[str]) -> np.ndarray:
    """후보 x 축 행렬. 서버와 같이 (x - min) / max(range, span)로 정규화한다. 높을수록 좋은 축은 1 - x."""
    matrix = np.zeros((len(options), len(axes)))
    for column, axis in enumerate(axes):
        values = np.array([option[axis] for option in options], dtype=float)
        if axis in HIGHER_IS_BETTER:
            values = 1.0 - values
        scale = max(values.max() - values.min(), SPANS[axis])
        matrix[:, column] = (values - values.min()) / scale
    return matrix


def choice_matrix(selected: Mapping[str, float], alternatives: Sequence[Mapping[str, float]],
                  axes: Sequence[str]) -> Optional[np.ndarray]:
    """고른 경로가 첫 행인 행렬. 후보 전체에 없는 축은 비교에서 빼므로 axes 중 일부만 쓴다."""
    options = [selected, *alternatives]
    usable = [axis for axis in axes if all(axis in option for option in options)]
    if not usable or len(options) < 2:
        return None
    matrix = np.zeros((len(options), len(axes)))
    columns = [axes.index(axis) for axis in usable]
    matrix[:, columns] = normalized_matrix(options, usable)
    return matrix


def fit_weights(choices: Sequence[Mapping], prior: Mapping[str, float],
                prior_strength: float = DEFAULT_PRIOR_STRENGTH) -> Dict[str, float]:
    """선택 기록으로 축별 가중치를 MAP 추정한다. 기록이 없으면 사전 가중치를 그대로 돌려준다."""
    axes = AXES
    theta0 = np.array([prior[axis] for axis in axes])
    matrices = [m for m in (choice_matrix(c['selected'], c.get('alternatives', []), axes)
                            for c in choices) if m is not None]
    if not matrices:
        return dict(zip(axes, theta0.tolist()))
    theta = theta0.copy()
    for _ in range(ITERATIONS):
        gradient = -prior_strength * (theta - theta0)
        for matrix in matrices:
            logits = -matrix @ theta
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            # d log P(선택) / d theta = -V_선택 + sum_k P_k V_k
            gradient += -matrix[0] + probs @ matrix
        # 보폭은 곡률 상한(사전항 세기 + 선택 수)의 역수. 고정 보폭이면 사전이 강할 때 진동한다
        theta = np.maximum(0.0, theta + gradient / (prior_strength + len(matrices)))
    return dict(zip(axes, theta.tolist()))


def predict_choice(options: Sequence[Mapping[str, float]], weights: Mapping[str, float]) -> int:
    """가중치로 점수가 가장 낮은 후보의 위치. 동점이면 앞선 후보."""
    usable = [axis for axis in AXES if all(axis in option for option in options)]
    matrix = normalized_matrix(options, usable)
    scores = matrix @ np.array([weights[axis] for axis in usable])
    return int(np.argmin(scores))

