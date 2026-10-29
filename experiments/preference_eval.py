"""취향 모델 비교 실험 (가상 사용자 시뮬레이션).

실제 사용자 기록이 아니라 가상 사용자로 만든 결과다. 선택 방식이 conditional logit과 같은 형태라
logit 모델에 유리할 수 있으므로, 절대 수치보다 기록이 늘 때의 추세와 방법 간 순서를 본다.

실행: python -m experiments.preference_eval
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipelines import preference_logit as logit  # noqa: E402
from app.pipelines.preference_v2 import AXES as V2_AXES, CODES, PreferenceRequestV2, compute_preference_v2  # noqa: E402

AXES_USED = ['time', 'fare', 'transfer', 'crowd', 'walk', 'express']
CODE_OF = dict(zip(V2_AXES, CODES))
HISTORY_SIZES = [0, 3, 10, 20, 50]
N_USERS = 100
N_TEST = 100
N_OPTIONS = 4
NOISE_LEVELS = [0.1, 0.25, 0.5]  # 선택의 무작위성(Gumbel 척도). 클수록 사용자가 덜 일관되게 고른다
NOISE = NOISE_LEVELS[1]


def make_choice_set(rng):
    """4개 후보 경로의 원시 수치. 후보 간 차이가 실제처럼 서로 상충하게 만든다."""
    options = []
    for _ in range(N_OPTIONS):
        options.append({
            'time': float(rng.uniform(900, 3600)),
            'fare': float(rng.choice([1400, 1550, 1750, 1950, 2150])),
            'transfer': float(rng.integers(0, 4)),
            'crowd': float(rng.uniform(20, 90)),
            'walk': float(rng.uniform(100, 1200)),
            'express': float(rng.uniform(0, 1)),
        })
    return options


def true_choice(options, true_weights, rng):
    matrix = logit.normalized_matrix(options, AXES_USED)
    utility = -matrix @ np.array([true_weights[a] for a in AXES_USED])
    utility = utility + NOISE * rng.gumbel(size=len(options))
    return int(np.argmax(utility))


def make_user(rng):
    """진짜 중요도를 뽑고, 설문은 가장 중요한 2개 축을 80% 확률로 맞게 고른다."""
    importance = {a: float(rng.beta(2, 2)) for a in AXES_USED}
    true_weights = {}
    for axis in logit.AXES:
        imp = importance.get(axis, 0.5)
        true_weights[axis] = (max(0.0, imp - 0.5) if axis in logit.HIGHER_IS_BETTER
                              else logit.BASE[axis] * 2 * imp)
        if axis in ('express',):
            true_weights[axis] = importance[axis]  # 급행은 선호가 아니라 크기로
    top = sorted(AXES_USED, key=lambda a: -importance[a] * (logit.BASE[a] or 0.5))[:2]
    survey = []
    for axis in AXES_USED:
        picked = axis in top
        if rng.random() < 0.2:
            picked = not picked  # 설문 응답 오류
        if picked:
            survey.append(CODE_OF[axis])
    return true_weights, survey


def survey_importance(survey):
    return {axis: (1.0 if CODE_OF[axis] in survey else 0.5) for axis in V2_AXES}


def snapshot(options, picked):
    selected = options[picked]
    return {'selected': selected, 'alternatives': [o for i, o in enumerate(options) if i != picked]}


def accuracy(weights, tests):
    hits = sum(logit.predict_choice(opts, weights) == chosen for opts, chosen in tests)
    return hits / len(tests)


def evaluate(prior_strength, seed, sizes=HISTORY_SIZES, n_users=N_USERS, noise=None):
    global NOISE
    NOISE = NOISE_LEVELS[1] if noise is None else noise
    rng = np.random.default_rng(seed)
    results = {m: {n: [] for n in sizes} for m in ('공통 기본값', '설문만', '현재 v2 혼합', 'logit MAP', '정답 가중치')}
    for _ in range(n_users):
        true_weights, survey = make_user(rng)
        tests = []
        for _ in range(N_TEST):
            opts = make_choice_set(rng)
            tests.append((opts, true_choice(opts, true_weights, rng)))
        history = []
        for _ in range(max(sizes)):
            opts = make_choice_set(rng)
            history.append(snapshot(opts, true_choice(opts, true_weights, rng)))
        survey_prior = logit.prior_weights(survey_importance(survey))
        default = logit.prior_weights({})
        oracle = accuracy(true_weights, tests)
        for n in sizes:
            past = history[:n]
            v2 = compute_preference_v2(PreferenceRequestV2(routePrefs=survey, choices=past))
            v2_weights = logit.prior_weights(dict(zip(v2.axisKeys, v2.vector)))
            fitted = logit.fit_weights(past, survey_prior, prior_strength)
            results['공통 기본값'][n].append(accuracy(default, tests))
            results['설문만'][n].append(accuracy(survey_prior, tests))
            results['현재 v2 혼합'][n].append(accuracy(v2_weights, tests))
            results['logit MAP'][n].append(accuracy(fitted, tests))
            results['정답 가중치'][n].append(oracle)
    return results


def table(results, sizes):
    lines = ['| 방법 | ' + ' | '.join(f'기록 {n}건' for n in sizes) + ' |', '| --- |' + ' --- |' * len(sizes)]
    for method, by_n in results.items():
        lines.append(f'| {method} | ' + ' | '.join(f'{np.mean(by_n[n]) * 100:.1f}%' for n in sizes) + ' |')
    return '\n'.join(lines)


def main():
    print('사전 세기 선택 (검증용 사용자 60명, 노이즈 0.25, seed 1)')
    best, best_score = None, -1
    for strength in (1.0, 4.0, 8.0, 16.0):
        res = evaluate(strength, seed=1, sizes=[5, 20], n_users=60)
        score = np.mean([np.mean(res['logit MAP'][n]) for n in (5, 20)])
        print(f'  prior_strength={strength}: 평균 정확도 {score * 100:.1f}%')
        if score > best_score:
            best, best_score = strength, score
    print(f'선택한 prior_strength = {best}\n')
    for noise in NOISE_LEVELS:
        results = evaluate(best, seed=2, noise=noise)
        print(f'평가 (노이즈 {noise}, 새 가상 사용자 {N_USERS}명, 사용자당 시험 선택 {N_TEST}개, 후보 {N_OPTIONS}개, seed 2)')
        print(table(results, HISTORY_SIZES))
        print()

if __name__ == '__main__':
    main()
