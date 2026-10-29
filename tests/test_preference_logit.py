import numpy as np

from app.pipelines import preference_logit as logit


def option(time, fare=1750.0, transfer=0.0):
    return {'time': time, 'fare': fare, 'transfer': transfer}


def test_no_choices_returns_the_prior():
    prior = logit.prior_weights({})
    assert logit.fit_weights([], prior) == prior


def test_prior_weights_match_the_server_formula():
    weights = logit.prior_weights({'time': 1.0, 'express': 0.8})
    assert weights['time'] == 2.0  # BASE 1 * 2 * 1.0
    assert weights['fare'] == 0.3  # 중립 0.5는 기본 가중치 그대로
    assert abs(weights['express'] - 0.3) < 1e-9  # 0.5를 넘는 만큼만
    assert logit.prior_weights({})['express'] == 0.0


def test_repeated_cheap_choices_raise_the_fare_weight():
    prior = logit.prior_weights({})
    # 항상 더 느리지만 더 싼 경로를 고른다
    choices = [
        {'selected': option(1500, fare=1400), 'alternatives': [option(900, fare=2100), option(1200, fare=1950)]}
        for _ in range(30)
    ]
    fitted = logit.fit_weights(choices, prior, prior_strength=1.0)
    assert fitted['fare'] > prior['fare']
    assert fitted['time'] < prior['time']


def test_few_choices_stay_near_the_prior():
    prior = logit.prior_weights({})
    choices = [{'selected': option(1500, fare=1400), 'alternatives': [option(900, fare=2100)]}]
    strong = logit.fit_weights(choices, prior, prior_strength=1000.0)
    assert all(abs(strong[a] - prior[a]) < 0.01 for a in logit.AXES)


def test_axes_missing_from_any_option_are_skipped():
    selected = {'time': 900.0, 'crowd': 30.0}
    alternatives = [{'time': 1200.0}]
    matrix = logit.choice_matrix(selected, alternatives, logit.AXES)
    assert matrix is not None
    assert not matrix[:, logit.AXES.index('crowd')].any()


def test_weights_never_go_negative():
    prior = logit.prior_weights({})
    choices = [{'selected': option(3000), 'alternatives': [option(900)]} for _ in range(50)]
    fitted = logit.fit_weights(choices, prior, prior_strength=0.1)
    assert min(fitted.values()) >= 0.0


def test_predict_picks_the_lowest_cost_option():
    weights = logit.prior_weights({})
    assert logit.predict_choice([option(1800), option(900)], weights) == 1
