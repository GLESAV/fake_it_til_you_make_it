"""Metrics and statistics."""

import numpy as np
import pytest

from fitymi.eval.metrics import (
    balanced_accuracy,
    evaluate,
    expected_calibration_error,
    quadratic_weighted_kappa,
)
from fitymi.eval.stats import (
    exchange_rate,
    holm_bonferroni,
    mean_ci,
    paired_bootstrap,
    trend_test,
)


def test_balanced_accuracy_ignores_class_prior():
    """The reason it is the primary endpoint: majority-guessing must not score well."""
    y_true = np.array([0] * 70 + [1] * 20 + [2] * 7 + [3] * 3)
    majority = np.zeros(100, dtype=int)
    # One class right, three at zero: 1/4 regardless of how skewed the prior is.
    assert balanced_accuracy(y_true, majority) == pytest.approx(0.25)
    assert (y_true == majority).mean() == pytest.approx(0.70)


def test_qwk_penalises_distant_confusions_more():
    y_true = np.array([0, 1, 2, 3] * 10)
    near = (y_true + 1) % 4
    far = 3 - y_true
    assert quadratic_weighted_kappa(y_true, near) > quadratic_weighted_kappa(y_true, far)


def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 2, 3, 3, 2, 1, 0])
    result = evaluate(y, y, np.eye(4)[y])
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert result.qwk == pytest.approx(1.0)
    assert result.mae == pytest.approx(0.0)


def test_ece_is_zero_for_a_calibrated_model():
    y = np.array([0, 1, 2, 3])
    probs = np.eye(4)[y]
    assert expected_calibration_error(y, probs) == pytest.approx(0.0, abs=1e-9)


def test_tail_recall_covers_the_severe_grades():
    y_true = np.array([0, 0, 2, 2, 3, 3])
    y_pred = np.array([0, 0, 2, 2, 0, 0])  # very severe entirely missed
    result = evaluate(y_true, y_pred)
    assert result.per_class_recall["very_severe"] == pytest.approx(0.0)
    assert result.tail_recall == pytest.approx(0.5)


def test_paired_bootstrap_detects_a_real_difference():
    rs = np.random.default_rng(0)
    y_true = rs.integers(0, 4, 400)
    good = np.where(rs.random(400) < 0.85, y_true, rs.integers(0, 4, 400))
    poor = np.where(rs.random(400) < 0.45, y_true, rs.integers(0, 4, 400))
    result = paired_bootstrap(y_true, good, poor, n_boot=400, seed=1)
    assert result.delta.point > 0
    assert result.delta.lo > 0


def test_paired_bootstrap_finds_no_difference_between_identical_models():
    rs = np.random.default_rng(1)
    y_true = rs.integers(0, 4, 200)
    pred = np.where(rs.random(200) < 0.7, y_true, rs.integers(0, 4, 200))
    result = paired_bootstrap(y_true, pred, pred, n_boot=200, seed=2)
    assert result.delta.point == pytest.approx(0.0)
    assert result.p_value > 0.5


def test_trend_test_recovers_a_decline():
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    scores = {
        0: [0.80, 0.78, 0.72, 0.65, 0.55],
        1: [0.81, 0.79, 0.74, 0.66, 0.52],
        2: [0.79, 0.77, 0.71, 0.63, 0.54],
    }
    result = trend_test(fractions, scores)
    assert result.slope.point < 0
    assert result.slope.hi < 0
    assert result.p_value < 0.05


def test_trend_test_finds_no_slope_when_there_is_none():
    fractions = [0.0, 0.5, 1.0]
    scores = {0: [0.70, 0.71, 0.70], 1: [0.71, 0.70, 0.71], 2: [0.70, 0.70, 0.71]}
    assert trend_test(fractions, scores).p_value > 0.05


def test_exchange_rate_interpolates_and_reports_infinity():
    assert exchange_rate([0, 1, 2, 4, 8], [0.60, 0.63, 0.66, 0.69, 0.70], 0.69) == pytest.approx(4.0)
    assert exchange_rate([0, 1, 2], [0.60, 0.61, 0.62], 0.90) == float("inf")


def test_holm_bonferroni_steps_down():
    out = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.30})
    assert out["a"]["reject"] is True
    assert out["b"]["reject"] is False
    assert out["c"]["reject"] is False


def test_mean_ci_needs_more_than_one_seed():
    single = mean_ci([0.5])
    assert np.isnan(single.lo)
    multi = mean_ci([0.50, 0.52, 0.48])
    assert multi.lo < multi.point < multi.hi
