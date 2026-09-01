from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "BinaryClassification"))

from risk_model import RiskModel  # noqa: E402
from timesfm_forecast import TimesFMFactorForecaster  # noqa: E402


class FakeEvaluator:
    def __init__(self):
        self.config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])
        self.calls = []

    def predict_batch(self, **kwargs):
        self.calls.append(kwargs)
        horizon = kwargs["horizon"]
        for context in kwargs["contexts"]:
            last = context[:, -1]
            point = np.repeat(last[:, None], horizon, axis=1)
            quantiles = np.stack([point - 0.01, point, point + 0.01], axis=2)
            yield SimpleNamespace(
                forecast=point,
                quantiles=quantiles if kwargs["return_quantiles"] else None,
            )


@pytest.fixture
def risk_model():
    index = pd.date_range("2025-01-01", periods=100, freq="B")
    history = pd.DataFrame(
        {
            "Sector A": np.linspace(-0.02, 0.01, len(index)),
            "Style": np.linspace(0.005, -0.003, len(index)),
        },
        index=index,
    )
    exposures = pd.DataFrame(
        {"Sector A": [1.0, 0.0], "Style": [0.5, -0.5]}, index=["AAA", "BBB"]
    )
    covariance = pd.DataFrame(np.eye(2), index=history.columns, columns=history.columns)
    return RiskModel(
        exposures,
        covariance,
        pd.Series([0.1, 0.1], index=exposures.index),
        pd.Series(["A", "B"], index=exposures.index),
        factor_returns=history,
    )


def test_joint_forecast_shapes_and_signed_return_settings(risk_model):
    evaluator = FakeEvaluator()
    forecaster = TimesFMFactorForecaster(risk_model, evaluator=evaluator)
    past = pd.DataFrame({"volume": np.arange(100)})
    known_future = pd.DataFrame({"event": np.arange(105)})

    result = forecaster.forecast(
        horizon=5,
        context_length=64,
        past_only_covariates=past,
        past_future_covariates=known_future,
    )

    assert result.point.shape == (5, 2)
    assert result.quantiles.shape == (5, 6)
    assert result.quantiles.columns.names == ["factor", "quantile"]
    call = evaluator.calls[0]
    assert call["contexts"][0].shape == (2, 64)
    assert call["past_only_covariates"][0].shape == (1, 64)
    assert call["past_future_covariates"][0].shape == (1, 69)
    assert call["make_positive"] is False
    assert call["use_symmetric_averaging"] is True


def test_portfolio_projection_uses_current_factor_exposures(risk_model):
    forecaster = TimesFMFactorForecaster(risk_model, evaluator=FakeEvaluator())
    forecast, portfolio = forecaster.forecast_portfolio(
        {"AAA": 0.6, "BBB": 0.4, "MISSING": 1.0}, horizon=3, context_length=32
    )

    expected_exposure = risk_model.exposures({"AAA": 0.6, "BBB": 0.4})
    pd.testing.assert_series_equal(portfolio.exposures, expected_exposure, check_names=False)
    pd.testing.assert_series_equal(
        portfolio.daily_return,
        forecast.point.mul(expected_exposure, axis=1).sum(axis=1).rename("point_return"),
    )
    assert portfolio.missing_symbols == ("MISSING",)


def test_backtest_compares_with_zero_baseline(risk_model):
    evaluator = FakeEvaluator()
    forecaster = TimesFMFactorForecaster(risk_model, evaluator=evaluator)
    metrics = forecaster.backtest(horizon=5, context_length=32, windows=3)

    assert list(metrics.index) == ["Sector A", "Style", "__overall__"]
    assert metrics.loc["__overall__", "n"] == 30
    assert np.isfinite(metrics.loc["__overall__", "mae_skill_vs_zero"])
    assert evaluator.calls[0]["return_quantiles"] is False


def test_numpy_covariate_length_is_validated(risk_model):
    forecaster = TimesFMFactorForecaster(risk_model, evaluator=FakeEvaluator())
    with pytest.raises(ValueError, match="expected 37"):
        forecaster.forecast(
            horizon=5,
            context_length=32,
            past_future_covariates=np.zeros((2, 36)),
        )


def test_joint_variate_limit_is_validated(risk_model):
    forecaster = TimesFMFactorForecaster(risk_model, evaluator=FakeEvaluator())
    with pytest.raises(ValueError, match="at most 32"):
        forecaster.forecast(
            horizon=5,
            context_length=32,
            past_only_covariates=np.zeros((31, 32)),
        )
