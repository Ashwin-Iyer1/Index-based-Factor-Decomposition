"""TimesFM 3 forecasting for the saved factor risk model.

This module deliberately sits on top of :mod:`risk_model`: TimesFM forecasts the
joint path of factor returns, while the existing Barra-style covariance model
continues to estimate risk. The TimesFM dependency is imported lazily so normal
risk-model use does not require PyTorch or a model-weight download.

TimesFM 3's default pretrained weights are currently licensed for
non-commercial, non-production use only. See the repository README before use.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_CHECKPOINT = "google/timesfm-3.0-pytorch"
DEFAULT_QUANTILES = tuple(np.arange(0.1, 1.0, 0.1).round(1))
MAX_CONTEXT_LENGTH = 15_360
MAX_VARIATES = 32


@dataclass(frozen=True)
class PortfolioForecast:
    """Point projection of factor forecasts through fixed portfolio exposures."""

    daily_return: pd.Series
    cumulative_return: pd.Series
    factor_contributions: pd.DataFrame
    exposures: pd.Series
    missing_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactorForecast:
    """A joint forecast of the model's factor-return series.

    ``point`` has forecast steps on rows and factors on columns. ``quantiles``
    has ``(factor, quantile)`` MultiIndex columns. TimesFM emits marginal
    quantiles for each factor; they must not be summed to construct a portfolio
    quantile because marginal quantiles do not preserve the joint distribution.
    """

    point: pd.DataFrame
    quantiles: pd.DataFrame | None
    context_start: Any
    context_end: Any
    checkpoint: str

    def project(self, exposures: pd.Series | dict[str, float]) -> PortfolioForecast:
        """Project point factor forecasts through a fixed factor-exposure vector."""
        exposures = pd.Series(exposures, dtype=float)
        unknown = exposures.index.difference(self.point.columns).tolist()
        if unknown:
            raise ValueError(f"unknown factor exposures: {', '.join(map(str, unknown))}")
        aligned = exposures.reindex(self.point.columns, fill_value=0.0)
        if not np.isfinite(aligned.to_numpy()).all():
            raise ValueError("factor exposures must all be finite")
        contributions = self.point.mul(aligned, axis=1)
        daily = contributions.sum(axis=1).rename("point_return")
        cumulative = ((1.0 + daily).cumprod() - 1.0).rename("cumulative_point_return")
        return PortfolioForecast(daily, cumulative, contributions, aligned)

    def save(self, path: str | Path) -> Path:
        """Save point forecasts, marginal quantiles, and run metadata as CSV/JSON."""
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        self.point.to_csv(output / "factor_point.csv")
        if self.quantiles is not None:
            self.quantiles.to_csv(output / "factor_quantiles.csv")
        metadata = {
            "checkpoint": self.checkpoint,
            "context_start": str(self.context_start),
            "context_end": str(self.context_end),
            "horizon": len(self.point),
            "factors": self.point.columns.tolist(),
            "quantiles_are_marginal": self.quantiles is not None,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return output


class TimesFMFactorForecaster:
    """Zero-shot multivariate forecaster for a fitted :class:`RiskModel`.

    Parameters
    ----------
    risk_model:
        Loaded ``RiskModel`` with ``factor_returns`` history.
    evaluator:
        Optional TimesFM-compatible evaluator. This injection point keeps unit
        tests light and permits reuse of an already-loaded checkpoint.
    checkpoint, device, batch_size:
        Passed to Google's ``TimesFM3Evaluator`` when ``evaluator`` is omitted.
    """

    def __init__(
        self,
        risk_model: Any,
        *,
        evaluator: Any | None = None,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        batch_size: int = 4,
    ) -> None:
        history = getattr(risk_model, "factor_returns", None)
        if history is None or history.empty:
            raise ValueError("risk model has no factor-return history")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.risk_model = risk_model
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self._evaluator = evaluator

    @property
    def evaluator(self) -> Any:
        if self._evaluator is None:
            try:
                from timesfm3 import ModelConfig, TimesFM3Evaluator
            except ImportError as exc:
                raise ImportError(
                    "TimesFM 3 is optional. Install it with "
                    "`pip install -r requirements-timesfm.txt`."
                ) from exc
            config = ModelConfig(
                checkpoint_path=self.checkpoint,
                per_core_batch_size=self.batch_size,
                device=self.device,
            )
            self._evaluator = TimesFM3Evaluator(config)
        return self._evaluator

    def forecast(
        self,
        horizon: int = 20,
        *,
        context_length: int = 512,
        factors: Iterable[str] | None = None,
        past_only_covariates: pd.DataFrame | pd.Series | np.ndarray | None = None,
        past_future_covariates: pd.DataFrame | pd.Series | np.ndarray | None = None,
        return_quantiles: bool = True,
        symmetric_averaging: bool = True,
    ) -> FactorForecast:
        """Forecast all requested factors jointly in one TimesFM query.

        Pandas covariates use time on rows and channels on columns. NumPy arrays
        follow TimesFM's native ``(channels, time)`` convention. Past-only input
        must cover the context; past-future input must cover context plus horizon.
        Longer pandas objects are tail-trimmed to the required length.
        """
        horizon, context_length = _validate_lengths(horizon, context_length)
        history = self._history(factors)
        context = history.tail(context_length)
        if len(context) < 32:
            raise ValueError("TimesFM needs at least 32 factor-return observations")

        po = _covariate_array(
            past_only_covariates, len(context), "past_only_covariates"
        )
        pf = _covariate_array(
            past_future_covariates,
            len(context) + horizon,
            "past_future_covariates",
        )
        covariate_count = (0 if po is None else po.shape[0]) + (
            0 if pf is None else pf.shape[0]
        )
        if len(context.columns) + covariate_count > MAX_VARIATES:
            raise ValueError(
                "TimesFM 3 supports at most 32 target + covariate channels per "
                "joint forward pass; select fewer factors or covariates"
            )
        output = self._predict_batch(
            [context.to_numpy(dtype=np.float32).T],
            horizon,
            [po] if po is not None else None,
            [pf] if pf is not None else None,
            return_quantiles,
            symmetric_averaging,
        )[0]
        return _format_output(
            output,
            factors=context.columns,
            horizon=horizon,
            checkpoint=self.checkpoint,
            context_start=context.index[0],
            context_end=context.index[-1],
            quantile_levels=self._quantile_levels(),
            include_quantiles=return_quantiles,
        )

    def forecast_portfolio(
        self,
        portfolio: dict[str, float] | pd.Series,
        **forecast_kwargs: Any,
    ) -> tuple[FactorForecast, PortfolioForecast]:
        """Jointly forecast factors and project their point path onto a portfolio."""
        forecast = self.forecast(**forecast_kwargs)
        weights, missing = self.risk_model._weights(portfolio)
        if weights.empty:
            raise ValueError("portfolio has no symbols in the risk-model universe")
        exposures = self.risk_model.B.loc[weights.index].T @ weights
        projected = forecast.project(exposures.reindex(forecast.point.columns))
        projected = PortfolioForecast(
            projected.daily_return,
            projected.cumulative_return,
            projected.factor_contributions,
            projected.exposures,
            tuple(map(str, missing)),
        )
        return forecast, projected

    def backtest(
        self,
        *,
        horizon: int = 20,
        context_length: int = 512,
        windows: int = 6,
        stride: int | None = None,
        factors: Iterable[str] | None = None,
        symmetric_averaging: bool = True,
    ) -> pd.DataFrame:
        """Rolling-origin point-forecast test against a zero-return baseline.

        Returns factor-level and overall MAE/RMSE/directional-accuracy metrics.
        Financial returns are difficult to forecast, so this check should precede
        any decision to consume TimesFM output downstream.
        """
        horizon, context_length = _validate_lengths(horizon, context_length)
        if windows < 1:
            raise ValueError("windows must be positive")
        stride = horizon if stride is None else stride
        if stride < 1:
            raise ValueError("stride must be positive")
        history = self._history(factors)
        last_origin = len(history) - horizon
        first_origin = last_origin - (windows - 1) * stride
        if first_origin < 32:
            raise ValueError("not enough history for the requested backtest windows")
        origins = list(range(first_origin, last_origin + 1, stride))
        return self._backtest_origins(
            history,
            origins,
            horizon,
            context_length,
            symmetric_averaging,
            split_kind="trailing_windows",
        )

    def backtest_holdout(
        self,
        test_start: str | pd.Timestamp,
        *,
        horizon: int = 20,
        context_length: int = 1536,
        stride: int | None = None,
        factors: Iterable[str] | None = None,
        symmetric_averaging: bool = True,
    ) -> pd.DataFrame:
        """Walk forward through a fixed, date-based holdout period.

        The first forecast begins on the first factor-return date at or after
        ``test_start``. Subsequent non-overlapping windows may use observations
        revealed earlier in the holdout, matching a live walk-forward process,
        but never use data from their own forecast horizon.

        TimesFM 3 is zero-shot: the pre-holdout period is model context, not a
        gradient-training set. The split nevertheless prevents evaluation on
        dates before the declared test boundary.
        """
        horizon, context_length = _validate_lengths(horizon, context_length)
        stride = horizon if stride is None else stride
        if stride < 1:
            raise ValueError("stride must be positive")
        history = self._history(factors)
        if not isinstance(history.index, pd.DatetimeIndex):
            raise ValueError("date-based holdout requires a DatetimeIndex")
        boundary = pd.Timestamp(test_start)
        if history.index.tz is not None:
            if boundary.tzinfo is None:
                boundary = boundary.tz_localize(history.index.tz)
            else:
                boundary = boundary.tz_convert(history.index.tz)
        elif boundary.tzinfo is not None:
            boundary = boundary.tz_localize(None)
        first_origin = int(history.index.searchsorted(boundary, side="left"))
        last_origin = len(history) - horizon
        if first_origin < 32:
            raise ValueError("holdout leaves fewer than 32 pre-test observations")
        if first_origin > last_origin:
            raise ValueError("test_start leaves no complete forecast horizon")
        origins = list(range(first_origin, last_origin + 1, stride))
        return self._backtest_origins(
            history,
            origins,
            horizon,
            context_length,
            symmetric_averaging,
            split_kind="fixed_holdout",
        )

    def _backtest_origins(
        self,
        history: pd.DataFrame,
        origins: list[int],
        horizon: int,
        context_length: int,
        symmetric_averaging: bool,
        *,
        split_kind: str,
    ) -> pd.DataFrame:
        """Evaluate a validated set of rolling forecast origins."""
        contexts = [
            history.iloc[max(0, origin - context_length):origin]
            .to_numpy(dtype=np.float32)
            .T
            for origin in origins
        ]
        outputs = self._predict_batch(
            contexts,
            horizon,
            None,
            None,
            False,
            symmetric_averaging,
        )
        predictions = np.stack([np.asarray(out.forecast) for out in outputs])
        actuals = np.stack(
            [history.iloc[origin:origin + horizon].to_numpy().T for origin in origins]
        )
        expected_shape = (len(origins), len(history.columns), horizon)
        if predictions.shape != expected_shape:
            raise ValueError(
                f"unexpected TimesFM backtest shape {predictions.shape}; "
                f"expected {expected_shape}"
            )
        error = predictions - actuals
        rows = []
        for i, factor in enumerate(history.columns):
            rows.append(_metric_row(factor, predictions[:, i], actuals[:, i], error[:, i]))
        rows.append(_metric_row("__overall__", predictions, actuals, error))
        metrics = pd.DataFrame(rows).set_index("factor")
        first_origin = origins[0]
        last_observation = origins[-1] + horizon - 1
        metrics.attrs.update({
            "split_kind": split_kind,
            "history_start": str(history.index[0]),
            "history_end": str(history.index[-1]),
            "train_end": str(history.index[first_origin - 1]),
            "test_start": str(history.index[first_origin]),
            "test_end": str(history.index[last_observation]),
            "n_windows": len(origins),
            "horizon": horizon,
            "stride": origins[1] - origins[0] if len(origins) > 1 else horizon,
            "context_length": context_length,
            "first_context_observations": min(context_length, first_origin),
        })
        return metrics

    def _history(self, factors: Iterable[str] | None) -> pd.DataFrame:
        history = self.risk_model.factor_returns.sort_index()
        if factors is not None:
            requested = list(dict.fromkeys(factors))
            missing = pd.Index(requested).difference(history.columns).tolist()
            if missing:
                raise ValueError(f"unknown factors: {', '.join(map(str, missing))}")
            if not requested:
                raise ValueError("at least one factor is required")
            history = history.loc[:, requested]
        history = history.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        empty = history.columns[history.notna().sum() == 0].tolist()
        if empty:
            raise ValueError(f"factors contain no observations: {', '.join(map(str, empty))}")
        return history

    def _predict_batch(
        self,
        contexts: list[np.ndarray],
        horizon: int,
        past_only_covariates: list[np.ndarray] | None,
        past_future_covariates: list[np.ndarray] | None,
        return_quantiles: bool,
        symmetric_averaging: bool,
    ) -> list[Any]:
        # Factor returns can be negative. TimesFM3Evaluator otherwise defaults
        # make_positive=True, which would silently clamp forecasts at zero.
        outputs = list(
            self.evaluator.predict_batch(
                contexts=contexts,
                horizon=horizon,
                past_only_covariates=past_only_covariates,
                past_future_covariates=past_future_covariates,
                return_quantiles=return_quantiles,
                use_symmetric_averaging=symmetric_averaging,
                make_positive=False,
                sort_quantiles=True,
                use_znorm=False,
                padding_mode="none",
            )
        )
        if len(outputs) != len(contexts):
            raise ValueError(
                f"TimesFM returned {len(outputs)} outputs for {len(contexts)} contexts"
            )
        return outputs

    def _quantile_levels(self) -> tuple[float, ...]:
        config = getattr(self.evaluator, "config", None)
        values = getattr(config, "quantiles", DEFAULT_QUANTILES)
        return tuple(float(q) for q in values)


def _validate_lengths(horizon: int, context_length: int) -> tuple[int, int]:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(context_length, int) or context_length < 32:
        raise ValueError("context_length must be an integer of at least 32")
    if context_length > MAX_CONTEXT_LENGTH:
        raise ValueError(f"context_length cannot exceed {MAX_CONTEXT_LENGTH}")
    return horizon, context_length


def _covariate_array(
    values: pd.DataFrame | pd.Series | np.ndarray | None,
    expected_length: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, pd.Series):
        if len(values) < expected_length:
            raise ValueError(f"{name} needs {expected_length} time steps")
        array = values.iloc[-expected_length:].to_numpy(dtype=np.float32)[None, :]
    elif isinstance(values, pd.DataFrame):
        if len(values) < expected_length:
            raise ValueError(f"{name} needs {expected_length} time steps")
        array = values.iloc[-expected_length:].to_numpy(dtype=np.float32).T
    else:
        array = np.asarray(values, dtype=np.float32)
        array = np.atleast_2d(array)
        if array.shape[-1] != expected_length:
            raise ValueError(
                f"{name} has {array.shape[-1]} time steps; expected {expected_length}"
            )
    if array.shape[-1] != expected_length:
        raise ValueError(f"{name} needs {expected_length} time steps")
    if np.any(np.isfinite(array).sum(axis=1) == 0):
        raise ValueError(f"each {name} channel needs at least one finite value")
    return array


def _format_output(
    output: Any,
    *,
    factors: pd.Index,
    horizon: int,
    checkpoint: str,
    context_start: Any,
    context_end: Any,
    quantile_levels: tuple[float, ...],
    include_quantiles: bool,
) -> FactorForecast:
    point_values = np.asarray(output.forecast)
    expected = (len(factors), horizon)
    if point_values.shape != expected:
        raise ValueError(
            f"unexpected TimesFM forecast shape {point_values.shape}; expected {expected}"
        )
    index = pd.RangeIndex(1, horizon + 1, name="forecast_step")
    point = pd.DataFrame(point_values.T, index=index, columns=factors)
    quantiles = None
    if include_quantiles:
        values = np.asarray(output.quantiles)
        expected_q = (len(factors), horizon, len(quantile_levels))
        if values.shape != expected_q:
            raise ValueError(
                f"unexpected TimesFM quantile shape {values.shape}; expected {expected_q}"
            )
        columns = pd.MultiIndex.from_product(
            [factors, quantile_levels], names=["factor", "quantile"]
        )
        quantiles = pd.DataFrame(
            values.transpose(1, 0, 2).reshape(horizon, -1),
            index=index,
            columns=columns,
        )
    return FactorForecast(
        point=point,
        quantiles=quantiles,
        context_start=context_start,
        context_end=context_end,
        checkpoint=checkpoint,
    )


def _metric_row(
    factor: str,
    prediction: np.ndarray,
    actual: np.ndarray,
    error: np.ndarray,
) -> dict[str, float | str]:
    valid = np.isfinite(prediction) & np.isfinite(actual)
    if not valid.any():
        return {
            "factor": factor,
            "timesfm_mae": np.nan,
            "zero_mae": np.nan,
            "mae_skill_vs_zero": np.nan,
            "timesfm_rmse": np.nan,
            "zero_rmse": np.nan,
            "directional_accuracy": np.nan,
            "n": 0,
        }
    pred = prediction[valid]
    act = actual[valid]
    err = error[valid]
    model_mae = float(np.mean(np.abs(err)))
    zero_mae = float(np.mean(np.abs(act)))
    return {
        "factor": factor,
        "timesfm_mae": model_mae,
        "zero_mae": zero_mae,
        "mae_skill_vs_zero": 1.0 - model_mae / zero_mae if zero_mae else np.nan,
        "timesfm_rmse": float(np.sqrt(np.mean(err**2))),
        "zero_rmse": float(np.sqrt(np.mean(act**2))),
        "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(act))),
        "n": int(valid.sum()),
    }
