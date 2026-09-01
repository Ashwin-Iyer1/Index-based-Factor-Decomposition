"""Command-line entry point for TimesFM factor and portfolio forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from risk_model import RiskModel
from timesfm_forecast import DEFAULT_CHECKPOINT, TimesFMFactorForecaster


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecast saved factor returns jointly with TimesFM 3."
    )
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--output-dir", default="timesfm_output")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", help="PyTorch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--factors", help="Comma-separated factor subset (default: all)")
    parser.add_argument(
        "--portfolio-json",
        help='JSON file containing symbol weights, e.g. {"AAPL": 0.5, "XOM": 0.5}',
    )
    parser.add_argument(
        "--past-only-covariates",
        help="CSV with time in the first column and past-only covariates in the rest",
    )
    parser.add_argument(
        "--past-future-covariates",
        help="CSV with time in the first column and known-future covariates in the rest",
    )
    parser.add_argument(
        "--backtest-windows",
        type=int,
        default=0,
        help="Also run this many rolling zero-baseline evaluation windows",
    )
    return parser


def _read_covariates(path: str | None) -> pd.DataFrame | None:
    return None if path is None else pd.read_csv(path, index_col=0)


def main() -> None:
    args = _parser().parse_args()
    model = RiskModel.load(args.model_dir)
    factors = None if not args.factors else [x.strip() for x in args.factors.split(",")]
    forecaster = TimesFMFactorForecaster(
        model,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )
    kwargs = {
        "horizon": args.horizon,
        "context_length": args.context_length,
        "factors": factors,
        "past_only_covariates": _read_covariates(args.past_only_covariates),
        "past_future_covariates": _read_covariates(args.past_future_covariates),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.portfolio_json:
        portfolio = json.loads(Path(args.portfolio_json).read_text())
        forecast, projection = forecaster.forecast_portfolio(portfolio, **kwargs)
        projection.daily_return.to_csv(output_dir / "portfolio_point.csv")
        projection.cumulative_return.to_csv(output_dir / "portfolio_cumulative_point.csv")
        projection.factor_contributions.to_csv(output_dir / "portfolio_factor_contributions.csv")
        if projection.missing_symbols:
            print("Ignored unknown symbols:", ", ".join(projection.missing_symbols))
    else:
        forecast = forecaster.forecast(**kwargs)
    forecast.save(output_dir)
    print(f"Saved {args.horizon}-step forecast to {output_dir.resolve()}")

    if args.backtest_windows:
        metrics = forecaster.backtest(
            horizon=args.horizon,
            context_length=args.context_length,
            windows=args.backtest_windows,
            factors=factors,
        )
        metrics.to_csv(output_dir / "backtest.csv")
        overall = metrics.loc["__overall__"]
        print(
            "Backtest overall: "
            f"MAE skill vs zero {overall['mae_skill_vs_zero']:+.1%}, "
            f"directional accuracy {overall['directional_accuracy']:.1%}"
        )


if __name__ == "__main__":
    main()
