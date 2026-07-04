import numpy as np

from power_price_risk import (
    EntropicRiskMeasure,
    ExpectationRiskMeasure,
    HedgeCost,
    LognormalPriceScenarioModel,
    MonteCarloCostOptimiser,
)


def main() -> None:
    forecast_curve = np.array(
        [
            72.0,
            68.0,
            65.0,
            63.0,
            62.0,
            66.0,
            78.0,
            92.0,
            101.0,
            96.0,
            88.0,
            82.0,
            79.0,
            77.0,
            80.0,
            91.0,
            108.0,
            119.0,
            112.0,
            98.0,
            87.0,
            80.0,
            76.0,
            73.0,
        ],
        dtype=float,
    )
    model = LognormalPriceScenarioModel(
        forecast_curve=forecast_curve,
        volatility=0.18,
    )
    cost_model = HedgeCost(forecast_curve=forecast_curve)

    mean_cost_optimiser = MonteCarloCostOptimiser(
        model=model,
        cost_model=cost_model,
        risk_measure=ExpectationRiskMeasure(),
    )
    hedge_grid = np.linspace(0.0, 1.0, 11)
    mean_costs = mean_cost_optimiser.cost_grid(
        hedge_fractions=hedge_grid,
        n_scenarios=40_000,
        seed=12_345_678,
    )

    risk_adjusted_optimiser = MonteCarloCostOptimiser(
        model=model,
        cost_model=cost_model,
        risk_measure=EntropicRiskMeasure(aversion=0.05),
    )
    result = risk_adjusted_optimiser.optimise(
        n_scenarios=40_000,
        seed=12_345_678,
        grid_size=501,
    )

    print("Mean residual cost range over hedge fraction:")
    print(f"  {np.ptp(mean_costs):.6f}")
    print()
    print("Risk-adjusted residual cost optimisation:")
    print(f"  hedge_fraction = {result['hedge_fraction']:.4f}")
    print(f"  cost = {result['cost']:.4f}")


if __name__ == "__main__":
    main()
