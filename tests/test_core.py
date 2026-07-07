import numpy as np
import pytest

from power_price_risk import (
    EntropicRiskMeasure,
    ExpectationRiskMeasure,
    HedgeCost,
    LognormalPriceScenarioModel,
)


def test_scenario_model() -> None:
    # Zero volatility should reproduce the forecast curve exactly and deterministically.
    forecast_curve = np.array([50.0, 75.0, 100.0])
    model = LognormalPriceScenarioModel(
        forecast_curve=forecast_curve,
        volatility=0.0,
    )

    prices = model.simulate_prices(n_scenarios=4, seed=123)
    repeated_prices = model.simulate_prices(n_scenarios=4, seed=123)

    assert prices.shape == (4, 3)
    np.testing.assert_allclose(prices, np.tile(forecast_curve, (4, 1)))
    np.testing.assert_allclose(prices, repeated_prices)


def test_hedge_cost() -> None:
    # Residual cost equals upside forecast error minus the hedge offset.
    cost_model = HedgeCost(forecast_curve=np.array([10.0, 20.0]))
    prices = np.array(
        [
            [12.0, 18.0],
            [8.0, 24.0],
        ]
    )

    costs = cost_model.evaluate(prices=prices, hedge_fraction=0.5)

    np.testing.assert_allclose(costs, np.array([1.0, 1.5]))


def test_entropic_risk_measure() -> None:
    # The entropic risk measure is larger than the mean for non-constant costs.
    costs = np.array([1.0, 3.0])

    mean_cost = ExpectationRiskMeasure().evaluate(costs=costs)
    entropic_cost = EntropicRiskMeasure(aversion=0.5).evaluate(costs=costs)

    assert mean_cost == pytest.approx(2.0)
    assert entropic_cost > mean_cost
