from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from power_price_risk._validation import FloatArray, finite_1d_array, positive_int
from power_price_risk.costs import HedgeCost
from power_price_risk.models import LognormalPriceScenarioModel
from power_price_risk.risk import EntropicRiskMeasure, ExpectationRiskMeasure


RiskMeasure = ExpectationRiskMeasure | EntropicRiskMeasure


@dataclass(frozen=True)
class MonteCarloCostOptimiser:
    model: LognormalPriceScenarioModel
    cost_model: HedgeCost
    risk_measure: RiskMeasure

    def cost_grid(
        self,
        *,
        hedge_fractions: FloatArray,
        n_scenarios: int,
        seed: int | None = None,
    ) -> FloatArray:
        hedge_fractions = finite_1d_array(
            hedge_fractions,
            name="hedge_fractions",
            unit_interval=True,
        )
        prices = self.model.simulate_prices(
            n_scenarios=n_scenarios,
            seed=seed,
        )
        return np.array(
            [
                self.risk_measure.evaluate(
                    costs=self.cost_model.evaluate(
                        prices=prices,
                        hedge_fraction=float(hedge_fraction),
                    )
                )
                for hedge_fraction in hedge_fractions
            ],
            dtype=float,
        )

    def optimise(
        self,
        *,
        n_scenarios: int,
        seed: int | None = None,
        grid_size: int = 101,
    ) -> dict[str, float | FloatArray]:
        grid_size = positive_int(grid_size, name="grid_size")
        hedge_fractions = np.linspace(0.0, 1.0, grid_size)
        costs = self.cost_grid(
            hedge_fractions=hedge_fractions,
            n_scenarios=n_scenarios,
            seed=seed,
        )
        best_index = int(np.argmin(costs))
        return {
            "hedge_fraction": float(hedge_fractions[best_index]),
            "cost": float(costs[best_index]),
            "hedge_fractions": hedge_fractions,
            "costs": costs,
        }
