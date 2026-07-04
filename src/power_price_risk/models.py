from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from power_price_risk._validation import (
    FloatArray,
    finite_1d_array,
    non_negative_float,
    positive_int,
)


@dataclass(frozen=True)
class LognormalPriceScenarioModel:
    """Synthetic hourly price scenarios around an explicit forecast curve."""

    forecast_curve: FloatArray
    volatility: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "forecast_curve",
            finite_1d_array(self.forecast_curve, name="forecast_curve", positive=True),
        )
        object.__setattr__(
            self,
            "volatility",
            non_negative_float(self.volatility, name="volatility"),
        )

    @property
    def horizon(self) -> int:
        return int(self.forecast_curve.size)

    def simulate_prices(
        self,
        *,
        n_scenarios: int,
        seed: int | None = None,
    ) -> FloatArray:
        n_scenarios = positive_int(n_scenarios, name="n_scenarios")
        rng = np.random.default_rng(seed)
        shocks = rng.normal(size=(n_scenarios, self.horizon))
        relative_errors = np.exp(
            -0.5 * self.volatility**2 + self.volatility * shocks
        )
        return self.forecast_curve * relative_errors
