from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from power_price_risk._validation import (
    FloatArray,
    finite_1d_array,
    finite_2d_array,
    unit_interval_float,
)


@dataclass(frozen=True)
class HedgeCost:
    """Residual cost from upward forecast errors after a static hedge."""

    forecast_curve: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "forecast_curve",
            finite_1d_array(self.forecast_curve, name="forecast_curve"),
        )

    def evaluate(self, *, prices: FloatArray, hedge_fraction: float) -> FloatArray:
        hedge_fraction = unit_interval_float(
            hedge_fraction,
            name="hedge_fraction",
        )
        prices = finite_2d_array(prices, name="prices")
        if prices.shape[1] != self.forecast_curve.size:
            raise ValueError("prices must match forecast_curve length")

        forecast_errors = prices - self.forecast_curve

        unhedged_upside_cost = np.maximum(forecast_errors, 0.0).mean(axis=1)
        hedge_offset = hedge_fraction * forecast_errors.mean(axis=1)
        return unhedged_upside_cost - hedge_offset
