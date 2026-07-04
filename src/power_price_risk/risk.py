from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from power_price_risk._validation import FloatArray, finite_1d_array, positive_float


class ExpectationRiskMeasure:
    """Mean residual cost benchmark."""

    def evaluate(self, *, costs: FloatArray) -> float:
        costs = finite_1d_array(costs, name="costs")
        return float(np.mean(costs))


@dataclass(frozen=True)
class EntropicRiskMeasure:
    """Entropic risk measure applied to residual costs."""

    aversion: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "aversion",
            positive_float(self.aversion, name="aversion"),
        )

    def evaluate(self, *, costs: FloatArray) -> float:
        costs = finite_1d_array(costs, name="costs")
        scaled = self.aversion * costs
        max_scaled = float(np.max(scaled))
        log_mean_exp = max_scaled + float(
            np.log(np.mean(np.exp(scaled - max_scaled)))
        )
        return log_mean_exp / self.aversion
