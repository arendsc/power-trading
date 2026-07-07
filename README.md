# Power Price Risk Case

This repository is a compact Python case study in price-risk modelling for power
trading. The first version deliberately starts from a synthetic benchmark: a
24-hour forecast curve, Monte Carlo realised price scenarios around that curve,
and a static hedge fraction selected by minimising residual cost.

## What this version demonstrates

- A small package layout with reusable modelling components.
- Monte Carlo simulation of synthetic hourly realised prices.
- Separation between scenario generation, residual cost evaluation, risk measures,
  and optimisation.
- A comparison between mean residual cost and a nonlinear risk-adjusted cost
  measure.
- A reproducible example script with fixed random seeds.

## Modelling idea

The example starts from an explicit 24-hour forecast curve. This represents a
stylised view of hourly delivery prices before realisation. The scenario model
then simulates possible realised prices around that curve.

For hour `h`, define the forecast error as

```text
realised_price_h - forecast_price_h
```

The model assumes an exposure to upward forecast errors: realised prices above
forecast create cost, while realised prices below forecast do not create the same
adverse effect. A static hedge fraction offsets part of that forecast-error
exposure. For each scenario, the residual cost is

```text
average positive forecast error - hedge_fraction * average forecast error
```

The hedge does not improve the forecast itself. It changes the financial exposure
to forecast errors. If prices realise above forecast, the hedge reduces the
upside cost. If prices realise below forecast, the hedge can give up favourable
outcomes.

The mean benchmark evaluates a hedge by its average residual cost. The
risk-adjusted benchmark uses an entropic risk measure applied to costs,

```text
(1 / gamma) * log(E[exp(gamma * cost)])
```

which penalises high-cost scenarios. The optimisation therefore chooses the hedge
fraction that minimises risk-adjusted residual cost, not the hedge fraction that
minimises forecast error.

## Structure

```text
.
├── examples/
│   └── run_case.py
├── tests/
│   └── test_core.py
├── src/
│   └── power_price_risk/
│       ├── costs.py
│       ├── models.py
│       ├── optimisation.py
│       ├── risk.py
│       └── _validation.py
└── pyproject.toml
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python3 examples/run_case.py
```

Without installing the package:

```bash
PYTHONPATH=src python3 examples/run_case.py
```

## Test

```bash
pytest
```

## Notes

The current model uses synthetic lognormal realisations around a fixed hourly
forecast curve.

Planned next steps:

- Add a mean-reverting price model as a more power-specific alternative to the
  independent lognormal benchmark.
- Add one compact trading-analysis layer, for example simple cost/risk diagnostics
  across hedge fractions.
