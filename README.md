# GP Vol Surface Alpha

**Detecting local mispricings on the equity options implied volatility surface via
Gaussian Process regression, validated against a Bayesian-Optimized SVI baseline.**

---

## TL;DR

- **Thesis**: market makers calibrate parametric models (SVI) mainly on liquid,
  near-the-money points and extrapolate into the wings. This extrapolation carries
  systematic bias, especially where institutional demand (e.g. pension funds buying
  OTM puts) creates persistent, localized pressure on the surface
  (Garleanu, Pedersen & Poteshman, 2009).
- **Method**: a Gaussian Process gives a non-parametric "consensus" estimate of the
  surface with calibrated uncertainty. Where the GP and an SVI baseline both disagree
  with the observed market price — beyond the bid-ask cost — we have a candidate signal.
- **Validation**: walk-forward backtest with a BO-tuned decision threshold,
  delta-hedged P&L isolation, deflated Sharpe ratio.
- **Status**: work in progress — see [Roadmap](#roadmap).

---

## Table of Contents

1. [Economic Motivation](#economic-motivation)
2. [Methodology](#methodology)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Results](#results)
7. [Limitations & Honest Caveats](#limitations--honest-caveats)
8. [Roadmap](#roadmap)
9. [References](#references)

---

## Economic Motivation

The Variance Risk Premium (Carr & Wu, 2009) means implied volatility is, on average,
priced above realized volatility — option sellers are compensated for bearing tail
risk. But the surface is not uniformly mispriced: the size of this premium varies
across strikes and maturities depending on liquidity and institutional demand
structure.

This project looks for *local* deviations from a model-consistent surface, filters
them for economic plausibility (calendar effects, liquidity, model agreement,
transaction costs), and tests whether the residual signal is tradable after
delta-hedging.

Full economic and mathematical writeup: [`docs/methodology.md`](docs/methodology.md)

## Methodology

| Step | What | Key tool |
|---|---|---|
| 1 | Build daily IV surface from the options chain | Black-Scholes inversion |
| 2 | Clean: liquidity + static no-arbitrage filters | Breeden-Litzenberger conditions |
| 3 | Fit a non-parametric reference surface | Gaussian Process (composite kernel) |
| 4 | Fit a parametric baseline | SVI, calibrated via Bayesian Optimization |
| 5 | Score local deviations | Composite score (z-score × calendar × confidence × consistency × net deviation) |
| 6 | Backtest | Walk-forward, BO-tuned threshold, delta-hedged P&L |
| 7 | Diagnose | Signal half-life, cross-sectional consistency, deflated Sharpe |

## Repository Structure

```
gp-vol-surface-alpha/
├── src/gpvol/
│   ├── data/            # fetch + clean options chains
│   ├── iv/               # Black-Scholes, IV inversion, moneyness
│   ├── surface/          # GP model + SVI model
│   ├── optimization/     # Bayesian Optimization
│   ├── signal/           # composite scoring
│   ├── backtest/         # walk-forward, delta-hedge, metrics
│   └── viz/              # plotly surfaces, equity curves
├── notebooks/            # thin orchestration layers, one per stage
├── tests/                # unit tests for every mathematical primitive
├── configs/default.yaml  # all tunable parameters
├── docs/methodology.md   # full economic + mathematical writeup
└── data/                 # gitignored — see data/README.md
```

## Installation

```bash
git clone https://github.com/lorenzo-tinivella/gp-vol-surface-alpha.git
cd gp-vol-surface-alpha
pip install -e ".[dev]"
```

## Quickstart

```python
from gpvol.data.loader import fetch_option_chain
from gpvol.iv.black_scholes import implied_vol_surface
from gpvol.surface.gp_model import GPSurface

chain = fetch_option_chain("SPY", date="2024-01-15")
surface = implied_vol_surface(chain)

gp = GPSurface()
gp.fit(surface)
mu, sigma = gp.predict_grid()
```

## Results

Pending — see [Roadmap](#roadmap).

## Limitations & Honest Caveats

- **Data quality**: built on `yfinance` EOD options data. Bid-ask spreads on
  illiquid strikes can exceed the magnitude of the signal itself —
  `docs/methodology.md` details how this is handled (net-of-spread scoring) and
  what production-grade data (OptionMetrics, CBOE DataShop) would change.
- **GP vs SVI disagreement ≠ "the market is wrong"**: the signal assumes the
  GP/SVI consensus is closer to fair value than the observed quote. This is a
  modeling assumption, tested indirectly via signal decay and cross-sectional
  consistency diagnostics — not a certainty.
- **Scope**: currently single-name / index equity options. The GP+BO methodology
  transfers structurally to FX vol surfaces or swaption grids, but that extension
  is not implemented here.

## Roadmap

- [ ] Data pipeline + IV surface construction (`gpvol.data`, `gpvol.iv`)
- [ ] GP surface model with composite kernel (`gpvol.surface.gp_model`)
- [ ] SVI calibration via Bayesian Optimization (`gpvol.surface.svi_model`, `gpvol.optimization`)
- [ ] Composite scoring function (`gpvol.signal`)
- [ ] Walk-forward backtest + delta-hedge P&L (`gpvol.backtest`)
- [ ] Diagnostics: signal half-life, cross-sectional consistency, deflated Sharpe

## References

Full bibliography: [`docs/methodology.md`](docs/methodology.md#full-bibliography)

## License

MIT — see [LICENSE](LICENSE)
