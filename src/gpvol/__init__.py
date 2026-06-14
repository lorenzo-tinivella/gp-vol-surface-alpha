"""
gpvol — Gaussian Process Volatility Surface Alpha Research

Research package for detecting local mispricings on the equity options
implied volatility surface, using a Gaussian Process as a non-parametric
reference model against an SVI baseline calibrated via Bayesian Optimization.

Modules
-------
data          : fetching and cleaning options chain data
iv            : Black-Scholes pricing, IV inversion, moneyness
surface       : GP and SVI surface models
optimization  : Bayesian Optimization (SVI calibration + threshold search)
signal        : composite scoring function
backtest      : walk-forward engine, delta-hedge, performance metrics
viz           : surface and result visualization
"""

__version__ = "0.1.0"
