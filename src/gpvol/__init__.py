"""
gpvol — Gaussian Process Volatility Surface Alpha Research

Package per la ricerca di mispricing locali sulla implied volatility
surface delle opzioni equity, usando un Gaussian Process come modello
non-parametrico di riferimento contro una baseline SVI calibrata via
Bayesian Optimization.

Moduli
------
data          : fetch e cleaning delle options chain
iv            : Black-Scholes pricing, inversione IV, moneyness
surface       : modelli GP e SVI della superficie
optimization  : Bayesian Optimization (calibrazione SVI + soglia)
signal        : composite scoring function
backtest      : walk-forward engine, delta-hedge, performance metrics
viz           : visualizzazione superfici e risultati
"""

__version__ = "0.1.0"
