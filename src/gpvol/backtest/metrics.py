"""
Performance metrics for the walk-forward backtest (Step 7).

All functions operate on pandas Series to match the daily-return time series
produced by the engine. Each function has an explicit docstring explaining
WHAT it measures and WHY that matters economically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, linregress

__all__ = [
    "sharpe_ratio",
    "max_drawdown",
    "deflated_sharpe_ratio",
    "signal_half_life",
]


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualised Sharpe ratio: mean / std * sqrt(periods_per_year).

    Returns 0.0 if std is zero (constant returns -- the strategy
    generates no volatility, so we cannot rank it against alternatives).
    """
    std = returns.std()
    if std == 0.0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough decline as a fraction of the peak value.

        MDD = max_t { (peak_t - equity_t) / peak_t }

    Returns a non-negative number. MDD = 0 means equity never declined.
    """
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return float(abs(drawdown.min()))


def deflated_sharpe_ratio(
    sr: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Corrects for the multiple-testing inflation that arises from trying
    n_trials configurations (e.g., different BO-optimised thresholds) and
    reporting only the best. Without this correction, a Sharpe of 1.5
    found by searching over 30 thresholds is not evidence of alpha -- it
    is the expected maximum of 30 independent samples.

    Returns the probability that the observed SR is genuinely significant,
    corrected for selection bias. DSR < raw_significance for n_trials > 1.

    Parameters
    ----------
    sr : annualised Sharpe ratio
    n_trials : number of configurations tested (e.g., n_bo_calls)
    n_obs : number of return observations (days in OOS period)
    skew, kurt : return distribution moments (default: Gaussian)
    """
    # For n_trials=1: standard significance test (no correction)
    if n_trials <= 1:
        return float(norm.cdf(sr * np.sqrt(n_obs)))

    # Expected maximum SR over n_trials iid Gaussian trials
    # (Bailey & Lopez de Prado 2014, Equation 2)
    euler_gamma = 0.5772156649
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected_max_sr = (1.0 - euler_gamma) * z1 + euler_gamma * z2

    # Non-normality adjustment
    sr_adj = sr * np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)

    dsr = norm.cdf(sr_adj * np.sqrt(n_obs) - expected_max_sr)
    return float(np.clip(dsr, 0.0, 1.0))


def signal_half_life(signal_series: pd.Series) -> float:
    """
    Mean-reversion half-life estimated from an AR(1) fit.

        x_t = alpha * x_{t-1} + epsilon
        half_life = -log(2) / log(alpha)

    A half-life of 2-10 days is consistent with market-maker repricing
    speed and institutional demand pressure (docs/methodology.md, Step 9).
    Very short (<1d): the signal is microstructure noise, not tradable at
    EOD resolution. Very long (>20d): a persistent structural premium, not
    a mean-reverting mispricing.
    """
    x = signal_series.values
    slope, *_ = linregress(x[:-1], x[1:])
    # Clip alpha to (0, 1) to avoid log domain errors
    alpha = float(np.clip(slope, 1e-6, 1.0 - 1e-6))
    return float(-np.log(2.0) / np.log(alpha))
