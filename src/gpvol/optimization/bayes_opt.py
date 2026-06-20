"""
Bayesian Optimisation for walk-forward threshold search (Step 7).

BO is the right tool here -- unlike SVI calibration (see svi_model.py),
the objective function for threshold search is EXPENSIVE:

    Sharpe(tau) requires simulating all trades above tau on the
    training window and computing their daily P&L time series.

With 252 training days and 5-10 contracts per day, each evaluation
is a genuine computation. BO finds the optimal tau in ~30 evaluations
instead of ~500 (grid search at 0.01 resolution over [0.01, 5.0]).

The 1-D search space (tau is a single scalar) is also ideal for BO:
in low dimensions the GP surrogate converges quickly, and the objective
is typically unimodal (higher thresholds filter out noise but also reduce
trade count, creating a peak somewhere in [0.5, 2.0]).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

from gpvol.backtest.metrics import sharpe_ratio

__all__ = ["optimize_threshold_bo"]


def _sharpe_for_threshold(panel: pd.DataFrame, tau: float) -> float:
    """
    Compute daily Sharpe of P&L for all trades with score > tau.

    P&L model (simplified, see docs/methodology.md Step 7):
        P&L_trade = direction * net_deviation
    Daily P&L = mean of P&L_trade for trades entered that day.
    Days with no qualifying trades contribute 0.
    """
    daily_pnl: list[float] = []

    for _, group in panel.groupby("date"):
        trades = group[group["score"] > tau]
        if len(trades) > 0:
            pnl = (trades["direction"] * trades["net_deviation"]).mean()
            daily_pnl.append(float(pnl))
        else:
            daily_pnl.append(0.0)

    if len(daily_pnl) < 2:
        return 0.0

    returns = pd.Series(daily_pnl)
    return sharpe_ratio(returns)


def optimize_threshold_bo(
    panel: pd.DataFrame,
    bounds: tuple[float, float] = (0.01, 5.0),
    n_calls: int = 30,
    random_state: int = 0,
) -> float:
    """
    Find the composite score threshold that maximises Sharpe on panel.

    Parameters
    ----------
    panel : scored DataFrame with columns [date, score, direction, net_deviation]
    bounds : (tau_min, tau_max) search range for the threshold
    n_calls : number of BO evaluations (default 30 -- typically sufficient for 1-D)
    random_state : seed for reproducibility

    Returns
    -------
    float
        Optimal threshold tau* in [bounds[0], bounds[1]].
    """
    lo, hi = bounds

    def objective(x: list) -> float:
        return -_sharpe_for_threshold(panel, x[0])   # gp_minimize minimises

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # n_initial_points must be < n_calls; default skopt value is 10
        n_initial = min(max(2, n_calls // 3), n_calls - 1)
        result = gp_minimize(
            objective,
            [Real(lo, hi)],
            n_calls=n_calls,
            n_initial_points=n_initial,
            random_state=random_state,
            verbose=False,
        )

    return float(result.x[0])
