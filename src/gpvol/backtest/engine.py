"""
Walk-forward backtest engine (Step 7).

Architecture
------------
The engine takes a scored panel (one row per contract per day, with
score/direction/net_deviation columns from gpvol.signal.scoring) and
runs a rolling walk-forward:

    for each window (train=[t-W, t], test=[t, t+Delta]):
        1. BO finds tau* that maximises Sharpe on the training trades
        2. Apply tau* to the test window -> OOS daily P&L
        3. Accumulate OOS returns across all windows

The output is a WalkForwardResults object with the OOS equity curve,
per-window thresholds, and aggregated performance metrics.

P&L model
---------
For each day d in the test window:
    trades = rows where score > tau*
    daily_pnl = mean(direction * net_deviation)  for trades

This assumes full mean reversion to mu_GP (the net_deviation is already
bid-ask adjusted). Delta-hedging (Step 8) refines this estimate.

No-lookahead guarantee
----------------------
The training set for window i is strictly [t-W, t). The test set is
[t, t+Delta). These are disjoint by construction, verified in
test_backtest.py::test_walk_forward_no_lookahead_bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gpvol.backtest.metrics import (
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
)
from gpvol.optimization.bayes_opt import optimize_threshold_bo

__all__ = ["WalkForwardConfig", "WalkForwardResults", "run_walkforward"]


@dataclass
class WalkForwardConfig:
    """
    Parameters for the walk-forward backtest.

    Attributes
    ----------
    train_window_days : number of trading days in the training window
    test_window_days  : number of trading days in each test (OOS) window
    threshold_bounds  : (tau_min, tau_max) search range for BO
    n_bo_calls        : number of BO evaluations per window
    random_state      : seed for BO reproducibility
    threshold_override: if set, skip BO and use this fixed threshold
                        (used in tests to verify filtering logic)
    """
    train_window_days: int = 252
    test_window_days: int = 63
    threshold_bounds: tuple[float, float] = (0.01, 5.0)
    n_bo_calls: int = 30
    random_state: int = 0
    threshold_override: float | None = None


@dataclass
class WalkForwardResults:
    """
    Output of run_walkforward.

    Attributes
    ----------
    oos_returns       : daily OOS P&L series (index = date)
    threshold_history : tau* selected by BO for each window
    window_date_pairs : list of (train_dates, test_dates) for lookahead audit
    sharpe            : annualised Sharpe of OOS returns
    deflated_sharpe   : DSR corrected for n_bo_calls (Bailey & Lopez de Prado)
    max_drawdown      : peak-to-trough OOS equity decline
    n_windows         : number of walk-forward windows completed
    n_trades          : total number of trades taken in OOS period
    """
    oos_returns: pd.Series
    threshold_history: list[float]
    window_date_pairs: list[tuple]
    sharpe: float
    deflated_sharpe: float
    max_drawdown: float
    n_windows: int
    n_trades: int


def run_walkforward(
    panel: pd.DataFrame,
    config: WalkForwardConfig,
) -> WalkForwardResults:
    """
    Run rolling walk-forward backtest on a scored panel.

    Parameters
    ----------
    panel : DataFrame with columns [date, score, direction, net_deviation].
        One row per scored contract per day. Produced by score_surface()
        in gpvol.signal.scoring after fitting GP and SVI models.
    config : WalkForwardConfig

    Returns
    -------
    WalkForwardResults
    """
    dates = sorted(panel["date"].unique())
    n_dates = len(dates)
    W = config.train_window_days
    Delta = config.test_window_days

    oos_returns_all: list[tuple[pd.Timestamp, float]] = []
    threshold_history: list[float] = []
    window_date_pairs: list[tuple] = []
    total_trades = 0

    # Roll forward in steps of test_window_days
    window_start = 0
    while window_start + W < n_dates:
        train_end = window_start + W
        test_end  = min(train_end + Delta, n_dates)

        train_dates = dates[window_start:train_end]
        test_dates  = dates[train_end:test_end]

        if len(test_dates) == 0:
            break

        window_date_pairs.append((list(train_dates), list(test_dates)))

        panel_train = panel[panel["date"].isin(train_dates)]
        panel_test  = panel[panel["date"].isin(test_dates)]

        # --- Find optimal threshold (BO or override) ---
        if config.threshold_override is not None:
            tau = config.threshold_override
        else:
            tau = optimize_threshold_bo(
                panel_train,
                bounds=config.threshold_bounds,
                n_calls=config.n_bo_calls,
                random_state=config.random_state,
            )

        threshold_history.append(tau)

        # --- Apply threshold to test window ---
        for date in test_dates:
            day_panel = panel_test[panel_test["date"] == date]
            trades = day_panel[day_panel["score"] > tau]

            if len(trades) > 0:
                pnl = (trades["direction"] * trades["net_deviation"]).mean()
                total_trades += len(trades)
            else:
                pnl = 0.0

            oos_returns_all.append((date, float(pnl)))

        window_start += Delta  # non-overlapping test windows

    if not oos_returns_all:
        empty = pd.Series([], dtype=float)
        return WalkForwardResults(
            oos_returns=empty,
            threshold_history=threshold_history,
            window_date_pairs=window_date_pairs,
            sharpe=0.0,
            deflated_sharpe=0.0,
            max_drawdown=0.0,
            n_windows=0,
            n_trades=0,
        )

    oos_dates, oos_pnl = zip(*oos_returns_all)
    oos_returns = pd.Series(list(oos_pnl), index=list(oos_dates), name="pnl")

    sr  = sharpe_ratio(oos_returns)
    dsr = deflated_sharpe_ratio(
        sr=sr,
        n_trials=config.n_bo_calls,
        n_obs=len(oos_returns),
    )
    mdd = max_drawdown((1 + oos_returns).cumprod())

    return WalkForwardResults(
        oos_returns=oos_returns,
        threshold_history=threshold_history,
        window_date_pairs=window_date_pairs,
        sharpe=sr,
        deflated_sharpe=dsr,
        max_drawdown=mdd,
        n_windows=len(threshold_history),
        n_trades=total_trades,
    )
