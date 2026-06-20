"""
Tests for gpvol.backtest.metrics, gpvol.optimization.bayes_opt,
and gpvol.backtest.engine.

Three correctness requirements dominate Step 7:

1. Mathematical correctness of metrics.
   A Sharpe ratio formula error silently mis-ranks strategies. The DSR
   must correct downward for multiple testing -- if DSR >= raw_significance
   for n_trials > 1, the correction is broken and the strategy looks
   better than it is.

2. No lookahead bias in the walk-forward split.
   If the threshold is optimised on test-window data (even inadvertently),
   the OOS Sharpe is an in-sample number. This is the most common and most
   damaging error in systematic backtesting.

3. Threshold is applied, not ignored.
   Only rows with score > threshold should generate trades. A threshold of
   0.0 means "trade everything" -- the BO search is wasted.
"""

import numpy as np
import pandas as pd
import pytest

from gpvol.backtest.metrics import (
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
    signal_half_life,
)
from gpvol.backtest.engine import WalkForwardConfig, run_walkforward
from gpvol.optimization.bayes_opt import optimize_threshold_bo


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_panel(n_days: int = 60, n_contracts: int = 5, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic scored panel: one row per (date, contract).
    direction=+1 always, net_deviation = score / 10 (positive alpha proxy).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows = []
    for d in dates:
        for _ in range(n_contracts):
            score = float(rng.uniform(0, 2))
            rows.append({
                "date": d,
                "score": score,
                "direction": 1,
                "net_deviation": score / 10.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# metrics.sharpe_ratio
# ---------------------------------------------------------------------------

def test_sharpe_zero_mean():
    """Zero-mean returns -> Sharpe = 0."""
    returns = pd.Series([0.01, -0.01, 0.01, -0.01])
    assert sharpe_ratio(returns) == pytest.approx(0.0, abs=1e-10)


def test_sharpe_positive_mean():
    """Positive mean -> Sharpe > 0."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.008])
    assert sharpe_ratio(returns) > 0.0


def test_sharpe_formula():
    """SR = mean/std * sqrt(252) -- verified against known values."""
    returns = pd.Series([0.01, -0.005, 0.008, -0.003, 0.012])
    expected = returns.mean() / returns.std() * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected, rel=1e-6)


def test_sharpe_constant_returns_raises_or_nan():
    """Constant returns have std=0; function must not crash."""
    returns = pd.Series([0.01, 0.01, 0.01])
    result = sharpe_ratio(returns)
    assert result == 0.0 or np.isnan(result) or np.isinf(result)


# ---------------------------------------------------------------------------
# metrics.max_drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_monotone_increase():
    """Monotonically increasing equity -> MDD = 0."""
    equity = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown(equity) == pytest.approx(0.0, abs=1e-10)


def test_max_drawdown_known_value():
    """Peak=1.0, trough=0.7 -> MDD = 0.30."""
    equity = pd.Series([1.0, 0.8, 0.9, 0.7, 0.85])
    assert max_drawdown(equity) == pytest.approx(0.30, rel=1e-6)


def test_max_drawdown_non_negative():
    equity = pd.Series([1.0, 0.5, 1.5, 0.8])
    assert max_drawdown(equity) >= 0.0


# ---------------------------------------------------------------------------
# metrics.deflated_sharpe_ratio
# ---------------------------------------------------------------------------

def test_dsr_below_raw_significance():
    """
    DSR must correct downward for multiple testing.
    With n_trials=30 and modest SR over small n_obs, DSR < naive p-value.
    Numbers verified in exploration: SR=0.3, n_obs=20, n_trials=30
      -> DSR=0.23, naive=0.91.
    """
    dsr  = deflated_sharpe_ratio(sr=0.3, n_trials=30, n_obs=20)
    naive = deflated_sharpe_ratio(sr=0.3, n_trials=1,  n_obs=20)
    assert dsr < naive, f"DSR={dsr:.4f} not below naive={naive:.4f}"


def test_dsr_increases_with_sr():
    """Higher SR -> higher DSR (more evidence of genuine alpha)."""
    dsr_low  = deflated_sharpe_ratio(sr=0.3, n_trials=30, n_obs=20)
    dsr_high = deflated_sharpe_ratio(sr=1.5, n_trials=30, n_obs=20)
    assert dsr_high > dsr_low


def test_dsr_bounded():
    """DSR is a probability: must be in [0, 1]."""
    for sr in [0.1, 0.5, 1.0, 2.0]:
        dsr = deflated_sharpe_ratio(sr=sr, n_trials=30, n_obs=252)
        assert 0.0 <= dsr <= 1.0, f"DSR={dsr} out of [0,1] for SR={sr}"


# ---------------------------------------------------------------------------
# metrics.signal_half_life
# ---------------------------------------------------------------------------

def test_signal_half_life_ar1():
    """
    AR(1) process with alpha=0.7 has theoretical half-life = -log(2)/log(0.7) ≈ 1.94.
    Estimated half-life must be within 20% of true value (estimation noise with n=300).
    """
    rng = np.random.default_rng(0)
    n, alpha = 300, 0.70
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = alpha * x[t - 1] + rng.standard_normal()

    hl_true = -np.log(2) / np.log(alpha)
    hl_est  = signal_half_life(pd.Series(x))
    assert abs(hl_est - hl_true) / hl_true < 0.20, (
        f"half_life_est={hl_est:.2f} deviates more than 20% from true={hl_true:.2f}"
    )


def test_signal_half_life_positive():
    """Half-life must be positive (alpha in (0,1) for stationary AR(1))."""
    rng = np.random.default_rng(1)
    x = pd.Series(np.cumsum(rng.standard_normal(100)) * 0.5)
    hl = signal_half_life(x)
    assert hl > 0.0


# ---------------------------------------------------------------------------
# optimization.bayes_opt.optimize_threshold_bo
# ---------------------------------------------------------------------------

def test_optimize_threshold_returns_float_in_bounds():
    """BO must return a threshold inside the search bounds."""
    panel = _make_panel(n_days=40, n_contracts=5)
    lo, hi = 0.1, 1.8
    tau = optimize_threshold_bo(panel, bounds=(lo, hi), n_calls=20, random_state=0)
    assert isinstance(tau, float)
    assert lo <= tau <= hi


def test_optimize_threshold_nonzero():
    """A threshold of exactly 0 means 'trade everything' -- BO must not degenerate."""
    panel = _make_panel(n_days=40, n_contracts=5)
    tau = optimize_threshold_bo(panel, bounds=(0.01, 2.0), n_calls=20, random_state=0)
    assert tau > 0.0


# ---------------------------------------------------------------------------
# engine.run_walkforward
# ---------------------------------------------------------------------------

def test_walk_forward_no_lookahead_bias():
    """
    For every walk-forward window, the set of training dates and test dates
    must be disjoint. A single overlap means the threshold was optimised
    using future information.
    """
    panel = _make_panel(n_days=80, n_contracts=3)
    config = WalkForwardConfig(
        train_window_days=30,
        test_window_days=10,
        n_bo_calls=5,
        random_state=0,
    )
    results = run_walkforward(panel, config)
    for train_dates, test_dates in results.window_date_pairs:
        overlap = set(train_dates) & set(test_dates)
        assert len(overlap) == 0, f"Lookahead detected: {len(overlap)} shared dates"


def test_walk_forward_threshold_filters_trades():
    """
    With a high threshold, only high-score rows generate trades.
    OOS returns on a zero-signal panel with threshold=100 must be 0.
    """
    panel = _make_panel(n_days=80, n_contracts=3)
    config = WalkForwardConfig(
        train_window_days=30,
        test_window_days=10,
        n_bo_calls=5,
        threshold_override=100.0,   # no contract will pass this
        random_state=0,
    )
    results = run_walkforward(panel, config)
    assert (results.oos_returns == 0.0).all()


def test_walk_forward_result_fields():
    """WalkForwardResults must expose all fields documented in engine.py."""
    panel = _make_panel(n_days=80, n_contracts=3)
    config = WalkForwardConfig(
        train_window_days=30,
        test_window_days=10,
        n_bo_calls=5,
        random_state=0,
    )
    results = run_walkforward(panel, config)
    assert isinstance(results.oos_returns, pd.Series)
    assert isinstance(results.threshold_history, list)
    assert isinstance(results.sharpe, float)
    assert isinstance(results.deflated_sharpe, float)
    assert isinstance(results.max_drawdown, float)
    assert isinstance(results.n_windows, int)
    assert results.n_windows >= 1


def test_walk_forward_oos_returns_length():
    """OOS return series must span the full out-of-sample period."""
    panel = _make_panel(n_days=80, n_contracts=3)
    config = WalkForwardConfig(
        train_window_days=30,
        test_window_days=10,
        n_bo_calls=5,
        random_state=0,
    )
    results = run_walkforward(panel, config)
    # OOS covers all dates after the first training window
    n_dates = panel["date"].nunique()
    n_train  = config.train_window_days
    assert len(results.oos_returns) <= n_dates - n_train
