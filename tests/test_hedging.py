"""
Tests for gpvol.backtest.hedging (updated with transaction costs).

The key economic insight driving these tests:
    commission_per_contract is the dominant cost for small positions.
    A $0.65 round-trip on a position generating $0.35 gross Gamma P&L
    makes the trade unprofitable. Tests verify this quantitatively via
    cost_breakdown() and via the simulated delta_hedge_pnl().

Test structure:
    - daily_gamma_pnl: formula correctness + cost deduction
    - delta_hedge_pnl: sign, length, cost impact, put/call symmetry
    - cost_breakdown:  analytical cost decomposition scenarios
"""

import numpy as np
import pandas as pd
import pytest

from gpvol.backtest.hedging import (
    cost_breakdown,
    daily_gamma_pnl,
    delta_hedge_pnl,
)

_S, _K, _T, _R, _SIGMA = 100.0, 100.0, 0.25, 0.04, 0.20
_GAMMA = 0.039448   # pre-computed, verified


# ---------------------------------------------------------------------------
# daily_gamma_pnl
# ---------------------------------------------------------------------------

def test_gamma_pnl_positive_when_realized_above_iv():
    pnl = daily_gamma_pnl(
        gamma=_GAMMA, S=_S,
        r_daily=0.25 / np.sqrt(252),
        sigma_iv=_SIGMA,
    )
    assert pnl > 0.0


def test_gamma_pnl_negative_when_realized_below_iv():
    pnl = daily_gamma_pnl(
        gamma=_GAMMA, S=_S,
        r_daily=0.15 / np.sqrt(252),
        sigma_iv=_SIGMA,
    )
    assert pnl < 0.0


def test_gamma_pnl_zero_at_breakeven():
    r_be = _SIGMA / np.sqrt(252)
    pnl = daily_gamma_pnl(gamma=_GAMMA, S=_S, r_daily=r_be, sigma_iv=_SIGMA)
    assert abs(pnl) < 1e-10


def test_gamma_pnl_borrow_reduces_pnl():
    """Higher borrow rate must lower the net P&L."""
    r_daily = 0.25 / np.sqrt(252)
    pnl_low  = daily_gamma_pnl(_GAMMA, _S, r_daily, _SIGMA, delta=0.55, borrow_rate=0.003)
    pnl_high = daily_gamma_pnl(_GAMMA, _S, r_daily, _SIGMA, delta=0.55, borrow_rate=0.10)
    assert pnl_low > pnl_high


def test_gamma_pnl_slippage_reduces_pnl():
    """Rebalancing slippage must lower the net P&L."""
    r_daily = 0.25 / np.sqrt(252)
    pnl_no_slip = daily_gamma_pnl(_GAMMA, _S, r_daily, _SIGMA,
                                   delta=0.55, delta_prev=0.54, slippage_pct=0.0)
    pnl_slip    = daily_gamma_pnl(_GAMMA, _S, r_daily, _SIGMA,
                                   delta=0.55, delta_prev=0.54, slippage_pct=0.01)
    assert pnl_no_slip > pnl_slip


def test_gamma_pnl_no_slippage_when_delta_unchanged():
    """If delta does not change, no rebalancing occurs and slippage = 0."""
    r_daily = 0.25 / np.sqrt(252)
    pnl_same_delta = daily_gamma_pnl(_GAMMA, _S, r_daily, _SIGMA,
                                      delta=0.55, delta_prev=0.55, slippage_pct=0.05)
    pnl_no_prev    = daily_gamma_pnl(_GAMMA, _S, r_daily, _SIGMA,
                                      delta=0.55, delta_prev=None, slippage_pct=0.05)
    # same-delta: slippage term is zero regardless of slippage_pct
    pnl_zero_slip  = daily_gamma_pnl(_GAMMA, _S, r_daily, _SIGMA,
                                      delta=0.55, delta_prev=0.55, slippage_pct=0.0)
    assert np.isclose(pnl_same_delta, pnl_zero_slip)


# ---------------------------------------------------------------------------
# delta_hedge_pnl
# ---------------------------------------------------------------------------

def _price_path(sigma_ann: float, n_days: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    daily_vol = sigma_ann / np.sqrt(252)
    returns = rng.normal(0.0, daily_vol, n_days)
    prices = _S * np.cumprod(1 + returns)
    return pd.Series(prices)


def test_pnl_positive_when_realized_above_iv():
    prices = _price_path(sigma_ann=0.30, n_days=20)
    pnl = delta_hedge_pnl(prices, _K, _T, _R, _SIGMA, commission_per_contract=0.0)
    assert pnl.sum() > 0.0


def test_pnl_negative_when_realized_below_iv():
    prices = _price_path(sigma_ann=0.10, n_days=20)
    pnl = delta_hedge_pnl(prices, _K, _T, _R, _SIGMA, commission_per_contract=0.0)
    assert pnl.sum() < 0.0


def test_pnl_returns_correct_length():
    prices = _price_path(sigma_ann=0.20, n_days=15)
    pnl = delta_hedge_pnl(prices, _K, _T, _R, _SIGMA)
    assert isinstance(pnl, pd.Series)
    assert len(pnl) == 15


def test_commission_reduces_net_pnl():
    """Higher commission must reduce cumulative P&L."""
    prices = _price_path(sigma_ann=0.30, n_days=20)
    pnl_zero = delta_hedge_pnl(prices, _K, _T, _R, _SIGMA, commission_per_contract=0.00)
    pnl_comm = delta_hedge_pnl(prices, _K, _T, _R, _SIGMA, commission_per_contract=0.65)
    assert pnl_zero.sum() > pnl_comm.sum()


def test_commission_charged_at_entry_and_exit():
    """Total commission deducted must equal 2 * commission_per_contract."""
    prices = _price_path(sigma_ann=0.20, n_days=10)
    comm = 0.50
    pnl_zero = delta_hedge_pnl(prices, _K, _T, _R, _SIGMA, commission_per_contract=0.00)
    pnl_comm = delta_hedge_pnl(prices, _K, _T, _R, _SIGMA, commission_per_contract=comm)
    difference = pnl_zero.sum() - pnl_comm.sum()
    assert np.isclose(difference, 2 * comm, atol=1e-10)


def test_high_commission_makes_trade_unprofitable():
    """
    $0.65 retail commission on a 20-day ATM position with modest vol
    advantage should make the trade unprofitable -- the commission
    ($1.30 round-trip) exceeds the gross Gamma P&L (~$0.35).
    This is the key break-even insight documented in the module docstring.
    """
    prices = _price_path(sigma_ann=0.25, n_days=20, seed=0)
    pnl = delta_hedge_pnl(prices, _K, _T, _R, _SIGMA,
                           commission_per_contract=0.65)
    assert pnl.sum() < 0.0, (
        f"Expected negative net P&L with $0.65 commission, got {pnl.sum():.4f}"
    )


def test_put_and_call_same_gamma_pnl():
    """Calls and puts at same strike have same Gamma -- net P&L must match."""
    prices = _price_path(sigma_ann=0.30, n_days=20, seed=7)
    pnl_c = delta_hedge_pnl(prices, _K, _T, _R, _SIGMA, "call",
                             commission_per_contract=0.0).sum()
    pnl_p = delta_hedge_pnl(prices, _K, _T, _R, _SIGMA, "put",
                             commission_per_contract=0.0).sum()
    assert np.isclose(pnl_c, pnl_p, rtol=0.05)


# ---------------------------------------------------------------------------
# cost_breakdown
# ---------------------------------------------------------------------------

def test_cost_breakdown_spy_high_retention():
    """SPY with zero commission retains > 85% of gross Gamma P&L."""
    result = cost_breakdown(
        S=_S, K=_K, T=_T, r=_R, sigma_iv=_SIGMA, sigma_real=0.25,
        n_days=20, borrow_rate=0.003, commission_per_contract=0.0,
        slippage_pct=0.0001,
    )
    assert result["pct_retained"] > 85.0, (
        f"SPY zero-comm should retain >85%, got {result['pct_retained']}%"
    )


def test_cost_breakdown_retail_commission_kills_pnl():
    """$0.65 retail commission makes the trade unprofitable (net < 0)."""
    result = cost_breakdown(
        S=_S, K=_K, T=_T, r=_R, sigma_iv=_SIGMA, sigma_real=0.25,
        n_days=20, borrow_rate=0.003, commission_per_contract=0.65,
        slippage_pct=0.0001,
    )
    assert result["net"] < 0.0, (
        f"Retail commission should make trade unprofitable, got net={result['net']}"
    )


def test_cost_breakdown_gross_positive_when_realized_above_iv():
    """Gross P&L must be positive when sigma_real > sigma_iv."""
    result = cost_breakdown(
        S=_S, K=_K, T=_T, r=_R, sigma_iv=0.20, sigma_real=0.25,
        n_days=20,
    )
    assert result["gross"] > 0.0


def test_cost_breakdown_components_sum_to_net():
    """gross + borrow + commission + slippage = net (accounting identity)."""
    result = cost_breakdown(
        S=_S, K=_K, T=_T, r=_R, sigma_iv=_SIGMA, sigma_real=0.25,
        n_days=20, borrow_rate=0.01, commission_per_contract=0.25,
        slippage_pct=0.0003,
    )
    total = result["gross"] + result["borrow"] + result["commission"] + result["slippage"]
    assert np.isclose(total, result["net"], atol=1e-6)
