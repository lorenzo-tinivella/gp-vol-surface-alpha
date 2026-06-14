"""
Tests for Black-Scholes pricing and implied volatility inversion.

These tests anchor the foundations of the entire pipeline: if the IV
inversion is wrong, every downstream surface, GP fit, and signal is wrong,
silently. Written before the implementation (TDD) to pin down the contract
of the functions in gpvol.iv.black_scholes.
"""

import numpy as np
import pytest

from gpvol.iv.black_scholes import bs_greeks, bs_price, implied_vol, log_moneyness


def test_bs_price_matches_known_value():
    """Textbook case: S=100, K=100, T=1, r=0.05, sigma=0.2 -> call ~10.4506."""
    price = bs_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    assert np.isclose(price, 10.4506, atol=1e-3)


def test_put_call_parity():
    """C - P = S - K * exp(-rT), independent of sigma."""
    S, K, T, r, sigma = 100, 95, 0.75, 0.03, 0.22
    call = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call")
    put = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put")
    assert np.isclose(call - put, S - K * np.exp(-r * T), atol=1e-8)


def test_implied_vol_roundtrip():
    """Price -> IV -> price should round-trip back to the starting point."""
    true_sigma = 0.25
    price = bs_price(S=100, K=105, T=0.5, r=0.03, sigma=true_sigma, option_type="put")
    recovered = implied_vol(price, S=100, K=105, T=0.5, r=0.03, option_type="put")
    assert np.isclose(recovered, true_sigma, atol=1e-4)


@pytest.mark.parametrize("sigma", [0.05, 0.15, 0.30, 0.80])
def test_implied_vol_handles_wide_vol_range(sigma):
    """IV inversion must converge over a realistic range, including
    deep ITM/OTM and high-vol regimes (earnings, crashes)."""
    price = bs_price(S=100, K=100, T=0.1, r=0.04, sigma=sigma, option_type="call")
    recovered = implied_vol(price, S=100, K=100, T=0.1, r=0.04, option_type="call")
    assert np.isclose(recovered, sigma, atol=1e-4)


def test_log_moneyness_atm_is_zero():
    """At F = K (forward-ATM), log-moneyness must be exactly zero."""
    S, K, T, r = 100, 100, 1.0, 0.0
    assert np.isclose(log_moneyness(S=S, K=K, T=T, r=r), 0.0)


def test_log_moneyness_sign():
    """k < 0 for strikes below the forward (OTM puts), k > 0 above (OTM calls)."""
    S, T, r = 100, 1.0, 0.02
    k_low = log_moneyness(S=S, K=90, T=T, r=r)
    k_high = log_moneyness(S=S, K=110, T=T, r=r)
    assert k_low < 0 < k_high


def test_delta_call_minus_put_equals_one():
    """delta_call - delta_put = 1 (structural identity: N(d1) - (N(d1)-1))."""
    delta_c, _, _ = bs_greeks(S=100, K=95, T=0.5, r=0.02, sigma=0.30, option_type="call")
    delta_p, _, _ = bs_greeks(S=100, K=95, T=0.5, r=0.02, sigma=0.30, option_type="put")
    assert np.isclose(delta_c - delta_p, 1.0, atol=1e-10)


def test_gamma_and_vega_positive():
    """Gamma and vega are always positive for vanilla options (BS property)."""
    _, gamma, vega = bs_greeks(S=100, K=100, T=0.25, r=0.03, sigma=0.20, option_type="call")
    assert gamma > 0
    assert vega > 0
