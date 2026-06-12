"""
Tests for Black-Scholes pricing and implied volatility inversion.

Questi test ancorano le fondamenta dell'intero pipeline: se l'inversione IV
e' sbagliata, ogni superficie, GP fit e segnale a valle e' sbagliato in modo
silenzioso. Scritti prima dell'implementazione (TDD) per fissare il contratto
delle funzioni in gpvol.iv.black_scholes.
"""

import numpy as np
import pytest

from gpvol.iv.black_scholes import bs_price, implied_vol, log_moneyness


def test_bs_price_matches_known_value():
    """Caso da manuale: S=100, K=100, T=1, r=0.05, sigma=0.2 -> call ~10.4506."""
    price = bs_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    assert np.isclose(price, 10.4506, atol=1e-3)


def test_put_call_parity():
    """C - P = S - K * exp(-rT), indipendentemente da sigma."""
    S, K, T, r, sigma = 100, 95, 0.75, 0.03, 0.22
    call = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call")
    put = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put")
    assert np.isclose(call - put, S - K * np.exp(-r * T), atol=1e-8)


def test_implied_vol_roundtrip():
    """Price -> IV -> price deve tornare al punto di partenza."""
    true_sigma = 0.25
    price = bs_price(S=100, K=105, T=0.5, r=0.03, sigma=true_sigma, option_type="put")
    recovered = implied_vol(price, S=100, K=105, T=0.5, r=0.03, option_type="put")
    assert np.isclose(recovered, true_sigma, atol=1e-4)


@pytest.mark.parametrize("sigma", [0.05, 0.15, 0.30, 0.80])
def test_implied_vol_handles_wide_vol_range(sigma):
    """IV inversion deve convergere su un range realistico, incluso
    deep ITM/OTM e regimi high-vol (earnings, crash)."""
    price = bs_price(S=100, K=100, T=0.1, r=0.04, sigma=sigma, option_type="call")
    recovered = implied_vol(price, S=100, K=100, T=0.1, r=0.04, option_type="call")
    assert np.isclose(recovered, sigma, atol=1e-4)


def test_log_moneyness_atm_is_zero():
    """A F = K (forward-ATM), il log-moneyness deve essere esattamente zero."""
    S, K, T, r = 100, 100, 1.0, 0.0
    assert np.isclose(log_moneyness(S=S, K=K, T=T, r=r), 0.0)


def test_log_moneyness_sign():
    """k < 0 per strike sotto il forward (puts OTM), k > 0 sopra (calls OTM)."""
    S, T, r = 100, 1.0, 0.02
    k_low = log_moneyness(S=S, K=90, T=T, r=r)
    k_high = log_moneyness(S=S, K=110, T=T, r=r)
    assert k_low < 0 < k_high
