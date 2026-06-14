"""
Cross-validation of Black-Scholes pricing and implied volatility against
vollib, an independent reference implementation.

The tests in test_black_scholes.py check internal consistency: algebraic
identities (put-call parity, delta identity) and a price -> IV -> price
roundtrip. All of these can pass even if bs_price carries a systematic
bias -- e.g. a wrong sign convention on r in d1/d2 -- because the roundtrip
only checks that bs_price and implied_vol are inverses of each other, not
that either is correct in an absolute sense. The single textbook anchor
(test_bs_price_matches_known_value) covers exactly one point.

This module closes that gap with a grid-based comparison against vollib,
whose implied_volatility wraps Peter Jaeckel's "Let's Be Rational"
algorithm -- a standard reference for IV inversion. Agreement to ~1e-6
(price) and ~1e-4 (IV) across a grid spanning the moneyness/maturity/rate/
vol region relevant to an equity IV surface is the second, independent
anchor referenced in docs/methodology.md, Step 1.

Grid design
-----------
Strikes span +/-15% moneyness, maturities from 1 week to 6 months, rates
from 0% to 4%, vols from 10% to 40% -- the region this project actually
operates in (Step 2's liquidity and no-arbitrage filters would remove
anything more extreme anyway).

For the IV cross-check, combinations where vega < _VEGA_FLOOR are excluded
from the grid at collection time. Deep ITM/OTM options at short maturities
have vega ~ 0: the option price is essentially insensitive to sigma, so IV
is not numerically identifiable by *any* algorithm. This is a statement
about floating-point conditioning, not about either library -- including
these points would make the test measure numerical noise, not correctness.
"""

import itertools

import numpy as np
import pytest
from vollib.black_scholes import black_scholes as vollib_price

from gpvol.iv.black_scholes import bs_greeks, bs_price, implied_vol

S = 100.0
_STRIKES = [85, 90, 95, 100, 105, 110, 115]
_MATURITIES = [7 / 365, 30 / 365, 90 / 365, 180 / 365]
_RATES = [0.0, 0.04]
_VOLS = [0.10, 0.20, 0.40]
_TYPES = ["call", "put"]

_VEGA_FLOOR = 1e-2

_FULL_GRID = list(itertools.product(_STRIKES, _MATURITIES, _RATES, _VOLS, _TYPES))


def _has_meaningful_vega(K, T, r, sigma, option_type) -> bool:
    _, _, vega = bs_greeks(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
    return vega >= _VEGA_FLOOR


_IV_GRID = [params for params in _FULL_GRID if _has_meaningful_vega(*params)]


@pytest.mark.parametrize("K,T,r,sigma,option_type", _FULL_GRID)
def test_bs_price_matches_vollib(K, T, r, sigma, option_type):
    """bs_price must agree with vollib's black_scholes to ~1e-6."""
    flag = "c" if option_type == "call" else "p"

    ours = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
    theirs = vollib_price(flag, S, K, T, r, sigma)

    assert np.isclose(ours, theirs, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("K,T,r,sigma,option_type", _IV_GRID)
def test_implied_vol_matches_vollib(K, T, r, sigma, option_type):
    """
    Cross-library roundtrip: price an option with vollib at a known sigma,
    recover sigma with gpvol's implied_vol, and check agreement.

    This is the converse of the roundtrip in test_black_scholes.py: there,
    bs_price feeds implied_vol (same codebase, so a shared bug could cancel
    out). Here, the market price comes from an independent pricer, so any
    systematic bias in bs_price's formula or implied_vol's inversion would
    show up as a mismatch against sigma.
    """
    flag = "c" if option_type == "call" else "p"

    market_price = vollib_price(flag, S, K, T, r, sigma)
    recovered = implied_vol(market_price, S=S, K=K, T=T, r=r, option_type=option_type)

    assert np.isclose(recovered, sigma, atol=1e-4)
