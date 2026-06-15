"""
Tests for gpvol.data.iv_surface.build_iv_surface.

build_iv_surface is the bridge between raw market data (gpvol.data) and the
Black-Scholes primitives (gpvol.iv.black_scholes, validated in
test_black_scholes.py / test_cross_validation.py). It applies implied_vol
and log_moneyness to every contract in a chain, producing the (k, T) -> IV
grid that gpvol.surface.gp_model fits on (Step 3).

Two failure modes, two behaviors:
- Schema violations (unknown option_type, missing required columns) raise
  immediately -- these indicate a bug in the caller, not a market quirk.
- Data-quality issues (missing quote, a price that violates a static
  no-arbitrage bound and makes implied_vol non-bracketable) become NaN.
  The row is kept: dropping rows is gpvol.data.cleaning's job (Step 2b),
  not this function's. len(output) == len(input) is the core invariant.
"""

import numpy as np
import pandas as pd
import pytest

from gpvol.data.iv_surface import build_iv_surface
from gpvol.iv.black_scholes import bs_price, log_moneyness

VALUATION_DATE = pd.Timestamp("2024-01-15")
S = 100.0
R = 0.04


def _expiry(days):
    return VALUATION_DATE + pd.Timedelta(days=days)


def _row(expiry, strike, option_type, bid, ask, **extra):
    row = {"expiry": expiry, "strike": strike, "option_type": option_type, "bid": bid, "ask": ask}
    row.update(extra)
    return row


def test_recovers_known_iv():
    """A contract priced at a known sigma must recover that sigma."""
    T_days = 30
    T = T_days / 365.0
    true_sigma = 0.22
    price = bs_price(S=S, K=100, T=T, r=R, sigma=true_sigma, option_type="call")

    chain = pd.DataFrame([_row(_expiry(T_days), 100, "call", price, price)])
    out = build_iv_surface(chain, underlying_price=S, valuation_date=VALUATION_DATE, r=R)

    assert np.isclose(out.loc[0, "iv"], true_sigma, atol=1e-4)


def test_log_moneyness_matches_scalar_function():
    """The log_moneyness column must match gpvol.iv.black_scholes.log_moneyness
    exactly -- this is the surface's coordinate system (Step 3)."""
    T_days = 60
    T = T_days / 365.0
    chain = pd.DataFrame([_row(_expiry(T_days), 95, "put", 4.0, 4.5)])
    out = build_iv_surface(chain, underlying_price=S, valuation_date=VALUATION_DATE, r=R)

    expected = log_moneyness(S=S, K=95, T=T, r=R)
    assert np.isclose(out.loc[0, "log_moneyness"], expected)


def test_time_to_maturity_in_years():
    chain = pd.DataFrame([_row(_expiry(73), 100, "call", 5.0, 5.2)])
    out = build_iv_surface(chain, underlying_price=S, valuation_date=VALUATION_DATE, r=R)
    assert np.isclose(out.loc[0, "T"], 73 / 365.0)


def test_expired_option_gets_nan_iv_but_row_is_kept():
    """T <= 0: IV is undefined (division by sqrt(T)). The row stays --
    dropping it is gpvol.data.cleaning's job, not this function's."""
    chain = pd.DataFrame([_row(VALUATION_DATE, 100, "call", 5.0, 5.2)])  # expiry == today -> T = 0
    out = build_iv_surface(chain, underlying_price=S, valuation_date=VALUATION_DATE, r=R)

    assert len(out) == 1
    assert np.isnan(out.loc[0, "iv"])
    assert np.isnan(out.loc[0, "log_moneyness"])


def test_missing_quote_gets_nan_iv():
    """bid/ask both NaN (no quote): mid is NaN, iv must be NaN, row kept."""
    chain = pd.DataFrame([_row(_expiry(30), 100, "call", np.nan, np.nan)])
    out = build_iv_surface(chain, underlying_price=S, valuation_date=VALUATION_DATE, r=R)

    assert len(out) == 1
    assert np.isnan(out.loc[0, "iv"])


def test_price_below_intrinsic_gets_nan_not_an_exception():
    """A mid price that violates the no-arbitrage lower bound makes
    implied_vol raise ValueError (test_black_scholes.py). build_iv_surface
    must catch this and return NaN, not propagate the exception."""
    T = 30 / 365.0
    # Deep ITM call, priced below intrinsic value S - K*exp(-rT):
    # no sigma can produce this price.
    intrinsic = S - 70 * np.exp(-R * T)
    bad_price = intrinsic - 5.0

    chain = pd.DataFrame([_row(_expiry(30), 70, "call", bad_price, bad_price)])
    out = build_iv_surface(chain, underlying_price=S, valuation_date=VALUATION_DATE, r=R)

    assert len(out) == 1
    assert np.isnan(out.loc[0, "iv"])


def test_mixed_valid_and_invalid_rows():
    """One bad row must not affect the others, and the row count is preserved."""
    T_days = 30
    T = T_days / 365.0
    good_price = bs_price(S=S, K=100, T=T, r=R, sigma=0.20, option_type="call")

    chain = pd.DataFrame([
        _row(_expiry(T_days), 100, "call", good_price, good_price),  # valid
        _row(VALUATION_DATE, 100, "put", 5.0, 5.2),                   # T = 0 -> NaN
        _row(_expiry(T_days), 90, "put", np.nan, np.nan),             # no quote -> NaN
    ])
    out = build_iv_surface(chain, underlying_price=S, valuation_date=VALUATION_DATE, r=R)

    assert len(out) == 3
    assert np.isclose(out.loc[0, "iv"], 0.20, atol=1e-4)
    assert np.isnan(out.loc[1, "iv"])
    assert np.isnan(out.loc[2, "iv"])


def test_unknown_option_type_raises():
    """A schema violation (not a market-data quirk) must fail loudly."""
    chain = pd.DataFrame([_row(_expiry(30), 100, "straddle", 5.0, 5.2)])
    with pytest.raises(ValueError, match="option_type"):
        build_iv_surface(chain, underlying_price=S, valuation_date=VALUATION_DATE, r=R)


def test_missing_required_column_raises():
    chain = pd.DataFrame([{"strike": 100, "option_type": "call", "bid": 5.0, "ask": 5.2}])
    with pytest.raises(ValueError, match="expiry"):
        build_iv_surface(chain, underlying_price=S, valuation_date=VALUATION_DATE, r=R)


def test_empty_chain_returns_empty_surface_with_expected_columns():
    chain = pd.DataFrame(columns=["expiry", "strike", "option_type", "bid", "ask"])
    out = build_iv_surface(chain, underlying_price=S, valuation_date=VALUATION_DATE, r=R)

    assert len(out) == 0
    for col in ["T", "mid", "iv", "log_moneyness"]:
        assert col in out.columns


def test_extra_columns_are_preserved():
    """open_interest, volume etc. (used by Step 2b's liquidity filter) must
    pass through untouched."""
    chain = pd.DataFrame([_row(_expiry(30), 100, "call", 5.0, 5.2, open_interest=150, volume=42)])
    out = build_iv_surface(chain, underlying_price=S, valuation_date=VALUATION_DATE, r=R)

    assert out.loc[0, "open_interest"] == 150
    assert out.loc[0, "volume"] == 42
