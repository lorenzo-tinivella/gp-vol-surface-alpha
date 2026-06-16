"""
Tests for gpvol.data.cleaning: filter_liquidity and filter_static_arbitrage.

Pipeline context
----------------
Both functions receive the output of build_iv_surface (Step 1), which
already carries T, mid, iv, log_moneyness. Their only job is to decide
which rows to DROP -- build_iv_surface never drops rows, so every
data-quality decision lands here (Step 2b).

Two functions, two responsibilities:
- filter_liquidity   : market-data quality gate (iv NaN, OI, spread, bid)
- filter_static_arbitrage : no-arbitrage structural gate
                            (butterfly convexity in price space;
                             calendar monotonicity in total-variance space)

Butterfly arithmetic (made explicit in the test):
  Interpolated price at K=100 between (K=90, mid=10.0) and (K=110, mid=0.1):
    interp = 10.0 + (0.1-10.0)*(100-90)/(110-90) = 5.05
  mid=4.0 at K=100 passes (4.0 <= 5.05); mid=6.0 violates (6.0 > 5.05).

Calendar arithmetic -- uses total variance w = iv^2 * T (Gatheral & Jacquier 2014):
  VIOLATION: w(T=30d) = 0.40^2 * 30/365 = 0.01315
             w(T=60d) = 0.20^2 * 60/365 = 0.00658  -> w decreasing -> violation
             -> shorter-dated contract (T=30d) is dropped (stale high-IV quote)
  VALID:     w(T=30d) = 0.20^2 * 30/365 = 0.00329
             w(T=60d) = 0.25^2 * 60/365 = 0.01027  -> w non-decreasing -> valid
"""

import numpy as np
import pandas as pd
import pytest

from gpvol.data.cleaning import filter_liquidity, filter_static_arbitrage

_EXPIRY_30 = pd.Timestamp("2024-02-14")
_EXPIRY_60 = pd.Timestamp("2024-03-15")
_T_30 = 30 / 365.0
_T_60 = 60 / 365.0


def _make_chain(rows: list) -> pd.DataFrame:
    """
    Build a minimal DataFrame that mimics the output of build_iv_surface.
    Default values represent a valid, liquid ATM call -- override per test.
    mid is auto-computed from bid/ask unless provided explicitly.
    Returns a properly-columned empty DataFrame when rows=[].
    """
    defaults = {
        "expiry": _EXPIRY_30,
        "strike": 100.0,
        "option_type": "call",
        "bid": 2.0,
        "ask": 2.1,
        "open_interest": 500,
        "iv": 0.20,
        "T": _T_30,
        "log_moneyness": 0.0,
    }
    if not rows:
        cols = list(defaults.keys()) + ["mid"]
        return pd.DataFrame(columns=cols)

    records = [{**defaults, **r} for r in rows]
    df = pd.DataFrame(records)
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
    return df


# ---------------------------------------------------------------------------
# filter_liquidity
# ---------------------------------------------------------------------------

def test_liquidity_drops_low_open_interest():
    df = _make_chain([{"open_interest": 10}])
    out = filter_liquidity(df, min_open_interest=50)
    assert len(out) == 0


def test_liquidity_drops_zero_bid():
    df = _make_chain([{"bid": 0.0, "ask": 0.0}])
    out = filter_liquidity(df)
    assert len(out) == 0


def test_liquidity_drops_wide_spread():
    # spread_pct = (2.0 - 1.0) / 1.5 = 0.667 > 0.5 -> drop
    df = _make_chain([{"bid": 1.0, "ask": 2.0}])
    out = filter_liquidity(df, max_spread_pct=0.5)
    assert len(out) == 0


def test_liquidity_drops_nan_iv():
    """iv=NaN means build_iv_surface could not invert the price:
    the contract already violated a no-arbitrage bound and its price
    is unreliable even for the butterfly check downstream."""
    df = _make_chain([{"iv": np.nan}])
    out = filter_liquidity(df)
    assert len(out) == 0


def test_liquidity_keeps_valid_row():
    df = _make_chain([{}])
    out = filter_liquidity(df)
    assert len(out) == 1


def test_liquidity_preserves_extra_columns():
    """Columns beyond the filter criteria must survive unchanged."""
    df = _make_chain([{"log_moneyness": -0.05, "T": _T_30}])
    out = filter_liquidity(df)
    assert out.loc[out.index[0], "log_moneyness"] == pytest.approx(-0.05)
    assert out.loc[out.index[0], "T"] == pytest.approx(_T_30)


# ---------------------------------------------------------------------------
# filter_static_arbitrage
# ---------------------------------------------------------------------------

def test_static_arbitrage_drops_nan_iv():
    """Defensive: NaN iv contracts are dropped even if they slipped past
    filter_liquidity. They cannot participate in the total-variance calendar
    check and their mid prices are unreliable for the butterfly check."""
    df = _make_chain([{"iv": np.nan}])
    out = filter_static_arbitrage(df)
    assert len(out) == 0


def test_butterfly_violation_drops_middle_strike():
    """
    Three calls, same expiry:
      K=90,  mid=10.0
      K=100, mid=6.0   <- violates: interp=5.05, 6.0 > 5.05
      K=110, mid=0.1
    Only K=100 should be dropped; wings are kept.
    """
    df = _make_chain([
        {"strike": 90.0,  "mid": 10.0, "bid": 9.90,  "ask": 10.10},
        {"strike": 100.0, "mid": 6.0,  "bid": 5.90,  "ask": 6.10},
        {"strike": 110.0, "mid": 0.1,  "bid": 0.05,  "ask": 0.15},
    ])
    out = filter_static_arbitrage(df)
    assert len(out) == 2
    assert 100.0 not in out["strike"].values


def test_valid_butterfly_keeps_all_strikes():
    """K=100 at mid=4.0 < interp=5.05 -> convex, all three kept."""
    df = _make_chain([
        {"strike": 90.0,  "mid": 10.0, "bid": 9.90,  "ask": 10.10},
        {"strike": 100.0, "mid": 4.0,  "bid": 3.90,  "ask": 4.10},
        {"strike": 110.0, "mid": 0.1,  "bid": 0.05,  "ask": 0.15},
    ])
    out = filter_static_arbitrage(df)
    assert len(out) == 3


def test_butterfly_not_checked_with_fewer_than_three_strikes():
    """Two strikes cannot form a triple -- both rows must pass through."""
    df = _make_chain([
        {"strike": 90.0,  "mid": 10.0, "bid": 9.90, "ask": 10.10},
        {"strike": 100.0, "mid": 9.5,  "bid": 9.40, "ask": 9.60},
    ])
    out = filter_static_arbitrage(df)
    assert len(out) == 2


def test_calendar_violation_drops_shorter_dated_option():
    """
    Total variance w = iv^2 * T must be non-decreasing in T.

    w(T=30d) = 0.40^2 * 30/365 = 0.01315
    w(T=60d) = 0.20^2 * 60/365 = 0.00658   <- w DECREASING -> violation

    The shorter-dated option (T=30d) is dropped: a high IV on a short-dated
    contract relative to a longer-dated one indicates a stale or distorted
    quote (e.g. an earnings spike that has already resolved).
    The longer-dated option (T=60d) is kept.
    """
    df = _make_chain([
        {"expiry": _EXPIRY_30, "T": _T_30, "iv": 0.40,
         "mid": 5.0, "bid": 4.90, "ask": 5.10},
        {"expiry": _EXPIRY_60, "T": _T_60, "iv": 0.20,
         "mid": 3.0, "bid": 2.90, "ask": 3.10},
    ])
    out = filter_static_arbitrage(df)
    assert len(out) == 1
    assert out["expiry"].iloc[0] == _EXPIRY_60


def test_valid_calendar_keeps_both_expiries():
    """
    w(T=30d) = 0.20^2 * 30/365 = 0.00329
    w(T=60d) = 0.25^2 * 60/365 = 0.01027   <- w INCREASING -> valid
    Both contracts are kept.
    """
    df = _make_chain([
        {"expiry": _EXPIRY_30, "T": _T_30, "iv": 0.20,
         "mid": 5.0, "bid": 4.90, "ask": 5.10},
        {"expiry": _EXPIRY_60, "T": _T_60, "iv": 0.25,
         "mid": 7.0, "bid": 6.90, "ask": 7.10},
    ])
    out = filter_static_arbitrage(df)
    assert len(out) == 2


def test_calendar_not_checked_with_single_expiry_per_strike():
    """Only one expiry at this strike -- no comparison possible, row kept."""
    df = _make_chain([
        {"expiry": _EXPIRY_30, "T": _T_30, "iv": 0.20,
         "mid": 5.0, "bid": 4.90, "ask": 5.10},
    ])
    out = filter_static_arbitrage(df)
    assert len(out) == 1


def test_empty_dataframe_returns_empty():
    df = _make_chain([])
    assert len(filter_liquidity(df)) == 0
    assert len(filter_static_arbitrage(df)) == 0
