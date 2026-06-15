"""
Construction of the implied volatility surface from a raw option chain.

build_iv_surface() is the bridge between gpvol.data (raw market data) and
gpvol.iv (Black-Scholes math, Step 1): it applies implied_vol and
log_moneyness to every contract in a chain, producing the (k, T) -> IV grid
that gpvol.surface.gp_model fits on (Step 3).

Contract
--------
- Never raises on bad market data: a contract whose mid price is missing,
  non-positive, or violates a static no-arbitrage bound (implied_vol raises
  ValueError -- see test_black_scholes.py) gets iv = NaN and
  log_moneyness = NaN. The row is kept -- dropping rows is
  gpvol.data.cleaning's job (Step 2b), not this function's.
  len(output) == len(input) always.
- DOES raise on schema violations (unknown option_type, missing required
  columns): those indicate a bug in the caller, not a market-data quirk,
  and should fail loudly.

Time-to-maturity convention: ACT/365 (calendar days / 365), no holiday
calendar. This is a simplification -- see README "Limitations".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gpvol.iv.black_scholes import implied_vol, log_moneyness

__all__ = ["build_iv_surface"]

_REQUIRED_COLUMNS = {"expiry", "strike", "option_type", "bid", "ask"}
_VALID_TYPES = {"call", "put"}


def build_iv_surface(
    chain: pd.DataFrame,
    underlying_price: float,
    valuation_date: pd.Timestamp,
    r: float,
) -> pd.DataFrame:
    """
    Add T, mid, iv, log_moneyness columns to a raw option chain.

    Parameters
    ----------
    chain : DataFrame with at least [expiry, strike, option_type, bid, ask].
        Extra columns (open_interest, volume, ...) pass through unchanged.
    underlying_price : spot price S, applied to every row.
    valuation_date : "today" for computing time to maturity.
    r : risk-free rate (continuous), applied to every row -- see
        gpvol.iv.black_scholes conventions (no dividends).

    Returns
    -------
    DataFrame
        Same rows as the input, plus:
        - T : years to expiry (ACT/365)
        - mid : (bid + ask) / 2
        - iv : implied volatility, NaN if T<=0 or mid is not bracketable
        - log_moneyness : k = ln(K/F), NaN if T<=0

    Raises
    ------
    ValueError
        If a required column is missing, or option_type contains values
        other than "call"/"put".
    """
    missing = _REQUIRED_COLUMNS - set(chain.columns)
    if missing:
        raise ValueError(f"chain is missing required columns: {sorted(missing)}")

    bad_types = set(chain["option_type"].unique()) - _VALID_TYPES
    if bad_types:
        raise ValueError(f"unknown option_type values: {sorted(bad_types)}")

    out = chain.copy()
    S = float(underlying_price)

    out["T"] = (pd.to_datetime(out["expiry"]) - pd.to_datetime(valuation_date)).dt.days / 365.0
    out["mid"] = (out["bid"] + out["ask"]) / 2.0

    if len(out) == 0:
        out["iv"] = pd.Series(dtype=float)
        out["log_moneyness"] = pd.Series(dtype=float)
        return out

    out["log_moneyness"] = out.apply(lambda row: _safe_log_moneyness(row, S, r), axis=1)
    out["iv"] = out.apply(lambda row: _safe_implied_vol(row, S, r), axis=1)

    return out


def _safe_log_moneyness(row: pd.Series, S: float, r: float) -> float:
    """k = ln(K/F); NaN if T<=0 (not a valid surface coordinate)."""
    T = row["T"]
    if T <= 0:
        return np.nan
    return log_moneyness(S=S, K=row["strike"], T=T, r=r)


def _safe_implied_vol(row: pd.Series, S: float, r: float) -> float:
    """implied_vol(mid); NaN if T<=0, mid is missing/non-positive, or mid
    is not bracketable (a static no-arbitrage violation -- see
    test_black_scholes.py for what makes implied_vol raise)."""
    T = row["T"]
    mid = row["mid"]

    if T <= 0 or not np.isfinite(mid) or mid <= 0:
        return np.nan

    try:
        return implied_vol(mid, S=S, K=row["strike"], T=T, r=r, option_type=row["option_type"])
    except ValueError:
        return np.nan
