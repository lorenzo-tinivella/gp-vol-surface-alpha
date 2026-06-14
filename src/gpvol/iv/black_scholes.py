"""
Black-Scholes pricing, Greeks, and implied volatility inversion.

This module is the foundation of the project: every surface, GP/SVI fit,
and downstream signal depends on the correctness of these functions. See
tests/test_black_scholes.py for the contracts (TDD) and docs/methodology.md,
Step 1, for the economic justification.

Conventions
-----------
- Continuous rates (r), no dividends. For dividend-paying equities, replace
  S with the forward S*exp(-q*T) -- not implemented here, see roadmap.
- option_type: "call" or "put".
- All inputs are scalar floats; vectorization over DataFrames
  (gpvol.data, gpvol.iv) happens by calling these functions row by row.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = ["bs_price", "bs_greeks", "implied_vol", "log_moneyness"]

_VALID_TYPES = {"call", "put"}


def _check_option_type(option_type: str) -> None:
    if option_type not in _VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be > 0")
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Black-Scholes price (no dividends, continuous rates).

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to maturity, in years
    r : risk-free rate (continuous)
    sigma : volatility (annualized)
    option_type : "call" or "put"

    Returns
    -------
    float
        Option price.

    Examples
    --------
    >>> round(bs_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"), 4)
    10.4506
    """
    _check_option_type(option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> tuple[float, float, float]:
    """
    Delta, Gamma, Vega.

    Gamma and Vega are identical for calls and puts (Black-Scholes property).
    Vega is the derivative of price with respect to sigma (per unit of vol,
    not per 1%). Used in gpvol.backtest.hedging for delta-hedging and the
    Gamma P&L decomposition (Step 8 of the methodology).

    Returns
    -------
    (delta, gamma, vega) : tuple[float, float, float]
    """
    _check_option_type(option_type)
    d1, _ = _d1_d2(S, K, T, r, sigma)

    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, vega


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    lo: float = 1e-6,
    hi: float = 5.0,
) -> float:
    """
    Inverts Black-Scholes: finds sigma such that bs_price(..., sigma) == price.

    Uses Brent's method (bracketed root-finding). Convergence is guaranteed
    because the BS price is strictly increasing in sigma (vega > 0 always for
    T>0), so the inversion is well-defined on any bracket [lo, hi] where the
    target price is reachable.

    Parameters
    ----------
    price : observed market price
    lo, hi : search bounds for sigma
        Default 1e-6 - 5.0 (0.0001% - 500%), wide enough to cover even
        extreme regimes (short-dated pre-earnings options, crashes).

    Returns
    -------
    float
        Implied sigma.

    Raises
    ------
    ValueError
        If the price is not bracketable on [lo, hi] -- typically indicates
        a no-arbitrage violation in the input price (see Step 2, static
        filters) or too narrow a range.
    """
    _check_option_type(option_type)

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type) - price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"Price {price} not bracketable in sigma over [{lo}, {hi}] "
            f"(f(lo)={f_lo:.6f}, f(hi)={f_hi:.6f}). "
            "Possible no-arbitrage violation or range too narrow."
        )

    return brentq(objective, lo, hi, xtol=1e-8, rtol=1e-10)


def log_moneyness(S: float, K: float, T: float, r: float) -> float:
    """
    Log-moneyness relative to the forward: k = ln(K / F),  F = S * exp(r*T).

    k == 0  ->  strike == forward (ATM forward)
    k <  0  ->  strike below the forward (OTM put region)
    k >  0  ->  strike above the forward (OTM call region)

    Standardizes the "strike" coordinate relative to the spot level, making
    the surface comparable over time. It is one of the two coordinates
    (together with log(T)) the GP fits on in gpvol.surface.gp_model
    (see docs/methodology.md, Steps 1 and 3).
    """
    F = S * np.exp(r * T)
    return float(np.log(K / F))
