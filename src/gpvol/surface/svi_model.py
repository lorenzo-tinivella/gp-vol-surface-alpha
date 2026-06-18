"""
SVI (Stochastic Volatility Inspired) parametric volatility surface model.

SVI serves as the parametric baseline against which the GP is compared in the
composite scoring step (Step 6). Where both models agree that an option is
mispriced relative to its neighbours, the signal is more credible than when
only one model flags it.

Parameterisation (Gatheral & Jacquier, 2014)
--------------------------------------------
The SVI raw parameterisation models total variance as a function of
log-moneyness k = ln(K/F):

    w(k; theta) = a + b * [rho*(k-m) + sqrt((k-m)^2 + sigma^2)]

where theta = (a, b, rho, m, sigma) and w = IV^2 * T is total variance.
To recover implied vol: IV(k, T) = sqrt(w(k) / T).

SVI is calibrated per expiry slice: for each maturity T, we find theta that
minimises the sum of squared total-variance residuals on the market points.

Calibration: scipy multi-start Nelder-Mead
------------------------------------------
BO is NOT used here. SVI calibration has a fast closed-form objective (each
evaluation is a vectorised formula, not a simulation) and a notoriously
multimodal loss landscape. Empirical comparison (see exploration notes):

    scipy 20 restarts: 0.96s   RMSE(w) = 1e-11  (exact recovery)
    BO 60 calls:       16.2s   RMSE(w) = 3e-3   (stuck in local minimum)

scipy multi-start is 17x faster and 8 orders of magnitude more accurate for
this problem. BO is used in Step 7 (threshold optimisation), where each
evaluation of the objective requires a full backtest (~seconds).

No-arbitrage conditions (Gatheral & Jacquier, 2014, Proposition 1)
-------------------------------------------------------------------
    b >= 0
    |rho| < 1
    sigma > 0
    a + b * sigma * sqrt(1 - rho^2) >= 0  [ensures w(k) >= 0 for all k]

The last condition ensures the minimum of w(k) is non-negative. The minimum
occurs at k* = m - rho*sigma / sqrt(1 - rho^2), giving w(k*) = a + b*sigma*sqrt(1-rho^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

__all__ = ["SVIParams", "svi_total_variance", "svi_vol", "check_no_arbitrage", "calibrate_svi"]

# Search bounds for each parameter during calibration
_BOUNDS = [
    (-0.10, 0.20),   # a : total variance level
    ( 0.00, 1.00),   # b : wing slope (>= 0)
    (-0.999, 0.999), # rho : skew (-1 < rho < 1)
    (-0.50, 0.50),   # m : ATM shift
    ( 0.01, 1.00),   # sigma : curvature (> 0)
]


@dataclass
class SVIParams:
    """
    SVI raw parameters for one expiry slice.

    Attributes
    ----------
    a     : total variance level (shifts the surface up/down)
    b     : wing slope (>= 0; controls ATM vol and wing steepness)
    rho   : skew parameter (-1 < rho < 1; negative for equity put skew)
    m     : ATM offset (shifts the smile along k)
    sigma : curvature / smoothness (> 0; controls ATM region shape)
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(k: np.ndarray | float, params: SVIParams) -> np.ndarray:
    """
    Compute total variance w(k) = a + b*[rho*(k-m) + sqrt((k-m)^2 + sigma^2)].

    Parameters
    ----------
    k : log-moneyness values, scalar or array
    params : SVIParams for this expiry slice

    Returns
    -------
    w : total variance (IV^2 * T), same shape as k
    """
    k = np.asarray(k, dtype=float)
    km = k - params.m
    return params.a + params.b * (params.rho * km + np.sqrt(km**2 + params.sigma**2))


def svi_vol(k: np.ndarray | float, T: float, params: SVIParams) -> np.ndarray:
    """
    Implied volatility IV(k, T) = sqrt(w(k) / T).

    Clips w to 0 before the square root to prevent NaN on near-zero
    floating-point artefacts at the no-arb boundary.
    """
    w = svi_total_variance(k, params)
    return np.sqrt(np.maximum(w, 0.0) / T)


def check_no_arbitrage(params: SVIParams) -> bool:
    """
    Check the four Gatheral-Jacquier (2014) no-arbitrage conditions.

    Returns True iff all conditions are satisfied.
    """
    if params.b < 0:
        return False
    if not (-1.0 < params.rho < 1.0):
        return False
    if params.sigma <= 0.0:
        return False
    if params.a + params.b * params.sigma * np.sqrt(1.0 - params.rho**2) < 0.0:
        return False
    return True


def calibrate_svi(
    k_vals: np.ndarray,
    iv_vals: np.ndarray,
    T: float,
    n_restarts: int = 20,
    random_state: int = 0,
) -> SVIParams:
    """
    Calibrate SVI parameters for one expiry slice.

    Minimises the sum of squared total-variance residuals:
        L(theta) = sum_i ( w_market_i - w_SVI(k_i; theta) )^2
    using scipy Nelder-Mead with n_restarts random starting points.

    Parameters
    ----------
    k_vals : log-moneyness values for this slice, shape (n,)
    iv_vals : market implied volatilities, shape (n,)
    T : time to maturity in years
    n_restarts : number of random starts (default 20 -- see module docstring)
    random_state : seed for reproducible random starts

    Returns
    -------
    SVIParams
        Calibrated parameters satisfying all no-arbitrage conditions.
        If no valid solution is found (degenerate slice), returns the
        best solution found even if it violates no-arb -- caller should
        validate with check_no_arbitrage().
    """
    k_vals = np.asarray(k_vals, dtype=float)
    w_market = iv_vals.astype(float) ** 2 * T

    def _loss(x: list) -> float:
        a, b, rho, m, sigma = x
        p = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
        if not check_no_arbitrage(p):
            return 1e6
        return float(np.sum((svi_total_variance(k_vals, p) - w_market) ** 2))

    rng = np.random.default_rng(random_state)
    best_loss, best_x = np.inf, None

    for _ in range(n_restarts):
        x0 = [rng.uniform(lo, hi) for lo, hi in _BOUNDS]
        result = minimize(
            _loss,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
        )
        if result.fun < best_loss:
            best_loss = result.fun
            best_x = result.x

    a, b, rho, m, sigma = best_x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
