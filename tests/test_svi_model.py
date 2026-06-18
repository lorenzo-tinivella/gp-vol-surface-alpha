"""
Tests for gpvol.surface.svi_model.

SVI is the parametric baseline in the composite score (Step 6): a signal
is stronger when both the GP and SVI agree that an option is mispriced.
Three properties must hold for SVI to be a reliable baseline:

1. The formula is correct (total variance w = IV^2 * T, not IV directly).
   A factor-of-T error would shift every SVI prediction by sqrt(T),
   producing spurious signals at every expiry.

2. check_no_arbitrage correctly gates calibrated parameters.
   Params that fail the conditions can produce w(k) < 0, which makes
   svi_vol() return NaN and crashes the composite score computation.

3. calibrate_svi recovers a known surface closely.
   RMSE(w) < 1e-4 in total variance space corresponds to < 0.05 vol-point
   error at typical maturities -- well below the signal threshold of ~1-2
   vol points used in the composite score.

No-arbitrage conditions (Gatheral & Jacquier, 2014):
    b >= 0
    |rho| < 1  (strict)
    sigma > 0
    a + b * sigma * sqrt(1 - rho^2) >= 0   [w(k) >= 0 everywhere]

Calibration design note
-----------------------
scipy multi-start (Nelder-Mead, 20 random starts) is used instead of BO.
See svi_model.py module docstring for the quantitative comparison:
scipy achieves RMSE(w) ~ 1e-11 in 0.96s; BO achieves 3e-3 in 16s on the
same problem. BO is reserved for Step 7 (threshold search in walk-forward)
where each objective evaluation requires a full backtest (~seconds).
"""

import numpy as np
import pytest

from gpvol.surface.svi_model import (
    SVIParams,
    calibrate_svi,
    check_no_arbitrage,
    svi_total_variance,
    svi_vol,
)

# ---------------------------------------------------------------------------
# Canonical valid params used throughout: ~20% ATM vol, negative skew, 3m
# ---------------------------------------------------------------------------

_VALID = SVIParams(a=0.004, b=0.050, rho=-0.40, m=0.00, sigma=0.10)
_T = 3 / 12  # 3-month expiry


# ---------------------------------------------------------------------------
# svi_total_variance -- formula correctness
# ---------------------------------------------------------------------------

def test_svi_atm_formula():
    """
    At k=0 with m=0 and rho=0 the formula simplifies analytically:
        w(0) = a + b * sqrt(sigma^2) = a + b * sigma
    This is a closed-form reference that pins the formula independently
    of any numerical computation.
    """
    p = SVIParams(a=0.004, b=0.050, rho=0.0, m=0.0, sigma=0.10)
    w = svi_total_variance(0.0, p)
    expected = p.a + p.b * p.sigma   # = 0.004 + 0.005 = 0.009
    assert np.isclose(w, expected, atol=1e-12)


def test_svi_vol_atm_matches_target():
    """
    _VALID params were chosen so that IV_ATM ≈ 20%.
    IV = sqrt(w / T) = sqrt((a + b*sigma) / T) = sqrt(0.009 / 0.25) ≈ 0.190.
    """
    iv_atm = svi_vol(0.0, _T, _VALID)
    assert np.isclose(iv_atm, np.sqrt((_VALID.a + _VALID.b * _VALID.sigma) / _T), atol=1e-10)
    assert 0.15 < iv_atm < 0.25, f"IV_ATM={iv_atm:.4f} outside realistic range"


def test_svi_total_variance_positive_everywhere():
    """
    For valid params, w(k) must be >= 0 for all k. Negative total variance
    would make svi_vol() return NaN, breaking the composite score.
    """
    k_grid = np.linspace(-2.0, 2.0, 200)
    w = svi_total_variance(k_grid, _VALID)
    assert (w >= 0).all(), f"w < 0 at {(w < 0).sum()} points"


def test_svi_total_variance_array_input():
    """svi_total_variance must accept and return numpy arrays, not just scalars."""
    k = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    w = svi_total_variance(k, _VALID)
    assert w.shape == k.shape


def test_svi_put_wing_steeper_than_call_wing():
    """
    Negative rho means the left wing (put side, k < 0) has higher total
    variance than the right wing (call side, k > 0). This is the standard
    equity skew pattern -- a sign error in rho would reverse it.
    """
    w_put_wing  = svi_total_variance(-0.3, _VALID)   # OTM put
    w_call_wing = svi_total_variance( 0.3, _VALID)   # OTM call
    assert w_put_wing > w_call_wing, "Expected steeper put wing for rho < 0"


# ---------------------------------------------------------------------------
# check_no_arbitrage
# ---------------------------------------------------------------------------

def test_no_arb_valid_params():
    assert check_no_arbitrage(_VALID) is True


def test_no_arb_negative_b():
    p = SVIParams(a=0.004, b=-0.01, rho=-0.40, m=0.00, sigma=0.10)
    assert check_no_arbitrage(p) is False


def test_no_arb_rho_at_boundary():
    """rho must be STRICTLY between -1 and 1 (degenerate wings at +/-1)."""
    assert check_no_arbitrage(SVIParams(a=0.004, b=0.05, rho=-1.0, m=0.0, sigma=0.10)) is False
    assert check_no_arbitrage(SVIParams(a=0.004, b=0.05, rho= 1.0, m=0.0, sigma=0.10)) is False


def test_no_arb_zero_sigma():
    p = SVIParams(a=0.004, b=0.050, rho=-0.40, m=0.00, sigma=0.0)
    assert check_no_arbitrage(p) is False


def test_no_arb_negative_minimum_variance():
    """
    a + b*sigma*sqrt(1 - rho^2) < 0 means w(k*) < 0 at the smile minimum.
    Example: a=-0.01, b=0.05, sigma=0.10, rho=0.40
        = -0.01 + 0.05*0.10*sqrt(1-0.16) = -0.01 + 0.00458 = -0.00542 < 0
    """
    p = SVIParams(a=-0.01, b=0.05, rho=0.40, m=0.0, sigma=0.10)
    assert check_no_arbitrage(p) is False


# ---------------------------------------------------------------------------
# calibrate_svi
# ---------------------------------------------------------------------------

def _make_synthetic_surface(params: SVIParams, T: float, n_points: int = 20):
    """Generate noiseless synthetic (k, IV) data from known SVI params."""
    k_vals = np.linspace(-0.3, 0.3, n_points)
    iv_vals = svi_vol(k_vals, T, params)
    return k_vals, iv_vals


def test_calibrate_recovers_known_params():
    """
    Fitting on noiseless SVI data must recover the true surface closely.
    RMSE(total variance) < 1e-4 corresponds to < 0.05 vol-point error
    at T=3m -- well below the 1-2 vol-point signal threshold in Step 6.
    """
    k_vals, iv_vals = _make_synthetic_surface(_VALID, _T)
    fitted = calibrate_svi(k_vals, iv_vals, _T, n_restarts=20, random_state=0)

    w_true   = svi_total_variance(k_vals, _VALID)
    w_fitted = svi_total_variance(k_vals, fitted)
    rmse_w   = np.sqrt(np.mean((w_fitted - w_true) ** 2))

    assert rmse_w < 1e-4, f"RMSE(total variance)={rmse_w:.2e}, expected < 1e-4"


def test_calibrate_returns_no_arbitrage_params():
    """calibrate_svi must always return params that satisfy no-arbitrage."""
    k_vals, iv_vals = _make_synthetic_surface(_VALID, _T)
    fitted = calibrate_svi(k_vals, iv_vals, _T, n_restarts=10, random_state=42)
    assert check_no_arbitrage(fitted), (
        f"Calibrated params violate no-arbitrage: {fitted}"
    )


def test_calibrate_reduces_loss_vs_initial():
    """
    The fitted total-variance RMSE must be smaller than the RMSE of an
    arbitrary starting point (here: flat surface with a=0.02, b=0).
    Checks that calibration actually optimises something.
    """
    k_vals, iv_vals = _make_synthetic_surface(_VALID, _T)
    w_market = iv_vals ** 2 * _T

    naive = SVIParams(a=0.02, b=0.0, rho=0.0, m=0.0, sigma=0.10)
    rmse_naive  = np.sqrt(np.mean((svi_total_variance(k_vals, naive) - w_market) ** 2))

    fitted = calibrate_svi(k_vals, iv_vals, _T, n_restarts=10, random_state=0)
    rmse_fitted = np.sqrt(np.mean((svi_total_variance(k_vals, fitted) - w_market) ** 2))

    assert rmse_fitted < rmse_naive, (
        f"fitted RMSE={rmse_fitted:.2e} is not better than naive={rmse_naive:.2e}"
    )
