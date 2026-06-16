"""
Tests for gpvol.surface.gp_model.GPSurface.

What we validate and why
------------------------
The GP is the analytical core of the project: every signal downstream
(composite score, walk-forward backtest) is a function of mu_GP and sigma_GP.
Three properties matter economically:

1. Posterior mean is accurate at observed points.
   A GP with a well-chosen kernel should interpolate smoothly through
   training data. RMSE at training points < 0.01 (1 vol point) is the
   same threshold used for the implied_vol roundtrip in test_black_scholes.py.

2. Uncertainty sigma_GP is calibrated: small where data is dense,
   large where data is sparse.
   This is the mechanism that down-weights signals in illiquid regions
   (confidence = 1/(1+sigma_GP) in the composite score, Step 6).
   If sigma_GP were uniformly small, all signals would have equal weight
   regardless of data density.

3. sigma_GP > 0 everywhere.
   sigma_GP enters the denominator of the z-score. A zero would produce
   division-by-zero in the signal computation.

Synthetic surface
-----------------
sigma(k, T) = 0.20 - 0.08*k + 0.04*sqrt(T)

Chosen to be:
- Smooth (GP should fit it well with a small number of training points)
- Realistic in shape (negative skew in k, positive term structure in T)
- Simple enough that tolerances are easy to reason about

Training grid: k in {-0.2, -0.1, 0, 0.1, 0.2}, T in {1/12, 3/12, 6/12, 1}
-> 20 points. Enough for a well-conditioned GP, fast to compute.

Tolerances (all justified by the exploration in notes/gp_exploration.py):
- RMSE at training < 0.01  (actual: ~3e-8)
- max sigma at training < 0.02  (actual: ~2e-5)
- sigma_far / sigma_center > 2  (actual: ~40)
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from gpvol.surface.gp_model import GPSurface


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _true_iv(k: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Synthetic volatility surface used in all GP tests."""
    return 0.20 - 0.08 * k + 0.04 * np.sqrt(T)


def _make_surface() -> pd.DataFrame:
    """
    20-point training surface on a regular (k, T) grid.
    Column names match the output of build_iv_surface (Step 1).
    """
    k_vals = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    T_vals = np.array([1 / 12, 3 / 12, 6 / 12, 12 / 12])
    K, TT = np.meshgrid(k_vals, T_vals, indexing="ij")
    k_flat = K.ravel()
    T_flat = TT.ravel()
    return pd.DataFrame({
        "log_moneyness": k_flat,
        "T": T_flat,
        "iv": _true_iv(k_flat, T_flat),
    })


def _fitted_gp() -> GPSurface:
    """Return a fitted GPSurface with fast settings for tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GPSurface(n_restarts_optimizer=0, random_state=0).fit(_make_surface())


# ---------------------------------------------------------------------------
# Fit and predict: correctness
# ---------------------------------------------------------------------------

def test_gp_recovers_training_iv():
    """
    Posterior mean at the 20 training points must be within 1 vol point
    of the true surface. If this fails, the kernel or coordinate system
    is fundamentally wrong -- every downstream signal would be unreliable.
    """
    surface = _make_surface()
    gp = _fitted_gp()
    mu, _ = gp.predict(surface["log_moneyness"].values, surface["T"].values)
    rmse = np.sqrt(np.mean((mu - surface["iv"].values) ** 2))
    assert rmse < 0.01, f"RMSE={rmse:.6f} exceeds 1 vol-point threshold"


def test_gp_uncertainty_low_at_training_points():
    """
    sigma_GP at training points must be < 0.02 (2 vol points).
    These are well-observed regions -- high uncertainty here would mean
    confidence = 1/(1+sigma) is too low even for liquid, well-priced options.
    """
    surface = _make_surface()
    gp = _fitted_gp()
    _, sigma = gp.predict(surface["log_moneyness"].values, surface["T"].values)
    assert sigma.max() < 0.02, f"max sigma at training={sigma.max():.6f}"


def test_gp_uncertainty_high_far_from_training():
    """
    sigma_GP at k=+/-0.5 (outside the training range of +/-0.2) must be
    at least 2x larger than sigma at the training center (k=0, T=3m).
    This is the mechanism that makes the composite score conservative in
    illiquid, data-sparse regions of the surface.
    """
    gp = _fitted_gp()
    _, sigma_center = gp.predict(np.array([0.0]), np.array([3 / 12]))
    _, sigma_far = gp.predict(np.array([0.5]), np.array([3 / 12]))
    ratio = sigma_far[0] / sigma_center[0]
    assert ratio > 2.0, f"uncertainty ratio far/center={ratio:.2f}, expected > 2"


def test_gp_sigma_always_positive():
    """
    sigma_GP must be strictly positive everywhere -- it enters the
    denominator of z = (IV_market - mu_GP) / sigma_GP in the scoring step.
    Tested on 200 random (k, T) points spanning the surface region.
    """
    gp = _fitted_gp()
    rng = np.random.default_rng(42)
    k_random = rng.uniform(-0.4, 0.4, 200)
    T_random = rng.uniform(1 / 52, 1.5, 200)
    _, sigma = gp.predict(k_random, T_random)
    assert (sigma > 0).all(), f"sigma <= 0 at {(sigma <= 0).sum()} points"


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

def test_gp_predict_output_shapes():
    """mu and sigma must both be 1-D arrays of length n."""
    gp = _fitted_gp()
    n = 15
    mu, sigma = gp.predict(
        np.linspace(-0.2, 0.2, n),
        np.full(n, 3 / 12),
    )
    assert mu.shape == (n,)
    assert sigma.shape == (n,)


def test_gp_predict_grid_returns_correct_shapes():
    """
    predict_grid must return four 2-D arrays of shape (n_k, n_T).
    Used directly by viz.plots.plot_iv_surface_3d -- wrong shapes
    would silently break the plotly surface rendering.
    """
    gp = _fitted_gp()
    n_k, n_T = 12, 8
    K_grid, T_grid, mu_grid, sigma_grid = gp.predict_grid(n_k=n_k, n_T=n_T)
    for arr, name in [(K_grid, "K_grid"), (T_grid, "T_grid"),
                      (mu_grid, "mu_grid"), (sigma_grid, "sigma_grid")]:
        assert arr.shape == (n_k, n_T), f"{name}.shape={arr.shape}"


# ---------------------------------------------------------------------------
# Coordinate system
# ---------------------------------------------------------------------------

def test_gp_uses_log_moneyness_column():
    """
    GPSurface reads the log_moneyness column from the DataFrame (k = ln(K/F)),
    not raw strike. This is the coordinate system that makes the surface
    comparable across time and underliers (docs/methodology.md, Step 1).
    Verified by checking that the stored training k values match the column.
    """
    surface = _make_surface()
    gp = _fitted_gp()
    np.testing.assert_array_equal(gp._k_train, surface["log_moneyness"].values)


def test_gp_uses_log_T_internally():
    """
    The GP feature matrix uses log(T), not T. Maturity spacing in practice
    is approximately log-uniform (weekly, monthly, quarterly, yearly) --
    log(T) makes these equidistant in feature space and improves length-scale
    estimation. Verified by checking the stored feature matrix.
    """
    surface = _make_surface()
    gp = _fitted_gp()
    expected_log_T = np.log(surface["T"].values)
    np.testing.assert_allclose(gp._X_train[:, 1], expected_log_T)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_gp_predict_before_fit_raises():
    """Calling predict() before fit() must raise RuntimeError, not AttributeError."""
    gp = GPSurface()
    with pytest.raises(RuntimeError, match="fitted"):
        gp.predict(np.array([0.0]), np.array([0.25]))


def test_gp_fit_raises_on_missing_columns():
    """fit() must raise ValueError listing the missing column names."""
    bad_surface = pd.DataFrame({"log_moneyness": [0.0], "T": [0.25]})
    with pytest.raises(ValueError, match="iv"):
        GPSurface().fit(bad_surface)


def test_gp_predict_grid_before_fit_raises():
    gp = GPSurface()
    with pytest.raises(RuntimeError, match="fitted"):
        gp.predict_grid()
