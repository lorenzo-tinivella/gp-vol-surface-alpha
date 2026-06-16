"""
Non-parametric volatility surface model using Gaussian Process regression.

GPSurface fits a GP on the cleaned (k, T) -> IV grid produced by Steps 1-2
(build_iv_surface -> filter_liquidity -> filter_static_arbitrage) and exposes
a predict() interface used by the composite scorer (Step 6) and the 3-D
surface visualisation (viz.plots.plot_iv_surface_3d).

Coordinate system
-----------------
The GP fits on two features:
  x1 = k = log_moneyness = ln(K/F)       (from build_iv_surface)
  x2 = log(T)                             (computed internally from T)

Using log(T) rather than T makes maturity spacing approximately uniform in
feature space: weekly, monthly, and yearly expiries are roughly equidistant
on a log scale, which produces well-conditioned length-scale estimates
(see test_gp_model.py::test_gp_uses_log_T_internally for verification).

Kernel
------
Composite kernel: RBF + Matern(nu=2.5) + WhiteKernel

  RBF          : smooth global structure of the surface
  Matern(5/2)  : local irregularities (C^2 differentiable, less smooth
                 than RBF -- better for skew and wings)
  WhiteKernel  : observation noise (bid-ask spread, quote staleness)

All hyperparameters (length scales, noise level) are optimised by maximising
the log marginal likelihood during fit().

Outputs
-------
  mu_GP(k, T)    : posterior mean -- "consensus" fair IV at each point
  sigma_GP(k, T) : posterior std dev -- uncertainty proxy for liquidity

Both are used in the composite score (Step 6):
  confidence     = 1 / (1 + sigma_GP)
  z_score        = (IV_market - mu_GP) / sigma_GP
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

__all__ = ["GPSurface"]

_REQUIRED_COLUMNS = {"log_moneyness", "T", "iv"}


class GPSurface:
    """
    Gaussian Process volatility surface.

    Parameters
    ----------
    n_restarts_optimizer : int
        Number of random restarts for kernel hyperparameter optimisation
        (sklearn's L-BFGS-B on the log marginal likelihood).
        Default 3 balances robustness vs speed; use 0 for fast unit tests.
    random_state : int or None
        Seed for reproducible hyperparameter initialisation.
    """

    def __init__(self, n_restarts_optimizer: int = 3, random_state: int = 0):
        self._n_restarts = n_restarts_optimizer
        self._random_state = random_state
        self._gp: GaussianProcessRegressor | None = None
        self._k_train: np.ndarray | None = None
        self._T_train: np.ndarray | None = None
        self._X_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, surface: pd.DataFrame) -> "GPSurface":
        """
        Fit on a cleaned surface DataFrame from Step 2.

        Parameters
        ----------
        surface : DataFrame with at minimum columns
            log_moneyness, T (years), iv.
            All rows are used -- caller is responsible for cleaning
            (filter_liquidity, filter_static_arbitrage).

        Returns
        -------
        self (for method chaining)
        """
        missing = _REQUIRED_COLUMNS - set(surface.columns)
        if missing:
            raise ValueError(f"surface is missing required columns: {sorted(missing)}")

        k = surface["log_moneyness"].values.astype(float)
        T = surface["T"].values.astype(float)
        y = surface["iv"].values.astype(float)

        log_T = np.log(T)
        X = np.column_stack([k, log_T])

        kernel = (
            RBF(length_scale=1.0, length_scale_bounds=(0.01, 10.0))
            + Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(0.01, 10.0))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 0.1))
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self._n_restarts,
            random_state=self._random_state,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y)

        self._gp = gp
        self._k_train = k
        self._T_train = T
        self._X_train = X

        return self

    def predict(
        self,
        k: np.ndarray,
        T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Posterior mean and standard deviation at arbitrary (k, T) points.

        Parameters
        ----------
        k : log-moneyness values, shape (n,) or scalar
        T : time-to-maturity in years, shape (n,) or scalar

        Returns
        -------
        (mu, sigma) : ndarrays of shape (n,)
            mu    : GP posterior mean (IV estimate)
            sigma : GP posterior std dev (uncertainty / illiquidity proxy)
        """
        self._check_fitted()
        k = np.asarray(k, dtype=float).ravel()
        log_T = np.log(np.asarray(T, dtype=float).ravel())
        X = np.column_stack([k, log_T])
        mu, sigma = self._gp.predict(X, return_std=True)
        return mu, sigma

    def predict_grid(
        self,
        n_k: int = 50,
        n_T: int = 20,
        k_range: tuple[float, float] | None = None,
        T_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict on a regular (k, T) grid for 3-D surface visualisation.

        Parameters
        ----------
        n_k, n_T : grid resolution along k and T axes
        k_range  : (k_min, k_max); defaults to training data range
        T_range  : (T_min, T_max); defaults to training data range

        Returns
        -------
        (K_grid, T_grid, mu_grid, sigma_grid) : 2-D arrays of shape (n_k, n_T)
            Passed directly to viz.plots.plot_iv_surface_3d.
        """
        self._check_fitted()

        if k_range is None:
            k_range = (float(self._k_train.min()), float(self._k_train.max()))
        if T_range is None:
            T_range = (float(self._T_train.min()), float(self._T_train.max()))

        k_vals = np.linspace(k_range[0], k_range[1], n_k)
        T_vals = np.linspace(T_range[0], T_range[1], n_T)

        K_grid, T_grid = np.meshgrid(k_vals, T_vals, indexing="ij")
        mu_flat, sigma_flat = self.predict(K_grid.ravel(), T_grid.ravel())

        return K_grid, T_grid, mu_flat.reshape(n_k, n_T), sigma_flat.reshape(n_k, n_T)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._gp is None:
            raise RuntimeError(
                "GPSurface must be fitted before calling predict() or predict_grid(). "
                "Call fit(surface) first."
            )
