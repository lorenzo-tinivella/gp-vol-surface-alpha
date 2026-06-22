"""
Smoke tests for gpvol.viz.plots.

Visualization functions are not amenable to numerical unit tests -- the
output is a figure object. What we verify:
- Functions return the correct type (plotly.Figure or matplotlib.Figure)
- Figures contain the expected number of traces / axes
- Functions handle optional parameters gracefully (market observations
  absent, benchmark absent)
- No crash on valid synthetic input

We do NOT test visual appearance or colour choices.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest

matplotlib.use("Agg")   # non-interactive backend for CI

from gpvol.viz.plots import (
    plot_equity_curve,
    plot_gp_vs_svi,
    plot_iv_surface_3d,
    plot_signal_decay,
)

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

_N_K, _N_T = 30, 15

def _make_grids():
    k = np.linspace(-0.3, 0.3, _N_K)
    T = np.linspace(1/12, 1.0, _N_T)
    K_grid, T_grid = np.meshgrid(k, T, indexing="ij")
    mu    = 0.20 - 0.08*K_grid + 0.04*np.sqrt(T_grid)
    sigma = 0.005 + 0.01*np.abs(K_grid)
    return K_grid, T_grid, mu, sigma


# ---------------------------------------------------------------------------
# plot_iv_surface_3d
# ---------------------------------------------------------------------------

def test_surface_returns_plotly_figure():
    K, T, mu, sigma = _make_grids()
    fig = plot_iv_surface_3d(K, T, mu, sigma)
    assert isinstance(fig, go.Figure)


def test_surface_has_three_traces_without_market():
    """Mean surface + upper band + lower band = 3 traces."""
    K, T, mu, sigma = _make_grids()
    fig = plot_iv_surface_3d(K, T, mu, sigma)
    assert len(fig.data) == 3


def test_surface_has_four_traces_with_market():
    """Adding market scatter adds a 4th trace."""
    K, T, mu, sigma = _make_grids()
    mk = np.array([-0.1, 0.0, 0.1])
    mT = np.array([0.25, 0.25, 0.25])
    mv = np.array([0.22, 0.20, 0.19])
    fig = plot_iv_surface_3d(K, T, mu, sigma, market_k=mk, market_T=mT, market_iv=mv)
    assert len(fig.data) == 4


def test_surface_z_shape():
    """z arrays must be (n_T, n_k) -- plotly convention, not (n_k, n_T)."""
    K, T, mu, sigma = _make_grids()
    fig = plot_iv_surface_3d(K, T, mu, sigma)
    mean_trace = fig.data[2]   # index 2 = mean surface (rendered last)
    assert mean_trace.z.shape == (_N_T, _N_K)


# ---------------------------------------------------------------------------
# plot_gp_vs_svi
# ---------------------------------------------------------------------------

def test_gp_vs_svi_returns_plotly_figure():
    k = np.linspace(-0.3, 0.3, 50)
    mu_gp  = 0.20 - 0.08*k
    sig_gp = 0.005 * np.ones(50)
    mu_svi = 0.20 - 0.07*k
    fig = plot_gp_vs_svi(k, mu_gp, sig_gp, mu_svi)
    assert isinstance(fig, go.Figure)


def test_gp_vs_svi_three_traces_without_market():
    """Band + GP line + SVI line = 3 traces."""
    k = np.linspace(-0.3, 0.3, 50)
    mu_gp  = 0.20 - 0.08*k
    sig_gp = 0.005 * np.ones(50)
    mu_svi = 0.20 - 0.07*k
    fig = plot_gp_vs_svi(k, mu_gp, sig_gp, mu_svi)
    assert len(fig.data) == 3


def test_gp_vs_svi_four_traces_with_market():
    k = np.linspace(-0.3, 0.3, 50)
    mu_gp  = 0.20 - 0.08*k
    sig_gp = 0.005 * np.ones(50)
    mu_svi = 0.20 - 0.07*k
    mk  = np.array([-0.15, 0.0, 0.15])
    miv = np.array([0.225, 0.200, 0.185])
    fig = plot_gp_vs_svi(k, mu_gp, sig_gp, mu_svi, market_k=mk, market_iv=miv)
    assert len(fig.data) == 4


# ---------------------------------------------------------------------------
# plot_equity_curve
# ---------------------------------------------------------------------------

def test_equity_curve_returns_matplotlib_figure():
    returns = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 100))
    fig = plot_equity_curve(returns)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


def test_equity_curve_two_axes():
    """Equity + drawdown = 2 subplots."""
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 80))
    fig = plot_equity_curve(returns)
    assert len(fig.axes) == 2
    plt.close("all")


def test_equity_curve_with_benchmark():
    returns   = pd.Series(np.random.default_rng(2).normal(0.001, 0.01, 80))
    benchmark = pd.Series(np.random.default_rng(3).normal(0.0005, 0.008, 80))
    fig = plot_equity_curve(returns, benchmark=benchmark)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_signal_decay
# ---------------------------------------------------------------------------

def test_signal_decay_returns_matplotlib_figure():
    np.random.seed(42)
    signal = pd.Series(np.cumsum(np.random.randn(200)) * 0.1)
    fig = plot_signal_decay(signal)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


def test_signal_decay_single_axis():
    signal = pd.Series(np.random.default_rng(0).standard_normal(150))
    fig = plot_signal_decay(signal, max_lags=10)
    assert len(fig.axes) == 1
    plt.close("all")
