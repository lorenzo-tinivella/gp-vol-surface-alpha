"""
Visualization utilities for the GP volatility surface project.

All functions return figure objects (plotly or matplotlib) without
displaying them, so they work identically in notebooks, scripts,
and the walk-forward reporting loop.

Four plots, four purposes:

    plot_iv_surface_3d   : the main deliverable -- GP mean surface with
                           uncertainty bands and market observations.
                           This is the visual that explains the project
                           in one glance.

    plot_gp_vs_svi       : per-slice cross-section showing GP (with band),
                           SVI baseline, and market points. Used to explain
                           the composite score logic in notebooks.

    plot_equity_curve    : OOS cumulative P&L with drawdown shading.
                           Standard walk-forward reporting plot.

    plot_signal_decay    : autocorrelation of the composite score signal
                           with theoretical AR(1) overlay. Verifies the
                           signal half-life is in the economically credible
                           2-10 day range (docs/methodology.md, Step 9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.figure

__all__ = [
    "plot_iv_surface_3d",
    "plot_gp_vs_svi",
    "plot_equity_curve",
    "plot_signal_decay",
]

# ---------------------------------------------------------------------------
# Colour palette (consistent across all plots)
# ---------------------------------------------------------------------------

_BLUE  = "#2563EB"   # GP mean line / surface
_RED   = "#DC2626"   # SVI baseline
_GREEN = "#16A34A"   # market observations
_GREY  = "#6B7280"   # secondary elements


# ---------------------------------------------------------------------------
# 1. 3D implied volatility surface
# ---------------------------------------------------------------------------

def plot_iv_surface_3d(
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    mu_grid: np.ndarray,
    sigma_grid: np.ndarray,
    market_k: np.ndarray | None = None,
    market_T: np.ndarray | None = None,
    market_iv: np.ndarray | None = None,
    title: str = "Implied Volatility Surface — GP Posterior",
) -> go.Figure:
    """
    3-D plot of the GP volatility surface with ±1σ uncertainty bands.

    Parameters
    ----------
    K_grid, T_grid : 2-D coordinate grids, shape (n_k, n_T).
        Output of GPSurface.predict_grid().
    mu_grid : GP posterior mean IV, shape (n_k, n_T).
    sigma_grid : GP posterior std dev, shape (n_k, n_T).
    market_k, market_T, market_iv : optional scatter of raw observations.
        If provided, plotted as a scatter3d layer over the surface.
    title : figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3-D figure. Call .show() in a notebook or
        .write_html("surface.html") to save.

    Notes
    -----
    plotly Surface expects z with shape (n_T, n_k) when
    x.shape=(n_k,) and y.shape=(n_T,), so mu_grid is transposed.
    This is verified in tests/test_viz.py.
    """
    k_vals = K_grid[:, 0]   # shape (n_k,)
    T_vals = T_grid[0, :]   # shape (n_T,)

    # z must be (n_T, n_k) for plotly Surface
    z_mean  = mu_grid.T
    z_upper = (mu_grid + sigma_grid).T
    z_lower = (mu_grid - sigma_grid).T

    fig = go.Figure()

    # Lower uncertainty band (rendered first so mean surface is on top)
    fig.add_trace(go.Surface(
        x=k_vals, y=T_vals, z=z_lower,
        name="−1σ uncertainty",
        colorscale=[[0, "rgba(37,99,235,0.0)"], [1, "rgba(37,99,235,0.15)"]],
        showscale=False,
        opacity=0.4,
        hovertemplate="k=%{x:.3f}<br>T=%{y:.3f}<br>IV=%{z:.4f}<extra>−1σ</extra>",
    ))

    # Upper uncertainty band
    fig.add_trace(go.Surface(
        x=k_vals, y=T_vals, z=z_upper,
        name="+1σ uncertainty",
        colorscale=[[0, "rgba(37,99,235,0.0)"], [1, "rgba(37,99,235,0.15)"]],
        showscale=False,
        opacity=0.4,
        hovertemplate="k=%{x:.3f}<br>T=%{y:.3f}<br>IV=%{z:.4f}<extra>+1σ</extra>",
    ))

    # GP mean surface (main)
    fig.add_trace(go.Surface(
        x=k_vals, y=T_vals, z=z_mean,
        name="GP posterior mean",
        colorscale="RdBu_r",
        showscale=True,
        colorbar=dict(title="IV", tickformat=".0%", len=0.7),
        opacity=1.0,
        hovertemplate="k=%{x:.3f}<br>T=%{y:.3f}<br>IV=%{z:.4f}<extra>GP mean</extra>",
    ))

    # Market observations (optional)
    if market_k is not None and market_iv is not None:
        fig.add_trace(go.Scatter3d(
            x=market_k, y=market_T, z=market_iv,
            mode="markers",
            name="Market IV",
            marker=dict(size=4, color=_GREEN, symbol="circle",
                        line=dict(width=0.5, color="white")),
            hovertemplate="k=%{x:.3f}<br>T=%{y:.3f}<br>IV=%{z:.4f}<extra>Market</extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis=dict(title="Log-moneyness k = ln(K/F)", tickformat=".2f"),
            yaxis=dict(title="Time to maturity T (years)", tickformat=".2f"),
            zaxis=dict(title="Implied volatility", tickformat=".0%"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
            bgcolor="rgba(255,255,255,1)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(x=0.01, y=0.99),
        width=900,
        height=650,
    )

    return fig


# ---------------------------------------------------------------------------
# 2. Per-slice GP vs SVI cross-section
# ---------------------------------------------------------------------------

def plot_gp_vs_svi(
    k_vals: np.ndarray,
    mu_gp: np.ndarray,
    sigma_gp: np.ndarray,
    mu_svi: np.ndarray,
    market_k: np.ndarray | None = None,
    market_iv: np.ndarray | None = None,
    T_label: str = "",
    title: str = "GP vs SVI — Implied Volatility Smile",
) -> go.Figure:
    """
    2-D smile cross-section at a single maturity: GP (with band), SVI, market.

    Parameters
    ----------
    k_vals : log-moneyness grid for this maturity slice, shape (n,)
    mu_gp, sigma_gp : GP posterior mean and std, shape (n,)
    mu_svi : SVI implied vol at same grid points, shape (n,)
    market_k, market_iv : optional market observations at this maturity
    T_label : maturity label shown in the legend (e.g. "T = 3m")
    title : figure title

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # GP uncertainty band (shaded region)
    fig.add_trace(go.Scatter(
        x=np.concatenate([k_vals, k_vals[::-1]]),
        y=np.concatenate([mu_gp + sigma_gp, (mu_gp - sigma_gp)[::-1]]),
        fill="toself",
        fillcolor="rgba(37,99,235,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="GP ±1σ",
        showlegend=True,
        hoverinfo="skip",
    ))

    # GP mean
    fig.add_trace(go.Scatter(
        x=k_vals, y=mu_gp,
        mode="lines",
        name=f"GP mean {T_label}",
        line=dict(color=_BLUE, width=2.5),
        hovertemplate="k=%{x:.3f}<br>IV=%{y:.4f}<extra>GP</extra>",
    ))

    # SVI baseline
    fig.add_trace(go.Scatter(
        x=k_vals, y=mu_svi,
        mode="lines",
        name=f"SVI {T_label}",
        line=dict(color=_RED, width=2, dash="dash"),
        hovertemplate="k=%{x:.3f}<br>IV=%{y:.4f}<extra>SVI</extra>",
    ))

    # Market observations
    if market_k is not None and market_iv is not None:
        fig.add_trace(go.Scatter(
            x=market_k, y=market_iv,
            mode="markers",
            name="Market",
            marker=dict(color=_GREEN, size=7, symbol="circle",
                        line=dict(width=1, color="white")),
            hovertemplate="k=%{x:.3f}<br>IV=%{y:.4f}<extra>Market</extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Log-moneyness k = ln(K/F)", tickformat=".2f",
                   zeroline=True, zerolinecolor="rgba(0,0,0,0.15)"),
        yaxis=dict(title="Implied volatility", tickformat=".0%"),
        legend=dict(x=0.02, y=0.98),
        hovermode="x unified",
        width=750,
        height=450,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# ---------------------------------------------------------------------------
# 3. Walk-forward equity curve with drawdown
# ---------------------------------------------------------------------------

def plot_equity_curve(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Walk-Forward OOS Equity Curve",
) -> matplotlib.figure.Figure:
    """
    Cumulative P&L curve with drawdown shading below the x-axis.

    Parameters
    ----------
    returns : daily OOS return series (output of run_walkforward)
    benchmark : optional benchmark return series (same index)
    title : figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    equity = (1 + returns).cumprod()
    peak   = equity.cummax()
    dd     = (equity - peak) / peak

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Equity curve
    ax1.plot(equity.index, equity.values, color=_BLUE, lw=2, label="Strategy")
    if benchmark is not None:
        bench_eq = (1 + benchmark).cumprod()
        ax1.plot(bench_eq.index, bench_eq.values, color=_GREY,
                 lw=1.5, ls="--", label="Benchmark")
    ax1.axhline(1.0, color="black", lw=0.7, ls="--", alpha=0.4)
    ax1.set_ylabel("Cumulative return (index = 1)")
    ax1.set_title(title, fontsize=13, pad=10)
    ax1.legend(framealpha=0)
    ax1.spines[["top", "right"]].set_visible(False)

    # Drawdown
    ax2.fill_between(dd.index, dd.values, 0, color=_RED, alpha=0.4)
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Signal decay / autocorrelation
# ---------------------------------------------------------------------------

def plot_signal_decay(
    signal_series: pd.Series,
    max_lags: int = 20,
    title: str = "Signal Decay — Autocorrelation of Composite Score",
) -> matplotlib.figure.Figure:
    """
    Autocorrelation of the composite score at lags 1..max_lags,
    with a fitted AR(1) decay curve overlaid.

    A half-life of 2-10 days (where autocorrelation crosses 0.5) is
    consistent with market-maker repricing speed (Step 9).

    Parameters
    ----------
    signal_series : time series of daily composite scores
    max_lags : maximum lag to display
    title : figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.stats import linregress

    x = signal_series.values
    lags = np.arange(1, max_lags + 1)

    # Sample autocorrelations
    acf = np.array([
        np.corrcoef(x[:-lag], x[lag:])[0, 1] for lag in lags
    ])

    # AR(1) fit: regress x[t] on x[t-1] -> estimated alpha
    slope, *_ = linregress(x[:-1], x[1:])
    alpha_hat  = float(np.clip(slope, 1e-6, 1.0 - 1e-6))
    half_life  = -np.log(2) / np.log(alpha_hat)
    ar1_curve  = alpha_hat ** lags

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(lags, acf, color=_BLUE, alpha=0.6, label="Sample ACF", width=0.6)
    ax.plot(lags, ar1_curve, color=_RED, lw=2, ls="--",
            label=f"AR(1) fit  α={alpha_hat:.3f}  half-life={half_life:.1f}d")
    ax.axhline(0.5, color=_GREY, lw=1, ls=":", alpha=0.8, label="Half-life threshold")
    ax.axhline(0.0, color="black", lw=0.7)

    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(framealpha=0)
    ax.set_xlim(0.5, max_lags + 0.5)
    ax.set_ylim(-0.2, 1.0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig
