"""
Delta-hedging and Gamma P&L decomposition (Step 8).

Purpose
-------
Delta-hedging eliminates directional (equity beta) exposure, leaving only
the Gamma P&L proportional to (sigma_realized^2 - sigma_IV^2). This is
the pure test of the vol alpha identified by the composite scorer.

Transaction costs
-----------------
Three costs are modelled. Understanding their relative magnitude is
critical for assessing strategy viability:

    borrow_rate (annualised)
        Daily cost of shorting the underlying shares for the delta-hedge.
        SPY / SPX:            0.003  (30bp)  -- near zero
        Liquid single name:   0.010-0.030
        Hard-to-borrow:       0.05-0.15+

    commission_per_contract
        Round-trip cost per option contract (entry + exit = 2x).
        Zero-commission brokers:  0.00   (IBKR Lite, Tastytrade)
        Retail (IBKR Pro):        0.65
        Institutional:            0.05-0.15
        WARNING: this is the dominant cost for small positions.
        A $0.65 commission on a position generating $0.35 in gross Gamma
        P&L over 20 days makes the trade unprofitable at retail rates.
        See cost_breakdown() for analytical scenario analysis.

    slippage_pct (per unit notional rebalanced)
        Bid-ask cost of adjusting the delta hedge daily.
        SPY:                  0.0001  (1bp)
        Liquid single name:   0.0003-0.0010
        Illiquid:             0.005+

Break-even (ATM call, S=100, T=3m, sigma_IV=20%, sigma_real=25%, 20 days):
    Gross Gamma P&L:                  $0.35 / contract
    SPY, zero-commission:             $0.33 retained  (92.7%)
    SPY, $0.65/contract retail:      -$0.97           (-276%)
    AAPL, $0.65/contract retail:     -$1.03           (-292%)

    Strategy requires zero-commission or institutional rates (<$0.15)
    to be viable. Documented in README Limitations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = ["daily_gamma_pnl", "delta_hedge_pnl", "cost_breakdown"]


# ---------------------------------------------------------------------------
# Core P&L functions
# ---------------------------------------------------------------------------

def daily_gamma_pnl(
    gamma: float,
    S: float,
    r_daily: float,
    sigma_iv: float,
    delta: float = 0.0,
    borrow_rate: float = 0.003,
    slippage_pct: float = 0.0001,
    delta_prev: float | None = None,
) -> float:
    """
    Net daily P&L of a delta-hedged option position.

        gross     = 0.5 * Gamma * S^2 * (r_daily^2 - sigma_IV^2/252)
        borrow    = borrow_rate/252 * |delta| * S
        slippage  = slippage_pct * |delta - delta_prev| * S
        net       = gross - borrow - slippage

    Parameters
    ----------
    gamma         : option gamma (positive for long vanilla)
    S             : spot price at start of day
    r_daily       : realized daily return (dS/S)
    sigma_iv      : annualized IV paid at entry (held constant)
    delta         : current delta (shares short for delta-neutral)
    borrow_rate   : annualized borrow cost on shorted shares
    slippage_pct  : bid-ask cost per unit of notional rebalanced
    delta_prev    : previous day delta; if None no slippage charged

    Returns
    -------
    float : net daily P&L
    """
    sigma_daily_sq = sigma_iv**2 / 252.0
    gross = 0.5 * gamma * S**2 * (r_daily**2 - sigma_daily_sq)

    borrow_cost = borrow_rate / 252.0 * abs(delta) * S

    if delta_prev is not None:
        slippage = slippage_pct * abs(delta - delta_prev) * S
    else:
        slippage = 0.0

    return gross - borrow_cost - slippage


def delta_hedge_pnl(
    S_path: pd.Series,
    K: float,
    T: float,
    r: float,
    sigma_iv: float,
    option_type: str = "call",
    borrow_rate: float = 0.003,
    commission_per_contract: float = 0.0,
    slippage_pct: float = 0.0001,
) -> pd.Series:
    """
    Simulate the daily net P&L of a delta-hedged option position.

    Commission is charged once on day 0 (entry) and once on the last
    day (exit): total round-trip = 2 * commission_per_contract.
    Borrow and slippage are charged daily.

    Parameters
    ----------
    S_path                  : daily spot price path
    K                       : strike
    T                       : time to maturity at entry (years)
    r                       : risk-free rate (continuous)
    sigma_iv                : implied vol at entry (annualized)
    option_type             : "call" or "put"
    borrow_rate             : annualized borrow rate (default 30bp)
    commission_per_contract : one-way commission; round-trip = 2x
    slippage_pct            : per-unit slippage on delta rebalancing

    Returns
    -------
    pd.Series : daily net P&L, same index as S_path
    """
    n = len(S_path)
    pnl_values = np.zeros(n)
    delta_prev = None

    for i in range(n):
        S_i = float(S_path.iloc[i])
        T_i = max(T - i / 252.0, 1e-6)

        gamma_i = _bs_gamma(S_i, K, T_i, r, sigma_iv)
        delta_i = _bs_delta(S_i, K, T_i, r, sigma_iv, option_type)

        if i == 0:
            # Entry: no price move yet, deduct entry commission
            pnl_values[i] = -commission_per_contract
        else:
            S_prev  = float(S_path.iloc[i - 1])
            r_daily = (S_i - S_prev) / S_prev
            pnl_values[i] = daily_gamma_pnl(
                gamma=gamma_i, S=S_i, r_daily=r_daily, sigma_iv=sigma_iv,
                delta=delta_i, borrow_rate=borrow_rate,
                slippage_pct=slippage_pct, delta_prev=delta_prev,
            )

        if i == n - 1:
            # Exit: deduct exit commission
            pnl_values[i] -= commission_per_contract

        delta_prev = delta_i

    return pd.Series(pnl_values, index=S_path.index, name="net_pnl")


# ---------------------------------------------------------------------------
# Analytical cost breakdown (no simulation needed)
# ---------------------------------------------------------------------------

def cost_breakdown(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma_iv: float,
    sigma_real: float,
    n_days: int,
    option_type: str = "call",
    borrow_rate: float = 0.003,
    commission_per_contract: float = 0.0,
    slippage_pct: float = 0.0001,
) -> dict:
    """
    Analytical cost breakdown for a delta-hedged position held n_days.

    Does NOT run a simulation — uses the Black-Scholes analytical values
    at inception to estimate each cost component. Useful for quick scenario
    analysis before committing to a full path simulation.

    Returns
    -------
    dict with keys: gross, borrow, commission, slippage, net, pct_retained
    """
    gamma = _bs_gamma(S, K, T, r, sigma_iv)
    delta = abs(_bs_delta(S, K, T, r, sigma_iv, option_type))

    r_real_sq      = (sigma_real / np.sqrt(252))**2
    sigma_daily_sq = sigma_iv**2 / 252.0
    gross_per_day  = 0.5 * gamma * S**2 * (r_real_sq - sigma_daily_sq)
    gross_total    = gross_per_day * n_days

    borrow_total   = borrow_rate / 252.0 * delta * S * n_days
    comm_total     = commission_per_contract * 2          # round-trip
    delta_change   = gamma * S * sigma_real / np.sqrt(252)
    slippage_total = slippage_pct * delta_change * S * n_days

    net = gross_total - borrow_total - comm_total - slippage_total
    pct = (net / gross_total * 100) if gross_total != 0 else float("nan")

    return {
        "gross":        round(gross_total, 4),
        "borrow":       round(-borrow_total, 4),
        "commission":   round(-comm_total, 4),
        "slippage":     round(-slippage_total, 4),
        "net":          round(net, 4),
        "pct_retained": round(pct, 1),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return float(norm.pdf(d1) / (S * sigma * sqrt_T))


def _bs_delta(S: float, K: float, T: float, r: float,
              sigma: float, option_type: str = "call") -> float:
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return float(norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0)
