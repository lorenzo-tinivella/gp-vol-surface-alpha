"""
Post-surface cleaning: liquidity and static no-arbitrage filters.

These functions operate on the output of build_iv_surface (columns: expiry,
strike, option_type, bid, ask, mid, T, iv, log_moneyness, open_interest, ...)
and return a subset of rows with violating contracts removed. They are the
last step before the GP fits on the data (Step 3).

Responsibility boundary
-----------------------
- build_iv_surface (iv_surface.py): assigns NaN to bad/missing data; never
  drops rows.
- filter_liquidity: drops rows with insufficient market activity. Contracts
  removed here are illiquid -- their prices are too wide or too thin to be
  reliable surface inputs regardless of no-arbitrage status.
- filter_static_arbitrage: drops rows that create risk-free arbitrage in
  the price grid. Contracts removed here could in principle be traded as
  static arbitrage by an arbitrageur with no transaction costs; in practice
  they are almost always stale or erroneous quotes.

No-arbitrage conditions implemented
------------------------------------
Butterfly (per expiry, per option_type):
  For consecutive strikes K1 < K2 < K3:
    mid(K2) <= linear_interp(mid(K1), mid(K3), at K2)
  i.e. prices must be convex in strike (d^2C/dK^2 >= 0, Breeden-Litzenberger).
  Only the middle point K2 is flagged; K1 and K3 are never blamed for a
  triple violation.

Calendar spread (per strike, per option_type):
  Total variance w(K, T) = iv^2 * T must be non-decreasing in T.
  If w(K, T_short) > w(K, T_long), the shorter-maturity contract is removed
  (short-dated options are more prone to stale quotes / event distortion).
  Only exact strike matches across expiries are compared (no interpolation).
  Rows with NaN iv are excluded from the calendar check.

Limitations
-----------
- Butterfly uses mid prices, not individual bid/ask -- a tighter check would
  compare the butterfly spread against the bid-ask cost, but that requires
  transaction-level data that may not always be available.
- Calendar check uses exact strike matching only. A full arbitrage-free SVI
  calibration (gpvol.surface.svi_model) provides a stricter global check.
- No cross-type checks (e.g. put-call parity is enforced implicitly by the
  BS IV inversion in build_iv_surface, not explicitly here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["filter_liquidity", "filter_static_arbitrage"]

_ARBITRAGE_REQUIRED = {"mid", "strike", "expiry", "option_type", "iv", "T"}
_BUTTERFLY_TOL = 1e-6   # numerical tolerance for the convexity check
_CALENDAR_TOL  = 1e-6   # numerical tolerance for the total-variance check


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_liquidity(
    df: pd.DataFrame,
    min_open_interest: int = 50,
    max_spread_pct: float = 0.20,
) -> pd.DataFrame:
    """
    Remove contracts with insufficient market activity.

    Parameters
    ----------
    df : output of build_iv_surface (or any DataFrame with bid, ask columns).
    min_open_interest : minimum open interest. Applied only if the
        ``open_interest`` column is present.
    max_spread_pct : maximum relative bid-ask spread: (ask - bid) / mid.
        Contracts with mid == 0 (bid == ask == 0) are always removed.

    Returns
    -------
    DataFrame
        Subset of df with illiquid contracts removed.
    """
    mask = pd.Series(True, index=df.index)

    if "iv" in df.columns:
        mask &= df["iv"].notna()

    if "open_interest" in df.columns:
        oi = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0)
        mask &= oi >= min_open_interest

    mid = (df["bid"] + df["ask"]) / 2.0
    spread = df["ask"] - df["bid"]
    spread_pct = spread / mid          # NaN where mid == 0
    mask &= spread_pct.notna() & (spread_pct <= max_spread_pct)

    return df[mask].copy()


def filter_static_arbitrage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove contracts that violate static no-arbitrage conditions.

    Applies two independent checks (butterfly, then calendar) and returns
    the subset of contracts that survive both. The order matters only for
    performance (butterfly first reduces the size of the calendar check).

    Parameters
    ----------
    df : output of build_iv_surface with at minimum:
        mid, strike, expiry, option_type, iv, T.

    Returns
    -------
    DataFrame
        Subset of df with arbitrage-violating contracts removed.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = _ARBITRAGE_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"filter_static_arbitrage requires columns: {sorted(missing)}"
        )

    if len(df) == 0:
        return df.copy()

    # Contracts with NaN iv already violated a BS no-arbitrage bound in
    # build_iv_surface. Drop them first: they cannot enter the calendar
    # total-variance check and their mid prices are unreliable for the
    # butterfly check.
    if "iv" in df.columns:
        df = df[df["iv"].notna()]

    if len(df) == 0:
        return df.copy()

    valid = pd.Series(True, index=df.index)

    # -- butterfly ---------------------------------------------------------
    for _, group in df.groupby(["expiry", "option_type"], sort=False):
        if len(group) < 3:
            continue
        invalid_idx = _butterfly_violations(group)
        valid.loc[invalid_idx] = False

    # -- calendar (only on butterfly-surviving rows) -----------------------
    surviving = df[valid]
    for _, group in surviving.groupby(["strike", "option_type"], sort=False):
        if len(group) < 2:
            continue
        invalid_idx = _calendar_violations(group)
        valid.loc[invalid_idx] = False

    return df[valid].copy()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _butterfly_violations(group: pd.DataFrame) -> list:
    """
    Return the index labels of contracts that violate convexity in strike
    within a single (expiry, option_type) group.

    Only middle points of triples are ever flagged.
    """
    g = group.sort_values("strike")
    strikes = g["strike"].values
    mids    = g["mid"].values
    labels  = g.index.tolist()
    n = len(strikes)

    bad = []
    for i in range(1, n - 1):
        K1, K2, K3 = strikes[i - 1], strikes[i], strikes[i + 1]
        C1, C2, C3 = mids[i - 1], mids[i], mids[i + 1]

        if np.isnan(C1) or np.isnan(C2) or np.isnan(C3):
            continue

        # Linear interpolation of price at K2 between (K1,C1) and (K3,C3)
        C2_interp = C1 + (C3 - C1) * (K2 - K1) / (K3 - K1)

        if C2 > C2_interp + _BUTTERFLY_TOL:
            bad.append(labels[i])

    return bad


def _calendar_violations(group: pd.DataFrame) -> list:
    """
    Return the index labels of shorter-expiry contracts that cause total
    variance w = iv^2 * T to decrease as T increases, within a single
    (strike, option_type) group.

    Rows with NaN iv or T <= 0 are excluded from the check.
    """
    with_iv = group[group["iv"].notna() & (group["T"] > 0)].sort_values("T")

    if len(with_iv) < 2:
        return []

    tv = (with_iv["iv"] ** 2 * with_iv["T"]).values
    idx = with_iv.index.tolist()

    bad = []
    for i in range(1, len(tv)):
        if tv[i] < tv[i - 1] - _CALENDAR_TOL:
            bad.append(idx[i - 1])   # shorter-expiry contract is at i-1

    return bad
