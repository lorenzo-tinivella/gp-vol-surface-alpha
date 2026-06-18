"""
Composite scoring function (Steps 5 and 6).

The score is the decision function that converts GP and SVI surface estimates
into a tradable signal. It combines five multipliers, each encoding one
economic condition that must hold for a trade to be credible:

    score(k, T) = z * cal * conf * cons * delta_net

    z          = |IV_market - mu_GP| / sigma_GP
                 How large is the deviation in units of GP uncertainty?

    cal        = calendar_weight(expiry, events)
                 Is the option free of known-event contamination (earnings,
                 FOMC)? If 0, the high IV is event risk, not a mispricing.

    conf       = 1 / (1 + sigma_GP)
                 How much should we trust the GP estimate here? Low where
                 the surface is data-sparse (illiquid strikes/expiries).

    cons       = consistency(IV_market, mu_GP, mu_SVI)
                 Do two independent models (non-parametric GP, parametric
                 SVI) agree on the direction of the mispricing? If not,
                 at least one model is wrong and we cannot tell which.

    delta_net  = max(|IV_market - mu_GP| - bid_ask/2, 0)
                 Does the signal survive transaction costs? Zero means
                 the spread absorbs the entire apparent gain.

Score is always >= 0 (it is a magnitude). The direction of the trade is
tracked separately in score_surface() as a +1 / -1 column.

Two uses:
- scalar functions (z_score, calendar_weight, ..., composite_score):
  clean, testable in isolation, used in notebooks for illustration
- score_surface(): vectorised over a full surface DataFrame, produces
  the ranked signal table used by the walk-forward backtest (Step 7)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "z_score",
    "calendar_weight",
    "confidence",
    "consistency",
    "net_deviation",
    "composite_score",
    "score_surface",
]


# ---------------------------------------------------------------------------
# Scalar component functions
# ---------------------------------------------------------------------------

def z_score(market_iv: float, mu_gp: float, sigma_gp: float) -> float:
    """
    Standardised absolute deviation of market IV from GP estimate.

        z = |IV_market - mu_GP| / sigma_GP

    z measures how many GP standard-deviations the market price is away from
    the model consensus. A threshold of z > 2 means "unlikely to be noise at
    the 95% level, under a Gaussian model for the GP residuals."
    """
    return abs(market_iv - mu_gp) / sigma_gp


def calendar_weight(
    expiry: pd.Timestamp,
    events: list[pd.Timestamp],
    buffer_days: int = 3,
) -> float:
    """
    Return 0.0 if expiry falls within buffer_days of any known event,
    1.0 otherwise.

    An option that spans an earnings date or FOMC meeting has elevated IV for
    a legitimate reason (event risk), not because of a market-maker
    interpolation error. Treating it as a mispricing signal would produce
    systematically wrong trades around every announcement.

    Parameters
    ----------
    expiry : option expiry date
    events : list of known event dates (FOMC, earnings, macro releases)
    buffer_days : symmetric window around each event (default 3)
    """
    for event in events:
        if abs((expiry - event).days) <= buffer_days:
            return 0.0
    return 1.0


def confidence(sigma_gp: float) -> float:
    """
    GP-uncertainty-based confidence weight.

        conf = 1 / (1 + sigma_GP)  in (0, 1]

    In liquid, data-dense regions sigma_GP is small -> conf near 1.
    In illiquid, data-sparse regions sigma_GP is large -> conf near 0.
    This is the mechanism that makes the composite score automatically
    conservative in the wings (Step 3, docs/methodology.md).
    """
    return 1.0 / (1.0 + sigma_gp)


def consistency(market_iv: float, mu_gp: float, mu_svi: float) -> float:
    """
    Return 1.0 if GP and SVI agree on the direction of the deviation,
    0.0 otherwise.

    The GP (non-parametric) and SVI (parametric, market-maker convention)
    are methodologically independent. If both say the market price is too
    high (or too low) relative to their estimates, there is a stronger case
    that the market is wrong rather than one of the models being wrong.

    When the models disagree, we cannot determine which is correct, so we
    suppress the signal entirely rather than act on ambiguous information.

    Special case: if market_iv equals either model exactly (deviation=0),
    there is no signal in that direction -- treated as inconsistent.
    """
    dev_gp  = market_iv - mu_gp
    dev_svi = market_iv - mu_svi
    if dev_gp == 0.0 or dev_svi == 0.0:
        return 0.0
    return 1.0 if np.sign(dev_gp) == np.sign(dev_svi) else 0.0


def net_deviation(
    market_iv: float,
    mu_gp: float,
    bid_ask_spread: float,
) -> float:
    """
    Gross deviation minus half the bid-ask spread, clipped to zero.

        delta_net = max(|IV_market - mu_GP| - bid_ask / 2, 0)

    A signal that cannot cover the round-trip transaction cost is not
    tradable. bid_ask / 2 is the one-way cost of entering the position
    (buying at the ask or selling at the bid).
    """
    gross = abs(market_iv - mu_gp)
    return max(gross - bid_ask_spread / 2.0, 0.0)


def composite_score(
    market_iv: float,
    mu_gp: float,
    sigma_gp: float,
    mu_svi: float,
    expiry: pd.Timestamp,
    events: list[pd.Timestamp],
    bid_ask_spread: float,
    buffer_days: int = 3,
) -> float:
    """
    Composite signal score: product of all five multipliers.

        score = z * cal * conf * cons * delta_net  >= 0

    Returns 0.0 if any single multiplier is zero:
    - cal=0   : option spans a known event (not a mispricing)
    - cons=0  : GP and SVI disagree (ambiguous signal)
    - delta_net=0 : signal absorbed by transaction costs

    A non-zero score represents a candidate trade. The caller ranks
    contracts by score and trades the top-N above the BO-optimised
    threshold (Step 7).
    """
    cal = calendar_weight(expiry, events, buffer_days)
    if cal == 0.0:
        return 0.0

    cons = consistency(market_iv, mu_gp, mu_svi)
    if cons == 0.0:
        return 0.0

    nd = net_deviation(market_iv, mu_gp, bid_ask_spread)
    if nd == 0.0:
        return 0.0

    z    = z_score(market_iv, mu_gp, sigma_gp)
    conf = confidence(sigma_gp)

    return z * cal * conf * cons * nd


# ---------------------------------------------------------------------------
# Vectorised surface scorer
# ---------------------------------------------------------------------------

def score_surface(
    surface: pd.DataFrame,
    mu_gp: np.ndarray,
    sigma_gp: np.ndarray,
    mu_svi: np.ndarray,
    events: list[pd.Timestamp],
    buffer_days: int = 3,
) -> pd.DataFrame:
    """
    Compute composite scores for every contract in the cleaned surface.

    Parameters
    ----------
    surface : DataFrame output of build_iv_surface -> filter_liquidity ->
        filter_static_arbitrage. Must have columns: iv, bid, ask, expiry.
    mu_gp : GP posterior mean for each row, shape (n,)
    sigma_gp : GP posterior std for each row, shape (n,)
    mu_svi : SVI implied vol for each row, shape (n,)
    events : list of known event dates (FOMC, earnings, macro releases)
    buffer_days : calendar filter window around each event

    Returns
    -------
    DataFrame
        Input surface with added columns:
        mu_gp, sigma_gp, mu_svi  : model estimates
        score                     : composite score (>= 0)
        direction                 : +1 (sell vol) or -1 (buy vol)
    """
    out = surface.copy()
    out["mu_gp"]    = mu_gp
    out["sigma_gp"] = sigma_gp
    out["mu_svi"]   = mu_svi

    bid_ask = (out["ask"] - out["bid"]).values
    expiries = out["expiry"].values

    scores    = np.zeros(len(out))
    directions = np.zeros(len(out), dtype=int)

    for i in range(len(out)):
        iv   = float(out["iv"].iloc[i])
        mgp  = float(mu_gp[i])
        sgp  = float(sigma_gp[i])
        msvi = float(mu_svi[i])
        exp  = pd.Timestamp(expiries[i])
        ba   = float(bid_ask[i])

        scores[i] = composite_score(
            market_iv=iv,
            mu_gp=mgp,
            sigma_gp=sgp,
            mu_svi=msvi,
            expiry=exp,
            events=events,
            bid_ask_spread=ba,
            buffer_days=buffer_days,
        )
        directions[i] = 1 if iv > mgp else -1

    out["score"]     = scores
    out["direction"] = directions

    return out
