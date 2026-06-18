"""
Tests for gpvol.signal.scoring (Steps 5 and 6).

The composite score is the decision function of the entire project:
    score = z * cal * conf * cons * delta_net

Every component has a precise economic meaning and a mathematical contract.
A bug in any one silently corrupts the signal -- a false positive looks like
alpha; a false negative loses a trade opportunity.

Component contracts tested here:
    z        = |IV_market - mu_GP| / sigma_GP         >= 0
    cal      = 0 if expiry within buffer_days of event, else 1.0
    conf     = 1 / (1 + sigma_GP)                     in (0, 1]
    cons     = 1 if sign(IV_market-mu_GP)==sign(IV_market-mu_SVI), else 0
    delta_net = max(|IV_market - mu_GP| - bid_ask/2, 0)  >= 0
    score    = product of all five                     >= 0

Score is always >= 0 (it is a magnitude).
Direction of the trade is tracked separately:
    direction = +1 (sell vol) if IV_market > mu_GP
    direction = -1 (buy vol)  if IV_market < mu_GP

The four zero conditions:
    cal=0    -> score=0 (option spans a known event -- not a mispricing)
    cons=0   -> score=0 (GP and SVI disagree on direction -- too ambiguous)
    delta_net=0 -> score=0 (signal too small to cover transaction costs)
    score=0 iff any multiplier is 0 -- this is tested explicitly.
"""

import numpy as np
import pandas as pd
import pytest

from gpvol.signal.scoring import (
    calendar_weight,
    composite_score,
    confidence,
    consistency,
    net_deviation,
    score_surface,
    z_score,
)

# ---------------------------------------------------------------------------
# Reference values (verified in exploration)
# ---------------------------------------------------------------------------
_VAL_DATE = pd.Timestamp("2024-01-15")
_FOMC     = _VAL_DATE + pd.Timedelta(days=1)   # event 1 day away


# ---------------------------------------------------------------------------
# z_score
# ---------------------------------------------------------------------------

def test_z_score_formula():
    assert np.isclose(z_score(market_iv=0.25, mu_gp=0.20, sigma_gp=0.02), 2.5)


def test_z_score_symmetric():
    """z-score is based on absolute deviation -- direction doesn't affect magnitude."""
    z_over  = z_score(market_iv=0.25, mu_gp=0.20, sigma_gp=0.02)
    z_under = z_score(market_iv=0.15, mu_gp=0.20, sigma_gp=0.02)
    assert np.isclose(z_over, z_under)


def test_z_score_zero_at_fair_value():
    assert z_score(market_iv=0.20, mu_gp=0.20, sigma_gp=0.02) == 0.0


def test_z_score_non_negative():
    assert z_score(market_iv=0.10, mu_gp=0.20, sigma_gp=0.05) >= 0.0


# ---------------------------------------------------------------------------
# calendar_weight
# ---------------------------------------------------------------------------

def test_calendar_weight_within_buffer():
    """Expiry 2 days from FOMC (buffer=3) -> weight 0."""
    expiry = _VAL_DATE + pd.Timedelta(days=2)
    w = calendar_weight(expiry, events=[_FOMC], buffer_days=3)
    assert w == 0.0


def test_calendar_weight_outside_buffer():
    """Expiry 30 days from FOMC -> weight 1."""
    expiry = _VAL_DATE + pd.Timedelta(days=30)
    w = calendar_weight(expiry, events=[_FOMC], buffer_days=3)
    assert w == 1.0


def test_calendar_weight_exactly_at_boundary():
    """Expiry exactly buffer_days from event is still within buffer (<=)."""
    expiry = _VAL_DATE + pd.Timedelta(days=4)   # 3 days from FOMC
    w = calendar_weight(expiry, events=[_FOMC], buffer_days=3)
    assert w == 0.0


def test_calendar_weight_no_events():
    """No events in calendar -> all options tradable."""
    expiry = _VAL_DATE + pd.Timedelta(days=30)
    w = calendar_weight(expiry, events=[], buffer_days=3)
    assert w == 1.0


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------

def test_confidence_formula():
    assert np.isclose(confidence(sigma_gp=0.02), 1 / (1 + 0.02))


def test_confidence_approaches_one_near_zero():
    assert confidence(sigma_gp=1e-6) > 0.999


def test_confidence_bounded():
    """confidence must be in (0, 1] for any sigma > 0."""
    for sigma in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
        c = confidence(sigma_gp=sigma)
        assert 0 < c <= 1.0, f"confidence={c} for sigma={sigma}"


# ---------------------------------------------------------------------------
# consistency
# ---------------------------------------------------------------------------

def test_consistency_same_sign_is_one():
    """Both GP and SVI say market is overpriced -> consistent."""
    c = consistency(market_iv=0.25, mu_gp=0.20, mu_svi=0.21)
    assert c == 1.0


def test_consistency_opposite_signs_is_zero():
    """GP says overpriced, SVI says underpriced -> inconsistent."""
    c = consistency(market_iv=0.25, mu_gp=0.20, mu_svi=0.28)
    assert c == 0.0


def test_consistency_both_underpriced():
    """Both models say market is underpriced -> consistent."""
    c = consistency(market_iv=0.18, mu_gp=0.20, mu_svi=0.21)
    assert c == 1.0


# ---------------------------------------------------------------------------
# net_deviation
# ---------------------------------------------------------------------------

def test_net_deviation_clears_spread():
    """Signal of 5 vol-pts, spread of 2 vol-pts -> net = 4 vol-pts."""
    nd = net_deviation(market_iv=0.25, mu_gp=0.20, bid_ask_spread=0.02)
    assert np.isclose(nd, 0.04)


def test_net_deviation_clipped_at_zero():
    """Signal smaller than half the spread -> net = 0, no trade."""
    nd = net_deviation(market_iv=0.21, mu_gp=0.20, bid_ask_spread=0.02)
    assert nd == 0.0


def test_net_deviation_non_negative():
    nd = net_deviation(market_iv=0.20, mu_gp=0.20, bid_ask_spread=0.05)
    assert nd >= 0.0


# ---------------------------------------------------------------------------
# composite_score
# ---------------------------------------------------------------------------

_VALID_KWARGS = dict(
    market_iv=0.25,
    mu_gp=0.20,
    sigma_gp=0.02,
    mu_svi=0.21,
    expiry=_VAL_DATE + pd.Timedelta(days=30),
    events=[_FOMC],
    bid_ask_spread=0.02,
    buffer_days=3,
)


def test_composite_score_positive_for_valid_signal():
    score = composite_score(**_VALID_KWARGS)
    assert score > 0.0


def test_composite_score_zero_when_cal_zero():
    """Option near a known event -> calendar = 0 -> score = 0."""
    kwargs = {**_VALID_KWARGS, "expiry": _VAL_DATE + pd.Timedelta(days=2)}
    assert composite_score(**kwargs) == 0.0


def test_composite_score_zero_when_inconsistent():
    """GP and SVI disagree -> consistency = 0 -> score = 0."""
    kwargs = {**_VALID_KWARGS, "mu_svi": 0.28}   # SVI says underpriced, GP says overpriced
    assert composite_score(**kwargs) == 0.0


def test_composite_score_zero_when_no_net_deviation():
    """Tiny signal absorbed by spread -> delta_net = 0 -> score = 0."""
    kwargs = {**_VALID_KWARGS, "market_iv": 0.201}   # only 0.001 above GP, spread=0.02
    assert composite_score(**kwargs) == 0.0


def test_composite_score_non_negative():
    assert composite_score(**_VALID_KWARGS) >= 0.0


# ---------------------------------------------------------------------------
# score_surface (vectorized)
# ---------------------------------------------------------------------------

def _make_surface():
    """Minimal surface DataFrame for vectorized scoring tests."""
    return pd.DataFrame({
        "log_moneyness": [-0.1, 0.0, 0.1],
        "T":             [0.25, 0.25, 0.25],
        "iv":            [0.25, 0.20, 0.18],
        "mid":           [5.0,  3.0,  2.5],
        "bid":           [4.9,  2.9,  2.4],
        "ask":           [5.1,  3.1,  2.6],
        "expiry":        [_VAL_DATE + pd.Timedelta(days=30)] * 3,
        "strike":        [90.0, 100.0, 110.0],
        "option_type":   ["put", "call", "call"],
    })


def test_score_surface_returns_expected_columns():
    """score_surface must add score and direction columns."""
    surface = _make_surface()
    mu_gp    = np.array([0.20, 0.20, 0.20])
    sigma_gp = np.array([0.02, 0.02, 0.02])
    mu_svi   = np.array([0.21, 0.20, 0.19])

    out = score_surface(surface, mu_gp, sigma_gp, mu_svi,
                        events=[_FOMC], buffer_days=3)

    for col in ("score", "direction", "mu_gp", "sigma_gp", "mu_svi"):
        assert col in out.columns, f"missing column: {col}"


def test_score_surface_preserves_row_count():
    surface = _make_surface()
    mu_gp    = np.array([0.20, 0.20, 0.20])
    sigma_gp = np.array([0.02, 0.02, 0.02])
    mu_svi   = np.array([0.21, 0.20, 0.19])

    out = score_surface(surface, mu_gp, sigma_gp, mu_svi,
                        events=[_FOMC], buffer_days=3)

    assert len(out) == len(surface)


def test_score_surface_direction_sign():
    """direction must be +1 where market > GP (sell vol) and -1 where market < GP."""
    surface = _make_surface()
    mu_gp    = np.array([0.20, 0.20, 0.20])
    sigma_gp = np.array([0.02, 0.02, 0.02])
    mu_svi   = np.array([0.21, 0.19, 0.19])

    out = score_surface(surface, mu_gp, sigma_gp, mu_svi,
                        events=[_FOMC], buffer_days=3)

    # iv=[0.25, 0.20, 0.18], mu_gp=0.20
    # row 0: 0.25 > 0.20 -> +1 (sell)
    # row 2: 0.18 < 0.20 -> -1 (buy)
    assert out.loc[0, "direction"] == 1
    assert out.loc[2, "direction"] == -1
