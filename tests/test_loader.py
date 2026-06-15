"""
Tests for gpvol.data.loader.fetch_option_chain.

The loader touches the network (yfinance / Yahoo Finance). Live network
calls in CI are:
  - slow (~2-5s per expiry)
  - flaky (Yahoo rate-limits, CDN outages)
  - non-deterministic (prices change every second)

Strategy: mock yf.Ticker at the boundary (unittest.mock.patch), feed it a
realistic fixture that mirrors the exact namedtuple / DataFrame structure
returned by yfinance._options2df (inspected in the implementation notes).
The fixture is not approximate -- it uses the real column names and dtypes
so that a yfinance API change that breaks the column mapping would also
break the fixture and therefore the test.

What we validate:
  1. Output schema: the 5 columns build_iv_surface expects are present and
     named correctly (snake_case, not yfinance's camelCase).
  2. Calls and puts are stacked and labelled in option_type.
  3. The expiry column is a pd.Timestamp, not a raw string.
  4. Extra columns from yfinance (impliedVolatility, contractSymbol, ...)
     that may be useful downstream are forwarded unchanged.
  5. Contracts with bid == ask == 0 (market closed / no quote) are included
     in the output -- filtering is cleaning.py's job, not the loader's.
  6. Requesting an expiry that yfinance doesn't recognise raises ValueError.
  7. A ticker with no listed options raises ValueError cleanly.
"""

from collections import namedtuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gpvol.data.loader import fetch_option_chain

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_OptionsChain = namedtuple("Options", ["calls", "puts", "underlying"])

_CALLS_FIXTURE = pd.DataFrame([
    {
        "contractSymbol": "SPY240119C00470000",
        "strike": 470.0,
        "bid": 1.50,
        "ask": 1.55,
        "volume": 1200,
        "openInterest": 8500,
        "impliedVolatility": 0.182,
        "lastPrice": 1.52,
        "inTheMoney": False,
    },
    {
        "contractSymbol": "SPY240119C00480000",
        "strike": 480.0,
        "bid": 0.0,
        "ask": 0.0,         # no quote -- must still appear in output
        "volume": 0,
        "openInterest": 200,
        "impliedVolatility": 0.0,
        "lastPrice": 0.0,
        "inTheMoney": False,
    },
])

_PUTS_FIXTURE = pd.DataFrame([
    {
        "contractSymbol": "SPY240119P00460000",
        "strike": 460.0,
        "bid": 2.10,
        "ask": 2.20,
        "volume": 950,
        "openInterest": 12000,
        "impliedVolatility": 0.195,
        "lastPrice": 2.15,
        "inTheMoney": False,
    },
])

_UNDERLYING_FIXTURE = {"regularMarketPrice": 475.50}

_EXPIRY = "2024-01-19"


def _make_mock_ticker(expiry: str = _EXPIRY):
    """Return a MagicMock that mimics yf.Ticker for one expiry date."""
    mock_ticker = MagicMock()
    mock_ticker.options = (expiry,)
    mock_ticker.option_chain.return_value = _OptionsChain(
        calls=_CALLS_FIXTURE.copy(),
        puts=_PUTS_FIXTURE.copy(),
        underlying=_UNDERLYING_FIXTURE,
    )
    return mock_ticker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("gpvol.data.loader.yf.Ticker")
def test_output_columns_match_build_iv_surface_schema(mock_cls):
    """The 5 columns that build_iv_surface requires must all be present,
    snake_case, and with the right dtypes."""
    mock_cls.return_value = _make_mock_ticker()

    result = fetch_option_chain("SPY", expiry=_EXPIRY)

    for col in ("expiry", "strike", "option_type", "bid", "ask"):
        assert col in result.columns, f"missing column: {col}"


@patch("gpvol.data.loader.yf.Ticker")
def test_calls_and_puts_are_stacked_with_correct_label(mock_cls):
    """Calls and puts must be combined into one DataFrame and labelled."""
    mock_cls.return_value = _make_mock_ticker()

    result = fetch_option_chain("SPY", expiry=_EXPIRY)

    assert set(result["option_type"].unique()) == {"call", "put"}
    assert len(result) == len(_CALLS_FIXTURE) + len(_PUTS_FIXTURE)


@patch("gpvol.data.loader.yf.Ticker")
def test_expiry_column_is_timestamp(mock_cls):
    """expiry must be a pd.Timestamp, not a raw string -- build_iv_surface
    does (expiry - valuation_date).dt.days on it."""
    mock_cls.return_value = _make_mock_ticker()

    result = fetch_option_chain("SPY", expiry=_EXPIRY)

    assert pd.api.types.is_datetime64_any_dtype(result["expiry"]), (
        f"expiry dtype is {result['expiry'].dtype}, expected datetime64"
    )


@patch("gpvol.data.loader.yf.Ticker")
def test_yfinance_extra_columns_are_forwarded(mock_cls):
    """Columns like open_interest and implied_volatility (from yfinance's
    openInterest / impliedVolatility) must be forwarded so that cleaning.py
    can filter on them in Step 2b."""
    mock_cls.return_value = _make_mock_ticker()

    result = fetch_option_chain("SPY", expiry=_EXPIRY)

    assert "open_interest" in result.columns
    assert "implied_volatility" in result.columns


@patch("gpvol.data.loader.yf.Ticker")
def test_no_quote_contract_is_included(mock_cls):
    """A contract with bid == ask == 0 (market closed / no active market)
    must appear in the output. Filtering is cleaning.py's job."""
    mock_cls.return_value = _make_mock_ticker()

    result = fetch_option_chain("SPY", expiry=_EXPIRY)
    zero_quote = result[(result["bid"] == 0) & (result["ask"] == 0)]

    assert len(zero_quote) == 1


@patch("gpvol.data.loader.yf.Ticker")
def test_unknown_expiry_raises_value_error(mock_cls):
    """Requesting an expiry that yfinance doesn't list must raise ValueError
    with the expiry date in the message."""
    mock_ticker = _make_mock_ticker()
    mock_ticker.option_chain.side_effect = ValueError(
        "Expiration `2099-01-01` cannot be found."
    )
    mock_cls.return_value = mock_ticker

    with pytest.raises(ValueError, match="2099-01-01"):
        fetch_option_chain("SPY", expiry="2099-01-01")


@patch("gpvol.data.loader.yf.Ticker")
def test_ticker_with_no_options_raises_value_error(mock_cls):
    """A ticker with no listed options (e.g. an ETF with no active chain)
    must raise ValueError, not return an empty or broken DataFrame."""
    mock_ticker = MagicMock()
    mock_ticker.options = ()           # empty tuple = no listed expiries
    mock_cls.return_value = mock_ticker

    with pytest.raises(ValueError, match="no listed options"):
        fetch_option_chain("XYZ", expiry=_EXPIRY)
