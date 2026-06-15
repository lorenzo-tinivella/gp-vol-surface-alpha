"""
Raw options data acquisition from Yahoo Finance (via yfinance).

Design boundaries
-----------------
This module owns exactly one responsibility: fetch one expiry's option chain
from Yahoo Finance and return it in the schema that build_iv_surface expects.
It does NOT:
  - compute IV, T, or log-moneyness  (iv_surface.py)
  - filter by liquidity or no-arbitrage constraints  (cleaning.py)
  - cache to disk  (planned: see roadmap)

The public API is intentionally narrow (one function, one expiry at a time)
so that callers control iteration over expiry dates and caching policy.

Column mapping: yfinance -> gpvol schema
----------------------------------------
yfinance camelCase       gpvol snake_case
--------------------     --------------------
openInterest          -> open_interest
impliedVolatility     -> implied_volatility
(calls df)            -> option_type = "call"
(puts df)             -> option_type = "put"

All other columns are forwarded as-is (strike, bid, ask, volume,
contractSymbol, lastPrice, inTheMoney, ...) so that downstream modules
can use them without re-fetching.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

__all__ = ["fetch_option_chain"]

# yfinance camelCase -> our snake_case; only what we normalise explicitly.
# Columns not in this map are forwarded under their original yfinance name.
_RENAME = {
    "openInterest": "open_interest",
    "impliedVolatility": "implied_volatility",
}


def fetch_option_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """
    Fetch one expiry's option chain from Yahoo Finance.

    Parameters
    ----------
    ticker : Yahoo Finance ticker symbol, e.g. "SPY", "AAPL".
    expiry : Expiry date string as returned by yf.Ticker.options,
        e.g. "2024-01-19".  Must be an exact match -- yfinance raises
        ValueError if the date is not in the listed expiries.

    Returns
    -------
    DataFrame
        Columns include at minimum:
          expiry (pd.Timestamp), strike (float), option_type ("call"/"put"),
          bid (float), ask (float), open_interest (int), implied_volatility
          (float), plus any other columns yfinance provides.

        Row count = len(calls) + len(puts) for this expiry.
        No rows are dropped -- zero-quote or zero-OI contracts are included
        (filtering is cleaning.py's responsibility).

    Raises
    ------
    ValueError
        If the ticker has no listed options, or if expiry is not in the
        ticker's listed expiry dates.
    """
    t = yf.Ticker(ticker)

    if not t.options:
        raise ValueError(
            f"Ticker '{ticker}' has no listed options. "
            "Check the symbol or try a different underlying."
        )

    chain = t.option_chain(expiry)

    calls = _normalise(chain.calls, option_type="call", expiry=expiry)
    puts = _normalise(chain.puts, option_type="put", expiry=expiry)

    return pd.concat([calls, puts], ignore_index=True)


def _normalise(df: pd.DataFrame, option_type: str, expiry: str) -> pd.DataFrame:
    """
    Rename yfinance columns to gpvol schema, add option_type and expiry.
    """
    out = df.rename(columns=_RENAME).copy()
    out["option_type"] = option_type
    out["expiry"] = pd.Timestamp(expiry)
    return out
