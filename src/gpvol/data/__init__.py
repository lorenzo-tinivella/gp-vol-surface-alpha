"""
Data acquisition, IV surface construction, and cleaning.

loader.py
    - fetch_option_chain(ticker, date) -> DataFrame
      yfinance wrapper with local caching

iv_surface.py
    - build_iv_surface(chain, underlying_price, valuation_date, r) -> DataFrame
      adds T, mid, iv, log_moneyness to a raw chain (Step 1)
      NaN on bad data (missing quote, non-bracketable price); raises on
      schema violations. Never drops rows: len(out) == len(in).

cleaning.py
    - filter_liquidity(df, min_oi, max_spread_pct) -> DataFrame
    - filter_static_arbitrage(df) -> DataFrame
      Butterfly:  d2C/dK2 >= 0
      Calendar:   dC/dT  >= 0
"""
