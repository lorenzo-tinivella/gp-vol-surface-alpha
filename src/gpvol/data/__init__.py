"""
Acquisizione e pulizia dati.

loader.py
    - fetch_option_chain(ticker, date) -> DataFrame
      wrapper su yfinance con caching locale

cleaning.py
    - filter_liquidity(df, min_oi, max_spread_pct) -> DataFrame
    - filter_static_arbitrage(df) -> DataFrame
      Butterfly:  d2C/dK2 >= 0
      Calendar:   dC/dT  >= 0
"""
