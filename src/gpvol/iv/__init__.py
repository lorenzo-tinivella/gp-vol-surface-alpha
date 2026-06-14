"""
Black-Scholes pricing and implied volatility inversion.

black_scholes.py
    - bs_price(S, K, T, r, sigma, option_type) -> float
    - bs_greeks(S, K, T, r, sigma, option_type) -> (delta, gamma, vega)
    - implied_vol(price, S, K, T, r, option_type) -> float
      inversion via scipy.optimize.brentq
    - log_moneyness(S, K, T, r) -> float
      k = ln(K / F),  F = S * exp(r*T)
"""
