"""
Composite scoring function.

scoring.py
    - z_score(market_iv, mu_gp, sigma_gp) -> float
    - calendar_weight(expiry, events, buffer_days=3) -> {0, 1}
    - confidence(sigma_gp) -> float           # 1 / (1 + sigma_gp)
    - consistency(market_iv, mu_gp, mu_svi) -> {0, 1}
    - net_deviation(market_iv, mu_gp, bid_ask_spread) -> float
      max(|market_iv - mu_gp| - spread/2, 0)
    - composite_score(...) -> float
      z * cal * conf * cons * net_deviation
"""
