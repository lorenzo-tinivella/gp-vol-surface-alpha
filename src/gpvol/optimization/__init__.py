"""
Wrapper di Bayesian Optimization.

bayes_opt.py
    - calibrate_svi_bo(market_iv, k_grid, bounds, n_calls) -> SVIParams
      minimizza sum((IV_market - IV_SVI)^2) via skopt.gp_minimize

    - optimize_threshold_bo(signals_df, pnl_fn, bounds, n_calls) -> float
      massimizza Sharpe OOS su finestra di training
      (usato dentro il walk-forward loop)
"""
