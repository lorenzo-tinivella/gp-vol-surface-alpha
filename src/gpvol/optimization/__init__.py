"""
Bayesian Optimization wrappers.

bayes_opt.py
    - calibrate_svi_bo(market_iv, k_grid, bounds, n_calls) -> SVIParams
      minimizes sum((IV_market - IV_SVI)^2) via skopt.gp_minimize

    - optimize_threshold_bo(signals_df, pnl_fn, bounds, n_calls) -> float
      maximizes out-of-sample Sharpe on the training window
      (used inside the walk-forward loop)
"""
