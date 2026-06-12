"""
Walk-forward engine, delta-hedge simulation, performance metrics.

engine.py
    - WalkForwardConfig(train_window_days, test_window_days)
    - run_walkforward(data, config) -> WalkForwardResults
      per ogni finestra: BO su train -> soglia tau* -> test OOS

hedging.py
    - delta_hedge_pnl(position, S_path, sigma_iv_entry, r) -> Series
      Gamma P&L: 0.5 * Gamma * S^2 * (r_t^2 - sigma_iv^2) * dt

metrics.py
    - sharpe_ratio(returns) -> float
    - deflated_sharpe_ratio(sr, n_trials, skew, kurt) -> float
    - max_drawdown(equity_curve) -> float
    - signal_half_life(signal_series) -> float
      da AR(1): half_life = -log(2) / log(alpha)
"""
