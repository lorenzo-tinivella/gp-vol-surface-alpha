"""
Surface models: the non-parametric reference (GP) and the
parametric baseline (SVI).

gp_model.py
    - GPSurface
      wraps sklearn.GaussianProcessRegressor
      kernel = RBF + Matern(nu=2.5) + WhiteKernel
      fit(X=[k, log(T)], y=IV)
      predict_grid() -> (mu_grid, sigma_grid)

svi_model.py
    - SVIParams: dataclass(a, b, rho, m, sigma)
    - svi_total_variance(k, params) -> float
      sigma^2(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    - check_no_arbitrage(params) -> bool
      b>=0, |rho|<1, sigma>0, a + b*sigma*sqrt(1-rho^2) >= 0
"""
