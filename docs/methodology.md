# GP Vol Surface Arbitrage — Project Reference

**Alpha Research | Options | Gaussian Process + Bayesian Optimization**

---

## Overview

The project targets systematic inefficiencies in the equity options
implied volatility surface using a Gaussian Process as a non-parametric
reference model, and Bayesian Optimization for calibration and trading
threshold selection. The economic alpha derives from the discrepancy
between market implied volatility and the GP estimate — adjusted for
liquidity, model consistency, and transaction costs — in the presence of
a well-documented Variance Risk Premium.

---

## Step 1 — Data Collection and Implied Volatility Calculation

### What we do
We download daily options chains (SPX or a liquid single name) and invert
the Black-Scholes formula to obtain the Implied Volatility (IV) for each
(strike K, maturity T) pair. We clean the data by removing contracts with
low open interest and prices that fall outside the bid-ask bounds.

### Mathematical ideas
- **Black-Scholes formula:** C = S·N(d1) − K·e^(−rT)·N(d2)
  where d1 = [ln(S/K) + (r + sigma^2/2)T] / (sigma·sqrt(T))
- **Numerical inversion:** IV = argmin_sigma |C_BS(sigma) − C_market|,
  solved with Brent's method (bracketed root-finding, guaranteed convergence)
- **Log-forward moneyness:** k = ln(K / F), where F = S·e^(rT) —
  standardizes the strike relative to the forward structure

### Economic ideas
- IV is not a historical volatility — it is the volatility that, when
  plugged into BS, reproduces the market price. It embeds both market
  expectations *and* a risk premium.
- Building the surface requires clean data: illiquid contracts have wide
  bid-ask spreads that make the computed IV unreliable and non-tradable.
- The log-forward moneyness transform removes the dependence on the spot
  level, making the surface comparable over time.

### Libraries
`yfinance`, `py_vollib`, `scipy.optimize.brentq`

### References
- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*
  Journal of Political Economy, 81(3), 637-654.
- Breeden, D. T., & Litzenberger, R. H. (1978). *Prices of State-Contingent Claims
  Implicit in Option Prices.* Journal of Business, 51(4), 621-651.

---

## Step 2 — Surface Construction and No-Arbitrage Constraints

### What we do
We organize the (k, T) -> IV points into a sparse grid and apply static
arbitrage checks. Points that violate the constraints are removed or
flagged before fitting the GP.

### Mathematical ideas
- **Butterfly arbitrage:** d^2C/dK^2 >= 0 for fixed K — the price curve as
  a function of strike must be convex. Equivalent to a non-negative
  risk-neutral density.
- **Calendar spread arbitrage:** dC/dT >= 0 for fixed K — an option with a
  later maturity cannot be worth less than one with an earlier maturity
  (same strike).
- **Breeden-Litzenberger condition:** q(S_T) = e^(rT) · d^2C/dK^2 — allows
  extracting the risk-neutral distribution of future prices from the
  surface.

### Economic ideas
- Violations of these constraints represent risk-free static arbitrage
  opportunities. In efficient markets they should not exist — when found,
  they are almost always artifacts of illiquid data or quoting errors.
- Removing these points does not impoverish the dataset: they are
  uninformative prices that would distort the GP fit.

### References
- Breeden, D. T., & Litzenberger, R. H. (1978). *Prices of State-Contingent Claims
  Implicit in Option Prices.* Journal of Business, 51(4), 621-651.
- Fengler, M. R. (2009). *Arbitrage-Free Smoothing of the Implied Volatility Surface.*
  Quantitative Finance, 9(4), 417-428.

---

## Step 3 — Gaussian Process on the Volatility Surface

### What we do
We fit a Gaussian Process Regressor on the cleaned sparse grid of points.
Input: X = [k, log(T)]. Output: a distribution over IV — a predicted mean
mu_GP(k,T) and standard deviation sigma_GP(k,T) at every point of the
surface.

### Mathematical ideas
- **GP as a prior over functions:** f ~ GP(m(x), kappa(x,x'))
  where m is the mean function (typically zero) and kappa is the
  covariance kernel.
- **Composite kernel:** kappa = RBF(l1) + Matern_5/2(l2) + WhiteNoise(sigma_n)
  - RBF captures the global smoothness of the surface
  - Matern captures local irregularities (less smooth than RBF)
  - WhiteNoise models bid-ask noise
- **Gaussian posterior:** once data is observed, the posterior is analytic:
  mu_GP(x*) = kappa(x*, X) · [kappa(X,X) + sigma_n^2 I]^-1 · y
  sigma^2_GP(x*) = kappa(x*,x*) − kappa(x*,X) · [kappa(X,X) + sigma_n^2 I]^-1 · kappa(X,x*)
- **Hyperparameter optimization** (length scales l1, l2, and noise sigma_n):
  maximize the log marginal likelihood:
  log p(y|X,theta) = -1/2 y^T K^-1 y - 1/2 log|K| - n/2 log(2*pi)

### Economic ideas
- The GP is an optimal interpolator in the mean-square sense: it uses all
  observed prices as mutual constraints to estimate IV at every point of
  the surface.
- sigma_GP(k,T) is a direct measure of local illiquidity: high where few
  contracts trade, low where the market is dense and informative.
- A significant deviation between IV_market and mu_GP signals that an
  option is priced inconsistently with its neighbors — a candidate
  mispricing.

### Libraries
`sklearn.gaussian_process`, `GPy`, `botorch` (PyTorch-based)

### References
- Cont, R., & da Fonseca, J. (2002). *Dynamics of Implied Volatility Surfaces.*
  Quantitative Finance, 2(1), 45-60.
- Cousin, A., Maatouk, H., & Rulliere, D. (2016). *Kriging of Financial Term-Structures.*
  European Journal of Operational Research, 255(2), 631-648.

---

## Step 4 — SVI Model as Parametric Baseline (Calibrated with BO)

### What we do
We calibrate the SVI (Stochastic Volatility Inspired) model on the daily
surface using Bayesian Optimization instead of grid search. SVI produces a
parametric, arbitrage-free surface that serves as the comparison term for
the GP.

### Mathematical ideas
- **SVI functional form:**
  sigma^2_SVI(k) = a + b · [rho·(k - m) + sqrt((k - m)^2 + xi^2)]
  where theta = (a, b, rho, m, xi) are the 5 parameters to calibrate
  (per maturity T).
  - a: total variance level
  - b: wing slope (ATM vol)
  - rho in (-1,1): skew correlation (asymmetry)
  - m: curve center (ATM offset)
  - xi: wing smoothness (curvature)
- **Bayesian Optimization:**
  Builds a GP surrogate model on the loss function L(theta) = sum((IV_market - IV_SVI)^2)
  and uses an acquisition function to choose where to evaluate L(theta) next:
  - Expected Improvement: EI(theta) = E[max(L(theta*) - L(theta), 0)]
  - Upper Confidence Bound: UCB(theta) = mu(theta) + beta·sigma(theta)
  BO finds the minimum in ~50 evaluations vs ~10^5 for grid search.
- **No-arbitrage conditions for SVI:**
  b >= 0, |rho| < 1, xi > 0, a + b·xi·sqrt(1-rho^2) >= 0

### Economic ideas
- SVI represents the market maker's "parametric prior": it is typically
  the model used to interpolate the surface in illiquid regions.
- Using BO instead of grid search is not just computational efficiency: it
  reduces the risk of overfitting the parameters to the same data we will
  use for the signal.
- The GP vs SVI comparison is at the heart of the signal: where the
  flexible model (GP) and the rigid one (SVI) agree on a mispricing, the
  signal is far more credible.

### Libraries
`scikit-optimize (skopt)`, `optuna`, `botorch`

### References
- Gatheral, J., & Jacquier, A. (2014). *Arbitrage-Free SVI Volatility Surfaces.*
  Quantitative Finance, 14(1), 59-71.
- Frazier, P. I. (2018). *A Tutorial on Bayesian Optimization.* arXiv:1807.02811.
- Garouani, M., & Bouneffa, M. (2024). *Automated Machine Learning Hyperparameters
  Tuning through Meta-Guided Bayesian Optimization.* Progress in Artificial
  Intelligence.

---

## Step 5 — Calendar Filter and Event Handling

### What we do
Before computing any signal, we flag options whose expiry falls within 3
days of a known event (earnings, FOMC, CPI release). The calendar
multiplier cal(T) = 0 for these options, regardless of the GP signal.

### Mathematical ideas
- **Formal definition:** cal(T) = 1[ min_{e in Events} |T - e| > delta ]
  where delta = 3 days and Events includes FOMC dates, earnings dates,
  macro releases.
- **Event vol decomposition:**
  sigma^2_total = sigma^2_daily · (T - t) + sigma^2_event · 1[event in [t, T]]
  The presence of an event systematically explains part of total IV.

### Economic ideas
- An option spanning an earnings date is not mispriced if its IV is high —
  it is correctly pricing event risk. It is not inconsistent with its
  neighbors; it belongs to a different category.
- This is the only filter that remains binary: no gradation is possible.
  An option covering an event is fundamentally different from the others.
- The logic separates structural vol (what we want to trade) from event
  vol (correctly priced, not tradable).

### References
- Garleanu, N., Pedersen, L. H., & Poteshman, A. M. (2009). *Demand-Based Option Pricing.*
  Review of Financial Studies, 22(10), 4259-4299.

---

## Step 6 — Composite Scoring Function

### What we do
We convert the four filters into continuous multipliers and combine them
into a single score for each option. A trade fires only if the score
exceeds a threshold tau, optimized via BO (Step 7).

### Mathematical ideas
- **Z-score of the deviation:**
  z(k,T) = [IV_market(k,T) - mu_GP(k,T)] / sigma_GP(k,T)
  Measures the deviation in units of model uncertainty.

- **Confidence (from GP uncertainty):**
  conf(k,T) = 1 / (1 + sigma_GP(k,T))   in (0, 1]

- **Consistency (GP and SVI agreement):**
  cons(k,T) = 1[ sign(IV_market - mu_GP) = sign(IV_market - mu_SVI) ]

- **Net deviation (bid-ask adjusted):**
  Delta_net(k,T) = max( |IV_market - mu_GP| - spread/2, 0 )

- **Composite score:**
  score(k,T) = z · cal · conf · cons · Delta_net

  Properties:
  - score = 0 if any multiplier is zero
  - score is continuous -> precise ranking of opportunities
  - position size proportional to score (natural position sizing)

### Economic ideas
Each multiplier has a precise economic interpretation. z measures how large
the mispricing is in statistical terms. cal confirms the mispricing is not
event vol (and therefore tradable). conf reflects whether the local market
is liquid enough for the GP estimate to be trusted. cons checks whether two
independent models agree that the market — not the models — is wrong.
Delta_net confirms the mispricing exceeds transaction costs, i.e. the trade
is actually profitable. The composite score is not merely a statistical
device: it formalizes the economic reasoning an experienced trader would
apply when looking at the surface.

---

## Step 7 — Walk-Forward Backtest with BO for the Threshold

### What we do
We use BO to find the optimal threshold tau on rolling training windows,
always validating on unseen data. Each window produces a tau*, applied to
the following out-of-sample period.

### Mathematical ideas
- **Walk-forward scheme:**
  For each t = t0, t0+Delta, t0+2*Delta, ...
  - Train: data from [t - W, t]  ->  BO finds tau*(t)
  - Test:  data from [t, t + Delta]  ->  apply tau*(t), measure OOS Sharpe
  Typically W = 252 days, Delta = 63 days (quarterly).

- **BO for optimizing tau:**
  max_tau  Sharpe_OOS(tau)  over  tau in [tau_min, tau_max]
  using a GP surrogate + UCB acquisition function.
  ~30-50 evaluations are sufficient (vs 500+ for grid search).

- **Deflated Sharpe Ratio (DSR):**
  corrects the Sharpe ratio for the number of configurations tested:
  DSR = SR · [1 - gamma(skewness, kurtosis)] / sqrt(V_trials)
  Prevents reporting inflated Sharpe ratios due to multiple testing.

- **Performance metrics:**
  - Annualized Sharpe: SR = E[R_daily] / sigma[R_daily] · sqrt(252)
  - Max drawdown: MDD = max_{t<=s} [V(t) - V(s)] / V(t)
  - Calmar ratio: SR / MDD
  - Hit rate: % of trades with P&L > 0
  - Signal half-life: estimated from the AR(1) autocorrelation of the signal

### Economic ideas
- Walk-forward replicates the only realistic situation: a manager who, in
  the past, could only have used data available up to that point in time.
- Without walk-forward, any backtest is inconsistent with reality because
  it uses future information (look-ahead bias).
- The DSR protects against data snooping: if 50 configurations are tested,
  the probability of finding a good one purely by chance is high. The DSR
  corrects for this inflation.

### References
- Bailey, D. H., & Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting, and Non-Normality.*
  Journal of Portfolio Management, 40(5), 94-107.
- White, H. (2000). *A Reality Check for Data Snooping.*
  Econometrica, 68(5), 1097-1126.

---

## Step 8 — Delta-Hedging and Isolating the Vol Alpha

### What we do
Every position is delta-hedged daily: we buy or sell the amount of
underlying needed to make the portfolio delta-neutral. This removes
directional exposure and isolates the P&L to the volatility component
alone.

### Mathematical ideas
- **BS Delta:**
  Delta_call = N(d1),   Delta_put = N(d1) - 1
  where d1 = [ln(S/K) + (r + sigma^2/2)T] / (sigma·sqrt(T))

- **P&L decomposition (Carr-Madan):**
  dPi = (Gamma/2)(dS)^2 · (sigma^2_realized - sigma^2_implied) · dt + residual
  The P&L of a delta-hedged position is proportional to the difference
  between realized volatility and the implied volatility paid at entry.

- **Daily Gamma P&L:**
  PnL_t ~= 0.5 · Gamma · S^2 · (r_t^2 - sigma^2_IV) · dt
  where r_t is the underlying's daily return.

- **Vega exposure:**
  Vega = dV/dsigma = S · N'(d1) · sqrt(T)
  Remaining after the delta hedge: exposure to moves in implied vol.

### Economic ideas
- Without delta-hedging, the strategy's P&L includes a large directional
  (equity beta) component that is not the alpha we are looking for.
- Delta-hedging turns the position into a "pure bet" on volatility: we
  profit if sigma_realized > sigma_implied (we bought cheap vol), or if
  sigma_implied moves toward the GP estimate (the mispricing corrects).
- Economically, the post-hedge P&L measures exclusively whether we were
  right about the GP estimate — it is the direct test of our alpha thesis.

### References
- Bakshi, G., & Kapadia, N. (2003). *Delta-Hedged Gains and the Negative Market
  Volatility Risk Premium.* Review of Financial Studies, 16(2), 527-566.
- Han, C.-H., & Wang, K. (2026). *Variance Risk Premia under Volatility Models.*
  Review of Quantitative Finance and Accounting.

---

## Step 9 — Signal Analysis and Diagnostics

### What we do
We verify that the signal has characteristics consistent with real alpha
rather than overfitting or noise: decay over time consistent with
arbitrage, cross-sectional consistency, and stability over time.

### Mathematical ideas
- **Signal decay (half-life):** estimate the speed of mispricing
  correction with an AR(1) model on the deviation:
  Delta_t = alpha · Delta_{t-1} + epsilon    ->    half_life = -log(2) / log(alpha)
  A half-life of 2-5 days is consistent with market-maker arbitrage.

- **Cross-sectional consistency:**
  correlation of the signal across different underlyings on the same day.
  A cross-sectionally consistent signal is far less likely to be pure noise.

- **Information coefficient (IC):**
  IC_t = corr(score_t, return_vol_{t+1})
  Measures the predictive power of the signal. IC > 0.05 is considered
  good in vol trading.

### Economic ideas
- A half-life that is too short (< 1 day) would suggest the signal is
  microstructure noise, not tradable with EOD data.
- A half-life that is too long (> 20 days) would suggest it is not a
  mispricing but a persistent economic structure that does not correct
  easily.
- A 2-10 day range is consistent with the speed of market-maker
  recalibration and with the persistence of institutional demand pressure.

### References
- Cont, R., & da Fonseca, J. (2002). *Dynamics of Implied Volatility Surfaces.*
  Quantitative Finance, 2(1), 45-60.

---

## Full Tech Stack

| Component           | Primary library              | Alternative            |
|---------------------|-------------------------------|-------------------------|
| Options data        | `yfinance`                    | `polygon.io` API        |
| Implied volatility  | `py_vollib`                   | `scipy` + custom BS      |
| Gaussian Process    | `sklearn.gaussian_process`    | `GPy`, `botorch`          |
| Bayesian opt.       | `scikit-optimize`              | `optuna`                  |
| Backtest            | `vectorbt`                     | custom `pandas`           |
| Visualization       | `plotly` (3D surface)          | `matplotlib`              |
| Delta hedge         | custom + `py_vollib`           |                          |

---

## Full Bibliography

### Pricing and the Volatility Surface
- **Black, F., & Scholes, M.** (1973). The Pricing of Options and Corporate Liabilities.
  *Journal of Political Economy*, 81(3), 637-654.
- **Breeden, D. T., & Litzenberger, R. H.** (1978). Prices of State-Contingent Claims
  Implicit in Option Prices. *Journal of Business*, 51(4), 621-651.
- **Fengler, M. R.** (2009). Arbitrage-Free Smoothing of the Implied Volatility Surface.
  *Quantitative Finance*, 9(4), 417-428.
- **Gatheral, J., & Jacquier, A.** (2014). Arbitrage-Free SVI Volatility Surfaces.
  *Quantitative Finance*, 14(1), 59-71.
- **Cont, R., & da Fonseca, J.** (2002). Dynamics of Implied Volatility Surfaces.
  *Quantitative Finance*, 2(1), 45-60.

### Variance Risk Premium and Demand Effects
- **Han, C.-H., & Wang, K.** (2026). Variance Risk Premia under Volatility Models.
  *Review of Quantitative Finance and Accounting*.
- **Bakshi, G., & Kapadia, N.** (2003). Delta-Hedged Gains and the Negative Market
  Volatility Risk Premium. *Review of Financial Studies*, 16(2), 527-566.
- **Garleanu, N., Pedersen, L. H., & Poteshman, A. M.** (2009). Demand-Based Option
  Pricing. *Review of Financial Studies*, 22(10), 4259-4299.

### Gaussian Processes and Non-Parametric Methods
- **Cousin, A., Maatouk, H., & Rulliere, D.** (2016). Kriging of Financial
  Term-Structures. *European Journal of Operational Research*, 255(2), 631-648.

### Bayesian Optimization
- **Frazier, P. I.** (2018). A Tutorial on Bayesian Optimization. *arXiv:1807.02811*.
- **Garouani, M., & Bouneffa, M.** (2024). Automated Machine Learning Hyperparameters
  Tuning through Meta-Guided Bayesian Optimization. *Progress in Artificial
  Intelligence*.

### Backtesting and Data Snooping
- **Bailey, D. H., & Lopez de Prado, M.** (2014). The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting, and Non-Normality.
  *Journal of Portfolio Management*, 40(5), 94-107.
- **White, H.** (2000). A Reality Check for Data Snooping.
  *Econometrica*, 68(5), 1097-1126.

---

*Internal reference document — GP Vol Surface Alpha Research Project*
