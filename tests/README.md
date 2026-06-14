# Tests

Before adding a test here, it's worth being explicit about what "correct"
means in this project -- because a research codebase has two very different
notions of correctness, and conflating them is a common source of false
confidence.

## Two scales, two questions

**Numerical correctness** (this directory): does the code compute what the
formulas say it should, to floating-point precision? A bug here -- a sign
error in `d1`, a wrong discounting convention -- would produce a
*systematic* bias on the order of 1e-2 to 1e-1 in implied vol. That is the
same order of magnitude as the trading signal itself (`docs/methodology.md`,
Step 6: net deviations of roughly 0.005-0.02 after bid-ask costs). If this
layer is wrong, every "mispricing" the GP finds downstream is partly or
wholly a pipeline artifact, not a market one.

**Economic / statistical plausibility** (mostly in notebooks, Step 9): does
the *signal* behave like alpha? Half-life in a 2-10 day range, cross-
sectional consistency, a GP uncertainty that's well-calibrated against
realized dispersion -- these are not pass/fail software properties. They are
research findings that depend on the data and the period, and they belong in
diagnostic notebooks, not in `pytest` assertions. A hard `assert sharpe >
1.0` in this directory would be a category error: it tests a result, not a
property of the code.

This directory exists to make the first scale a non-issue, so that when the
Step 9 diagnostics show something interesting (or don't), we know it
reflects the market and the model -- not a bug three layers down.

## Three kinds of check

| Kind | Typical tolerance | Answers | Example |
|---|---|---|---|
| Algebraic identity | ~1e-8 to 1e-10 (machine precision) | "Is there a bug in the algebra?" | put-call parity; `delta_call - delta_put = 1` |
| External anchor | ~1e-3 to 1e-6 | "Does this match a known, independent reference?" | textbook BS value; cross-validation vs `vollib` |
| Internal roundtrip | ~1e-4 (solver tolerance) | "Are the forward and inverse functions consistent with *each other*?" | `price -> implied_vol -> price` |

A passing roundtrip means the two functions agree with each other. A passing
external anchor means they agree with the world. Both are needed: a shared
bug in a forward/inverse pair can make a roundtrip pass while both functions
are simultaneously wrong. See the module docstring in
`test_cross_validation.py` for the concrete case.

## Map of test modules

This table is a living index. When a new module is added, it gets a row
describing *what question it answers* -- not a restatement of its docstring.

| File | Validates | A failure means |
|---|---|---|
| `test_black_scholes.py` | BS pricing, greeks, IV inversion: algebraic identities + internal roundtrip | A bug in the core pricing/inversion formulas |
| `test_cross_validation.py` | Same functions against `vollib` (Jaeckel's reference algorithm), on a realistic moneyness/maturity/rate/vol grid | Formulas are internally consistent but disagree with standard convention (e.g. wrong forward, wrong discounting) |
| `test_gp_model.py` *(planned)* | GP posterior mean/variance recovers a known synthetic surface | GP fit, kernel choice, or coordinate system is wrong |
| `test_svi_model.py` *(planned)* | BO-calibrated SVI parameters satisfy no-arbitrage conditions (`b>=0, |rho|<1, ...`) | Calibration converges to an arbitrageable parametrization |
| `test_scoring.py` *(planned)* | Composite score is zero iff any filter multiplier is zero; monotonicity in each input | Scoring logic diverges from `docs/methodology.md`, Step 6 |
| `test_backtest.py` *(planned)* | Walk-forward train/test index separation (no look-ahead); Sharpe/DSR/max-drawdown formulas on synthetic series with known closed-form answers | Look-ahead bias, or a metric formula error |

## Running

```bash
pytest -v          # full suite
pytest -v -k bs     # only Black-Scholes-related tests
```
