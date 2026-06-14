# Notebooks

Each notebook is a thin orchestration layer: it imports from `gpvol`,
calls the functions, and produces figures. The logic lives in `src/gpvol/`.

| Notebook | Methodology steps | Content |
|---|---|---|
| 01_data_and_iv | 1-2 | fetch options chain, compute IV, no-arb cleaning |
| 02_surface_models | 3-4 | fit GP, calibrate SVI via BO, comparison |
| 03_signal_construction | 5-6 | calendar filter, composite score |
| 04_walkforward_backtest | 7 | walk-forward loop, BO on the threshold |
| 05_hedging_and_diagnostics | 8-9 | delta-hedge P&L, signal decay, DSR |
