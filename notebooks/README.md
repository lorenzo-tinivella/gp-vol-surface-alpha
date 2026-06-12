# Notebooks

Ogni notebook è un thin orchestration layer: importa da `gpvol`,
chiama le funzioni, produce figure. La logica vive in `src/gpvol/`.

| Notebook | Punti metodologia | Contenuto |
|---|---|---|
| 01_data_and_iv | 1-2 | fetch options chain, calcolo IV, cleaning no-arb |
| 02_surface_models | 3-4 | fit GP, calibrazione SVI via BO, confronto |
| 03_signal_construction | 5-6 | calendar filter, composite score |
| 04_walkforward_backtest | 7 | walk-forward loop, BO sulla soglia |
| 05_hedging_and_diagnostics | 8-9 | delta-hedge P&L, signal decay, DSR |
