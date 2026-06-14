# Data

`raw/` and `processed/` are gitignored — do not commit market data.

To populate:
```bash
python -m gpvol.data.loader --ticker SPY --start 2022-01-01 --end 2024-12-31
```

Output:
- `raw/<ticker>_<date>.parquet` — raw option chain
- `processed/<ticker>_iv_surface.parquet` — cleaned IV surface (post no-arb filter)
