# Data

`raw/` e `processed/` sono gitignored — non committare dati di mercato.

Per popolare:
```bash
python -m gpvol.data.loader --ticker SPY --start 2022-01-01 --end 2024-12-31
```

Output:
- `raw/<ticker>_<date>.parquet` — option chain grezza
- `processed/<ticker>_iv_surface.parquet` — IV surface pulita (post no-arb filter)
