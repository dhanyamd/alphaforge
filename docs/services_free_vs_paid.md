# Free vs paid services — how AlphaForge maps to the KAI/Kling stack

You can take this project **fully cloud + production-shaped for $0**, then swap in
paid services only where you actually need them. The architecture already isolates
every external dependency behind an interface, so each swap is a drop-in.

## The mapping

| Layer (JD wording) | Free tier we use / can use | Paid production equivalent | Swap point in code |
|---|---|---|---|
| Market data | synthetic generator; **yfinance** (free EOD); Alpha Vantage / Tiingo free tiers | Bloomberg, Refinitiv, FactSet, Polygon, Databento | `data/sources/*` (`DataSource.fetch`) |
| Fundamentals | yfinance / SEC EDGAR (free) | Compustat, FactSet Fundamentals | `data/sources/*` |
| Alt-data | synthetic alt-signal | RavenPack, credit-card panels, supply-chain feeds | `data/sources/*` |
| ID mapping | **OpenFIGI (free)** + our security master | CUSIP Global Services, ISIN (licensed) | `refdata/` |
| Warehouse | **DuckDB + Parquet** (local) → **MotherDuck** (free cloud DuckDB) | **Snowflake**, Databricks Delta | `storage/warehouse.py::Warehouse` |
| Object store | local disk → **Cloudflare R2** (10GB free) / **AWS S3** (5GB/12mo) | AWS S3 (at scale) | `Warehouse` path |
| Distributed compute | Polars/NumPy (local) → **Databricks Community Edition** (free Spark) | Databricks, EMR, Ray | `risk/`, `data/loaders.py` |
| Tick/as-of engine | **DuckDB ASOF** | **KDB+/q** (licensed) | `storage/asof.py` |
| Orchestration | **GitHub Actions** cron (free) | Airflow, Dagster, Databricks Workflows | `backtest/pipeline.py` |
| Compute box | local; AWS EC2 t2.micro (750h/12mo); Modal/Render free | EC2 fleets, HTCondor grid | — |
| Big SQL | local DuckDB → **BigQuery** (1TB/mo free forever) | Snowflake, BigQuery (at scale) | `Warehouse` |

## What actually costs money (be deliberate)
- **Real alt-data** is the expensive part of quant (RavenPack etc., $$$$). Free
  proxies: SEC EDGAR filings, Google Trends, Reddit/news scraping, GDELT.
- **Snowflake / KDB+** are paid; MotherDuck + DuckDB-ASOF cover the learning and
  even modest production needs for free.
- **Survivorship-bias-free historical data with point-in-time fundamentals** is
  the single most valuable paid thing in this space — CRSP/Compustat via WRDS
  (often free through a university). Worth knowing it exists.

## Recommended $0 "looks production-grade" path
1. Local DuckDB/Parquet for iteration (now).
2. Add a `MotherDuckWarehouse(Warehouse)` subclass → cloud warehouse, free.
3. Push the Parquet lake to Cloudflare R2 / S3 free tier; DuckDB reads it directly.
4. A **GitHub Actions** workflow on a cron runs `generate-data → audit → backtest`
   daily and commits the tearsheet — *real, recent, scheduled activity* on your
   GitHub, which is exactly the portfolio proof the JD asks for.
5. (Optional) run the heavy historical factor build once on Databricks Community
   Edition to show you can drive a real Spark cluster.
