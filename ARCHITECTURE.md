# AlphaForge — System Architecture

This document is the system-design view: the blocks, the data contracts between
them, the point-in-time discipline that ties them together, and where the cloud
seams are. (For the *why* / the quant theory, read `CONCEPTS.md`.)

## The one-line mental model

> We get paid to **outperform a benchmark**. The value we add is **alpha**.
> The whole machine is a **breadth machine**: make thousands of small,
> independent, slightly-better-than-even bets, every rebalance, across the whole
> universe — and let `IR ≈ IC·√breadth` compound modest skill into a real edge.

## Block diagram

```
                ┌─────────────── feedback / decay loop (factors decay) ───────────────┐
                │                                                                      │
   ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────────┐
   │  DATA + │→ │  ALPHA   │→ │   RISK   │→ │  PORTFOLIO   │→ │   IMPL /  │→ │ PERFORMANCE  │
   │ SIGNALS │  │ FORECAST │  │  MODEL   │  │ CONSTRUCTION │  │  TRADING  │  │  ANALYSIS    │
   └────┬────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └─────┬─────┘  └──────────────┘
   data/ │     alpha/│       risk/ │     portfolio/│      execution/│        performance/
        signals/                                                              │
   ┌────┴────┐  ┌────┴───────┐                                          (feeds research)
   │(IV) SEC │  │(V) RESEARCH│
   │ MATCHING│  │ → PROD pod │
   │ refdata/│  └────────────┘
   └─────────┘
   ───────────────────────────────────────────────────────────────────────────────────
   (VI) COMPUTE: Polars/NumPy vectorized  (Spark/Databricks  analog)        backtest/
   (VII) STORAGE: Parquet + DuckDB warehouse  ·  DuckDB ASOF join (KDB+ analog)   storage/
```

## Data contracts (what flows between blocks)


| Boundary              | Shape (long format unless noted)                                                                                                                                                                                              |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| source → lake         | `prices(date,security_id,close,volume,ret,shares_out,dollar_volume,mktcap)`, `fundamentals(...)`, `altdata(...)`, `security_static(security_id,sector)`, `security_master(security_id,ticker,cusip,isin,valid_from,valid_to)` |
| signals → alpha       | `panel(date,security_id, value,momentum,size,quality,low_vol,altdata)` (raw)                                                                                                                                                  |
| alpha → optimizer     | `panel(date,security_id, <clean signals>, fvol, alpha)`                                                                                                                                                                       |
| risk → optimizer      | snapshot `(B: N×K, F: K×K, d: N)` per rebalance date                                                                                                                                                                          |
| optimizer → execution | `w_target: N`                                                                                                                                                                                                                 |
| execution → engine    | `realized_weights: N`, `cost_return: float`                                                                                                                                                                                   |
| engine → performance  | daily real/paper/benchmark return series                                                                                                                                                                                      |


## Point-in-time discipline (the thing that makes a backtest honest)

Every read that crosses a time boundary is an **ASOF backward join**: it may only
see data published *at or before* the as-of date. This is enforced in three places:

1. **Security matching** (`refdata/`) — a vendor key maps to the internal id whose
  `[valid_from, valid_to)` window contains the date.
2. **Fundamentals / alt-data → signals** (`signals/library.py`) — values are
  attached as-of, respecting the reporting/publication lag.
3. **Risk model** (`risk/model.py`) — each rebalance's covariance uses only the
  trailing window of factor returns; exposures are taken as-of the date.

## Cloud seams (free-tier ready, zero upstream changes)

Two interfaces are the only things that know *where* data lives:

- `storage/warehouse.py::Warehouse` — subclass it (`MotherDuckWarehouse`,
`S3DeltaWarehouse`) and the path/engine changes; signals/risk/optimizer don't.
- `data/sources/base.py::DataSource` — implement `fetch()` against Snowflake/a
paid vendor; loaders/audit/universe don't change.

See `docs/services_free_vs_paid.md` for the free-tier mapping.

## Execution order (the pipeline DAG)

```
generate-data (ingest vendors → lake, build security master)
      ↓
audit (coverage %, nulls, outliers, staleness)   ← halts if a feed is broken
      ↓
build_universe (liquidity gate)        build_alphas (signals → clean → refine → blend)
      └───────────────┬───────────────────────────┘
                      ↓
              RiskModel.fit (cross-sectional factor returns, once)
                      ↓
   for each rebalance date:  snapshot → optimize → simulate fills → accrue P&L
                      ↓
   performance: IR, fundamental law, attribution, factor-IC decay → feeds research
```

`backtest/pipeline.py` is a minimal topological-sort runner that mirrors how this
DAG is wired as a Databricks/Airflow workflow in production.