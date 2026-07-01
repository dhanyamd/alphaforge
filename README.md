# AlphaForge — an end-to-end multi-factor equity trading machine

> A from-scratch, runnable implementation of *the* standard production quant
> architecture — **"The Multifactor Machine"** — with every block of the real
> pipeline built and wired into a single backtest. Built to learn how a quant
> research engineer's work actually fits together, 0 → 100.

It runs **offline today** on a synthetic market that has *real factor structure*
(so the signals genuinely predict returns), and every data/storage boundary is
an interface you can later point at **Snowflake / S3 / MotherDuck / Databricks**
(see [`docs/services_free_vs_paid.md`](docs/services_free_vs_paid.md)).

## The architecture (what you're building)

```
DATA+SIGNALS → ALPHA FORECASTS → RISK MODEL → PORTFOLIO CONSTRUCTION → IMPL/TRADING → PERFORMANCE
     │              │                                                                      │
 (IV) Security  (V) Research→Prod                                                          │
   Matching        (the pod)                                                               │
     └───────────────────────── feedback / decay loop (factors decay) ─────────────────────┘
                  (VI) SPARK / compute  ·  (VII) STORAGE: Parquet/Delta + KDB(asof)
```

Each architecture block maps to one Python subpackage:

| Block | Package | What it does |
|---|---|---|
| (I) Data + Signals | `data/`, `signals/` | vendor loaders, **data audit**, universe, raw signals |
| (IV) Security matching | `refdata/` | **point-in-time** identifier crosswalk (no look-ahead) |
| (II) Alpha forecasts | `alpha/` | Grinold refine `α = vol·IC·score`, multi-factor blend |
| (III) Risk model | `risk/` | Barra-style cross-sectional factor model `V = BΣ_fB'+D` |
| (IV) Portfolio construction | `portfolio/` | CVXPY mean-variance optimizer + constraints |
| (V) Implementation/trading | `execution/` | square-root **market-impact** cost model, fills, shortfall |
| (VI) Performance | `performance/` | IR, fundamental law, attribution, **factor decay** |
| (VII) Storage | `storage/` | Parquet/DuckDB warehouse + **ASOF join** (KDB analog) |
| Engine | `backtest/` | the daily loop that wires it all together |

Read [`CONCEPTS.md`](CONCEPTS.md) for the full block-by-block teaching walkthrough,
and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the system design.

## Quickstart

```bash
# 1. install (uses uv; falls back to pip)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev,viz]"      # or: pip install -e ".[dev,viz]"

# 2. build the local data lake (synthetic vendors -> Parquet/DuckDB)
alphaforge generate-data

# 3. audit data quality (coverage %, outliers, nulls, staleness)
alphaforge audit

# 4. run the full end-to-end multifactor backtest
alphaforge backtest

# 5. reprint the latest tearsheet
alphaforge report
```

Or with the Makefile: `make install && make gen && make audit && make backtest`.

### Switch to real equities
```bash
uv pip install -e ".[realdata]"
# set data.source: "yfinance" in config/backtest.yaml, then:
alphaforge generate-data && alphaforge backtest
```

## What "good" looks like
The backtest prints a tearsheet: annualized return/vol, Sharpe, **information
ratio**, max drawdown, the **fundamental-law** IC·√breadth check, the
**implementation shortfall** (what trading cost you), and a **factor-IC decay**
table (the signal that feeds the research loop). Because the synthetic market has
embedded factor premia, a correctly wired machine produces a *positive but
modest* IR — exactly like reality.

## Status / roadmap
- [x] Full classic multifactor machine, end-to-end, runnable offline
- [ ] Cloud adapters: `MotherDuckWarehouse`, `S3` Parquet, Databricks Spark job
- [ ] GitHub Actions cron: daily loader + factor refresh on free cloud storage
- [ ] Autonomous AI researcher (LLM proposes/evaluates factors) — the "KAI" layer
