# Running AlphaForge on Databricks (free Community Edition)

This is the production path: the **same** AlphaForge logic, running on real
**Apache Spark** and writing to **Delta Lake** — exactly the on-prem→AWS+Databricks
migration the transcript describes. You don't run Spark on small local data; you
run it where the data and the cluster live.

## Why Spark/Databricks (and why NOT locally)

Spark pays a big fixed cost — spin up a cluster, shuffle data across machines —
that only pays off on data too big for one machine (terabytes of alt-data). Our
200-name synthetic universe fits in memory thousands of times over, so locally
DuckDB/pandas is *faster*. Spark is for **scale**, not small data. Build & prove
locally → deploy to Databricks when the data outgrows one box.

## Option A — Databricks Community Edition (free, recommended)

1. Sign up: [https://www.databricks.com/try-databricks](https://www.databricks.com/try-databricks) → **Community Edition**
  (free forever; no credit card; a single small cluster).
2. **Create a cluster** (latest runtime — Delta + Spark are preinstalled).
3. **Import the notebook:** Workspace → Import → `databricks/alphaforge_pipeline_notebook.py`.
4. Install AlphaForge on the cluster: either
  - `%pip install alphaforge` (after you publish a wheel), or
  - build a wheel locally (`python -m build`) and upload it under
  Cluster → Libraries → Install New → Python Whl.
5. **Attach** the notebook to the cluster and **Run All**.
6. Open the **Spark UI** (cluster → Spark UI) and find the join stage — you'll see
  `BroadcastHashJoin` with **no Exchange (shuffle)**. That's the security-matching
   broadcast join from the architecture, working at scale.

## Option B — run Spark locally (only if you insist)

Spark needs a JVM. On macOS:

```bash
brew install openjdk@17
export JAVA_HOME="$(/usr/libexec/java_home -v 17)"
pip install -e ".[spark]"          # installs pyspark + delta-spark
```

Then drive `alphaforge.spark.pipeline_spark.run_spark_pipeline` from a script.
(Reminder: on our toy data this is *slower* than the local DuckDB path — it's here
to learn the API, not for speed.)

## What maps to what (local → Databricks)


| Local (this repo)                       | Databricks / production                                    |
| --------------------------------------- | ---------------------------------------------------------- |
| `storage/warehouse.py` (DuckDB+Parquet) | `spark/delta_warehouse.py` (Delta Lake)                    |
| pandas/Polars transforms                | `spark/pipeline_spark.py` (PySpark)                        |
| `data/loaders.py`                       | the notebook, scheduled as a **Workflow/Job**              |
| `backtest/pipeline.py` (DAG runner)     | Databricks **Workflows** (real DAG, cron)                  |
| local disk                              | DBFS / **S3**                                              |
| ASOF join (DuckDB)                      | broadcast join + window funcs (Spark); KDB+ for live ticks |


> "I built the factor pipeline local-first on DuckDB for fast iteration, then
> ported the compute layer to PySpark on Databricks with Delta Lake — point-in-time
> security matching as a broadcast join (no shuffle), window-function signals,
> partition-by-date writes, and time travel for reproducible historical runs."

