# Databricks notebook source
# MAGIC %md
# MAGIC # AlphaForge — Spark + Delta pipeline (Databricks)
# MAGIC
# MAGIC This notebook runs the AlphaForge data layer on **real Spark**, writing to
# MAGIC **Delta Lake** — the production version of the local DuckDB pipeline. Import
# MAGIC it into Databricks (Community Edition is free), attach a cluster, Run All.
# MAGIC
# MAGIC It demonstrates the architecture's Spark concepts: lazy evaluation, a
# MAGIC **broadcast join** for point-in-time security matching, window-function
# MAGIC signals, and partition-by-date Delta writes with time travel.

# COMMAND ----------
# MAGIC %md ## 1. Generate sample data into DBFS
# MAGIC On a real desk this comes from vendor feeds on S3. Here we generate the same
# MAGIC synthetic lake and land it as Parquet in DBFS so Spark can read it.

# COMMAND ----------
# If alphaforge isn't installed on the cluster: %pip install alphaforge  (or upload the wheel)
from alphaforge.config import load_config
from alphaforge.data.loaders import ingest
from alphaforge.storage.warehouse import Warehouse

LAKE = "/dbfs/alphaforge_raw"          # local-file view of DBFS
DBFS_LAKE = "dbfs:/alphaforge_raw"     # spark view of the same path

cfg = load_config("config/backtest.yaml")
cfg.storage.lake_path = LAKE
ingest(cfg, Warehouse(LAKE))           # writes prices/fundamentals/altdata/security_master as Parquet

# COMMAND ----------
# MAGIC %md ## 2. Grab the Spark session + a Delta warehouse

# COMMAND ----------
from alphaforge.spark.session import get_spark
from alphaforge.spark.delta_warehouse import DeltaWarehouse

spark = get_spark()
delta = DeltaWarehouse(spark, root="dbfs:/alphaforge_delta")

# COMMAND ----------
# MAGIC %md ## 3. Run the distributed pipeline
# MAGIC prices → **broadcast-join** security matching → window-function signals →
# MAGIC Delta (partitioned by date). Watch the Spark UI: the join shows as
# MAGIC `BroadcastHashJoin` with **no shuffle**.

# COMMAND ----------
from alphaforge.spark.pipeline_spark import run_spark_pipeline

run_spark_pipeline(spark, DBFS_LAKE, delta)

# COMMAND ----------
# MAGIC %md ## 4. Read it back + prove TIME TRAVEL (point-in-time)

# COMMAND ----------
enriched = delta.read_table("prices_enriched")
print("rows:", enriched.count())
enriched.select("date", "security_id", "close", "momentum", "low_vol").show(5)

# Delta time travel: query the table exactly as it was at version 0.
v0 = delta.read_table("prices_enriched", version=0)
print("version 0 rows:", v0.count())

# COMMAND ----------
# MAGIC %md ## 5. (Schedule it)
# MAGIC In Workflows, wrap this notebook as a **Job** on a cron (e.g. after the NY
# MAGIC close) so the factor re-runs daily on fresh data — the "race the trading
# MAGIC window" loop from the transcript.
