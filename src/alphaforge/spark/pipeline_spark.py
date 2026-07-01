"""PySpark data pipeline — the distributed twin of `data/loaders.py` + signals.

This is the code that, in the transcript, the quant DEVELOPER writes: take the
researcher's logic and make it run distributed over terabytes. It demonstrates
the exact Spark concepts from the architecture whiteboard:

  * **lazy evaluation** — transformations build a plan (a DAG); nothing runs until
    an *action* (write/count). Spark's Catalyst optimizer then rewrites the whole
    plan (predicate pushdown, join reordering) before a byte is read.
  * **BROADCAST JOIN** for security matching — the security-master table is small,
    so we `broadcast()` it to every executor and the join happens locally with
    **no shuffle**. This is literally the "(IV) join — broadcast → no shuffle"
    arrow on the diagram, and the #1 Spark optimization you'll be asked about.
  * **window functions** — momentum/vol are per-security time-series features;
    `Window.partitionBy(security).orderBy(date)` computes them in parallel across
    partitions.
  * **partition-by-date writes** — cheap daily appends + fast historical scans.

Every function takes/returns a Spark DataFrame and stays lazy until the final
Delta write. Run it on Databricks; locally it needs a JVM + `.[spark]`.
"""
from __future__ import annotations


def transform_prices(spark, prices_path: str):
    """Read raw vendor prices (Parquet on S3/DBFS) and derive trading fields.

    Stays lazy: this just registers a plan. Column/date filters pushed down to the
    Parquet reader so we only scan what's needed.
    """
    from pyspark.sql import functions as F

    df = spark.read.parquet(prices_path)
    return (
        df.withColumn("dollar_volume", F.col("close") * F.col("volume"))
          .withColumn("mktcap", F.col("close") * F.col("shares_out"))
    )


def match_securities(prices_sdf, security_master_sdf):
    """Point-in-time security matching via a BROADCAST JOIN (no shuffle).

    The security master is tiny relative to the price history, so we broadcast it
    to every executor. We join on security_id and then keep only the row whose
    [valid_from, valid_to) window contains the price date — the point-in-time
    firewall, done in a distributed, shuffle-free way.
    """
    from pyspark.sql import functions as F

    sm = F.broadcast(security_master_sdf)   # <-- the key line: ship the small table everywhere
    joined = prices_sdf.join(sm, on="security_id", how="left")
    valid = joined.where(
        (F.col("date") >= F.col("valid_from")) & (F.col("date") < F.col("valid_to"))
    )
    return valid.drop("valid_from", "valid_to")


def add_momentum(prices_sdf, skip: int = 21, lookback: int = 252):
    """12-1 momentum via a window function, computed in parallel per security.

    Window partitioned by security, ordered by date. `lag` reaches back the
    required number of rows — all knowable at t (no look-ahead).
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy("security_id").orderBy("date")
    return (
        prices_sdf
        .withColumn("c_skip", F.lag("close", skip).over(w))
        .withColumn("c_look", F.lag("close", lookback).over(w))
        .withColumn("momentum", F.col("c_skip") / F.col("c_look") - F.lit(1.0))
        .drop("c_skip", "c_look")
    )


def add_rolling_vol(prices_sdf, window_days: int = 63):
    """Trailing realized volatility via a rolling window (rowsBetween)."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = (
        Window.partitionBy("security_id").orderBy("date")
        .rowsBetween(-(window_days - 1), 0)
    )
    return prices_sdf.withColumn("vol_63", F.stddev("ret").over(w)).withColumn(
        "low_vol", -F.col("vol_63")
    )


def run_spark_pipeline(spark, lake_root: str, out: "object") -> None:
    """End-to-end Spark ingest: prices -> match -> signals -> Delta (partitioned).

    `out` is a DeltaWarehouse. This is the job you'd schedule as a Databricks
    Workflow that fires after the New York close, in time for the next market open
    — the "race the clock" trading window from the transcript.
    """
    from alphaforge.logging import get_logger

    log = get_logger(__name__)

    prices = transform_prices(spark, f"{lake_root}/prices")
    sm = spark.read.parquet(f"{lake_root}/security_master")

    matched = match_securities(prices, sm)
    enriched = add_rolling_vol(add_momentum(matched))

    # The ACTION — only now does Spark actually execute the whole DAG.
    out.write_table("prices_enriched", enriched, partition_col="date")
    log.info("spark.pipeline.done", lake_root=lake_root)
