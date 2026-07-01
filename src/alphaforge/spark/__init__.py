"""(VI/VII) The Spark + Delta Lake path — the SAME pipeline, at scale.

Everything in the local pipeline (DuckDB + pandas) is for *iteration speed* on
small data. When the data outgrows one machine (terabytes of alt-data), you run
the identical logic on **Apache Spark** writing to **Delta Lake** — on
**Databricks**. This package is the Spark twin of the local data layer:

  * `session.py`        — get/inherit a SparkSession (works on Databricks as-is)
  * `delta_warehouse.py`— a Delta Lake warehouse (partitioned, ACID, time-travel)
  * `pipeline_spark.py` — PySpark data loader + security-matching BROADCAST JOIN
                          + window-function signals (lazy, partition-pruned)

It runs on Databricks Community Edition (free) out of the box. Locally it needs a
JVM (`brew install openjdk@17`) + `pip install -e ".[spark]"` — but the whole
point is that you DON'T run Spark on small local data; you run it where the data
and the cluster live. See `databricks/README_DATABRICKS.md`.
"""
