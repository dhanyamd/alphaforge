"""Get a SparkSession — transparently on Databricks or locally.

On Databricks a `spark` session already exists (the platform injects it), so we
just grab it. Locally we build one configured for Delta Lake. This is the one
place that knows *where* Spark is running; everything else is portable.
"""
from __future__ import annotations


def get_spark(app_name: str = "alphaforge"):
    """Return an active SparkSession, configured for Delta Lake.

    Order of preference:
      1. an existing session (this is what happens on Databricks),
      2. a locally-built session with the Delta extension wired in.
    """
    from pyspark.sql import SparkSession

    active = SparkSession.getActiveSession()
    if active is not None:
        return active

    builder = (
        SparkSession.builder.appName(app_name)
        # Delta Lake wiring — on Databricks this is preconfigured.
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Sensible local defaults; on a cluster these come from the cluster config.
        .config("spark.sql.shuffle.partitions", "8")
    )
    try:
        # configure_spark_with_delta_pip auto-pulls the matching delta-spark jars.
        from delta import configure_spark_with_delta_pip

        return configure_spark_with_delta_pip(builder).getOrCreate()
    except ImportError:
        return builder.getOrCreate()
