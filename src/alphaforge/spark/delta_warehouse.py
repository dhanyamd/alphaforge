"""Delta Lake warehouse — the production twin of the local DuckDB warehouse.

Same role as `storage/warehouse.py`, but backed by **Delta Lake** on Spark. Delta
adds, on top of plain Parquet:
  * **ACID transactions** — a write fully happens or not at all; readers never see
    a half-written table; concurrent jobs don't corrupt each other.
  * **schema enforcement** — bad-shaped data is rejected, not silently appended.
  * **time travel** — query the table AS OF any past version/timestamp, which is
    the point-in-time guarantee you need for reproducible historical runs.
  * **partition overwrite** — cleanly replace just the dates a vendor restated.

The method names mirror `Warehouse` so the same pipeline code can target either
backend. On Databricks you'd point `root` at a DBFS/S3 path like
`dbfs:/alphaforge` or `s3://bucket/alphaforge`.
"""
from __future__ import annotations


class DeltaWarehouse:
    def __init__(self, spark, root: str = "dbfs:/alphaforge") -> None:
        self.spark = spark
        self.root = root.rstrip("/")

    def _path(self, table: str) -> str:
        return f"{self.root}/{table}"

    #  writes 
    def write_table(self, table: str, sdf, partition_col: str = "date", mode: str = "overwrite") -> None:
        """Write a Spark DataFrame as a Delta table, partitioned by date.

        `replaceWhere`/dynamic partition overwrite is how you absorb a vendor
        restatement without rebuilding the whole table.
        """
        (
            sdf.write.format("delta")
            .mode(mode)
            .partitionBy(partition_col)
            .option("overwriteSchema", "true")
            .save(self._path(table))
        )

    def write_unpartitioned(self, table: str, sdf, mode: str = "overwrite") -> None:
        sdf.write.format("delta").mode(mode).option("overwriteSchema", "true").save(self._path(table))

    #  reads 
    def read_table(self, table: str, version: int | None = None):
        """Read a Delta table; `version` enables TIME TRAVEL (point-in-time reads)."""
        reader = self.spark.read.format("delta")
        if version is not None:
            reader = reader.option("versionAsOf", version)   # query the table as it was
        return reader.load(self._path(table))

    def table_exists(self, table: str) -> bool:
        try:
            self.spark.read.format("delta").load(self._path(table)).limit(1).count()
            return True
        except Exception:
            return False
