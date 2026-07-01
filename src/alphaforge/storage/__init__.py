"""(VII) STORAGE — the layer underneath everything.

Two tools for two jobs (exactly as real firms run it):

  * `warehouse.py` — Parquet + DuckDB. The *warehouse*: columnar, cheap, scales
    out, remembers the full history, partitioned by date. This is our local
    stand-in for an S3 + Delta Lake / Snowflake setup. Swap the read/write
    methods for Snowflake or `deltalake` later and nothing upstream changes.

  * `asof.py` — the *engine* primitive: an as-of join ("give me the value that
    was in effect as of this timestamp"). This is the single operation KDB+ is
    famous for; DuckDB ships it as `ASOF JOIN`.
"""

from alphaforge.storage.warehouse import Warehouse

__all__ = ["Warehouse"]
