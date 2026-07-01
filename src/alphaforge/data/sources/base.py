"""The `DataSource` interface + the canonical `MarketData` bundle.

Every source (synthetic, yfinance, Snowflake, ...) returns the *same* set of
tidy long-format tables keyed by (date, security_id). Downstream code only ever
sees this contract, never the vendor's quirks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class MarketData:
    """The standardized output of any data source.

    All frames are long format with a `date` (datetime64) and `security_id`
    (str) key, so they Hive-partition cleanly and join with as-of semantics.
    """

    prices: pd.DataFrame        # date, security_id, close, volume, ret, shares_out
    fundamentals: pd.DataFrame  # date, security_id, book_to_price, gross_profit, total_assets
    altdata: pd.DataFrame       # date, security_id, alt_score
    static: pd.DataFrame        # security_id, sector   (slowly/never changing)

    def security_ids(self) -> list[str]:
        return sorted(self.static["security_id"].unique().tolist())


class DataSource(ABC):
    """Abstract base for all market-data sources."""

    @abstractmethod
    def fetch(self, start: str, end: str) -> MarketData:
        """Return a `MarketData` bundle covering [start, end]."""
        raise NotImplementedError
