"""The security master — a point-in-time identifier crosswalk.

Real vendors disagree on how they name the same company: one uses CUSIP, one
ISIN, one SEDOL, one a Bloomberg/FIGI ticker, one a website URL. Worse, those
IDs *change over time* (mergers, ticker changes, re-domiciles). The security
master is the table that reconciles all of it and — crucially — stamps every
mapping with a ``[valid_from, valid_to)`` window.

Matching a vendor record then becomes an **as-of join**: given a vendor key and
a date, return the internal ``security_id`` that was valid on that date.

In a real firm you'd seed this from CUSIP Global Services / OpenFIGI. Here we
synthesize one with realistic ID-change events so you can see the mechanics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.storage.asof import asof_join
from alphaforge.storage.warehouse import Warehouse

# A sentinel "open interval" end date — the mapping is still valid today.
OPEN_END = pd.Timestamp("2099-12-31")


def _fake_cusip(rng: np.random.Generator) -> str:
    chars = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join(rng.choice(list(chars), size=9))


def build_security_master(
    security_ids: list[str],
    start: str,
    end: str,
    seed: int = 7,
    id_change_prob: float = 0.15,
) -> pd.DataFrame:
    """Build a point-in-time crosswalk: (security_id, cusip, isin, ticker, valid_from, valid_to).

    With probability ``id_change_prob`` a name experiences a mid-history ID
    change (think M&A), producing two rows with non-overlapping date windows for
    the same internal ``security_id``.
    """
    rng = np.random.default_rng(seed)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows: list[dict] = []
    for sid in security_ids:
        ticker = sid  # internal id doubles as a clean ticker in the synthetic world
        if rng.random() < id_change_prob:
            # Two regimes split at a random mid-history date (an M&A / reincorporation).
            split = start_ts + (end_ts - start_ts) * rng.uniform(0.3, 0.7)
            split = pd.Timestamp(split.date())
            rows.append(_row(sid, ticker + ".OLD", _fake_cusip(rng), start_ts, split))
            rows.append(_row(sid, ticker, _fake_cusip(rng), split, OPEN_END))
        else:
            rows.append(_row(sid, ticker, _fake_cusip(rng), start_ts, OPEN_END))
    return pd.DataFrame(rows)


def _row(sid: str, ticker: str, cusip: str, vf: pd.Timestamp, vt: pd.Timestamp) -> dict:
    # ISIN = country + national-number(CUSIP) + check; we fake a plausible US ISIN.
    return {
        "security_id": sid,
        "ticker": ticker,
        "cusip": cusip,
        "isin": f"US{cusip}0",
        "valid_from": vf,
        "valid_to": vt,
    }


class SecurityMaster:
    """Loads the crosswalk and resolves vendor keys -> internal IDs, point-in-time."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.df["valid_from"] = pd.to_datetime(self.df["valid_from"])
        self.df["valid_to"] = pd.to_datetime(self.df["valid_to"])

    @classmethod
    def from_warehouse(cls, wh: Warehouse, table: str = "security_master") -> "SecurityMaster":
        return cls(wh.read_table(table))

    def match(self, vendor_records: pd.DataFrame, key: str, date_col: str = "date") -> pd.DataFrame:
        """Resolve ``vendor_records[key]`` (e.g. cusip/isin/ticker) to ``security_id``.

        Performs a point-in-time match: a record dated D maps to the internal ID
        whose ``[valid_from, valid_to)`` window contains D. Records that match no
        window (coverage gap) get a null ``security_id`` and are reported by the
        data-audit layer — this is exactly the "coverage %" metric that flags a
        broken vendor feed.
        """
        recs = vendor_records.copy()
        recs[date_col] = pd.to_datetime(recs[date_col])
        recs = recs.sort_values(date_col)

        # Build the right side: one row per (key value, valid_from) so the as-of
        # join on date picks the window in effect.
        right = (
            self.df.rename(columns={key: "_key"})[["_key", "security_id", "valid_from", "valid_to"]]
            .sort_values("valid_from")
        )
        # ASOF on date within each key, then enforce valid_to (window upper bound).
        merged = asof_join(
            recs.rename(columns={key: "_key"}),
            right.rename(columns={"valid_from": date_col}),
            on=date_col,
            by="_key",
            direction="backward",
        )
        # Null out matches that fell past the window end (record after ID retired).
        past_window = merged["valid_to"].notna() & (merged[date_col] >= merged["valid_to"])
        merged.loc[past_window, "security_id"] = None
        return merged.rename(columns={"_key": key}).drop(columns=["valid_to"])

    def coverage(self, matched: pd.DataFrame) -> float:
        """Fraction of vendor rows successfully mapped to an internal ID."""
        if matched.empty:
            return 0.0
        return float(matched["security_id"].notna().mean())
