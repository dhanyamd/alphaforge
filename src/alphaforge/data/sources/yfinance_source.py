"""Optional real-equity source (free EOD via yfinance).

Enable with `pip install -e ".[realdata]"`. This shows the *same* `DataSource`
contract pointed at a real vendor: it downloads prices, derives a crude
fundamentals/alt-data stand-in, and returns the identical `MarketData` bundle.
When you later swap in Snowflake or a paid vendor, you implement exactly this
one method and the rest of AlphaForge is untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.sources.base import DataSource, MarketData
from alphaforge.logging import get_logger

log = get_logger(__name__)

# A small, liquid, sector-diverse default universe. Edit freely.
# A broader, sector-diverse set of large liquid US names (more breadth for the
# fundamental law). All existed and traded across 2015-2020.
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Technology",
    "META": "Technology", "ORCL": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "INTC": "Technology", "CSCO": "Technology", "AMD": "Technology", "QCOM": "Technology",
    # Consumer
    "AMZN": "Consumer", "WMT": "Consumer", "PG": "Consumer", "KO": "Consumer", "HD": "Consumer",
    "MCD": "Consumer", "NKE": "Consumer", "SBUX": "Consumer", "DIS": "Consumer", "COST": "Consumer",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "AXP": "Financials", "C": "Financials",
    # Healthcare
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    # Industrials
    "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "RTX": "Industrials", "BA": "Industrials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    # Materials
    "LIN": "Materials", "FCX": "Materials", "NEM": "Materials", "APD": "Materials",
}
# ~140 large, liquid US names across all sectors (all traded 2015-2020). Real
# sectors are pulled from yfinance .info at load time; this list is just the
# investable set. More names = more breadth (the fundamental law lever).
DEFAULT_TICKERS = [
    # Tech / Comm
    "AAPL","MSFT","NVDA","GOOGL","GOOG","META","ORCL","CRM","ADBE","INTC","CSCO","AMD","QCOM",
    "TXN","AVGO","IBM","NOW","INTU","AMAT","MU","ADI","LRCX","KLAC","NXPI","ADSK","CTSH","ACN",
    "T","VZ","TMUS","CMCSA","NFLX","DIS","CHTR","EA",
    # Consumer
    "AMZN","WMT","PG","KO","PEP","HD","MCD","NKE","SBUX","COST","TGT","LOW","BKNG","MDLZ",
    "CL","MO","PM","KMB","GIS","EL","YUM","DG","ROST","TJX","F","GM",
    # Financials
    "JPM","BAC","WFC","GS","MS","AXP","C","BLK","SCHW","USB","PNC","TFC","CB","MMC","SPGI","CME","ICE",
    # Healthcare
    "JNJ","PFE","UNH","ABBV","MRK","TMO","ABT","LLY","DHR","BMY","AMGN","MDT","GILD","CVS","CI","ISRG","SYK","ZTS",
    # Energy
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","WMB","KMI",
    # Industrials
    "CAT","GE","HON","UPS","RTX","BA","UNP","LMT","DE","MMM","GD","NOC","EMR","ETN","CSX","FDX","ITW",
    # Utilities
    "NEE","DUK","SO","D","AEP","EXC","SRE","XEL",
    # Materials / RE
    "LIN","FCX","NEM","APD","SHW","ECL","DOW","DD","NUE","PLD","AMT","CCI","SPG",
]


class YFinanceSource(DataSource):
    def __init__(self, tickers: list[str] | None = None) -> None:
        self.tickers = tickers or DEFAULT_TICKERS

    def fetch(self, start: str, end: str) -> MarketData:
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install extras: pip install -e '.[realdata]'") from e

        log.info("yfinance.download", n=len(self.tickers), start=start, end=end)
        raw = yf.download(self.tickers, start=start, end=end, auto_adjust=True, progress=False)
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()

        prices = (
            close.stack().rename("close").reset_index()
            .rename(columns={"Date": "date", "Ticker": "security_id", "level_1": "security_id"})
        )
        vol = (
            volume.stack().rename("volume").reset_index()
            .rename(columns={"Date": "date", "Ticker": "security_id", "level_1": "security_id"})
        )
        prices = prices.merge(vol, on=["date", "security_id"], how="left")
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.sort_values(["security_id", "date"])
        prices["ret"] = prices.groupby("security_id")["close"].pct_change().fillna(0.0)
        ids = list(prices["security_id"].unique())

        # ---- REAL fundamentals from yfinance .info (per ticker, robust) ------
        # NOTE ON POINT-IN-TIME: .info returns only the *latest* valuation, so we
        # anchor book value with today's priceToBook. This introduces mild
        # look-ahead (we use today's book in 2015). It's an honest free-data
        # compromise: value becomes TIME-VARYING via price (cheap when the stock
        # has fallen), and quality uses the real profit margin. A production desk
        # would use point-in-time fundamentals (Compustat / SEC EDGAR).
        pb, pm, mc, sec = {}, {}, {}, {}
        for t in ids:
            try:
                info = yf.Ticker(t).info
                pb[t] = info.get("priceToBook")
                pm[t] = info.get("profitMargins")
                mc[t] = info.get("marketCap")
                sec[t] = info.get("sector")            # REAL sector, for sector-neutrality
            except Exception:  # noqa: BLE001 — a flaky ticker shouldn't kill the run
                pb[t] = pm[t] = mc[t] = sec[t] = None

        # Real-ish shares outstanding = marketCap / latest price -> mktcap moves with price.
        last_px = prices.groupby("security_id")["close"].last().to_dict()
        so = {t: (mc[t] / last_px[t]) if mc.get(t) and last_px.get(t) else 1e9 for t in ids}
        prices["shares_out"] = prices["security_id"].map(so)

        # Book/share anchored at the first date; book_to_price = book / price each month.
        first_px = prices.groupby("security_id")["close"].first().to_dict()
        book = {t: (first_px[t] / pb[t]) if pb.get(t) else None for t in ids}
        monthly = (
            prices.set_index("date").groupby("security_id")["close"]
            .resample("MS").first().reset_index().dropna(subset=["close"])
        )
        monthly["book_to_price"] = monthly.apply(
            lambda r: (book[r.security_id] / r["close"]) if book.get(r.security_id) else np.nan, axis=1
        )
        # quality = gross_profitability = gross_profit/total_assets; set so it equals
        # the real profit margin (total_assets=1).
        monthly["gross_profit"] = monthly["security_id"].map(
            lambda t: pm[t] if pm.get(t) is not None else np.nan
        )
        monthly["total_assets"] = 1.0
        funda = monthly[["date", "security_id", "book_to_price", "gross_profit", "total_assets"]]

        # Alt-data has no free source -> neutral (0), so it neither helps nor injects noise.
        alt = prices[["date", "security_id"]].copy()
        alt["alt_score"] = 0.0

        static = pd.DataFrame({
            "security_id": ids,
            "sector": [sec.get(t) or SECTOR_MAP.get(t, "Other") for t in ids],
        })
        return MarketData(prices=prices, fundamentals=funda, altdata=alt, static=static)
