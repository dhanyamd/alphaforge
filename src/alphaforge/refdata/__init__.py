"""(IV) SECURITY MATCHING — map messy vendor entities to internal IDs, point-in-time.

The single most important idea here is **point-in-time**: an identifier is only
valid for a date *range*. Apple-the-listing under one CUSIP from 2010-2015, a
different one after an M&A event. When a factor runs over history it must ask
"what was true *as of that date*" — never "what is true today". Getting this
wrong injects look-ahead bias and silently inflates every backtest.
"""

from alphaforge.refdata.security_master import SecurityMaster

__all__ = ["SecurityMaster"]
