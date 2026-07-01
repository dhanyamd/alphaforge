"""Structured logging — the cheapest observability you can buy.

In production every stage of the pipeline emits structured (key=value) logs so
that when a factor halts at 2am you can grep the run and see *exactly* which
vendor file, which date partition, and which audit check failed. We use
`structlog` to get JSON-able, contextual logs instead of bare prints.
"""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog once at process start."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),  # pretty in a terminal; swap for JSONRenderer in prod
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
