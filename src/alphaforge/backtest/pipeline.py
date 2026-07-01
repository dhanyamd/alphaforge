"""A tiny DAG pipeline runner — the local analog of a Databricks workflow.

In production each block of the machine is a notebook/task wired into a workflow
that triggers on a schedule (and only runs once its upstreams succeed). We mirror
that shape with a minimal dependency-ordered runner so the data-loader -> audit
-> signals -> ... chain is explicit, logged, and re-runnable. It's deliberately
small — the point is to show the *shape* (DAG of stages with dependencies), not
to reimplement Airflow.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from alphaforge.logging import get_logger

log = get_logger(__name__)


@dataclass
class Stage:
    name: str
    fn: Callable[[dict[str, Any]], Any]
    depends_on: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(self) -> None:
        self.stages: dict[str, Stage] = {}
        self.results: dict[str, Any] = {}

    def add(self, name: str, fn: Callable, depends_on: list[str] | None = None) -> "Pipeline":
        self.stages[name] = Stage(name, fn, depends_on or [])
        return self

    def _order(self) -> list[str]:
        """Topological sort so every stage runs after its dependencies."""
        order, seen = [], set()

        def visit(n: str, path: tuple[str, ...]) -> None:
            if n in seen:
                return
            if n in path:
                raise ValueError(f"cycle detected at stage {n}")
            for dep in self.stages[n].depends_on:
                visit(dep, (*path, n))
            seen.add(n)
            order.append(n)

        for name in self.stages:
            visit(name, ())
        return order

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = context or {}
        for name in self._order():
            log.info("pipeline.stage.start", stage=name)
            self.results[name] = self.stages[name].fn(ctx)
            ctx[name] = self.results[name]
            log.info("pipeline.stage.done", stage=name)
        return self.results
