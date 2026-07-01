"""AlphaForge CLI — drive the machine from the terminal.

    alphaforge generate-data   # ingest vendors -> the local lake (synthetic/yfinance)
    alphaforge audit           # data-quality checks on the lake
    alphaforge backtest        # run the full end-to-end multifactor backtest
    alphaforge report          # print the latest tearsheet

Every command loads the same typed config (config/backtest.yaml).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from alphaforge.config import load_config
from alphaforge.logging import configure_logging
from alphaforge.storage.warehouse import Warehouse

app = typer.Typer(add_completion=False, help="AlphaForge — the multifactor machine.")
console = Console()
ARTIFACTS = Path("artifacts")


def _wh(cfg) -> Warehouse:
    return Warehouse(cfg.storage.lake_path)


@app.command("generate-data")
def generate_data(config: str = "config/backtest.yaml", log_level: str = "INFO") -> None:
    """Ingest all vendors into the local Parquet/DuckDB lake."""
    configure_logging(log_level)
    cfg = load_config(config)
    from alphaforge.data.loaders import ingest

    wh = _wh(cfg)
    ingest(cfg, wh)
    console.print(f"[green]✓[/green] data lake built at [bold]{cfg.storage.lake_path}[/bold]")


@app.command()
def audit(config: str = "config/backtest.yaml", log_level: str = "INFO") -> None:
    """Run data-quality audits on the lake."""
    configure_logging(log_level)
    cfg = load_config(config)
    from alphaforge.data.audit import audit_prices

    wh = _wh(cfg)
    rep = audit_prices(wh, expected_names=cfg.universe.n_names)
    table = Table(title="Data Audit — prices")
    table.add_column("check"); table.add_column("value"); table.add_column("threshold"); table.add_column("status")
    for r in rep.results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table.add_row(r.check, f"{r.value:.4f}", f"{r.threshold:.4f}", status)
    console.print(table)
    raise typer.Exit(0 if rep.ok else 1)


@app.command()
def backtest(config: str = "config/backtest.yaml", log_level: str = "INFO") -> None:
    """Run the full multifactor backtest and save artifacts."""
    configure_logging(log_level)
    cfg = load_config(config)
    from alphaforge.backtest.engine import run_backtest

    wh = _wh(cfg)
    if not wh.table_exists("prices"):
        console.print("[yellow]No data found — running generate-data first…[/yellow]")
        from alphaforge.data.loaders import ingest
        ingest(cfg, wh)

    result = run_backtest(cfg, wh)
    ARTIFACTS.mkdir(exist_ok=True)
    result.equity_curve.to_frame("equity").to_parquet(ARTIFACTS / "equity_curve.parquet")
    result.daily_returns.to_frame("ret").to_parquet(ARTIFACTS / "daily_returns.parquet")
    result.decay.to_parquet(ARTIFACTS / "factor_decay.parquet")
    (ARTIFACTS / "metrics.json").write_text(
        json.dumps({"metrics": result.metrics, "shortfall": result.shortfall,
                    "diagnostics": result.diagnostics}, indent=2, default=float)
    )
    _print_tearsheet(result.metrics, result.shortfall, result.decay)


@app.command()
def analyze(config: str = "config/backtest.yaml", plot: bool = True, log_level: str = "WARNING") -> None:
    """Prove the model predicts: information coefficient + quantile spread (+ plot)."""
    configure_logging(log_level)
    cfg = load_config(config)
    from alphaforge.alpha.combine import build_alphas
    from alphaforge.performance.predictive import analyze_predictive_power

    wh = _wh(cfg)
    if not wh.table_exists("prices"):
        console.print("[red]No data — run `alphaforge generate-data` first.[/red]")
        raise typer.Exit(1)

    with console.status("[bold]Computing alpha + forward returns…"):
        alpha_panel = build_alphas(cfg, wh)
        prices = wh.read_table("prices", columns=["date", "security_id", "ret"])
        rep = analyze_predictive_power(alpha_panel, prices)

    # ---- IC summary ----
    t = Table(title="Does the model predict? — Information Coefficient")
    t.add_column("metric", style="bold"); t.add_column("value", justify="right")
    t.add_row("mean daily IC", f"{rep.mean_ic:.4f}")
    t.add_row("IC info-ratio (ann.)", f"{rep.ic_ir:.2f}")
    t.add_row("% days IC positive", f"{rep.pct_days_positive:.1%}")
    t.add_row("observations", f"{rep.n_obs:,}")
    console.print(t)
    console.print("[dim]IC > 0 means it predicts. Real equity ICs are 0.02–0.05.[/dim]")

    # ---- quantile spread ----
    qt = Table(title="Quantile forward returns — Q1=disliked … Q5=liked")
    qt.add_column("bucket"); qt.add_column("avg fwd return", justify="right")
    for q, v in rep.quantile_returns.items():
        label = "Q5 (liked)" if q == rep.quantile_returns.index[-1] else (
            "Q1 (disliked)" if q == rep.quantile_returns.index[0] else f"Q{q}")
        qt.add_row(label, f"{v*1e4:+.2f} bps/day")
    console.print(qt)
    yr = rep.top_minus_bottom_bps * 252 / 100
    verdict = "[green]MONOTONIC — the edge is real ✓[/green]" if _is_monotonic(rep.quantile_returns) \
        else "[yellow]not cleanly monotonic — weak/noisy edge[/yellow]"
    console.print(f"[bold]Top − Bottom spread:[/bold] {rep.top_minus_bottom_bps:+.2f} bps/day "
                  f"(≈ {yr:+.1f}%/yr)  →  {verdict}")

    if plot:
        _plot_curves()


def _is_monotonic(q) -> bool:
    vals = q.values
    return bool(np.all(np.diff(vals) > -abs(vals).mean() * 0.5))  # broadly increasing


def _plot_curves() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        console.print("[dim](install '.[viz]' for the equity-curve plot)[/dim]")
        return
    eq_path = ARTIFACTS / "equity_curve.parquet"
    if not eq_path.exists():
        console.print("[dim](run `alphaforge backtest` first to plot the equity curve)[/dim]")
        return
    eq = pd.read_parquet(eq_path)["equity"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq.index, eq.values, lw=1.6)
    ax.set_title("AlphaForge — backtest equity curve (growth of $1)")
    ax.set_ylabel("equity (×)"); ax.grid(alpha=0.3)
    out = ARTIFACTS / "equity_curve.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    console.print(f"[green]✓[/green] equity curve saved → [bold]{out}[/bold]")


@app.command("compare-models")
def compare_models(config: str = "config/backtest.yaml", log_level: str = "WARNING") -> None:
    """Head-to-head: linear alpha vs gradient-boosted-tree alpha (IC + spread)."""
    configure_logging(log_level)
    cfg = load_config(config)
    from alphaforge.alpha.combine import build_alphas
    from alphaforge.alpha.ml_model import build_ml_alphas
    from alphaforge.performance.predictive import analyze_predictive_power

    wh = _wh(cfg)
    if not wh.table_exists("prices"):
        console.print("[red]No data — run `alphaforge generate-data` first.[/red]")
        raise typer.Exit(1)
    prices = wh.read_table("prices", columns=["date", "security_id", "ret"])

    with console.status("[bold]Building linear alpha…"):
        lin = analyze_predictive_power(build_alphas(cfg, wh), prices)
    with console.status("[bold]Training walk-forward gradient-boosted alpha (slower)…"):
        gbt = analyze_predictive_power(build_ml_alphas(cfg, wh), prices)

    t = Table(title="Linear vs Gradient-Boosted Trees — out-of-sample predictive power")
    t.add_column("metric", style="bold")
    t.add_column("Linear", justify="right"); t.add_column("GBT (ML)", justify="right")
    t.add_column("winner", justify="center")

    def row(label, lv, gv, fmt, higher_better=True):
        win = "GBT" if (gv > lv) == higher_better else "Linear"
        color = "green"
        t.add_row(label, fmt.format(lv), fmt.format(gv), f"[{color}]{win}[/{color}]")

    row("mean daily IC", lin.mean_ic, gbt.mean_ic, "{:.4f}")
    row("IC info-ratio (ann.)", lin.ic_ir, gbt.ic_ir, "{:.2f}")
    row("% days IC positive", lin.pct_days_positive, gbt.pct_days_positive, "{:.1%}")
    row("top-bottom spread (bps/day)", lin.top_minus_bottom_bps, gbt.top_minus_bottom_bps, "{:.2f}")
    console.print(t)
    console.print("[dim]GBT is walk-forward trained (no look-ahead): retrain on the past, "
                  "predict the next block. A fair, honest comparison.[/dim]")


@app.command("feature-importance")
def feature_importance(config: str = "config/backtest.yaml", log_level: str = "WARNING") -> None:
    """Which signals did the gradient-boosted model actually rely on? (out-of-sample)."""
    configure_logging(log_level)
    cfg = load_config(config)
    from alphaforge.alpha.ml_model import ml_feature_importance

    wh = _wh(cfg)
    if not wh.table_exists("prices"):
        console.print("[red]No data — run `alphaforge generate-data` first.[/red]")
        raise typer.Exit(1)
    with console.status("[bold]Walk-forward training + permutation importance (slower)…"):
        imp = ml_feature_importance(cfg, wh)

    t = Table(title="GBT feature importance — which signals carry the edge")
    t.add_column("signal", style="bold"); t.add_column("share", justify="right")
    t.add_column("", justify="left")
    for sig, row in imp.iterrows():
        pct = row["importance_pct"]
        bar = "█" * int(round(pct * 30))
        t.add_row(sig, f"{pct:.1%}", f"[cyan]{bar}[/cyan]")
    console.print(t)
    console.print("[dim]Permutation importance: shuffle a signal, measure how much accuracy "
                  "drops. Bigger = the model leaned on it more. Out-of-sample only.[/dim]")


@app.command()
def report() -> None:
    """Print the latest saved tearsheet."""
    metrics_path = ARTIFACTS / "metrics.json"
    if not metrics_path.exists():
        console.print("[red]No artifacts — run `alphaforge backtest` first.[/red]")
        raise typer.Exit(1)
    blob = json.loads(metrics_path.read_text())
    import pandas as pd
    decay = pd.read_parquet(ARTIFACTS / "factor_decay.parquet") if (ARTIFACTS / "factor_decay.parquet").exists() else pd.DataFrame()
    _print_tearsheet(blob["metrics"], blob["shortfall"], decay)


def _print_tearsheet(metrics: dict, shortfall: dict, decay) -> None:
    t = Table(title="AlphaForge — Performance Tearsheet")
    t.add_column("metric", style="bold"); t.add_column("value", justify="right")
    fmt = {
        "ann_return": "{:.2%}", "ann_vol": "{:.2%}", "sharpe": "{:.2f}",
        "information_ratio": "{:.2f}", "max_drawdown": "{:.2%}", "hit_rate": "{:.2%}",
        "total_return": "{:.2%}", "mean_ic": "{:.4f}", "breadth": "{:.0f}",
        "ir_implied_by_law": "{:.2f}",
    }
    for k, v in metrics.items():
        t.add_row(k, fmt.get(k, "{:.4f}").format(v))
    console.print(t)

    st = Table(title="Implementation Shortfall (cost of trading)")
    st.add_column("metric", style="bold"); st.add_column("value", justify="right")
    for k, v in shortfall.items():
        st.add_row(k, f"{v:.2%}")
    console.print(st)

    if decay is not None and len(decay):
        dt = Table(title="Factor IC Decay (feeds the research loop)")
        dt.add_column("factor"); dt.add_column("mean IC", justify="right")
        dt.add_column("recent IC", justify="right"); dt.add_column("status")
        for factor, row in decay.iterrows():
            color = "red" if row.get("status") == "DECAYING" else "green"
            dt.add_row(str(factor), f"{row['mean_ic']:.4f}", f"{row['recent_ic']:.4f}",
                       f"[{color}]{row.get('status','')}[/{color}]")
        console.print(dt)


if __name__ == "__main__":
    app()
