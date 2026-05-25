"""Visualizaciones del backtest con matplotlib."""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin display (compatible con servidores)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_equity_curve(
    trades_df: pd.DataFrame,
    initial_balance: float,
    save_path: Path,
    title: str = "Equity Curve",
) -> None:
    """Grafico de equity curve + drawdown subplots."""
    df = trades_df.sort_values("entry_time").copy()
    df["cum_pnl"]     = df["net_pnl_usd"].cumsum()
    df["equity"]      = initial_balance + df["cum_pnl"]
    df["running_max"] = df["equity"].cummax()
    df["drawdown"]    = df["equity"] - df["running_max"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax1.plot(df["entry_time"], df["equity"], color="steelblue", linewidth=1.5, label="Equity")
    ax1.fill_between(df["entry_time"], initial_balance, df["equity"], alpha=0.2, color="steelblue")
    ax1.axhline(initial_balance, color="gray", linestyle="--", alpha=0.5, label=f"Inicial ${initial_balance:,.0f}")
    ax1.set_ylabel("Equity (USD)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.fill_between(df["entry_time"], 0, df["drawdown"], color="crimson", alpha=0.6)
    ax2.set_ylabel("Drawdown (USD)")
    ax2.set_xlabel("Fecha")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Equity curve guardado: %s", save_path)


def plot_pnl_distribution(trades_df: pd.DataFrame, save_path: Path) -> None:
    """Histograma de PnL por trade con media y mediana."""
    pnls = trades_df["net_pnl_usd"]
    fig, ax = plt.subplots(figsize=(10, 6))

    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    ax.hist(wins,   bins=20, color="steelblue", edgecolor="white", alpha=0.8, label=f"Wins ({len(wins)})")
    ax.hist(losses, bins=20, color="crimson",   edgecolor="white", alpha=0.8, label=f"Losses ({len(losses)})")

    ax.axvline(0,           color="black", linestyle="-",  linewidth=1, alpha=0.5)
    ax.axvline(pnls.mean(), color="green", linestyle="--", linewidth=1.5, label=f"Media {pnls.mean():+.2f}")
    ax.axvline(pnls.median(), color="orange", linestyle="--", linewidth=1.5, label=f"Mediana {pnls.median():+.2f}")

    ax.set_xlabel("PnL por trade (USD)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion de PnL por trade")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("PnL distribution guardado: %s", save_path)


def plot_exit_reasons(trades_df: pd.DataFrame, save_path: Path) -> None:
    """Pie chart de razones de salida (TP/SL/forced_close)."""
    counts = trades_df["exit_reason"].value_counts()
    color_map = {"tp": "#2ecc71", "sl": "#e74c3c", "forced_close": "#f39c12"}
    colors = [color_map.get(r, "#95a5a6") for r in counts.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.8,
    )
    for t in autotexts:
        t.set_fontsize(11)
    ax.set_title("Razones de salida")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Exit reasons guardado: %s", save_path)


def plot_monthly_pnl(trades_df: pd.DataFrame, save_path: Path) -> None:
    """Barras de PnL mensual neto."""
    df = trades_df.copy()
    df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    monthly = df.groupby("month")["net_pnl_usd"].sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue" if v >= 0 else "crimson" for v in monthly.values]
    ax.bar(monthly.index.astype(str), monthly.values, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mes")
    ax.set_ylabel("PnL neto (USD)")
    ax.set_title("PnL mensual neto")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Monthly PnL guardado: %s", save_path)


def generate_full_report(
    trades_df: pd.DataFrame,
    metrics: dict,
    initial_balance: float,
    output_dir: Path,
    title_prefix: str = "v3",
) -> None:
    """Genera todos los graficos del reporte en una carpeta."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if trades_df.empty:
        logger.warning("Sin trades — no se generan graficos.")
        return

    plot_equity_curve(
        trades_df, initial_balance,
        output_dir / "equity_curve.png",
        title=f"Equity Curve — {title_prefix}",
    )
    plot_pnl_distribution(trades_df, output_dir / "pnl_distribution.png")
    plot_exit_reasons(trades_df, output_dir / "exit_reasons.png")
    plot_monthly_pnl(trades_df, output_dir / "monthly_pnl.png")
    logger.info("Reporte completo guardado en %s", output_dir)
