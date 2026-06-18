"""Analisis detallado mes a mes del backtest.

Solo lectura — NO modifica trades, solo los agrupa y resume.
Capa de analisis posterior sobre el output de run_backtest() (engine.py),
sin tocar el motor ni la logica de la estrategia.
"""
import pandas as pd


def compute_monthly_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa trades por mes y calcula metricas mensuales.

    Args:
        trades_df: DataFrame de trades del backtest (no se modifica).

    Returns:
        DataFrame con metricas mensuales (mes, trades, win rate, PnL, etc.)
    """
    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["year_month"] = df["entry_time"].dt.to_period("M").astype(str)

    rows = []
    for ym, g in df.groupby("year_month"):
        wins = g[g["net_pnl_usd"] > 0]
        losses = g[g["net_pnl_usd"] <= 0]
        gw = wins["net_pnl_usd"].sum()
        gl = losses["net_pnl_usd"].sum()
        pf = abs(gw / gl) if gl != 0 else float("inf")

        equity = g["net_pnl_usd"].cumsum()
        running_max = equity.cummax()
        dd = (equity - running_max).min()

        rows.append({
            "year_month": ym,
            "n_trades": len(g),
            "n_buy": (g["direction"] == "BUY").sum() if "direction" in g.columns else 0,
            "n_sell": (g["direction"] == "SELL").sum() if "direction" in g.columns else 0,
            "n_wins": len(wins),
            "n_losses": len(losses),
            "win_rate_pct": len(wins) / len(g) * 100,
            "pnl_neto_usd": g["net_pnl_usd"].sum(),
            "avg_win": wins["net_pnl_usd"].mean() if len(wins) > 0 else 0,
            "avg_loss": losses["net_pnl_usd"].mean() if len(losses) > 0 else 0,
            "profit_factor": pf,
            "max_dd_intra_mes": dd,
            "tp_count": (g["exit_reason"] == "tp").sum(),
            "sl_count": (g["exit_reason"] == "sl").sum(),
            "forced_close_count": (g["exit_reason"] == "forced_close").sum(),
            "best_trade": g["net_pnl_usd"].max(),
            "worst_trade": g["net_pnl_usd"].min(),
        })

    monthly = pd.DataFrame(rows).sort_values("year_month").reset_index(drop=True)
    monthly["pnl_acumulado_usd"] = monthly["pnl_neto_usd"].cumsum()
    return monthly


def print_monthly_report(monthly: pd.DataFrame, initial_balance: float = 2000.0) -> None:
    """Imprime tabla mensual."""
    print()
    print("=" * 120)
    print(f"{'REPORTE MENSUAL DETALLADO':^120}")
    print("=" * 120)
    print()
    print(f"{'Mes':<10} {'Trades':>7} {'B/S':>7} {'W/L':>9} {'WR%':>6} "
          f"{'PnL':>10} {'Acum':>10} {'PF':>6} {'DD':>9} {'TP/SL/FC':>11}")
    print("-" * 120)

    for _, row in monthly.iterrows():
        bs = f"{row['n_buy']}/{row['n_sell']}"
        wl = f"{row['n_wins']}/{row['n_losses']}"
        wr = f"{row['win_rate_pct']:.1f}"
        pnl = f"{row['pnl_neto_usd']:+.0f}"
        acum = f"{row['pnl_acumulado_usd']:+.0f}"
        pf = f"{row['profit_factor']:.2f}" if row['profit_factor'] != float('inf') else "inf"
        dd = f"{row['max_dd_intra_mes']:+.0f}"
        exits = f"{row['tp_count']}/{row['sl_count']}/{row['forced_close_count']}"
        print(f"{row['year_month']:<10} {row['n_trades']:>7} {bs:>7} {wl:>9} {wr:>6} "
              f"{pnl:>10} {acum:>10} {pf:>6} {dd:>9} {exits:>11}")

    print("-" * 120)
    total_pnl = monthly['pnl_neto_usd'].sum()
    print(f"TOTAL: {monthly['n_trades'].sum()} trades | "
          f"PnL total: ${total_pnl:+,.2f} | "
          f"Equity final: ${initial_balance + total_pnl:,.2f}")
    print("=" * 120)

    best = monthly.loc[monthly['pnl_neto_usd'].idxmax()]
    worst = monthly.loc[monthly['pnl_neto_usd'].idxmin()]
    print()
    print(f"Mejor mes: {best['year_month']} con ${best['pnl_neto_usd']:+,.2f} "
          f"({best['n_trades']} trades, WR {best['win_rate_pct']:.1f}%)")
    print(f"Peor mes:  {worst['year_month']} con ${worst['pnl_neto_usd']:+,.2f} "
          f"({worst['n_trades']} trades, WR {worst['win_rate_pct']:.1f}%)")

    positive = (monthly['pnl_neto_usd'] > 0).sum()
    negative = (monthly['pnl_neto_usd'] <= 0).sum()
    print(f"Meses positivos: {positive} / {len(monthly)} ({positive/len(monthly)*100:.1f}%)")
    print(f"Meses negativos: {negative} / {len(monthly)} ({negative/len(monthly)*100:.1f}%)")


def compute_yearly_breakdown(monthly: pd.DataFrame) -> pd.DataFrame:
    """Agrupa el reporte mensual por año."""
    df = monthly.copy()
    df["year"] = df["year_month"].str[:4]

    yearly = df.groupby("year").agg({
        "n_trades": "sum",
        "n_wins": "sum",
        "n_losses": "sum",
        "pnl_neto_usd": "sum",
        "tp_count": "sum",
        "sl_count": "sum",
        "forced_close_count": "sum",
    }).reset_index()

    yearly["win_rate_pct"] = yearly["n_wins"] / yearly["n_trades"] * 100
    yearly["pnl_acumulado_usd"] = yearly["pnl_neto_usd"].cumsum()
    return yearly


def print_yearly_report(yearly: pd.DataFrame, initial_balance: float = 2000.0) -> None:
    """Imprime tabla anual."""
    print()
    print("=" * 100)
    print(f"{'REPORTE ANUAL':^100}")
    print("=" * 100)
    print(f"{'Año':<6} {'Trades':>8} {'W/L':>10} {'WR%':>7} "
          f"{'PnL Año':>12} {'PnL Acum':>12} {'TP/SL/FC':>14}")
    print("-" * 100)

    for _, row in yearly.iterrows():
        wl = f"{row['n_wins']}/{row['n_losses']}"
        wr = f"{row['win_rate_pct']:.1f}"
        pnl = f"${row['pnl_neto_usd']:+,.0f}"
        acum = f"${row['pnl_acumulado_usd']:+,.0f}"
        exits = f"{row['tp_count']}/{row['sl_count']}/{row['forced_close_count']}"
        print(f"{row['year']:<6} {row['n_trades']:>8} {wl:>10} {wr:>7} "
              f"{pnl:>12} {acum:>12} {exits:>14}")

    print("-" * 100)
