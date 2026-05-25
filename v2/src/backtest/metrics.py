"""Metricas profesionales de evaluacion de estrategia."""
import numpy as np
import pandas as pd


def compute_full_metrics(
    trades_df: pd.DataFrame,
    initial_balance: float = 2000.0,
) -> dict:
    """Calcula el set completo de metricas estilo fondo cuantitativo.

    Args:
        trades_df: DataFrame con columnas entry_time, net_pnl_usd, exit_reason.
        initial_balance: balance inicial asumido para calcular equity curve.

    Returns:
        Dict con todas las metricas. Incluye 'equity_curve' como lista de dicts.
    """
    if trades_df.empty:
        return {"n_trades": 0, "error": "No trades"}

    df = trades_df.sort_values("entry_time").copy()
    df["cumulative_pnl"] = df["net_pnl_usd"].cumsum()
    df["equity"]         = initial_balance + df["cumulative_pnl"]

    n      = len(df)
    wins   = int((df["net_pnl_usd"] > 0).sum())
    losses = int((df["net_pnl_usd"] <= 0).sum())

    gross_wins   = float(df.loc[df["net_pnl_usd"] > 0,  "net_pnl_usd"].sum())
    gross_losses = float(df.loc[df["net_pnl_usd"] <= 0, "net_pnl_usd"].sum())
    profit_factor = abs(gross_wins / gross_losses) if gross_losses != 0 else float("inf")

    avg_win  = float(df.loc[df["net_pnl_usd"] > 0,  "net_pnl_usd"].mean()) if wins   > 0 else 0.0
    avg_loss = float(df.loc[df["net_pnl_usd"] <= 0, "net_pnl_usd"].mean()) if losses > 0 else 0.0

    # Drawdown
    running_max = df["equity"].cummax()
    drawdown    = df["equity"] - running_max
    max_dd_usd  = float(drawdown.min())
    # Evitar division por cero si running_max es 0
    max_dd_pct  = float((drawdown / running_max.replace(0, np.nan) * 100).min())

    # Sharpe por trade
    ret    = df["net_pnl_usd"]
    sharpe = float(ret.mean() / ret.std()) if ret.std() > 0 else 0.0

    # Sharpe anualizado: estimar frecuencia de trades
    trading_days = max(
        (df["entry_time"].max() - df["entry_time"].min()).days, 1
    )
    trades_per_year  = (n / trading_days) * 252
    sharpe_annual    = sharpe * float(np.sqrt(trades_per_year)) if trades_per_year > 0 else 0.0

    # Retorno total y Calmar
    final_equity       = float(df["equity"].iloc[-1])
    total_return_pct   = (final_equity - initial_balance) / initial_balance * 100
    calmar = abs(total_return_pct / max_dd_pct) if max_dd_pct < 0 else float("inf")

    return {
        "n_trades":           n,
        "wins":               wins,
        "losses":             losses,
        "win_rate_pct":       wins / n * 100,
        "pnl_total_usd":      float(df["net_pnl_usd"].sum()),
        "final_equity_usd":   final_equity,
        "total_return_pct":   total_return_pct,
        "profit_factor":      profit_factor,
        "expectancy_usd":     float(ret.mean()),
        "avg_win_usd":        avg_win,
        "avg_loss_usd":       avg_loss,
        "max_drawdown_usd":   max_dd_usd,
        "max_drawdown_pct":   max_dd_pct,
        "sharpe_per_trade":   sharpe,
        "sharpe_annualized":  sharpe_annual,
        "calmar_ratio":       calmar,
        "exit_distribution":  df["exit_reason"].value_counts().to_dict(),
        "equity_curve":       df[["entry_time", "equity", "cumulative_pnl"]].to_dict(orient="records"),
    }


def print_metrics_report(metrics: dict) -> None:
    """Imprime las metricas en formato tabular legible."""
    if metrics.get("n_trades", 0) == 0:
        print("Sin trades.")
        return
    print()
    print("=" * 60)
    print("         REPORTE DE BACKTEST")
    print("=" * 60)
    print(f"Trades ejecutados      : {metrics['n_trades']}")
    print(f"Wins / Losses          : {metrics['wins']} / {metrics['losses']}")
    print(f"Win rate               : {metrics['win_rate_pct']:.1f}%")
    print(f"PnL total (USD)        : {metrics['pnl_total_usd']:+.2f}")
    print(f"Equity final (USD)     : {metrics['final_equity_usd']:.2f}")
    print(f"Retorno total          : {metrics['total_return_pct']:+.2f}%")
    print(f"Profit factor          : {metrics['profit_factor']:.2f}")
    print(f"Expectancy/trade       : {metrics['expectancy_usd']:+.2f}")
    print(f"Avg win / Avg loss     : {metrics['avg_win_usd']:+.2f} / {metrics['avg_loss_usd']:+.2f}")
    print(f"Max drawdown (USD)     : {metrics['max_drawdown_usd']:+.2f}")
    print(f"Max drawdown (%)       : {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe (por trade)     : {metrics['sharpe_per_trade']:.3f}")
    print(f"Sharpe anualizado      : {metrics['sharpe_annualized']:.2f}")
    print(f"Calmar ratio           : {metrics['calmar_ratio']:.2f}")
    print(f"Exit reasons           : {metrics['exit_distribution']}")
    print("=" * 60)
