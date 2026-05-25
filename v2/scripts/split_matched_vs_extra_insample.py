"""Diagnostico: comparar performance de señales matched (que el operador tomo)
vs señales extra (que el operador NO tomo).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from v2.config.settings import DATA_FEATURES, DATA_GROUND_TRUTH
from v2.src.backtest.engine import BacktestConfig, run_backtest
from v2.src.strategies.asian_momentum_v3 import (
    StrategyParamsV3,
    apply_strategy_v3_to_snapshots,
)

TOLERANCE_MIN = 60


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = pd.read_parquet(DATA_GROUND_TRUTH / "trades.parquet")
    trades_is = trades[trades["sample"] == "in_sample"].copy()
    trades_is["time_open_utc"] = pd.to_datetime(trades_is["time_open_utc"], utc=True)

    snapshots = pd.read_parquet(DATA_FEATURES / "snapshots_M1.parquet")
    if snapshots.index.tz is None:
        snapshots.index = snapshots.index.tz_localize("UTC")

    ohlc_m1 = pd.read_parquet(DATA_GROUND_TRUTH / "ohlc_M1.parquet")
    if ohlc_m1.index.tz is None:
        ohlc_m1.index = ohlc_m1.index.tz_localize("UTC")
    ohlc_m1.columns = [c.lower() for c in ohlc_m1.columns]

    t_start  = trades_is["time_open_utc"].min() - pd.Timedelta(days=2)
    t_end    = trades_is["time_open_utc"].max() + pd.Timedelta(days=1)
    snaps_is = snapshots[(snapshots.index >= t_start) & (snapshots.index <= t_end)].copy()

    return trades_is, snaps_is, ohlc_m1


def mark_matched_to_real(bt_df: pd.DataFrame, real_trades: pd.DataFrame) -> pd.DataFrame:
    """Marca cada trade ejecutado como matched si su entry_time está dentro de
    ±TOLERANCE_MIN de un trade real del mismo tipo."""
    real_buy  = real_trades[real_trades["type"] == "BUY"]["time_open_utc"].sort_values().values
    real_sell = real_trades[real_trades["type"] == "SELL"]["time_open_utc"].sort_values().values

    def _is_matched(row: pd.Series) -> bool:
        ts  = pd.Timestamp(row["entry_time"]).to_datetime64()
        arr = real_buy if row["direction"] == "BUY" else real_sell
        if len(arr) == 0:
            return False
        diffs = (arr.astype("datetime64[ns]") - ts.astype("datetime64[ns]")) / np.timedelta64(1, "m")
        return bool(np.any(np.abs(diffs) <= TOLERANCE_MIN))

    bt_df = bt_df.copy()
    bt_df["matched"] = bt_df.apply(_is_matched, axis=1)
    return bt_df


def compute_group_metrics(group: pd.DataFrame, name: str) -> dict:
    n = len(group)
    if n == 0:
        return {"name": name, "n": 0, "wins": 0, "losses": 0,
                "win_rate_pct": 0.0, "pnl_total_usd": 0.0,
                "profit_factor": float("nan"), "avg_win_usd": 0.0,
                "avg_loss_usd": 0.0, "expectancy_usd": 0.0,
                "max_drawdown_usd": 0.0, "exit_distribution": {}}

    wins   = int((group["net_pnl_usd"] > 0).sum())
    losses = int((group["net_pnl_usd"] <= 0).sum())

    gross_wins   = float(group.loc[group["net_pnl_usd"] > 0,  "net_pnl_usd"].sum())
    gross_losses = float(group.loc[group["net_pnl_usd"] <= 0, "net_pnl_usd"].sum())
    pf = abs(gross_wins / gross_losses) if gross_losses != 0 else float("inf")

    g = group.sort_values("entry_time").copy()
    g["cum_pnl"]   = g["net_pnl_usd"].cumsum()
    running_max    = g["cum_pnl"].cummax()
    max_dd         = float((g["cum_pnl"] - running_max).min())

    return {
        "name":              name,
        "n":                 n,
        "wins":              wins,
        "losses":            losses,
        "win_rate_pct":      wins / n * 100,
        "pnl_total_usd":     float(group["net_pnl_usd"].sum()),
        "profit_factor":     pf,
        "avg_win_usd":       float(group.loc[group["net_pnl_usd"] > 0,  "net_pnl_usd"].mean()) if wins   > 0 else 0.0,
        "avg_loss_usd":      float(group.loc[group["net_pnl_usd"] <= 0, "net_pnl_usd"].mean()) if losses > 0 else 0.0,
        "expectancy_usd":    float(group["net_pnl_usd"].mean()),
        "max_drawdown_usd":  max_dd,
        "exit_distribution": group["exit_reason"].value_counts().to_dict(),
    }


def print_table(metrics_list: list[dict]) -> None:
    col_w = 20
    lbl_w = 28
    print(f"{'Metrica':<{lbl_w}}", end="")
    for m in metrics_list:
        print(f"{m['name']:>{col_w}}", end="")
    print()
    print("-" * (lbl_w + col_w * len(metrics_list)))

    def _row(label: str, vals: list[str]) -> None:
        print(f"{label:<{lbl_w}}", end="")
        for v in vals:
            print(f"{v:>{col_w}}", end="")
        print()

    _row("N trades",           [str(m["n"])                                  for m in metrics_list])
    _row("Wins / Losses",      [f"{m['wins']}W / {m['losses']}L"             for m in metrics_list])
    _row("Win rate %",         [f"{m['win_rate_pct']:.1f}%"                  for m in metrics_list])
    _row("PnL NET (USD)",      [f"{m['pnl_total_usd']:+.2f}"                 for m in metrics_list])
    _row("Profit factor",      [f"{m['profit_factor']:.2f}"                  for m in metrics_list])
    _row("Expectancy/trade",   [f"{m['expectancy_usd']:+.2f}"                for m in metrics_list])
    _row("Avg win (USD)",      [f"{m['avg_win_usd']:+.2f}"                   for m in metrics_list])
    _row("Avg loss (USD)",     [f"{m['avg_loss_usd']:+.2f}"                  for m in metrics_list])
    _row("Max drawdown (USD)", [f"{m['max_drawdown_usd']:+.2f}"              for m in metrics_list])
    _row("Exit reasons",       [
        ", ".join(f"{k}:{v}" for k, v in sorted(m["exit_distribution"].items()))
        for m in metrics_list
    ])


def main() -> pd.DataFrame:
    print("Cargando datos...")
    trades_real, snaps_is, ohlc_m1 = load_data()
    print(f"  Trades reales IS: {len(trades_real)}  |  Snapshots: {len(snaps_is):,}")

    print("Generando señales v3...")
    signals = apply_strategy_v3_to_snapshots(snaps_is, StrategyParamsV3())
    signals["timestamp_utc"] = pd.to_datetime(signals["timestamp_utc"], utc=True)
    print(f"  Señales v3: {len(signals)}")

    config = BacktestConfig(
        sl_pips_buy=260.7, sl_pips_sell=342.8,
        tp_pips_buy=272.1, tp_pips_sell=289.8,
    )

    print("Ejecutando backtest...")
    bt = run_backtest(signals, ohlc_m1, config)
    print(f"  Trades ejecutados: {len(bt)}")

    print("Marcando matched vs extra...")
    bt = mark_matched_to_real(bt, trades_real)
    n_matched = int(bt["matched"].sum())
    n_extra   = len(bt) - n_matched
    print(f"  Matched: {n_matched}  |  Extra: {n_extra}")

    matched_df = bt[bt["matched"]].copy()
    extra_df   = bt[~bt["matched"]].copy()

    m_total   = compute_group_metrics(bt,         f"TOTAL ({len(bt)})")
    m_matched = compute_group_metrics(matched_df, f"MATCHED ({n_matched})")
    m_extra   = compute_group_metrics(extra_df,   f"EXTRA ({n_extra})")

    print()
    print("=" * 88)
    print("   SPLIT DIAGNOSTIC: MATCHED vs EXTRA — backtest v3 in-sample")
    print("=" * 88)
    print()
    print_table([m_total, m_matched, m_extra])
    print("=" * 88)

    # Interpretacion
    pnl_m = m_matched["pnl_total_usd"]
    pnl_e = m_extra["pnl_total_usd"]
    pf_m  = m_matched["profit_factor"]
    pf_e  = m_extra["profit_factor"]
    wr_m  = m_matched["win_rate_pct"]
    wr_e  = m_extra["win_rate_pct"]

    print()
    print("=== INTERPRETACION ===")
    if pnl_m > 0 and pnl_e < 0:
        print(f"  Matched rentables: +${pnl_m:.2f} | PF={pf_m:.2f} | WR={wr_m:.1f}%")
        print(f"  Extra pierden    :  ${pnl_e:.2f} | PF={pf_e:.2f} | WR={wr_e:.1f}%")
        print("  -> Las señales que el operador tomo son rentables.")
        print("  -> Las extras destruyen valor. Hay margen para añadir un filtro de calidad.")
    elif pnl_m > 0 and pnl_e > 0:
        if abs(pnl_m - pnl_e) / max(abs(pnl_m), abs(pnl_e)) < 0.3:
            print(f"  Ambos grupos rentables y similares: matched=${pnl_m:+.2f}, extra=${pnl_e:+.2f}.")
            print("  -> Sistema robusto: la logica es valida incluso en señales no tomadas por el operador.")
            print("  -> Pasar directamente a OOS.")
        else:
            print(f"  Matched: +${pnl_m:.2f} (PF={pf_m:.2f} WR={wr_m:.1f}%) "
                  f"| Extra: +${pnl_e:.2f} (PF={pf_e:.2f} WR={wr_e:.1f}%)")
            print("  -> Matched supera a Extra pero ambos positivos.")
            print("  -> El operador selecciona las mejores oportunidades, pero el sistema")
            print("     captura alpha incluso en sus descartes. Validar en OOS.")
    elif pnl_m <= 0:
        print(f"  Matched no rentable: ${pnl_m:.2f}")
        print("  -> El sistema no replica el alpha del operador con SL/TP fijos.")
        print("  -> Problema: el operador ajusta SL/TP dinamicamente; los fijos median no son suficientes.")

    # Guardar
    out_path = DATA_GROUND_TRUTH.parent / "backtest_v3_split.parquet"
    bt.to_parquet(out_path, index=False)
    print(f"\nGuardado en: {out_path}")

    return bt


if __name__ == "__main__":
    main()
