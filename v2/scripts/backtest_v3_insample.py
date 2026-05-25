"""Backtest profesional v3 in-sample.

Ejecuta las 54 señales v3, simula SL/TP con costos reales, y reporta metricas.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from v2.config.settings import DATA_FEATURES, DATA_GROUND_TRUTH
from v2.src.backtest.engine import BacktestConfig, PIP, run_backtest
from v2.src.strategies.asian_momentum_v3 import (
    StrategyParamsV3,
    apply_strategy_v3_to_snapshots,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades_all = pd.read_parquet(DATA_GROUND_TRUTH / "trades.parquet")
    trades_is  = trades_all[trades_all["sample"] == "in_sample"].copy()
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


def compute_metrics(bt: pd.DataFrame, real_trades: pd.DataFrame) -> dict:
    if len(bt) == 0:
        return {}

    bt = bt.sort_values("entry_time").copy()
    bt["cum_pnl"] = bt["net_pnl_usd"].cumsum()

    n      = len(bt)
    wins   = int((bt["net_pnl_usd"] > 0).sum())
    losses = int((bt["net_pnl_usd"] <= 0).sum())

    gross_wins   = bt.loc[bt["net_pnl_usd"] > 0,  "net_pnl_usd"].sum()
    gross_losses = bt.loc[bt["net_pnl_usd"] <= 0, "net_pnl_usd"].sum()
    pf = abs(gross_wins / gross_losses) if gross_losses != 0 else float("inf")

    running_max = bt["cum_pnl"].cummax()
    max_dd      = float((bt["cum_pnl"] - running_max).min())

    ret     = bt["net_pnl_usd"]
    sharpe  = float(ret.mean() / ret.std()) if ret.std() > 0 else 0.0

    return {
        "n_trades":          n,
        "wins":              wins,
        "losses":            losses,
        "win_rate_pct":      wins / n * 100,
        "profit_total_usd":  float(bt["net_pnl_usd"].sum()),
        "profit_factor":     pf,
        "avg_win_usd":       float(bt.loc[bt["net_pnl_usd"] > 0,  "net_pnl_usd"].mean()) if wins   > 0 else 0.0,
        "avg_loss_usd":      float(bt.loc[bt["net_pnl_usd"] <= 0, "net_pnl_usd"].mean()) if losses > 0 else 0.0,
        "expectancy_usd":    float(ret.mean()),
        "max_drawdown_usd":  max_dd,
        "sharpe_per_trade":  sharpe,
        "exit_dist":         bt["exit_reason"].value_counts().to_dict(),
        "real_pnl":          float(real_trades["profit"].sum()),
        "real_win_rate":     float((real_trades["profit"] > 0).mean() * 100),
    }


def main() -> tuple[pd.DataFrame, dict]:
    print("Cargando datos...")
    trades_real, snaps_is, ohlc_m1 = load_data()
    print(f"Trades reales in-sample : {len(trades_real)}")
    print(f"Snapshots in-sample     : {len(snaps_is):,}")
    print(f"Barras M1 disponibles   : {len(ohlc_m1):,}")

    print("\nGenerando señales v3...")
    signals = apply_strategy_v3_to_snapshots(snaps_is, StrategyParamsV3())
    signals["timestamp_utc"] = pd.to_datetime(signals["timestamp_utc"], utc=True)
    print(f"Señales v3              : {len(signals)}")

    # SL/TP empíricos del análisis previo (medianas del operador)
    config = BacktestConfig(
        sl_pips_buy=260.7,
        sl_pips_sell=342.8,
        tp_pips_buy=272.1,
        tp_pips_sell=289.8,
        slippage_entry_pips=0.5,
        slippage_exit_pips=0.5,
        spread_pips=3.0,
        volume_lots=0.01,
        max_hold_hours=12,
    )

    print("\nConfig backtest:")
    print(f"  SL BUY  : {config.sl_pips_buy:.1f} pips  |  TP BUY  : {config.tp_pips_buy:.1f} pips")
    print(f"  SL SELL : {config.sl_pips_sell:.1f} pips  |  TP SELL : {config.tp_pips_sell:.1f} pips")
    print(f"  Slippage: {config.slippage_entry_pips}/{config.slippage_exit_pips} pips (entry/exit)")
    print(f"  Spread  : {config.spread_pips} pips round-trip")
    print(f"  Volume  : {config.volume_lots} lotes (1 oz, $0.10/pip)")
    print(f"  Max hold: {config.max_hold_hours}h")

    print("\nEjecutando backtest...")
    bt = run_backtest(signals, ohlc_m1, config)
    print(f"Trades ejecutados       : {len(bt)}")

    metrics = compute_metrics(bt, trades_real)

    print()
    print("=" * 70)
    print("      BACKTEST v3 IN-SAMPLE (con costos realistas)")
    print("=" * 70)
    print(f"Trades ejecutados        : {metrics['n_trades']}")
    print(f"Wins / Losses            : {metrics['wins']} / {metrics['losses']}")
    print(f"Win rate                 : {metrics['win_rate_pct']:.1f}%")
    print(f"Profit total NET (USD)   : {metrics['profit_total_usd']:+.2f}")
    print(f"Profit factor            : {metrics['profit_factor']:.2f}")
    print(f"Expectancy por trade     : {metrics['expectancy_usd']:+.2f} USD")
    print(f"Avg win / Avg loss       : {metrics['avg_win_usd']:+.2f} / {metrics['avg_loss_usd']:+.2f}")
    print(f"Max drawdown (USD)       : {metrics['max_drawdown_usd']:+.2f}")
    print(f"Sharpe por trade         : {metrics['sharpe_per_trade']:.3f}")
    print(f"Exit reasons             : {metrics['exit_dist']}")
    print()
    print("=== Comparacion vs OPERADOR REAL ===")
    print(f"Operador real PnL total  : {metrics['real_pnl']:+.2f}")
    print(f"Backtest v3 PnL total    : {metrics['profit_total_usd']:+.2f}")
    print(f"Diferencia               : {metrics['profit_total_usd'] - metrics['real_pnl']:+.2f}")
    print(f"Operador real win rate   : {metrics['real_win_rate']:.1f}%")
    print(f"Backtest v3 win rate     : {metrics['win_rate_pct']:.1f}%")
    print("=" * 70)

    # Tabla de trades detallada
    print("\n=== DETALLE DE TRADES (primeros 15) ===")
    cols = ["entry_time", "direction", "entry_price", "sl_price", "tp_price",
            "exit_price", "exit_reason", "duration_min", "gross_pips", "net_pnl_usd"]
    print(bt[cols].head(15).to_string(index=False))

    # Guardar
    out_path = DATA_GROUND_TRUTH.parent / "data" / "backtest_v3_insample.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(out_path, index=False)
    print(f"\nResultados guardados en: {out_path}")

    return bt, metrics


if __name__ == "__main__":
    main()
