"""Validacion out-of-sample de la estrategia v3 (FINAL, NO MODIFICAR).

Aplica v3 sin cambios al periodo 28/04 -> 12/05/2026.
Reporta metricas comparables con el in-sample.
"""
import sys
from datetime import datetime, timezone
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

SPLIT_CUTOFF_UTC = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
TOLERANCE_MIN = 60


def load_oos_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = pd.read_parquet(DATA_GROUND_TRUTH / "trades.parquet")
    trades_oos = trades[trades["sample"] == "out_of_sample"].copy()
    trades_oos["time_open_utc"] = pd.to_datetime(trades_oos["time_open_utc"], utc=True)

    snapshots = pd.read_parquet(DATA_FEATURES / "snapshots_M1.parquet")
    if snapshots.index.tz is None:
        snapshots.index = snapshots.index.tz_localize("UTC")

    t_start   = trades_oos["time_open_utc"].min() - pd.Timedelta(days=2)
    t_end     = trades_oos["time_open_utc"].max() + pd.Timedelta(days=1)
    snaps_oos = snapshots[(snapshots.index >= t_start) & (snapshots.index <= t_end)].copy()

    ohlc_m1 = pd.read_parquet(DATA_GROUND_TRUTH / "ohlc_M1.parquet")
    if ohlc_m1.index.tz is None:
        ohlc_m1.index = ohlc_m1.index.tz_localize("UTC")
    ohlc_m1.columns = [c.lower() for c in ohlc_m1.columns]

    return trades_oos, snaps_oos, ohlc_m1


def mark_matched(bt_df: pd.DataFrame, real_trades: pd.DataFrame) -> pd.DataFrame:
    real_buy  = real_trades[real_trades["type"] == "BUY"]["time_open_utc"].sort_values().values
    real_sell = real_trades[real_trades["type"] == "SELL"]["time_open_utc"].sort_values().values

    def _is_matched(row: pd.Series) -> bool:
        ts  = pd.Timestamp(row["entry_time"]).to_datetime64().astype("datetime64[ns]")
        arr = real_buy if row["direction"] == "BUY" else real_sell
        if len(arr) == 0:
            return False
        diffs = (arr.astype("datetime64[ns]") - ts) / np.timedelta64(1, "m")
        return bool(np.any(np.abs(diffs) <= TOLERANCE_MIN))

    bt_df = bt_df.copy()
    bt_df["matched"] = bt_df.apply(_is_matched, axis=1)
    return bt_df


def compute_metrics(df: pd.DataFrame, name: str) -> dict:
    n = len(df)
    if n == 0:
        return {"name": name, "n": 0, "wins": 0, "losses": 0,
                "win_rate_pct": 0.0, "pnl_total_usd": 0.0,
                "profit_factor": float("nan"), "expectancy_usd": 0.0,
                "avg_win_usd": 0.0, "avg_loss_usd": 0.0,
                "max_drawdown_usd": 0.0, "exit_distribution": {}}

    wins   = int((df["net_pnl_usd"] > 0).sum())
    losses = int((df["net_pnl_usd"] <= 0).sum())
    gw     = float(df.loc[df["net_pnl_usd"] > 0,  "net_pnl_usd"].sum())
    gl     = float(df.loc[df["net_pnl_usd"] <= 0, "net_pnl_usd"].sum())
    pf     = abs(gw / gl) if gl != 0 else float("inf")

    g = df.sort_values("entry_time").copy()
    g["cum"] = g["net_pnl_usd"].cumsum()
    max_dd   = float((g["cum"] - g["cum"].cummax()).min())

    return {
        "name":              name,
        "n":                 n,
        "wins":              wins,
        "losses":            losses,
        "win_rate_pct":      wins / n * 100,
        "pnl_total_usd":     float(df["net_pnl_usd"].sum()),
        "profit_factor":     pf,
        "expectancy_usd":    float(df["net_pnl_usd"].mean()),
        "avg_win_usd":       float(df.loc[df["net_pnl_usd"] > 0,  "net_pnl_usd"].mean()) if wins   > 0 else 0.0,
        "avg_loss_usd":      float(df.loc[df["net_pnl_usd"] <= 0, "net_pnl_usd"].mean()) if losses > 0 else 0.0,
        "max_drawdown_usd":  max_dd,
        "exit_distribution": df["exit_reason"].value_counts().to_dict(),
    }


def main() -> tuple[pd.DataFrame, dict, dict]:
    print("=" * 70)
    print("     VALIDACION OOS — Aplicando v3 SIN CAMBIOS al periodo OOS")
    print("=" * 70)
    print()

    print("Cargando datos OOS (28/04 -> 12/05/2026)...")
    trades_oos, snaps_oos, ohlc_m1 = load_oos_data()
    n_buy_oos  = int((trades_oos["type"] == "BUY").sum())
    n_sell_oos = int((trades_oos["type"] == "SELL").sum())
    print(f"  Trades OOS reales : {len(trades_oos)}  ({n_buy_oos} BUY, {n_sell_oos} SELL)")
    print(f"  Snapshots OOS     : {len(snaps_oos):,}")
    print(f"  Periodo           : {trades_oos['time_open_utc'].min()} -> {trades_oos['time_open_utc'].max()}")

    print("\nAplicando v3 (parametros default, SIN MODIFICAR)...")
    signals_oos = apply_strategy_v3_to_snapshots(snaps_oos, StrategyParamsV3())
    signals_oos["timestamp_utc"] = pd.to_datetime(signals_oos["timestamp_utc"], utc=True)
    n_sig_buy  = int((signals_oos["signal"] == "BUY").sum())
    n_sig_sell = int((signals_oos["signal"] == "SELL").sum())
    print(f"  Señales generadas : {len(signals_oos)}  ({n_sig_buy} BUY, {n_sig_sell} SELL)")

    # SL/TP empíricos del in-sample (fijos, no re-estimados)
    config = BacktestConfig(
        sl_pips_buy=260.7, sl_pips_sell=342.8,
        tp_pips_buy=272.1, tp_pips_sell=289.8,
        slippage_entry_pips=0.5, slippage_exit_pips=0.5,
        spread_pips=3.0, volume_lots=0.01, max_hold_hours=12,
    )

    print("Ejecutando backtest OOS...")
    bt = run_backtest(signals_oos, ohlc_m1, config)
    print(f"  Trades ejecutados : {len(bt)}")

    if len(bt) == 0:
        print("\nATENCION: v3 NO genero ninguna señal ejecutable en OOS.")
        return pd.DataFrame(), {}, {}

    bt = mark_matched(bt, trades_oos)
    n_matched = int(bt["matched"].sum())
    n_extra   = len(bt) - n_matched

    # Recall OOS real
    real_buy  = trades_oos[trades_oos["type"] == "BUY"]["time_open_utc"].sort_values().values
    real_sell = trades_oos[trades_oos["type"] == "SELL"]["time_open_utc"].sort_values().values
    matched_real: set[str] = set()
    for _, row in bt.iterrows():
        ts  = pd.Timestamp(row["entry_time"]).to_datetime64().astype("datetime64[ns]")
        arr = real_buy if row["direction"] == "BUY" else real_sell
        if len(arr) == 0:
            continue
        diffs = (arr.astype("datetime64[ns]") - ts) / np.timedelta64(1, "m")
        for i in np.where(np.abs(diffs) <= TOLERANCE_MIN)[0]:
            matched_real.add(str(arr[i]))

    n_real_detected = len(matched_real)
    recall_oos = n_real_detected / len(trades_oos) * 100

    misses = [
        t for _, t in trades_oos.iterrows()
        if str(t["time_open_utc"].to_datetime64()) not in matched_real
    ]
    misses_df = pd.DataFrame(misses)

    # Metricas
    m_total   = compute_metrics(bt,                         f"TOTAL OOS ({len(bt)})")
    m_matched = compute_metrics(bt[bt["matched"]].copy(),   f"MATCHED OOS ({n_matched})")
    m_extra   = compute_metrics(bt[~bt["matched"]].copy(),  f"EXTRA OOS ({n_extra})")

    real_oos_pnl     = float(trades_oos["profit"].sum())
    real_oos_wr      = float((trades_oos["profit"] > 0).mean() * 100)

    # ── Reporte ───────────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("           RESULTADO OOS — v3 sin modificaciones")
    print("=" * 90)
    print()
    print(f"RECALL OOS          : {n_real_detected}/{len(trades_oos)} trades reales detectados ({recall_oos:.1f}%)")
    print(f"Señales generadas   : {len(signals_oos)}  | Ejecutadas: {len(bt)}")
    print(f"Matched / Extra     : {n_matched} / {n_extra}")
    print()
    print(f"--- BACKTEST OOS TOTAL ({len(bt)} trades) ---")
    print(f"  Wins / Losses     : {m_total['wins']} / {m_total['losses']}")
    print(f"  Win rate          : {m_total['win_rate_pct']:.1f}%")
    print(f"  PnL net USD       : {m_total['pnl_total_usd']:+.2f}")
    print(f"  Profit factor     : {m_total['profit_factor']:.2f}")
    print(f"  Expectancy/trade  : {m_total['expectancy_usd']:+.2f}")
    print(f"  Max drawdown      : {m_total['max_drawdown_usd']:+.2f}")
    print(f"  Exit reasons      : {m_total['exit_distribution']}")
    print()
    print(f"--- BACKTEST OOS MATCHED ({n_matched} trades) ---")
    print(f"  Wins / Losses     : {m_matched['wins']} / {m_matched['losses']}")
    print(f"  Win rate          : {m_matched['win_rate_pct']:.1f}%")
    print(f"  PnL net USD       : {m_matched['pnl_total_usd']:+.2f}")
    print(f"  Profit factor     : {m_matched['profit_factor']:.2f}")
    print(f"  Exit reasons      : {m_matched['exit_distribution']}")
    print()
    print(f"--- BACKTEST OOS EXTRAS ({n_extra} trades) ---")
    print(f"  Wins / Losses     : {m_extra['wins']} / {m_extra['losses']}")
    print(f"  Win rate          : {m_extra['win_rate_pct']:.1f}%")
    print(f"  PnL net USD       : {m_extra['pnl_total_usd']:+.2f}")
    print(f"  Profit factor     : {m_extra['profit_factor']:.2f}")
    print()
    print(f"--- OPERADOR REAL OOS ({len(trades_oos)} trades) ---")
    print(f"  Win rate          : {real_oos_wr:.1f}%")
    print(f"  PnL net USD       : {real_oos_pnl:+.2f}")

    print()
    print("=" * 90)
    print("           COMPARATIVA IN-SAMPLE vs OUT-OF-SAMPLE")
    print("=" * 90)
    W = 30
    print(f"{'Metrica':<{W}}{'IN-SAMPLE':>22}{'OUT-OF-SAMPLE':>22}")
    print("-" * (W + 44))
    rows = [
        ("Trades reales",                "30",              str(len(trades_oos))),
        ("Recall regla",                 "73.3% (22/30)",   f"{recall_oos:.1f}% ({n_real_detected}/{len(trades_oos)})"),
        ("Trades ejecutados backtest",   "54",              str(len(bt))),
        ("Matched / Extra",              "21 / 33",         f"{n_matched} / {n_extra}"),
        ("Win rate total",               "55.6%",           f"{m_total['win_rate_pct']:.1f}%"),
        ("PF total backtest",            "1.15",            f"{m_total['profit_factor']:.2f}"),
        ("PF MATCHED",                   "2.64",            f"{m_matched['profit_factor']:.2f}"),
        ("PnL total backtest",           "+$107.36",        f"{m_total['pnl_total_usd']:+.2f}"),
        ("PnL operador real",            "+$421.40",        f"{real_oos_pnl:+.2f}"),
        ("Max drawdown",                 "-$140.99",        f"{m_total['max_drawdown_usd']:+.2f}"),
    ]
    for label, is_val, oos_val in rows:
        print(f"{label:<{W}}{is_val:>22}{oos_val:>22}")

    # ── Veredicto ─────────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("           VEREDICTO")
    print("=" * 90)
    pf_total   = m_total["profit_factor"]
    pf_matched = m_matched["profit_factor"]
    pnl_total  = m_total["pnl_total_usd"]

    if pf_matched >= 1.5 and recall_oos >= 50:
        print("  OOS VALIDA LA ESTRATEGIA")
        print(f"    - Recall OOS solido ({recall_oos:.0f}% >= 50%)")
        print(f"    - PF MATCHED OOS ({pf_matched:.2f}) confirma el alpha")
        print(f"    - El operador es replicable mecanicamente con esta regla")
        print(f"    - PROXIMO PASO: implementar EA MQL5 y testear en demo")
    elif pf_total >= 1.0 and recall_oos >= 50:
        print("  OOS PARCIALMENTE VALIDA")
        print(f"    - Recall OOS aceptable ({recall_oos:.0f}%)")
        print(f"    - PF total OOS modesto ({pf_total:.2f})")
        print(f"    - La estrategia funciona pero gana poco neto de costos")
        print(f"    - PROXIMO PASO: explorar SL/TP dinamicos antes de EA")
    else:
        print("  OOS NO VALIDA")
        print(f"    - PF OOS < 1 o recall < 50%: estrategia no generaliza")
        print(f"    - El edge in-sample era artificial o sobreajustado")
        print(f"    - PROXIMO PASO: revisar hipotesis o aceptar no replicable")

    print()
    if len(misses_df) > 0:
        print(f"=== TRADES OOS SIN MATCH ({len(misses_df)}) ===")
        cols = ["time_open_utc", "type", "price_open", "sl_initial", "tp_initial", "profit"]
        print(misses_df[cols].to_string(index=False))

    out_path = DATA_GROUND_TRUTH.parent / "backtest_v3_oos.parquet"
    bt.to_parquet(out_path, index=False)
    print(f"\nResultados guardados en: {out_path}")

    return bt, m_total, m_matched


if __name__ == "__main__":
    main()
