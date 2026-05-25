"""Ablation study: que aporta cada filtro?

Compara v2, v3, v3.1, v3.2 sobre los 30 trades in-sample.
Reporta tabla profesional con metricas comparables.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from v2.config.settings import DATA_FEATURES, DATA_GROUND_TRUTH
from v2.src.strategies.asian_momentum_v2 import (
    StrategyParamsV2,
    apply_strategy_v2_to_snapshots,
)
from v2.src.strategies.asian_momentum_v3 import (
    StrategyParamsV3,
    apply_strategy_v3_to_snapshots,
)
from v2.src.strategies.asian_momentum_v3_1 import (
    StrategyParamsV3_1,
    apply_strategy_v3_1_to_snapshots,
)

TOLERANCE_MIN = 60


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_parquet(DATA_GROUND_TRUTH / "trades.parquet")
    trades = trades[trades["sample"] == "in_sample"].copy()
    trades["time_open_utc"] = pd.to_datetime(trades["time_open_utc"], utc=True)

    snapshots = pd.read_parquet(DATA_FEATURES / "snapshots_M1.parquet")
    if snapshots.index.tz is None:
        snapshots.index = snapshots.index.tz_localize("UTC")

    t_start = trades["time_open_utc"].min() - pd.Timedelta(days=2)
    t_end   = trades["time_open_utc"].max() + pd.Timedelta(days=1)
    snaps_is = snapshots[(snapshots.index >= t_start) & (snapshots.index <= t_end)].copy()

    return trades, snaps_is


def compute_match_stats(signals_df: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Computa metricas de matching señales vs trades reales."""
    real_buy  = trades[trades["type"] == "BUY"]["time_open_utc"].sort_values().values
    real_sell = trades[trades["type"] == "SELL"]["time_open_utc"].sort_values().values

    matched: dict[str, set] = {"BUY": set(), "SELL": set()}
    drifts: list[float] = []

    for _, row in signals_df.iterrows():
        ts  = row["timestamp_utc"]
        arr = real_buy if row["signal"] == "BUY" else real_sell
        if len(arr) == 0:
            continue
        diffs_min = (arr.astype("datetime64[ns]") - np.datetime64(ts)) / np.timedelta64(1, "m")
        for i in np.where(np.abs(diffs_min) <= TOLERANCE_MIN)[0]:
            matched[row["signal"]].add(str(arr[i]))
            drifts.append(float(diffs_min[i]))

    n_buy_real  = int((trades["type"] == "BUY").sum())
    n_sell_real = int((trades["type"] == "SELL").sum())
    n_matched   = len(matched["BUY"]) + len(matched["SELL"])

    return {
        "signals_total":            len(signals_df),
        "signals_buy":              int((signals_df["signal"] == "BUY").sum())  if len(signals_df) else 0,
        "signals_sell":             int((signals_df["signal"] == "SELL").sum()) if len(signals_df) else 0,
        "matched_total":            n_matched,
        "matched_buy":              len(matched["BUY"]),
        "matched_sell":             len(matched["SELL"]),
        "recall_total":             n_matched / len(trades) * 100,
        "recall_buy":               len(matched["BUY"])  / n_buy_real  * 100,
        "recall_sell":              len(matched["SELL"]) / n_sell_real * 100,
        "ratio_signals_per_trade":  len(signals_df) / len(trades),
        "drift_median":             float(np.median(drifts)) if drifts else float("nan"),
        "drift_mean":               float(np.mean(drifts))   if drifts else float("nan"),
        "drift_n":                  len(drifts),
    }


def main() -> list[tuple]:
    print("Cargando datos...")
    trades, snaps_is = load_data()
    print(f"Trades in-sample: {len(trades)} | Snapshots: {len(snaps_is):,}")
    print()

    configs_raw: list[tuple[str, str, pd.DataFrame]] = []

    # ── v2 baseline ───────────────────────────────────────────────────────────
    print("Aplicando v2 baseline...")
    sig_v2 = apply_strategy_v2_to_snapshots(
        snaps_is, StrategyParamsV2(), deduplicate_within_minutes=30,
    )
    sig_v2["timestamp_utc"] = pd.to_datetime(sig_v2["timestamp_utc"], utc=True)
    configs_raw.append(("v2_baseline", "(sin filtros extra)", sig_v2))

    # ── v3 cooldown=90 ───────────────────────────────────────────────────────
    print("Aplicando v3 (horario + max2 + cooldown=90)...")
    sig_v3 = apply_strategy_v3_to_snapshots(snaps_is, StrategyParamsV3())
    configs_raw.append(("v3_cooldown90", "horario+max2+cooldown90", sig_v3))

    # ── v3.1 cooldown=0 ──────────────────────────────────────────────────────
    print("Aplicando v3.1 (horario + max2, sin cooldown)...")
    sig_v31 = apply_strategy_v3_1_to_snapshots(snaps_is, StrategyParamsV3_1())
    configs_raw.append(("v3_1_no_cooldown", "horario+max2", sig_v31))

    # ── v3.2 solo horario (max_trades=99, cooldown=0) ─────────────────────────
    print("Aplicando v3.2 (solo horario, sin otros filtros)...")
    p32 = StrategyParamsV3_1(max_trades_per_session_day=99, cooldown_minutes_within_day=0)
    sig_v32 = apply_strategy_v3_1_to_snapshots(snaps_is, p32)
    configs_raw.append(("v3_2_only_hours", "solo horario", sig_v32))

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    print()
    W = 104
    print("=" * W)
    print(" " * 30 + "ABLATION STUDY (in-sample, 30 trades)")
    print("=" * W)
    hdr = (f"{'Config':<22}{'Filtros':<28}{'Recall':>9}"
           f"{'BUYr':>7}{'SELLr':>7}{'Senales':>9}{'Ratio':>7}{'Drift med':>11}{'N drifts':>10}")
    print(hdr)
    print("-" * W)

    rows: list[tuple] = []
    for name, descr, sig in configs_raw:
        st = compute_match_stats(sig, trades)
        rows.append((name, descr, st))
        print(
            f"{name:<22}{descr:<28}{st['recall_total']:>8.1f}%"
            f"{st['recall_buy']:>6.0f}%{st['recall_sell']:>6.0f}%"
            f"{st['signals_total']:>9}{st['ratio_signals_per_trade']:>6.1f}x"
            f"{st['drift_median']:>10.1f}{st['drift_n']:>10}"
        )
    print("=" * W)

    # ── Analisis incremental ──────────────────────────────────────────────────
    def _get(n: str) -> dict:
        return next(st for nm, _, st in rows if nm == n)

    s_v2  = _get("v2_baseline")
    s_v3  = _get("v3_cooldown90")
    s_v31 = _get("v3_1_no_cooldown")
    s_v32 = _get("v3_2_only_hours")

    print()
    print("=== Aporte incremental de cada filtro ===")
    print(f"  Filtro horario  (v2 -> v3.2):  "
          f"recall {s_v2['recall_total']:.1f}% -> {s_v32['recall_total']:.1f}% "
          f"({s_v32['recall_total']-s_v2['recall_total']:+.1f} pp) | "
          f"senales {s_v2['signals_total']} -> {s_v32['signals_total']} "
          f"({s_v32['signals_total']-s_v2['signals_total']:+d})")
    print(f"  Filtro max2/dia (v3.2 -> v3.1): "
          f"recall {s_v32['recall_total']:.1f}% -> {s_v31['recall_total']:.1f}% "
          f"({s_v31['recall_total']-s_v32['recall_total']:+.1f} pp) | "
          f"senales {s_v32['signals_total']} -> {s_v31['signals_total']} "
          f"({s_v31['signals_total']-s_v32['signals_total']:+d})")
    print(f"  Filtro cooldown (v3.1 -> v3):  "
          f"recall {s_v31['recall_total']:.1f}% -> {s_v3['recall_total']:.1f}% "
          f"({s_v3['recall_total']-s_v31['recall_total']:+.1f} pp) | "
          f"senales {s_v31['signals_total']} -> {s_v3['signals_total']} "
          f"({s_v3['signals_total']-s_v31['signals_total']:+d})")

    print()
    print("=== Veredicto ===")
    print(f"  Candidato optimo: v3.1 — recall {s_v31['recall_total']:.1f}% | "
          f"{s_v31['signals_total']} senales | ratio {s_v31['ratio_signals_per_trade']:.1f}x")
    if s_v31["recall_total"] >= 85:
        print("  -> Recall >= 85%. Hipotesis confirmada. Pasar a backtest.")
    elif s_v31["recall_total"] >= 70:
        print("  -> Recall 70-85%. Hipotesis parcial. Revisar misses antes del backtest.")
    else:
        print("  -> Recall < 70%. Requiere revision de features o parametros.")

    return rows


if __name__ == "__main__":
    main()
