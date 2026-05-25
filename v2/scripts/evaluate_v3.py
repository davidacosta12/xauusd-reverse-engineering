"""Aplica v3, hace matching con trades reales, imprime tabla comparativa v1/v2/v3."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from v2.config.settings import DATA_GROUND_TRUTH, DATA_FEATURES
from v2.src.strategies.asian_momentum_v3 import (
    StrategyParamsV3,
    apply_strategy_v3_to_snapshots,
)

TOLERANCE_MIN = 60


def evaluate() -> tuple[pd.DataFrame, np.ndarray, float]:
    # ── Cargar trades in-sample ───────────────────────────────────────────────
    trades = pd.read_parquet(DATA_GROUND_TRUTH / "trades.parquet")
    trades = trades[trades["sample"] == "in_sample"].copy()
    trades["time_open_utc"] = pd.to_datetime(trades["time_open_utc"], utc=True)
    n_buys_real  = int((trades["type"] == "BUY").sum())
    n_sells_real = int((trades["type"] == "SELL").sum())

    # ── Cargar snapshots in-sample ────────────────────────────────────────────
    snapshots = pd.read_parquet(DATA_FEATURES / "snapshots_M1.parquet")
    if snapshots.index.tz is None:
        snapshots.index = snapshots.index.tz_localize("UTC")
    t_start = trades["time_open_utc"].min() - pd.Timedelta(days=2)
    t_end   = trades["time_open_utc"].max() + pd.Timedelta(days=1)
    snaps_is = snapshots[(snapshots.index >= t_start) & (snapshots.index <= t_end)].copy()
    print(f"Snapshots in-sample: {len(snaps_is):,} filas")

    # ── Aplicar v3 ────────────────────────────────────────────────────────────
    params = StrategyParamsV3()
    print(f"Parametros v3:")
    print(f"  core_session            : {params.core_session_start_hour}h - {params.core_session_end_hour}h UTC")
    print(f"  max_trades_per_day      : {params.max_trades_per_session_day}")
    print(f"  cooldown_minutes        : {params.cooldown_minutes_within_day}")
    print(f"  min_distance_asian_mid  : {params.min_distance_from_asian_mid_pips} pips")
    print(f"  bb_pct_b thresholds     : buy>{params.bb_pct_b_buy_threshold} sell<{params.bb_pct_b_sell_threshold}")
    print("Aplicando v3 ...")
    signals = apply_strategy_v3_to_snapshots(snaps_is, params)
    n_sig      = len(signals)
    n_buy_sig  = int((signals["signal"] == "BUY").sum())  if n_sig > 0 else 0
    n_sell_sig = int((signals["signal"] == "SELL").sum()) if n_sig > 0 else 0

    # ── Matching: señal dentro de ±TOLERANCE_MIN de un trade real ─────────────
    real_buy  = trades[trades["type"] == "BUY"]["time_open_utc"].sort_values().values
    real_sell = trades[trades["type"] == "SELL"]["time_open_utc"].sort_values().values

    matched_keys: dict[str, set] = {"BUY": set(), "SELL": set()}
    drifts: list[float] = []

    for _, row in signals.iterrows():
        ts  = row["timestamp_utc"]
        arr = real_buy if row["signal"] == "BUY" else real_sell
        if len(arr) == 0:
            continue
        diffs_min = (arr.astype("datetime64[ns]") - np.datetime64(ts)) / np.timedelta64(1, "m")
        for i in np.where(np.abs(diffs_min) <= TOLERANCE_MIN)[0]:
            matched_keys[row["signal"]].add(str(arr[i]))
            drifts.append(float(diffs_min[i]))

    n_match_buy  = len(matched_keys["BUY"])
    n_match_sell = len(matched_keys["SELL"])
    n_match      = n_match_buy + n_match_sell
    buy_recall   = n_match_buy  / n_buys_real  * 100 if n_buys_real  > 0 else 0.0
    sell_recall  = n_match_sell / n_sells_real * 100 if n_sells_real > 0 else 0.0
    total_recall = n_match / len(trades) * 100

    drift_arr    = np.array(drifts)
    drift_median = float(np.median(drift_arr)) if len(drift_arr) > 0 else float("nan")
    drift_mean   = float(drift_arr.mean())     if len(drift_arr) > 0 else float("nan")
    drift_p10    = float(np.percentile(drift_arr, 10)) if len(drift_arr) > 0 else float("nan")
    drift_p90    = float(np.percentile(drift_arr, 90)) if len(drift_arr) > 0 else float("nan")
    ratio        = n_sig / len(trades)

    # ── Imprimir resultados ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("           EVALUACION ESTRATEGIA v3 (in-sample)")
    print("=" * 70)
    print(f"Trades in-sample            : {len(trades)} ({n_buys_real} BUY, {n_sells_real} SELL)")
    print(f"Senales v3 generadas        : {n_sig} ({n_buy_sig} BUY, {n_sell_sig} SELL)")
    print(f"Ratio senales/trades        : {ratio:.1f}x")
    print()
    print(f"BUY recall                  : {n_match_buy}/{n_buys_real} ({buy_recall:.1f}%)")
    print(f"SELL recall                 : {n_match_sell}/{n_sells_real} ({sell_recall:.1f}%)")
    print(f"TOTAL recall                : {n_match}/{len(trades)} ({total_recall:.1f}%)")
    print()
    if len(drift_arr) > 0:
        print("DRIFT temporal (senal->trade, minutos):")
        print(f"  Media   : {drift_mean:+.1f}")
        print(f"  Mediana : {drift_median:+.1f}")
        print(f"  P10/P90 : {drift_p10:+.1f} / {drift_p90:+.1f}")
    print()
    print("Comparativa historica:")
    print(f"  v1: 40.0% recall | ~252 senales | ratio ~8.4x")
    print(f"  v2: 96.7% recall |  231 senales | ratio  7.7x | mediana  5.0 min")
    print(f"  v3: {total_recall:.1f}% recall | {n_sig:>4d} senales | ratio {ratio:>4.1f}x | mediana {drift_median:+.1f} min")
    print("=" * 70)

    # ── Trades sin match ──────────────────────────────────────────────────────
    def _is_matched(r: pd.Series) -> bool:
        return str(r["time_open_utc"].to_datetime64()) in matched_keys[r["type"]]

    misses = trades[~trades.apply(_is_matched, axis=1)]
    if len(misses) > 0:
        print(f"\n{len(misses)} TRADES SIN MATCH:")
        print(misses[["time_open_utc", "type", "price_open"]].to_string(index=False))
    else:
        print("\nSin misses — todos los trades detectados.")

    # ── Distribucion horaria ──────────────────────────────────────────────────
    if n_sig > 0:
        signals["hour_utc"] = signals["timestamp_utc"].dt.hour
        print("\nDistribucion horaria de senales v3:")
        hour_dist = signals.groupby(["hour_utc", "signal"]).size().unstack(fill_value=0)
        print(hour_dist.to_string())

    return signals, drift_arr, total_recall


if __name__ == "__main__":
    evaluate()
