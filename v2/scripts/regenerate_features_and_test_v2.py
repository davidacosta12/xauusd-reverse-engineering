"""Regenera snapshots_M1.parquet con asian levels rolling y evalúa estrategia v2.

Pasos:
1. Carga OHLC ground truth (M1, M5, M15, H1, H4)
2. Recalcula TODOS los features (replicando notebook 01) con la nueva
   compute_asian_levels rolling
3. Guarda snapshots_M1.parquet actualizado
4. Aplica estrategia v2 a los snapshots in-sample
5. Hace matching contra los 30 trades reales (±60 min)
6. Imprime recall v1 (histórico hardcoded) vs recall v2
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

from v2.config.settings import DATA_GROUND_TRUTH, DATA_FEATURES
from v2.src.features.levels import (
    compute_pdh_pdl, compute_weekly_levels, compute_monthly_levels,
    compute_asian_levels, compute_distances_to_levels,
)
from v2.src.features.microstructure import compute_microstructure_features
from v2.src.features.technical import compute_technical_indicators
from v2.src.strategies.asian_momentum_v2 import (
    StrategyParamsV2, apply_strategy_v2_to_snapshots,
)

RECALL_V1 = 40.0  # resultado histórico de la estrategia v1 (hardcoded)
TOLERANCE_MINUTES = 60


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_gt(tf: str) -> pd.DataFrame:
    path = DATA_GROUND_TRUTH / f"ohlc_{tf}.parquet"
    if not path.exists():
        logger.warning("No encontrado: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    logger.info("  %s: %d barras", tf, len(df))
    return df


def _align_to_base(base: pd.DataFrame, feat: pd.DataFrame, name: str) -> pd.DataFrame:
    """Alinea feat al índice de base usando merge_asof backward."""
    if feat.empty:
        return pd.DataFrame(index=base.index)
    left = pd.DataFrame({"_ts": base.index}).sort_values("_ts")
    right = feat.reset_index()
    ts_col = right.columns[0]
    right = right.rename(columns={ts_col: "_ts_feat"}).sort_values("_ts_feat")
    merged = pd.merge_asof(
        left, right, left_on="_ts", right_on="_ts_feat", direction="backward",
    ).drop(columns=["_ts_feat"])
    merged.index = base.index
    result = merged.drop(columns=["_ts"])
    logger.info("  ✓ %s: %d columnas", name, result.shape[1])
    return result


def _match_signals_to_trades(signals_df: pd.DataFrame, trades_is: pd.DataFrame) -> pd.DataFrame:
    """Para cada trade in-sample, busca la señal del mismo tipo más cercana (±TOLERANCE_MINUTES)."""
    rows = []
    for _, trade in trades_is.iterrows():
        t_open = trade["time_open_utc"]
        t_type = trade["type"]
        same = signals_df[signals_df["signal"] == t_type].copy()
        same["diff_min"] = (same["timestamp_utc"] - t_open).dt.total_seconds() / 60
        window = same[same["diff_min"].abs() <= TOLERANCE_MINUTES]
        if len(window) > 0:
            closest = window.loc[window["diff_min"].abs().idxmin()]
            rows.append({
                "position_id": trade["position_id"],
                "time_open_utc": t_open,
                "type": t_type,
                "nearest_signal_ts": closest["timestamp_utc"],
                "diff_min": closest["diff_min"],
                "match": True,
            })
        else:
            rows.append({
                "position_id": trade["position_id"],
                "time_open_utc": t_open,
                "type": t_type,
                "nearest_signal_ts": pd.NaT,
                "diff_min": np.nan,
                "match": False,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Cargar OHLC ────────────────────────────────────────────────────────
    logger.info("=== Cargando OHLC ground truth ===")
    m1  = _load_gt("M1")
    m5  = _load_gt("M5")
    m15 = _load_gt("M15")
    h1  = _load_gt("H1")
    h4  = _load_gt("H4")

    if m1.empty or m15.empty:
        raise SystemExit("M1 o M15 vacíos. Ejecuta el notebook 00 primero.")

    # ── 2. Calcular features ──────────────────────────────────────────────────
    logger.info("=== Calculando features ===")

    logger.info("Niveles institucionales...")
    pdh_pdl  = compute_pdh_pdl(m15)
    weekly   = compute_weekly_levels(m15)
    monthly  = compute_monthly_levels(m15)
    asian    = compute_asian_levels(m15, m5_df=m5 if not m5.empty else None)
    all_levels = pd.concat([pdh_pdl, weekly, monthly, asian], axis=1)

    logger.info("Distancias a niveles...")
    distances = compute_distances_to_levels(m15, all_levels)

    logger.info("Microestructura...")
    micro: dict[str, pd.DataFrame] = {}
    for tf, df in [("m5", m5), ("m15", m15), ("h1", h1)]:
        if not df.empty:
            micro[tf] = compute_microstructure_features(df, prefix=tf)

    logger.info("Indicadores técnicos...")
    tech: dict[str, pd.DataFrame] = {}
    for tf, df in [("m15", m15), ("h1", h1), ("h4", h4)]:
        if not df.empty:
            tech[tf] = compute_technical_indicators(df, prefix=tf)

    # ── 3. Alinear al índice M1 y guardar ────────────────────────────────────
    logger.info("=== Alineando al índice M1 ===")
    base = m1
    blocks: list[pd.DataFrame] = [
        _align_to_base(base, all_levels, "levels"),
        _align_to_base(base, distances, "distances"),
    ]
    for name, df in micro.items():
        blocks.append(_align_to_base(base, df, f"micro_{name}"))
    for name, df in tech.items():
        blocks.append(_align_to_base(base, df, f"tech_{name}"))

    snapshots = pd.concat([b for b in blocks if not b.empty], axis=1)
    snapshots = snapshots.loc[:, ~snapshots.columns.duplicated()]

    out_path = DATA_FEATURES / "snapshots_M1.parquet"
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(out_path)
    size_mb = out_path.stat().st_size / 1024**2
    logger.info("snapshots_M1.parquet guardado: %d filas × %d cols (%.1f MB)",
                len(snapshots), snapshots.shape[1], size_mb)

    # ── 4. Cargar trades in-sample ────────────────────────────────────────────
    trades_all = pd.read_parquet(DATA_GROUND_TRUTH / "trades.parquet")
    trades_is = trades_all[trades_all["sample"] == "in_sample"].copy()
    trades_is["time_open_utc"] = pd.to_datetime(trades_is["time_open_utc"], utc=True)
    n_buy  = (trades_is["type"] == "BUY").sum()
    n_sell = (trades_is["type"] == "SELL").sum()
    logger.info("Trades in-sample: %d (BUY=%d SELL=%d)", len(trades_is), n_buy, n_sell)

    # ── 5. Filtrar snapshots al periodo in-sample ─────────────────────────────
    IS_START = trades_is["time_open_utc"].min() - pd.Timedelta(hours=1)
    IS_END   = trades_is["time_open_utc"].max() + pd.Timedelta(hours=1)
    snaps_is = snapshots.loc[IS_START:IS_END]
    logger.info("Snapshots in-sample: %d filas", len(snaps_is))

    # ── 6. Aplicar estrategia v2 ──────────────────────────────────────────────
    params = StrategyParamsV2()
    logger.info("=== Aplicando estrategia v2 (puede tardar ~30s) ===")
    signals_df = apply_strategy_v2_to_snapshots(snaps_is, params, deduplicate_within_minutes=30)
    logger.info("Señales generadas v2 (tras dedup 30min): %d", len(signals_df))

    # ── 7. Matching signals vs trades ─────────────────────────────────────────
    match_report = _match_signals_to_trades(signals_df, trades_is)

    matched   = match_report[match_report["match"]]
    n_match   = len(matched)
    n_trades  = len(trades_is)

    n_match_buy  = (matched["type"] == "BUY").sum()
    n_match_sell = (matched["type"] == "SELL").sum()

    recall_total = n_match / n_trades * 100
    recall_buy   = n_match_buy  / n_buy  * 100 if n_buy  > 0 else 0.0
    recall_sell  = n_match_sell / n_sell * 100 if n_sell > 0 else 0.0

    diffs = matched["diff_min"].dropna()
    drift_mean   = diffs.mean()   if len(diffs) > 0 else float("nan")
    drift_median = diffs.median() if len(diffs) > 0 else float("nan")

    improvement = recall_total - RECALL_V1

    # ── 8. Imprimir resultado ─────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("     COMPARATIVA v1 -> v2")
    print("=" * 55)
    print(f"  Recall v1 (histórico)        : {RECALL_V1:.1f}%")
    print(f"  Recall v2 total              : {recall_total:.1f}%  ({n_match}/{n_trades})")
    print(f"    BUY recall                 : {recall_buy:.1f}%  ({n_match_buy}/{n_buy})")
    print(f"    SELL recall                : {recall_sell:.1f}%  ({n_match_sell}/{n_sell})")
    print(f"  Señales totales v2           : {len(signals_df)}")
    print(f"    BUY                        : {(signals_df['signal']=='BUY').sum()}")
    print(f"    SELL                       : {(signals_df['signal']=='SELL').sum()}")
    print(f"  Drift temporal (media)       : {drift_mean:.1f} min")
    print(f"  Drift temporal (mediana)     : {drift_median:.1f} min")
    print(f"  Mejora vs v1                 : {improvement:+.1f} pp")
    print("=" * 55)

    if recall_total >= 60:
        print("  VEREDICTO: Hipótesis v2 viable. Pasar a refinar.")
    elif recall_total >= 30:
        print("  VEREDICTO: Hipótesis v2 parcial. Analizar misses.")
    else:
        print("  VEREDICTO: Hipótesis v2 inadecuada. Revisar features.")
    print("=" * 55)

    # Detalle de misses
    missed = match_report[~match_report["match"]]
    if len(missed) > 0:
        print(f"\nTrades SIN match ({len(missed)}):")
        print(missed[["time_open_utc", "type"]].to_string(index=False))


if __name__ == "__main__":
    main()
