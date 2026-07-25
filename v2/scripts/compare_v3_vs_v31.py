"""Backtest comparativo v3 (cooldown=90 min) vs v3.1 (cooldown=0).

Lee DIRECTAMENTE del cache Twelve Data (sin API calls).
Procesa un periodo a la vez para mantener RAM < 5 GB.

NO modifica ningun archivo validado. Solo crea resultados en
v2/data/comparison_v3_vs_v31/.
"""
import gc
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import psutil

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

from v2.src.backtest.engine import BacktestConfig, run_backtest
from v2.src.backtest.metrics import compute_full_metrics, print_metrics_report
from v2.src.features.pipeline import build_snapshots
from v2.src.strategies.asian_momentum_v3 import (
    StrategyParamsV3,
    apply_strategy_v3_to_snapshots,
)
from v2.src.strategies.asian_momentum_v3_1 import (
    StrategyParamsV3_1,
    apply_strategy_v3_1_to_snapshots,
)

# ── Configuracion ──────────────────────────────────────────────────────────────
INITIAL_BALANCE = 2000.0
CONFIG = BacktestConfig(
    sl_pips_buy=260.7,
    sl_pips_sell=342.8,
    tp_pips_buy=272.1,
    tp_pips_sell=289.8,
    slippage_entry_pips=0.5,
    slippage_exit_pips=0.5,
    spread_pips=3.0,
    volume_lots=0.02,
    max_hold_hours=12,
)

CACHE_DIR = Path("v2/data/twelve_data")
INTERVAL_FILENAMES = {
    "1min":  "XAU_USD_1min.parquet",
    "5min":  "XAU_USD_5min.parquet",
    "15min": "XAU_USD_15min.parquet",
    "1h":    "XAU_USD_1h.parquet",
    "4h":    "XAU_USD_4h.parquet",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def load_from_cache(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Lee todos los timeframes del cache local, filtrados al rango solicitado."""
    ohlc: dict[str, pd.DataFrame] = {}
    for key, fname in INTERVAL_FILENAMES.items():
        fpath = CACHE_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Cache no encontrado: {fpath}")
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index().loc[start:end]
        logger.info("  %s: %d velas (%s -> %s)",
                    key, len(df),
                    df.index.min().date() if len(df) > 0 else "?",
                    df.index.max().date() if len(df) > 0 else "?")
        ohlc[key] = df
    return ohlc


def prep_m1_for_engine(m1: pd.DataFrame) -> pd.DataFrame:
    """Prepara M1 para run_backtest: tick_volume en lugar de volume."""
    m1_bt = m1.copy()
    if "volume" in m1_bt.columns and "tick_volume" not in m1_bt.columns:
        m1_bt = m1_bt.rename(columns={"volume": "tick_volume"})
    return m1_bt


def run_period(
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga, construye snapshots, aplica v3 y v3.1, ejecuta backtests.

    Returns (trades_v3, trades_v31). Si RAM < 4 GB al inicio, devuelve DataFrames vacios.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  PERIODO: {label} ({start.date()} -> {end.date()})")
    print(f"  RAM disponible: {ram_gb():.1f} GB")
    print(sep)

    free = ram_gb()
    if free < 4.0:
        logger.warning("RAM INSUFICIENTE: %.1f GB < 4 GB requeridos. Saltando periodo.", free)
        return pd.DataFrame(), pd.DataFrame()

    # 1. Cargar desde cache
    logger.info("Cargando datos desde cache (sin API calls)...")
    ohlc = load_from_cache(start, end)

    m1 = ohlc["1min"]
    if m1.empty:
        logger.warning("Sin datos M1 para el periodo. Saltando.")
        return pd.DataFrame(), pd.DataFrame()

    m1_bt = prep_m1_for_engine(m1)

    # 2. Construir snapshots (paso de mayor RAM)
    logger.info("Construyendo snapshots (10-20 min estimado)...")
    snapshots = build_snapshots(
        m1=ohlc["1min"],
        m5=ohlc["5min"],
        m15=ohlc["15min"],
        h1=ohlc["1h"],
        h4=ohlc["4h"],
    )
    logger.info("Snapshots: %d filas x %d cols | RAM: %.1f GB",
                len(snapshots), snapshots.shape[1], ram_gb())

    # Liberar raw OHLC (ya incorporado en snapshots)
    del ohlc, m1
    gc.collect()

    # 3. Aplicar estrategias (solo filtrado, muy rapido)
    logger.info("Aplicando v3 (cooldown=90 min)...")
    params_v3 = StrategyParamsV3()
    sigs_v3 = apply_strategy_v3_to_snapshots(snapshots, params_v3)
    sigs_v3["timestamp_utc"] = pd.to_datetime(sigs_v3["timestamp_utc"], utc=True)
    n_buy_v3  = int((sigs_v3["signal"] == "BUY").sum())
    n_sell_v3 = int((sigs_v3["signal"] == "SELL").sum())
    logger.info("  Señales v3: %d (%d BUY, %d SELL)", len(sigs_v3), n_buy_v3, n_sell_v3)

    logger.info("Aplicando v3.1 (cooldown=0)...")
    params_v31 = StrategyParamsV3_1()
    sigs_v31 = apply_strategy_v3_1_to_snapshots(snapshots, params_v31)
    sigs_v31["timestamp_utc"] = pd.to_datetime(sigs_v31["timestamp_utc"], utc=True)
    n_buy_v31  = int((sigs_v31["signal"] == "BUY").sum())
    n_sell_v31 = int((sigs_v31["signal"] == "SELL").sum())
    logger.info("  Señales v3.1: %d (%d BUY, %d SELL)", len(sigs_v31), n_buy_v31, n_sell_v31)

    del snapshots
    gc.collect()

    # 4. Ejecutar backtests
    logger.info("Backtest v3...")
    trades_v3 = run_backtest(sigs_v3, m1_bt, CONFIG)
    logger.info("  Trades v3: %d", len(trades_v3))

    logger.info("Backtest v3.1...")
    trades_v31 = run_backtest(sigs_v31, m1_bt, CONFIG)
    logger.info("  Trades v3.1: %d", len(trades_v31))

    del m1_bt, sigs_v3, sigs_v31
    gc.collect()
    logger.info("Periodo completado | RAM liberada: %.1f GB", ram_gb())

    return trades_v3, trades_v31


# ── Reporte comparativo ────────────────────────────────────────────────────────

def print_comparison(
    trades_v3: pd.DataFrame,
    trades_v31: pd.DataFrame,
    label: str,
) -> None:
    sep = "=" * 70

    # -- Metricas individuales --
    print(f"\n{sep}")
    print(f"  v3 (cooldown=90 min) — {label}")
    print(sep)
    m_v3 = compute_full_metrics(trades_v3, initial_balance=INITIAL_BALANCE)
    print_metrics_report(m_v3)

    print(f"\n{sep}")
    print(f"  v3.1 (cooldown=0) — {label}")
    print(sep)
    m_v31 = compute_full_metrics(trades_v31, initial_balance=INITIAL_BALANCE)
    print_metrics_report(m_v31)

    # -- Tabla de diferencias --
    print(f"\n{sep}")
    print(f"  TABLA DIFERENCIAS (v3.1 - v3) — {label}")
    print(sep)

    metricas = [
        ("Total trades",        "n_trades",         ".0f"),
        ("Win rate %",          "win_rate_pct",     ".1f"),
        ("PnL total ($)",       "pnl_total_usd",    ".2f"),
        ("Retorno total %",     "total_return_pct", ".2f"),
        ("Profit Factor",       "profit_factor",    ".3f"),
        ("Expectancy/trade",    "expectancy_usd",   ".2f"),
        ("Avg win ($)",         "avg_win_usd",      ".2f"),
        ("Avg loss ($)",        "avg_loss_usd",     ".2f"),
        ("Max DD ($)",          "max_drawdown_usd", ".2f"),
        ("Max DD %",            "max_drawdown_pct", ".2f"),
        ("Sharpe (por trade)",  "sharpe_per_trade", ".3f"),
        ("Sharpe anualizado",   "sharpe_annualized",".2f"),
        ("Calmar ratio",        "calmar_ratio",     ".2f"),
    ]

    print(f"\n{'Metrica':<26} {'v3 (cd=90)':>13} {'v3.1 (cd=0)':>13} {'Delta':>10}")
    print("-" * 65)
    for label_m, key, fmt in metricas:
        v3_val  = m_v3.get(key,  None)
        v31_val = m_v31.get(key, None)
        if v3_val is not None and v31_val is not None:
            fmt_str = f"{{:{fmt}}}"
            delta = v31_val - v3_val
            delta_str = f"{delta:+.2f}"
            print(f"{label_m:<26} {fmt_str.format(v3_val):>13} {fmt_str.format(v31_val):>13} {delta_str:>10}")
        else:
            print(f"{label_m:<26} {'N/A':>13} {'N/A':>13} {'-':>10}")

    print(f"\nExit distribution v3:   {m_v3.get('exit_distribution', {})}")
    print(f"Exit distribution v3.1: {m_v31.get('exit_distribution', {})}")

    # -- PnL por año --
    print(f"\n{sep}")
    print("  PnL POR AÑO")
    print(sep)

    for name, trades in [("v3 (cd=90)", trades_v3), ("v3.1 (cd=0)", trades_v31)]:
        t = trades.copy()
        t["year"] = pd.to_datetime(t["entry_time"]).dt.year
        agg = t.groupby("year")["net_pnl_usd"].agg(
            trades="count",
            pnl="sum",
            win_rate=lambda x: (x > 0).mean() * 100,
        )
        print(f"\n{name}:")
        print(f"  {'Año':>4}  {'Trades':>7}  {'PnL ($)':>10}  {'WR%':>7}")
        print(f"  {'-'*4}  {'-'*7}  {'-'*10}  {'-'*7}")
        for yr, row in agg.iterrows():
            print(f"  {yr:>4}  {int(row['trades']):>7}  {row['pnl']:>10.2f}  {row['win_rate']:>6.1f}%")
        print(f"  {'TOTAL':>4}  {len(t):>7}  {t['net_pnl_usd'].sum():>10.2f}  {(t['net_pnl_usd']>0).mean()*100:>6.1f}%")

    # -- Trades adicionales en v3.1 (los que el cooldown bloqueaba) --
    print(f"\n{sep}")
    print("  TRADES ADICIONALES EN v3.1 (los que cooldown=90 bloqueaba)")
    print(sep)

    t3  = trades_v3.copy()
    t31 = trades_v31.copy()
    t3["entry_min"]  = pd.to_datetime(t3["entry_time"]).dt.floor("min")
    t31["entry_min"] = pd.to_datetime(t31["entry_time"]).dt.floor("min")

    times_v3  = set(t3["entry_min"])
    times_v31 = set(t31["entry_min"])
    extra_times = times_v31 - times_v3

    print(f"\nTrades adicionales en v3.1: {len(extra_times)}")

    if extra_times:
        extra = t31[t31["entry_min"].isin(extra_times)].copy()
        pnl_extra  = extra["net_pnl_usd"].sum()
        wr_extra   = (extra["net_pnl_usd"] > 0).mean() * 100
        avg_extra  = extra["net_pnl_usd"].mean()
        print(f"PnL total de trades adicionales: ${pnl_extra:.2f}")
        print(f"Win rate de trades adicionales:  {wr_extra:.1f}%")
        print(f"PnL promedio por trade adicional: ${avg_extra:.2f}")

        print(f"\nDetalle (primeros 30):")
        cols = [c for c in ["entry_time", "direction", "entry_price", "net_pnl_usd", "exit_reason"]
                if c in extra.columns]
        print(extra[cols].head(30).to_string(index=False))
    else:
        print("(sin diferencias detectables al nivel de minuto)")

    return m_v3, m_v31


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("   BACKTEST COMPARATIVO v3 (cooldown=90) vs v3.1 (cooldown=0)")
    print(f"   RAM disponible al inicio: {ram_gb():.1f} GB")
    print("   Fuente: cache Twelve Data local (sin API calls)")
    print("   Periodos: 2023-05-29/2026-05-28  +  2021-05-29/2023-05-28")
    print("=" * 70)

    # Verificar cache completo
    for fname in INTERVAL_FILENAMES.values():
        if not (CACHE_DIR / fname).exists():
            logger.error("Cache no encontrado: %s", CACHE_DIR / fname)
            sys.exit(1)
    logger.info("Cache OK (5 timeframes)")

    all_v3:  list[pd.DataFrame] = []
    all_v31: list[pd.DataFrame] = []

    # ── Periodo 1: 3 años (2023-2026) ─────────────────────────────────────────
    start_3y = pd.Timestamp("2023-05-29", tz="UTC")
    end_3y   = pd.Timestamp("2026-05-28 23:59:59", tz="UTC")

    t3_v3, t3_v31 = run_period("3 años", start_3y, end_3y)
    if not t3_v3.empty:
        all_v3.append(t3_v3)
        all_v31.append(t3_v31)
        logger.info("Periodo 3y: v3=%d trades, v3.1=%d trades", len(t3_v3), len(t3_v31))

    logger.info("RAM tras periodo 3y: %.1f GB", ram_gb())

    # ── Periodo 2: 2 años anteriores (2021-2023) ───────────────────────────────
    start_2y = pd.Timestamp("2021-05-29", tz="UTC")
    end_2y   = pd.Timestamp("2023-05-28 23:59:59", tz="UTC")

    t2_v3, t2_v31 = run_period("2 años anteriores", start_2y, end_2y)
    if not t2_v3.empty:
        all_v3.append(t2_v3)
        all_v31.append(t2_v31)
        logger.info("Periodo 2y: v3=%d trades, v3.1=%d trades", len(t2_v3), len(t2_v31))

    logger.info("RAM tras periodo 2y: %.1f GB", ram_gb())

    if not all_v3:
        logger.error("No se obtuvieron resultados para ningun periodo (RAM insuficiente).")
        sys.exit(1)

    # ── Combinar y reportar ────────────────────────────────────────────────────
    trades_v3  = pd.concat(all_v3,  ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    trades_v31 = pd.concat(all_v31, ignore_index=True).sort_values("entry_time").reset_index(drop=True)

    logger.info("Combinados: v3=%d trades, v3.1=%d trades", len(trades_v3), len(trades_v31))

    m_v3, m_v31 = print_comparison(trades_v3, trades_v31, label="5 años combinados (2021-2026)")

    # ── Guardar resultados ─────────────────────────────────────────────────────
    out_dir = Path("v2/data/comparison_v3_vs_v31")
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_v3.to_parquet(out_dir / "trades_v3_5y.parquet",  index=False)
    trades_v31.to_parquet(out_dir / "trades_v31_5y.parquet", index=False)

    # Guardar métricas como JSON (excluir equity_curve para legibilidad)
    def metrics_to_json(m: dict) -> dict:
        return {k: (float(v) if isinstance(v, (int, float)) else
                    str(v) if not isinstance(v, dict) else v)
                for k, v in m.items() if k != "equity_curve"}

    summary = {"v3_cooldown90": metrics_to_json(m_v3), "v3_1_cooldown0": metrics_to_json(m_v31)}
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Resultados guardados en: %s", out_dir.resolve())
    print(f"\n=== FIN BACKTEST COMPARATIVO ===")
    print(f"Archivos en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
