"""Backtest de 3 anos (2023-05-29 -> 2026-05-28) con reporte mensual detallado.

Reusa el pipeline validado (estrategia v3, features, engine, metricas) SIN
modificarlo. Este script solo orquesta una pasada unica end-to-end y anade
el reporte mensual/anual como capa de analisis posterior (monthly_report.py).

Recursos esperados:
- ~1.1M velas M1 (3 anos)
- Snapshots finales: ~1.4 GB en RAM
- Pico durante build_snapshots (merge_asof + concat de bloques): ~4-5 GB
- Tiempo total estimado: 45-75 min (descarga + features + backtest)

Recomendaciones antes de lanzar:
- Cerrar Chrome / IDE pesados para liberar RAM
- Verificar que hay >6 GB libres (Administrador de tareas)
- No interrumpir el proceso durante build_snapshots (es la fase critica de RAM)

Uso:
    python -m v2.scripts.backtest_3years
"""
import gc
import logging
import sys
from pathlib import Path

import pandas as pd

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
from v2.src.backtest.monthly_report import (
    compute_monthly_breakdown,
    compute_yearly_breakdown,
    print_monthly_report,
    print_yearly_report,
)
from v2.src.backtest.visualization import generate_full_report
from v2.src.data.dukascopy_extractor import DukascopyExtractor
from v2.src.data.dukascopy_to_features import fetch_all_timeframes
from v2.src.features.pipeline import build_snapshots
from v2.src.strategies.asian_momentum_v3 import (
    StrategyParamsV3,
    apply_strategy_v3_to_snapshots,
)

INITIAL_BALANCE = 2000.0


def main() -> None:
    start = pd.Timestamp("2023-05-29", tz="UTC")
    end   = pd.Timestamp("2026-05-28 23:59:59", tz="UTC")

    print("=" * 70)
    print("   BACKTEST 3 ANOS -- XAUUSD Asian Momentum v3")
    print(f"   Periodo : {start.date()} -> {end.date()}")
    print("   Fuente  : Dukascopy (datos institucionales, sin rate limits)")
    print("   ESTIMADO: 45-75 min total, ~4-5 GB RAM pico")
    print("   ESTRATEGIA: SIN modificar (defaults validados v3)")
    print("=" * 70)

    # -- 1. Descarga (reusa DukascopyExtractor / fetch_all_timeframes) -----------
    logger.info("=== DESCARGA (con cache local) ===")
    extractor = DukascopyExtractor()
    ohlc = fetch_all_timeframes(extractor, "XAUUSD", start.to_pydatetime(), end.to_pydatetime())

    for iv, df in ohlc.items():
        logger.info("  %s: %d velas", iv, len(df))

    m1 = ohlc.get("1min", pd.DataFrame())
    if m1.empty:
        logger.error("Sin datos M1. Verifica dukascopy-python instalado.")
        sys.exit(1)

    # -- 2. Feature engineering (pipeline validado, pasada unica) ----------------
    logger.info("=== FEATURE ENGINEERING (10-20 min para 3 anos, fase critica de RAM) ===")
    snapshots = build_snapshots(
        m1=ohlc["1min"],
        m5=ohlc["5min"],
        m15=ohlc["15min"],
        h1=ohlc["1h"],
        h4=ohlc["4h"],
    )
    logger.info("Snapshots: %d filas x %d columnas", len(snapshots), snapshots.shape[1])

    # Liberar OHLC crudo (ya alineado dentro de snapshots) antes del backtest
    m1_for_bt = m1.copy()
    m1_for_bt.columns = [c.lower() for c in m1_for_bt.columns]
    if "volume" in m1_for_bt.columns and "tick_volume" not in m1_for_bt.columns:
        m1_for_bt = m1_for_bt.rename(columns={"volume": "tick_volume"})

    del ohlc, m1
    gc.collect()

    # -- 3. Estrategia v3 (NO modificada — parametros default validados) ---------
    logger.info("=== ESTRATEGIA v3 (parametros default, sin tocar) ===")
    signals = apply_strategy_v3_to_snapshots(snapshots, StrategyParamsV3())
    signals["timestamp_utc"] = pd.to_datetime(signals["timestamp_utc"], utc=True)
    n_buy  = int((signals["signal"] == "BUY").sum())
    n_sell = int((signals["signal"] == "SELL").sum())
    logger.info("Senales generadas: %d (%d BUY, %d SELL)", len(signals), n_buy, n_sell)

    if signals.empty:
        logger.warning("Sin senales en el periodo.")
        sys.exit(0)

    del snapshots
    gc.collect()

    # -- 4. Backtest (motor validado, sin modificar) -----------------------------
    logger.info("=== BACKTEST ===")
    config = BacktestConfig(
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
    trades = run_backtest(signals, m1_for_bt, config)
    logger.info("Trades ejecutados: %d", len(trades))

    if trades.empty:
        logger.warning("Sin trades ejecutados — no se generan reportes.")
        sys.exit(0)

    # -- 5. Metricas agregadas (modulo validado) ---------------------------------
    metrics = compute_full_metrics(trades, initial_balance=INITIAL_BALANCE)
    print_metrics_report(metrics)

    # -- 6. NUEVO: reporte mensual detallado --------------------------------------
    monthly = compute_monthly_breakdown(trades)
    print_monthly_report(monthly, initial_balance=INITIAL_BALANCE)

    # -- 7. NUEVO: reporte anual ---------------------------------------------------
    yearly = compute_yearly_breakdown(monthly)
    print_yearly_report(yearly, initial_balance=INITIAL_BALANCE)

    # -- 8. Guardar resultados -----------------------------------------------------
    out_dir = Path(f"v2/data/backtest_3years/XAUUSD_{start.date()}_{end.date()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    trades.to_parquet(out_dir / "trades.parquet", index=False)
    monthly.to_csv(out_dir / "monthly_breakdown.csv", index=False)
    yearly.to_csv(out_dir / "yearly_breakdown.csv", index=False)

    metrics_save = {k: v for k, v in metrics.items() if k != "equity_curve"}
    pd.Series(metrics_save).to_json(out_dir / "metrics.json", indent=2)

    logger.info("Resultados guardados en: %s", out_dir)

    # -- 9. Graficos (modulo validado) ---------------------------------------------
    logger.info("=== GENERANDO GRAFICOS ===")
    generate_full_report(
        trades, metrics, INITIAL_BALANCE,
        output_dir=out_dir / "plots",
        title_prefix=f"v3 XAUUSD {start.date()}/{end.date()} (3 anos)",
    )

    print(f"\nTodos los resultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
