"""Backtest profesional usando datos de Twelve Data.

Flujo completo:
1. Descargar OHLC del periodo solicitado (Twelve Data API, con cache)
2. Generar features (pipeline existente validado)
3. Aplicar estrategia v3 (NO modificada)
4. Ejecutar backtest con engine existente (NO modificado)
5. Calcular metricas profesionales
6. Generar graficos
7. Guardar resultados

Uso:
    python -m v2.scripts.backtest_twelvedata --start 2026-04-01 --end 2026-04-30

    # Ver todas las opciones:
    python -m v2.scripts.backtest_twelvedata --help
"""
import argparse
import logging
import sys
from datetime import datetime
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

from v2.config.settings import DATA_FEATURES
from v2.src.backtest.engine import BacktestConfig, run_backtest
from v2.src.backtest.metrics import compute_full_metrics, print_metrics_report
from v2.src.backtest.visualization import generate_full_report
from v2.src.data.twelve_data_extractor import TwelveDataExtractor
from v2.src.data.twelve_data_to_features import (
    build_snapshots_from_twelvedata,
    fetch_all_timeframes,
)
from v2.src.strategies.asian_momentum_v3 import (
    StrategyParamsV3,
    apply_strategy_v3_to_snapshots,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest de la estrategia v3 usando datos de Twelve Data"
    )
    p.add_argument("--symbol",    default="XAU/USD",    help="Simbolo Twelve Data (default: XAU/USD)")
    p.add_argument("--start",     default="2026-04-01", help="Fecha inicio YYYY-MM-DD")
    p.add_argument("--end",       default="2026-04-30", help="Fecha fin YYYY-MM-DD")
    p.add_argument("--sl-buy",    type=float, default=260.7, help="SL pips BUY (default: 260.7)")
    p.add_argument("--tp-buy",    type=float, default=272.1, help="TP pips BUY (default: 272.1)")
    p.add_argument("--sl-sell",   type=float, default=342.8, help="SL pips SELL (default: 342.8)")
    p.add_argument("--tp-sell",   type=float, default=289.8, help="TP pips SELL (default: 289.8)")
    p.add_argument("--volume",    type=float, default=0.02,  help="Lotes (default: 0.02)")
    p.add_argument("--spread",    type=float, default=3.0,   help="Spread asumido en pips (default: 3.0)")
    p.add_argument("--slip-entry",type=float, default=0.5,   help="Slippage entry en pips (default: 0.5)")
    p.add_argument("--slip-exit", type=float, default=0.5,   help="Slippage exit en pips (default: 0.5)")
    p.add_argument("--balance",   type=float, default=2000.0,help="Balance inicial USD (default: 2000)")
    p.add_argument("--force-refresh", action="store_true",   help="Ignorar cache y re-descargar")
    p.add_argument("--no-plots",  action="store_true",       help="No generar graficos")
    p.add_argument("--output-dir",default=None,              help="Directorio para guardar resultados")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt   = datetime.strptime(args.end + " 23:59:59", "%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print(f"   BACKTEST Twelve Data — {args.symbol}")
    print(f"   Periodo : {args.start} -> {args.end}")
    print(f"   SL/TP   : BUY {args.sl_buy}/{args.tp_buy} pips | SELL {args.sl_sell}/{args.tp_sell} pips")
    print(f"   Volumen : {args.volume} lotes | Spread asumido: {args.spread} pips")
    print("=" * 70)

    # ── 1. Descargar OHLC ────────────────────────────────────────────────────
    logger.info("=== DESCARGA (con cache local) ===")
    extractor = TwelveDataExtractor()
    ohlc = fetch_all_timeframes(
        extractor, args.symbol, start_dt, end_dt,
    )

    for iv, df in ohlc.items():
        logger.info("  %s: %d velas", iv, len(df))

    m1 = ohlc.get("1min", pd.DataFrame())
    if m1.empty:
        logger.error("Sin datos M1. Verifica la API key y el simbolo.")
        sys.exit(1)

    # ── 2. Generar features ──────────────────────────────────────────────────
    logger.info("=== FEATURE ENGINEERING ===")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"v2/data/backtest_twelve/{args.symbol.replace('/', '_')}_{args.start}_{args.end}")
    snap_path = out_dir / "snapshots_M1.parquet"

    snapshots = build_snapshots_from_twelvedata(ohlc, save_path=snap_path)
    logger.info("Snapshots: %d filas x %d columnas", len(snapshots), snapshots.shape[1])

    # ── 3. Aplicar estrategia v3 ─────────────────────────────────────────────
    logger.info("=== ESTRATEGIA v3 ===")
    signals = apply_strategy_v3_to_snapshots(snapshots, StrategyParamsV3())
    signals["timestamp_utc"] = pd.to_datetime(signals["timestamp_utc"], utc=True)
    n_buy  = int((signals["signal"] == "BUY").sum())
    n_sell = int((signals["signal"] == "SELL").sum())
    logger.info("Señales generadas: %d (%d BUY, %d SELL)", len(signals), n_buy, n_sell)

    if signals.empty:
        logger.warning("Sin señales en el periodo. Considera ampliar el rango de fechas.")
        sys.exit(0)

    # ── 4. Backtest ──────────────────────────────────────────────────────────
    logger.info("=== BACKTEST ===")
    config = BacktestConfig(
        sl_pips_buy=args.sl_buy,
        sl_pips_sell=args.sl_sell,
        tp_pips_buy=args.tp_buy,
        tp_pips_sell=args.tp_sell,
        slippage_entry_pips=args.slip_entry,
        slippage_exit_pips=args.slip_exit,
        spread_pips=args.spread,
        volume_lots=args.volume,
        max_hold_hours=12,
    )

    # El engine usa ohlc_m1 con columnas lowercase
    m1_for_bt = m1.copy()
    m1_for_bt.columns = [c.lower() for c in m1_for_bt.columns]
    if "volume" in m1_for_bt.columns and "tick_volume" not in m1_for_bt.columns:
        m1_for_bt = m1_for_bt.rename(columns={"volume": "tick_volume"})

    bt = run_backtest(signals, m1_for_bt, config)
    logger.info("Trades ejecutados: %d", len(bt))

    # ── 5. Metricas ──────────────────────────────────────────────────────────
    metrics = compute_full_metrics(bt, initial_balance=args.balance)
    print_metrics_report(metrics)

    # ── 6. Guardar resultados ─────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    bt_path = out_dir / "backtest_results.parquet"
    if not bt.empty:
        bt.to_parquet(bt_path, index=False)
        logger.info("Resultados guardados: %s", bt_path)

    # Guardar metricas sin la equity_curve (no serializable directamente)
    metrics_save = {k: v for k, v in metrics.items() if k != "equity_curve"}
    pd.Series(metrics_save).to_json(out_dir / "metrics.json", indent=2)

    # ── 7. Graficos ───────────────────────────────────────────────────────────
    if not args.no_plots and not bt.empty:
        logger.info("=== GENERANDO GRAFICOS ===")
        generate_full_report(
            bt, metrics, args.balance,
            output_dir=out_dir / "plots",
            title_prefix=f"v3 {args.symbol} {args.start}/{args.end}",
        )
        logger.info("Graficos en: %s/plots/", out_dir)

    print(f"\nTodos los resultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
