"""Adaptador: descarga OHLC de Twelve Data y construye snapshots con features.

REUSA pipeline.py que a su vez reusa los modulos de features ya validados.
No duplica ninguna logica de feature engineering.

Mapeo de intervalos Twelve Data -> nomenclatura interna:
  1min  -> M1  (base de snapshots)
  5min  -> M5  (microestructura)
  15min -> M15 (niveles, microestructura, tecnicos)
  1h    -> H1  (microestructura, tecnicos)
  4h    -> H4  (tecnicos)
"""
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from v2.src.data.twelve_data_extractor import TwelveDataExtractor, RATE_LIMIT_SLEEP_SEC
from v2.src.features.pipeline import build_snapshots

logger = logging.getLogger(__name__)

# Intervalos requeridos y su nombre interno
_INTERVAL_MAP = {
    "1min":  "m1",
    "5min":  "m5",
    "15min": "m15",
    "1h":    "h1",
    "4h":    "h4",
}


def fetch_all_timeframes(
    extractor: TwelveDataExtractor,
    symbol: str,
    start: datetime,
    end: datetime,
    sleep_between: float = RATE_LIMIT_SLEEP_SEC,
) -> dict[str, pd.DataFrame]:
    """Descarga OHLC en todos los timeframes necesarios para el pipeline.

    Args:
        extractor: instancia de TwelveDataExtractor ya configurada
        symbol: ej. "XAU/USD"
        start, end: rango temporal
        sleep_between: segundos entre requests (respetar rate limit)

    Returns:
        Dict {interval_name: DataFrame} con OHLCV para cada timeframe.
    """
    ohlc: dict[str, pd.DataFrame] = {}
    intervals = list(_INTERVAL_MAP.keys())

    for i, interval in enumerate(intervals):
        logger.info("Descargando %s %s...", symbol, interval)
        df = extractor.fetch(symbol, interval, start, end)
        ohlc[interval] = df
        logger.info("  -> %d velas", len(df))
        if i < len(intervals) - 1:
            time.sleep(sleep_between)

    return ohlc


def build_snapshots_from_twelvedata(
    ohlc: dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Construye snapshots M1 con features a partir de datos de Twelve Data.

    Args:
        ohlc: dict devuelto por fetch_all_timeframes
        save_path: si se provee, guarda el resultado en parquet

    Returns:
        DataFrame de snapshots listo para aplicar la estrategia v3.
    """
    m1  = ohlc.get("1min",  pd.DataFrame())
    m5  = ohlc.get("5min",  pd.DataFrame())
    m15 = ohlc.get("15min", pd.DataFrame())
    h1  = ohlc.get("1h",    pd.DataFrame())
    h4  = ohlc.get("4h",    pd.DataFrame())

    if m1.empty:
        raise ValueError("Sin datos M1 (1min). No se pueden construir snapshots.")
    if m15.empty:
        raise ValueError("Sin datos M15 (15min). Los niveles asiaticos requieren M15.")

    snapshots = build_snapshots(m1=m1, m5=m5, m15=m15, h1=h1, h4=h4)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        snapshots.to_parquet(save_path)
        logger.info("Snapshots guardados: %s (%d filas)", save_path, len(snapshots))

    return snapshots
