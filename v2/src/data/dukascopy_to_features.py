"""Adaptador: Dukascopy OHLC -> snapshots M1 con 163 features.

Equivalente a twelve_data_to_features.py pero usando DukascopyExtractor.
Sin rate limits: no hay sleep entre timeframes.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from v2.src.data.dukascopy_extractor import DukascopyExtractor
from v2.src.features.pipeline import build_snapshots

logger = logging.getLogger(__name__)

_INTERVAL_MAP = {
    "1min":  "m1",
    "5min":  "m5",
    "15min": "m15",
    "1h":    "h1",
    "4h":    "h4",
}


def fetch_all_timeframes(
    extractor: DukascopyExtractor,
    symbol: str,
    start: datetime,
    end: datetime,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Descarga OHLC en todos los timeframes necesarios para el pipeline.

    Args:
        extractor: instancia de DukascopyExtractor ya configurada
        symbol: "XAUUSD" o "XAU/USD"
        start, end: rango temporal
        force_refresh: si True, ignora cache y re-descarga todo

    Returns:
        Dict {interval_name: DataFrame} con OHLCV para cada timeframe.
    """
    ohlc: dict[str, pd.DataFrame] = {}

    for interval in _INTERVAL_MAP:
        logger.info("Descargando %s %s...", symbol, interval)
        df = extractor.fetch(symbol, interval, start, end, force_refresh=force_refresh)
        ohlc[interval] = df
        logger.info("  -> %d velas", len(df))

    return ohlc


def build_snapshots_from_dukascopy(
    ohlc: dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Construye snapshots M1 con features a partir de datos de Dukascopy.

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
