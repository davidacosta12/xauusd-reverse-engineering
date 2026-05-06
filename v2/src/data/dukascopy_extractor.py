"""Extractor de datos XAUUSD desde Dukascopy (fuente de referencia externa).

Usa `dukascopy-python` >= 4.0.1 para descargar velas históricas OHLCV
y las normaliza al mismo formato que el extractor MT5.

API real de la versión 4.0.1 (verificada 2026-05-06):
    import dukascopy_python
    df = dukascopy_python.fetch(
        instrument="XAU/USD",
        interval=dukascopy_python.INTERVAL_MIN_15,
        offer_side=dukascopy_python.OFFER_SIDE_BID,
        start=datetime(..., tzinfo=timezone.utc),
        end=datetime(..., tzinfo=timezone.utc),
    )
    # → columnas: open, high, low, close, volume
    # → índice: timestamp (DatetimeIndex UTC tz-aware)

Propósito: validación cross-fuente. Los precios Dukascopy pueden diferir
ligeramente del broker MEX Atlantic (spread, feed) — eso es esperado.
"""
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from v2.src.utils.timezone import normalize_index_utc
from v2.config.settings import (
    DATA_RAW_DUKA,
    PERIOD_START_UTC,
    PERIOD_END_UTC,
    TIMEFRAMES,
)

logger = logging.getLogger(__name__)

# Instrumento XAUUSD en la API de Dukascopy free service
_DUKA_INSTRUMENT = "XAU/USD"

# Mapa timeframe nombre → constante de dukascopy_python (verificado 2026-05-06)
# NOTA: son INTERVAL_MIN_* NO INTERVAL_MINUTE_*
_DUKA_TF_MAP: dict[str, str] = {
    "M1":  "1MIN",
    "M5":  "5MIN",
    "M15": "15MIN",
    "M30": "30MIN",
    "H1":  "1HOUR",
    "H4":  "4HOUR",
    "D1":  "1DAY",
}


def _get_interval_const(timeframe: str) -> str:
    """Devuelve la constante de intervalo de dukascopy_python para el TF dado."""
    import dukascopy_python as dk

    mapping = {
        "M1":  dk.INTERVAL_MIN_1,
        "M5":  dk.INTERVAL_MIN_5,
        "M15": dk.INTERVAL_MIN_15,
        "M30": dk.INTERVAL_MIN_30,
        "H1":  dk.INTERVAL_HOUR_1,
        "H4":  dk.INTERVAL_HOUR_4,
        "D1":  dk.INTERVAL_DAY_1,
    }
    if timeframe not in mapping:
        raise ValueError(
            f"Timeframe '{timeframe}' no soportado. Opciones: {list(mapping)}"
        )
    return mapping[timeframe]


def _check_dukascopy_installed() -> bool:
    """Verifica que dukascopy-python esté instalado."""
    try:
        import dukascopy_python  # noqa: F401
        return True
    except ImportError:
        return False


def extract_dukascopy_ohlc(
    timeframe: str,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    save_parquet: bool = True,
    max_retries: int = 3,
    retry_delay_base: float = 2.0,
) -> pd.DataFrame:
    """Descarga velas OHLCV de Dukascopy para XAUUSD (XAU/USD).

    Usa `dukascopy_python.fetch()` con OFFER_SIDE_BID (precio bid, estándar
    para datos históricos de referencia).

    Args:
        timeframe: nombre del timeframe ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1').
        start_utc: inicio del periodo en UTC. Default: PERIOD_START_UTC.
        end_utc: fin del periodo en UTC. Default: PERIOD_END_UTC.
        save_parquet: si True, guarda el resultado en data/raw_dukascopy/.
        max_retries: intentos máximos ante fallos de red.
        retry_delay_base: segundos base para backoff exponencial (2^attempt).

    Returns:
        DataFrame con índice DatetimeIndex UTC (tz-aware, nombre 'time_utc') y columnas:
            open, high, low, close, tick_volume, real_volume, spread
        DataFrame vacío si falla tras todos los reintentos.
    """
    import dukascopy_python as dk

    if timeframe not in _DUKA_TF_MAP:
        raise ValueError(
            f"Timeframe '{timeframe}' no soportado. Opciones: {list(_DUKA_TF_MAP)}"
        )

    start_utc = start_utc or PERIOD_START_UTC
    end_utc = end_utc or PERIOD_END_UTC

    # Asegurar que start/end son tz-aware UTC (fetch() lo requiere)
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)

    interval = _get_interval_const(timeframe)

    df = pd.DataFrame()
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Dukascopy %s %s: descargando (intento %d/%d) | %s → %s",
                _DUKA_INSTRUMENT,
                timeframe,
                attempt,
                max_retries,
                start_utc.strftime("%Y-%m-%d"),
                end_utc.strftime("%Y-%m-%d"),
            )
            df = dk.fetch(
                instrument=_DUKA_INSTRUMENT,
                interval=interval,
                offer_side=dk.OFFER_SIDE_BID,
                start=start_utc,
                end=end_utc,
            )
            if df is not None and len(df) > 0:
                break
            logger.warning(
                "Dukascopy devolvió respuesta vacía para %s %s (intento %d).",
                _DUKA_INSTRUMENT,
                timeframe,
                attempt,
            )
        except Exception as exc:
            last_exc = exc
            wait = retry_delay_base ** attempt
            if attempt < max_retries:
                logger.warning(
                    "Error Dukascopy %s (intento %d/%d): %s. Reintentando en %.1fs...",
                    timeframe,
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Dukascopy %s %s falló tras %d intentos: %s. Continuando sin este timeframe.",
                    _DUKA_INSTRUMENT,
                    timeframe,
                    max_retries,
                    last_exc,
                )
                return pd.DataFrame()

    if df is None or (hasattr(df, "__len__") and len(df) == 0):
        logger.error(
            "Dukascopy devolvió datos vacíos para %s %s.", _DUKA_INSTRUMENT, timeframe
        )
        return pd.DataFrame()

    df = _normalize_dukascopy_df(df)

    logger.info(
        "Dukascopy %s %s: %d velas | %s → %s",
        _DUKA_INSTRUMENT,
        timeframe,
        len(df),
        df.index.min(),
        df.index.max(),
    )

    if save_parquet:
        DATA_RAW_DUKA.mkdir(parents=True, exist_ok=True)
        out_path = DATA_RAW_DUKA / f"ohlc_{timeframe}.parquet"
        df.to_parquet(out_path)
        logger.info(
            "Dukascopy %s guardado en %s (%.1f KB)",
            timeframe,
            out_path,
            out_path.stat().st_size / 1024,
        )

    return df


def _normalize_dukascopy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el DataFrame de dukascopy_python.fetch() al formato estándar.

    dukascopy_python 4.0.1 devuelve:
        - índice: 'timestamp' (DatetimeIndex UTC tz-aware)
        - columnas: open, high, low, close, volume

    Salida estándar del proyecto:
        - índice: 'time_utc' (DatetimeIndex UTC tz-aware)
        - columnas: open, high, low, close, tick_volume, real_volume, spread
    """
    df = df.copy()

    # Normalizar índice a UTC con nombre estándar
    df = normalize_index_utc(df)
    df.index.name = "time_utc"

    # Mapear 'volume' → 'tick_volume' (Dukascopy no provee real_volume ni spread)
    if "volume" in df.columns:
        df = df.rename(columns={"volume": "tick_volume"})

    # Añadir columnas sintéticas que Dukascopy no provee
    df["real_volume"] = 0
    df["spread"] = 0

    # Garantizar presencia de todas las columnas OHLC
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            logger.warning("Columna '%s' faltante en output de Dukascopy.", col)
            df[col] = float("nan")

    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0

    cols = ["open", "high", "low", "close", "tick_volume", "real_volume", "spread"]
    return df[cols].sort_index()


def extract_all_timeframes(
    timeframes: list[str] | None = None,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Extrae OHLCV de Dukascopy para todos los timeframes especificados.

    Los timeframes que fallen se loggean pero no interrumpen el proceso.

    Args:
        timeframes: lista de nombres. Default: todos excepto D1.
        start_utc: inicio del periodo en UTC.
        end_utc: fin del periodo en UTC.

    Returns:
        Dict mapeando nombre de timeframe → DataFrame OHLCV.
    """
    if not _check_dukascopy_installed():
        raise ImportError(
            "dukascopy-python no está instalado. Ejecuta: pip install dukascopy-python"
        )

    timeframes = timeframes or [tf for tf in TIMEFRAMES if tf != "D1"]

    results: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        logger.info("--- Descargando Dukascopy %s ---", tf)
        try:
            df = extract_dukascopy_ohlc(tf, start_utc=start_utc, end_utc=end_utc)
            if not df.empty:
                results[tf] = df
        except Exception as exc:
            logger.error("Error no recuperable en Dukascopy %s: %s", tf, exc)

    return results
