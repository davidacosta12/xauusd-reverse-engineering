"""Conversión robusta de zona horaria para el proyecto v2.

Regla fundamental: TODO se almacena y procesa en UTC.
El servidor MT5 (MEX Atlantic) corre en GMT+3 — confirmado el 2026-05-05.
"""
from datetime import datetime, timezone, timedelta

import pandas as pd

MT5_GMT_OFFSET = 3  # horas — GMT+3 confirmado


def server_to_utc(ts) -> pd.Timestamp:
    """Convierte timestamp en hora server (GMT+3) a UTC.

    Acepta str, datetime, pd.Timestamp — naive o tz-aware.
    Si es naive lo trata como hora server (GMT+3) antes de convertir.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone(timedelta(hours=MT5_GMT_OFFSET)))
    return ts.tz_convert(timezone.utc)


def utc_to_server(ts) -> pd.Timestamp:
    """Convierte UTC → hora server (GMT+3).

    Acepta str, datetime, pd.Timestamp — naive o tz-aware.
    Si es naive lo trata como UTC.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.tz_convert(timezone(timedelta(hours=MT5_GMT_OFFSET)))


def df_server_to_utc(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    """Convierte un DataFrame de hora server a UTC.

    Args:
        df: DataFrame con tiempo en hora server.
        time_col: nombre de columna de tiempo. Si None usa el índice.

    Returns:
        Copia del DataFrame con tiempo en UTC.
    """
    df = df.copy()
    server_tz = timezone(timedelta(hours=MT5_GMT_OFFSET))

    if time_col is None:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)
        if idx.tz is None:
            idx = idx.tz_localize(server_tz)
        df.index = idx.tz_convert(timezone.utc)
    else:
        col = pd.to_datetime(df[time_col])
        if col.dt.tz is None:
            col = col.dt.tz_localize(server_tz)
        df[time_col] = col.dt.tz_convert(timezone.utc)

    return df


def get_session_utc(ts_utc: pd.Timestamp) -> str:
    """Devuelve la sesión de trading activa para un timestamp UTC.

    Sesiones (horarios en UTC):
        asia             22:00 – 05:59
        asia_london_gap  06:00 – 06:59
        london           07:00 – 11:59
        london_ny_overlap 12:00 – 12:59
        ny               13:00 – 20:59
        ny_close         21:00 – 21:59

    Args:
        ts_utc: Timestamp tz-aware en UTC.

    Returns:
        Nombre de la sesión como string.

    Raises:
        ValueError: si ts_utc es naive.
    """
    if ts_utc.tzinfo is None:
        raise ValueError("ts_utc debe ser tz-aware UTC. Usa server_to_utc() primero.")

    h = ts_utc.hour
    if h >= 22 or h < 6:
        return "asia"
    if h == 6:
        return "asia_london_gap"
    if 7 <= h < 12:
        return "london"
    if h == 12:
        return "london_ny_overlap"
    if 13 <= h < 21:
        return "ny"
    return "ny_close"


def normalize_index_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que el índice de un DataFrame es DatetimeIndex UTC.

    Útil para normalizar datos recibidos de fuentes externas (Dukascopy, yfinance, etc.)
    que pueden venir sin tz o en otras zonas horarias.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        # Dukascopy entrega UTC por defecto — asumimos UTC si no hay info
        df.index = df.index.tz_localize(timezone.utc)
    else:
        df.index = df.index.tz_convert(timezone.utc)
    return df
