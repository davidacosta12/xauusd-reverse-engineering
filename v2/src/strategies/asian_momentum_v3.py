"""Estrategia v3: v2 + filtros operacionales.

Filtros añadidos sobre v2:
1. Sesion core 22:00-02:59 UTC (operador no opera 03-04h UTC)
2. Maximo 2 entradas por dia UTC
3. Cooldown minimo entre senales del mismo dia

Mantiene la logica direccional de v2 (momentum manda, VWAP filtra).
"""
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from v2.src.strategies.asian_momentum_v2 import (
    Signal,
    SignalType,
    StrategyParamsV2,
    evaluate_signal_v2,
)


@dataclass
class StrategyParamsV3(StrategyParamsV2):
    """v3 hereda todos los parametros de v2 y añade los nuevos filtros."""
    # Filtro 1: ventana horaria CORE (mas restrictiva que la de v2)
    # v2 usaba 22-06 UTC; v3 limita a 22-03 UTC (sesion asiatica activa real)
    core_session_start_hour: int = 22
    core_session_end_hour: int = 3

    # Filtro 2: max trades por dia de sesion
    max_trades_per_session_day: int = 2

    # Filtro 3: cooldown entre senales del mismo dia (minutos)
    cooldown_minutes_within_day: int = 90


def _is_in_core_session(ts_utc: pd.Timestamp, params: StrategyParamsV3) -> bool:
    """22:00-02:59 UTC (cruza medianoche)."""
    h = ts_utc.hour
    s, e = params.core_session_start_hour, params.core_session_end_hour
    if s < e:
        return s <= h < e
    return h >= s or h < e


def _session_date(ts_utc: pd.Timestamp, params: StrategyParamsV3) -> object:
    """Define el 'dia de sesion' para agrupar señales.

    La sesion asiatica del 19 marzo empieza el 18 marzo a las 22:00 UTC.
    Si hora >= core_session_start_hour (22) → session_date = dia siguiente.
    Si hora < core_session_end_hour (3)   → session_date = dia actual.
    """
    h = ts_utc.hour
    if h >= params.core_session_start_hour:
        return (ts_utc + pd.Timedelta(days=1)).date()
    return ts_utc.date()


def apply_strategy_v3_to_snapshots(
    snapshots_df: pd.DataFrame,
    params: StrategyParamsV3 | None = None,
) -> pd.DataFrame:
    """Aplica v3: evalua señal con v2, luego aplica filtros operacionales.

    Args:
        snapshots_df: DataFrame con indice DatetimeIndex UTC y todas las features.
        params: parametros v3 (default: StrategyParamsV3()).

    Returns:
        DataFrame con columnas: timestamp_utc, signal, reason, session_date.
        Solo incluye señales aceptadas tras aplicar todos los filtros.
    """
    params = params or StrategyParamsV3()

    # 1. Generar señales crudas con la logica de v2
    raw: list[dict] = []
    for ts, row in snapshots_df.iterrows():
        sig = evaluate_signal_v2(row, params)
        if sig.signal is not None:
            raw.append({"timestamp_utc": ts, "signal": sig.signal, "reason": sig.reason})

    if not raw:
        return pd.DataFrame(columns=["timestamp_utc", "signal", "reason", "session_date"])

    sig_df = pd.DataFrame(raw)
    sig_df["timestamp_utc"] = pd.to_datetime(sig_df["timestamp_utc"], utc=True)
    sig_df = sig_df.sort_values("timestamp_utc").reset_index(drop=True)

    # 2. Filtro 1: solo señales en sesion core (22:00-02:59 UTC)
    sig_df = sig_df[
        sig_df["timestamp_utc"].apply(lambda t: _is_in_core_session(t, params))
    ].copy()

    if sig_df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "signal", "reason", "session_date"])

    # 3. Asignar session_date
    sig_df["session_date"] = sig_df["timestamp_utc"].apply(lambda t: _session_date(t, params))

    # 4. Filtros 2 + 3: max N trades por dia + cooldown
    accepted: list[dict] = []
    last_ts_per_day: dict[object, pd.Timestamp] = {}
    count_per_day: dict[object, int] = {}

    for _, row in sig_df.iterrows():
        day = row["session_date"]
        ts  = row["timestamp_utc"]

        if count_per_day.get(day, 0) >= params.max_trades_per_session_day:
            continue

        last_ts = last_ts_per_day.get(day)
        if last_ts is not None:
            if (ts - last_ts).total_seconds() / 60 < params.cooldown_minutes_within_day:
                continue

        accepted.append(row.to_dict())
        last_ts_per_day[day] = ts
        count_per_day[day] = count_per_day.get(day, 0) + 1

    if not accepted:
        return pd.DataFrame(columns=["timestamp_utc", "signal", "reason", "session_date"])

    return pd.DataFrame(accepted).reset_index(drop=True)
