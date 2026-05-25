"""Estrategia v1: Asian Range Breakout con confirmación de momentum.

Hipótesis (derivada del análisis de z-scores de los 30 trades in-sample):
El operador entra en la dirección de la ruptura del rango asiático,
con confirmación de momentum vía Bollinger, Stochastic y velas consecutivas.

Esta es la PRIMERA versión. No optimizada. Parámetros redondeados a valores
estadísticamente significativos (no tunados al dataset).
"""
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

SignalType = Literal["BUY", "SELL", None]


@dataclass
class StrategyParams:
    """Parámetros de la estrategia v1. Valores redondeados; NO optimizados."""
    # Distancia mínima al Asian VWAP para considerar ruptura
    vwap_distance_threshold_pips: float = 200.0

    # Confirmación de momentum
    bb_pct_b_buy_threshold: float = 0.50    # mitad superior de Bollinger para BUY
    bb_pct_b_sell_threshold: float = 0.40   # mitad inferior para SELL

    stoch_k_buy_threshold: float = 50.0     # Stoch en mitad alta para BUY
    stoch_k_sell_threshold: float = 40.0    # Stoch en mitad baja para SELL

    # Mínimo de velas consecutivas en la dirección
    min_consecutive_bull_for_buy: int = 1
    min_consecutive_bear_for_sell: int = 1

    # Restricción de sesión (en UTC). La estrategia parece operar en sesión asiática.
    # 22:00-06:00 UTC = sesión asiática (cruza medianoche)
    session_start_utc_hour: int = 22
    session_end_utc_hour: int = 6


@dataclass
class Signal:
    """Resultado de evaluar la estrategia en un timestamp."""
    timestamp_utc: pd.Timestamp
    signal: SignalType   # 'BUY', 'SELL', o None
    reason: str          # explicación humana de por qué (útil para debug)


def is_in_asian_session(timestamp_utc: pd.Timestamp, params: StrategyParams) -> bool:
    """Devuelve True si el timestamp está dentro de la sesión asiática (cruza medianoche UTC)."""
    h = timestamp_utc.hour
    if params.session_start_utc_hour < params.session_end_utc_hour:
        return params.session_start_utc_hour <= h < params.session_end_utc_hour
    # Cruza medianoche: ej. 22-06 UTC
    return h >= params.session_start_utc_hour or h < params.session_end_utc_hour


def evaluate_signal(snapshot: pd.Series, params: StrategyParams) -> Signal:
    """Evalúa la estrategia en un único snapshot M1 y devuelve la señal.

    Args:
        snapshot: fila del DataFrame snapshots_M1 (pd.Series con todas las features).
                  El índice debe ser un pd.Timestamp UTC tz-aware.
        params: parámetros de la estrategia.

    Returns:
        Signal con type 'BUY', 'SELL' o None y razón.
    """
    ts = snapshot.name  # timestamp UTC del snapshot

    if not is_in_asian_session(ts, params):
        return Signal(ts, None, "out_of_session")

    required = [
        "dist_to_asian_vwap_pips_signed",
        "m15_bb_pct_b",
        "m15_stoch_k",
        "m5_consecutive_bull",
        "m15_consecutive_bear",
    ]
    if any(pd.isna(snapshot.get(f)) for f in required):
        return Signal(ts, None, "insufficient_data")

    dist_vwap = snapshot["dist_to_asian_vwap_pips_signed"]
    bb_pct_b  = snapshot["m15_bb_pct_b"]
    stoch_k   = snapshot["m15_stoch_k"]
    cons_bull = snapshot["m5_consecutive_bull"]
    cons_bear = snapshot["m15_consecutive_bear"]

    is_buy = (
        dist_vwap > params.vwap_distance_threshold_pips
        and bb_pct_b > params.bb_pct_b_buy_threshold
        and stoch_k > params.stoch_k_buy_threshold
        and cons_bull >= params.min_consecutive_bull_for_buy
    )
    if is_buy:
        return Signal(
            ts, "BUY",
            f"vwap={dist_vwap:.0f} bb%={bb_pct_b:.2f} stoch={stoch_k:.0f} cons_bull={cons_bull:.0f}",
        )

    is_sell = (
        dist_vwap < -params.vwap_distance_threshold_pips
        and bb_pct_b < params.bb_pct_b_sell_threshold
        and stoch_k < params.stoch_k_sell_threshold
        and cons_bear >= params.min_consecutive_bear_for_sell
    )
    if is_sell:
        return Signal(
            ts, "SELL",
            f"vwap={dist_vwap:.0f} bb%={bb_pct_b:.2f} stoch={stoch_k:.0f} cons_bear={cons_bear:.0f}",
        )

    return Signal(ts, None, "no_signal_conditions_met")


def apply_strategy_to_snapshots(
    snapshots_df: pd.DataFrame,
    params: Optional[StrategyParams] = None,
    deduplicate_within_minutes: int = 30,
) -> pd.DataFrame:
    """Aplica la estrategia a todo el DataFrame de snapshots M1.

    Args:
        snapshots_df: DataFrame con índice DatetimeIndex UTC tz-aware y features.
        params: parámetros (default: StrategyParams()).
        deduplicate_within_minutes: si hay señales del mismo tipo dentro de N minutos,
                                    solo se conserva la primera (evita rachas de 100 BUYs).

    Returns:
        DataFrame con columnas: timestamp_utc, signal, reason.
        Solo incluye filas con signal no-None.
    """
    params = params or StrategyParams()

    signals: list[dict] = []
    for ts, row in snapshots_df.iterrows():
        sig = evaluate_signal(row, params)
        if sig.signal is not None:
            signals.append({"timestamp_utc": ts, "signal": sig.signal, "reason": sig.reason})

    if not signals:
        return pd.DataFrame(columns=["timestamp_utc", "signal", "reason"])

    signals_df = pd.DataFrame(signals).sort_values("timestamp_utc").reset_index(drop=True)
    signals_df["timestamp_utc"] = pd.to_datetime(signals_df["timestamp_utc"], utc=True)

    if deduplicate_within_minutes > 0:
        keep = []
        last_kept: dict[str, Optional[pd.Timestamp]] = {"BUY": None, "SELL": None}
        for _, row in signals_df.iterrows():
            last = last_kept[row["signal"]]
            if last is None or (row["timestamp_utc"] - last).total_seconds() / 60 >= deduplicate_within_minutes:
                keep.append(True)
                last_kept[row["signal"]] = row["timestamp_utc"]
            else:
                keep.append(False)
        signals_df = signals_df[keep].reset_index(drop=True)

    return signals_df
