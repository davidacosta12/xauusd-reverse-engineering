"""Estrategia v2: Asian Range con dirección por MOMENTUM.

Diferencias clave vs v1:
- La dirección (BUY/SELL) viene del momentum (BB%B, velas consecutivas), NO del VWAP.
- El asian_mid se usa solo como filtro de OPORTUNIDAD (precio lejos del centro asiático).
- Compatible con valores rolling de Asian Range (computados durante la formación).
"""
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

SignalType = Literal["BUY", "SELL", None]


@dataclass
class StrategyParamsV2:
    """Parámetros de la estrategia v2. Valores redondeados; NO optimizados."""
    # Filtro de OPORTUNIDAD: precio lejos del centro asiático
    min_distance_from_asian_mid_pips: float = 100.0

    # Confirmación de momentum (BB%B)
    bb_pct_b_buy_threshold: float = 0.55
    bb_pct_b_sell_threshold: float = 0.45

    # Velas consecutivas
    min_consecutive_bull_for_buy: int = 1
    min_consecutive_bear_for_sell: int = 1

    # Restricción de sesión asiática (cruza medianoche UTC: 22:00-06:00)
    session_start_utc_hour: int = 22
    session_end_utc_hour: int = 6


@dataclass
class Signal:
    """Resultado de evaluar la estrategia en un timestamp."""
    timestamp_utc: pd.Timestamp
    signal: SignalType
    reason: str


def is_in_asian_session(ts_utc: pd.Timestamp, params: StrategyParamsV2) -> bool:
    """Devuelve True si el timestamp está dentro de la sesión asiática."""
    h = ts_utc.hour
    if params.session_start_utc_hour < params.session_end_utc_hour:
        return params.session_start_utc_hour <= h < params.session_end_utc_hour
    return h >= params.session_start_utc_hour or h < params.session_end_utc_hour


def evaluate_signal_v2(snapshot: pd.Series, params: StrategyParamsV2) -> Signal:
    """Evalúa la estrategia v2 en un único snapshot M1.

    Args:
        snapshot: fila del DataFrame snapshots_M1 (pd.Series con features).
        params: parámetros de la estrategia.

    Returns:
        Signal con tipo 'BUY', 'SELL' o None y razón.
    """
    ts = snapshot.name

    if not is_in_asian_session(ts, params):
        return Signal(ts, None, "out_of_session")

    required = [
        "dist_to_asian_mid_pips_abs",
        "m15_bb_pct_b",
        "m5_consecutive_bull",
        "m15_consecutive_bear",
    ]
    if any(pd.isna(snapshot.get(f)) for f in required):
        return Signal(ts, None, "insufficient_data")

    dist_mid_abs = snapshot["dist_to_asian_mid_pips_abs"]
    bb_pct_b  = snapshot["m15_bb_pct_b"]
    cons_bull = snapshot["m5_consecutive_bull"]
    cons_bear = snapshot["m15_consecutive_bear"]

    if dist_mid_abs < params.min_distance_from_asian_mid_pips:
        return Signal(
            ts, None,
            f"too_close_to_asian_mid ({dist_mid_abs:.0f} < {params.min_distance_from_asian_mid_pips:.0f})",
        )

    is_buy  = (bb_pct_b > params.bb_pct_b_buy_threshold
               and cons_bull >= params.min_consecutive_bull_for_buy)
    is_sell = (bb_pct_b < params.bb_pct_b_sell_threshold
               and cons_bear >= params.min_consecutive_bear_for_sell)

    if is_buy and is_sell:
        return Signal(ts, None, "conflicting_signals")
    if is_buy:
        return Signal(
            ts, "BUY",
            f"dist_mid={dist_mid_abs:.0f} bb%={bb_pct_b:.2f} cons_bull={cons_bull:.0f}",
        )
    if is_sell:
        return Signal(
            ts, "SELL",
            f"dist_mid={dist_mid_abs:.0f} bb%={bb_pct_b:.2f} cons_bear={cons_bear:.0f}",
        )

    return Signal(ts, None, "no_momentum_signal")


def apply_strategy_v2_to_snapshots(
    snapshots_df: pd.DataFrame,
    params: Optional[StrategyParamsV2] = None,
    deduplicate_within_minutes: int = 30,
) -> pd.DataFrame:
    """Aplica v2 a todos los snapshots M1.

    Args:
        snapshots_df: DataFrame con índice DatetimeIndex UTC tz-aware y features.
        params: parámetros (default: StrategyParamsV2()).
        deduplicate_within_minutes: elimina señales del mismo tipo dentro de N min.

    Returns:
        DataFrame con columnas: timestamp_utc, signal, reason. Solo señales no-None.
    """
    params = params or StrategyParamsV2()

    signals: list[dict] = []
    for ts, row in snapshots_df.iterrows():
        sig = evaluate_signal_v2(row, params)
        if sig.signal is not None:
            signals.append({"timestamp_utc": ts, "signal": sig.signal, "reason": sig.reason})

    if not signals:
        return pd.DataFrame(columns=["timestamp_utc", "signal", "reason"])

    signals_df = pd.DataFrame(signals).sort_values("timestamp_utc").reset_index(drop=True)
    signals_df["timestamp_utc"] = pd.to_datetime(signals_df["timestamp_utc"], utc=True)

    if deduplicate_within_minutes > 0:
        keep: list[bool] = []
        last_kept: dict[str, Optional[pd.Timestamp]] = {"BUY": None, "SELL": None}
        for _, row in signals_df.iterrows():
            last = last_kept[row["signal"]]
            gap = (row["timestamp_utc"] - last).total_seconds() / 60 if last else float("inf")
            if gap >= deduplicate_within_minutes:
                keep.append(True)
                last_kept[row["signal"]] = row["timestamp_utc"]
            else:
                keep.append(False)
        signals_df = signals_df[keep].reset_index(drop=True)

    return signals_df
