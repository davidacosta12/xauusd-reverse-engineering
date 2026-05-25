"""Features de microestructura de mercado para XAUUSD — v2.

Para cada bar T, TODOS los features usan datos de barras cerradas (T-1 y anteriores).
Implementación: compute sobre serie completa, luego .shift(1) en bloque al final.
"""
import logging

import numpy as np
import pandas as pd

from v2.config.settings import PIP_VALUE

logger = logging.getLogger(__name__)


def _true_range(df: pd.DataFrame) -> pd.Series:
    """True Range: max(H-L, |H-Cp|, |L-Cp|) donde Cp = close anterior."""
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    """ATR(window) como media móvil simple del True Range."""
    return _true_range(df).rolling(window=window, min_periods=window).mean()


def _count_consecutive(direction: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Cuenta barras consecutivas bull/bear (incluyendo la barra actual).

    Returns:
        (consecutive_bull, consecutive_bear) — ambas Series con mismo índice.
        0 cuando la dirección no coincide.
    """
    # Grupo: cada cambio de dirección incrementa el ID de grupo
    group = (direction != direction.shift(1)).cumsum()
    # Conteo 1-based dentro del grupo
    consec = direction.groupby(group).cumcount() + 1
    bull = consec.where(direction > 0, 0)
    bear = consec.where(direction < 0, 0)
    return bull, bear


def compute_microstructure_features(
    ohlc_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Calcula features de microestructura para un DataFrame OHLCV.

    Args:
        ohlc_df: DataFrame con columnas open, high, low, close, tick_volume.
                 Indexado por DatetimeIndex UTC tz-aware.
        prefix: prefijo de columnas, ej. 'm5', 'm15', 'h1'.

    Returns:
        DataFrame con 13 features, mismo índice que ohlc_df.
        Todas las features están shifteadas en 1 barra (sin lookahead):
        el valor en fila T corresponde a lo observable al abrir la barra T.

    Features:
        {p}_atr_14, {p}_atr_21, {p}_atr_50
        {p}_realized_vol_10, {p}_realized_vol_20
        {p}_body_pct, {p}_upper_wick_pct, {p}_lower_wick_pct
        {p}_range_expansion_5, {p}_range_expansion_20
        {p}_consecutive_bull, {p}_consecutive_bear
        {p}_volume_rel_20
        {p}_gap_pct
    """
    df = ohlc_df.copy()
    p = prefix
    features: dict[str, pd.Series] = {}

    # ── ATR ──────────────────────────────────────────────────────────────────
    features[f"{p}_atr_14"] = _atr(df, 14)
    features[f"{p}_atr_21"] = _atr(df, 21)
    features[f"{p}_atr_50"] = _atr(df, 50)

    # ── Volatilidad realizada (std de log-returns) ────────────────────────────
    log_ret = np.log(df["close"] / df["close"].shift(1))
    features[f"{p}_realized_vol_10"] = log_ret.rolling(10, min_periods=10).std()
    features[f"{p}_realized_vol_20"] = log_ret.rolling(20, min_periods=20).std()

    # ── Body / Wick ratios ───────────────────────────────────────────────────
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    features[f"{p}_body_pct"] = body / bar_range
    features[f"{p}_upper_wick_pct"] = upper_wick / bar_range
    features[f"{p}_lower_wick_pct"] = lower_wick / bar_range

    # ── Range expansion ──────────────────────────────────────────────────────
    # Expansión = range actual / media de los 5 (o 20) rangos anteriores
    mean_range_5 = bar_range.shift(1).rolling(5, min_periods=5).mean()
    mean_range_20 = bar_range.shift(1).rolling(20, min_periods=20).mean()
    features[f"{p}_range_expansion_5"] = bar_range / mean_range_5
    features[f"{p}_range_expansion_20"] = bar_range / mean_range_20

    # ── Barras consecutivas ──────────────────────────────────────────────────
    direction = np.sign(df["close"] - df["open"])
    bull_streak, bear_streak = _count_consecutive(direction)
    features[f"{p}_consecutive_bull"] = bull_streak
    features[f"{p}_consecutive_bear"] = bear_streak

    # ── Volumen relativo ─────────────────────────────────────────────────────
    vol = df["tick_volume"].replace(0, np.nan)
    mean_vol_20 = vol.shift(1).rolling(20, min_periods=20).mean()
    features[f"{p}_volume_rel_20"] = vol / mean_vol_20

    # ── Gap ──────────────────────────────────────────────────────────────────
    # gap_pct = (open_T - close_{T-1}) / close_{T-1}
    features[f"{p}_gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # ── Construir DataFrame y aplicar shift(1) en bloque ─────────────────────
    result = pd.DataFrame(features, index=ohlc_df.index)
    result = result.shift(1)

    logger.debug(
        "Microestructura %s: %d filas × %d features",
        prefix,
        len(result),
        len(result.columns),
    )
    return result
