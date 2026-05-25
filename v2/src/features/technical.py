"""Indicadores técnicos clásicos para XAUUSD — v2.

Usa la librería `ta` (technical-analysis-library-python).
Todos los outputs aplican .shift(1) en bloque al final: el valor en fila T
corresponde al indicador calculado al cierre de la barra T-1 (sin lookahead).
"""
import logging

import numpy as np
import pandas as pd

import ta.momentum as tam
import ta.trend as tat
import ta.volatility as tav

from v2.config.settings import PIP_VALUE

logger = logging.getLogger(__name__)


def compute_technical_indicators(
    ohlc_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Calcula indicadores técnicos clásicos para un DataFrame OHLCV.

    Args:
        ohlc_df: DataFrame con columnas open, high, low, close, tick_volume.
                 Indexado por DatetimeIndex UTC tz-aware.
        prefix: prefijo de columnas, ej. 'm15', 'h1', 'h4'.

    Returns:
        DataFrame con ~28 features, mismo índice que ohlc_df.
        Todas las features están shifteadas 1 barra (sin lookahead).

    Features (para prefix='m15'):
        RSI:      m15_rsi_14, m15_rsi_14_overbought, m15_rsi_14_oversold
        MACD:     m15_macd_line, m15_macd_signal, m15_macd_hist, m15_macd_hist_increasing
        BB(20):   m15_bb_upper, m15_bb_lower, m15_bb_mid, m15_bb_pct_b, m15_bb_width
        EMAs:     m15_ema_9, m15_ema_21, m15_ema_50, m15_ema_200
                  m15_ema_9_above_21, m15_ema_21_above_50, m15_ema_50_above_200
                  m15_dist_close_to_ema_21_pips, m15_dist_close_to_ema_50_pips
                  m15_dist_close_to_ema_200_pips
        ADX:      m15_adx_14, m15_adx_14_strong
        Stoch:    m15_stoch_k, m15_stoch_d, m15_stoch_overbought, m15_stoch_oversold
    """
    h = ohlc_df["high"]
    l = ohlc_df["low"]
    c = ohlc_df["close"]
    p = prefix
    features: dict[str, pd.Series] = {}

    # ── RSI(14) ──────────────────────────────────────────────────────────────
    rsi = tam.RSIIndicator(close=c, window=14, fillna=False).rsi()
    features[f"{p}_rsi_14"] = rsi
    features[f"{p}_rsi_14_overbought"] = (rsi >= 70).astype(float)
    features[f"{p}_rsi_14_oversold"] = (rsi <= 30).astype(float)

    # ── MACD(12, 26, 9) ──────────────────────────────────────────────────────
    macd_ind = tat.MACD(close=c, window_slow=26, window_fast=12, window_sign=9, fillna=False)
    macd_line = macd_ind.macd()
    macd_sig = macd_ind.macd_signal()
    macd_hist = macd_ind.macd_diff()
    features[f"{p}_macd_line"] = macd_line
    features[f"{p}_macd_signal"] = macd_sig
    features[f"{p}_macd_hist"] = macd_hist
    features[f"{p}_macd_hist_increasing"] = (macd_hist > macd_hist.shift(1)).astype(float)

    # ── Bollinger Bands(20, 2σ) ──────────────────────────────────────────────
    bb = tav.BollingerBands(close=c, window=20, window_dev=2, fillna=False)
    bb_up = bb.bollinger_hband()
    bb_lo = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    features[f"{p}_bb_upper"] = bb_up
    features[f"{p}_bb_lower"] = bb_lo
    features[f"{p}_bb_mid"] = bb_mid
    # %B: posición del close en la banda [0, 1]
    bb_width_raw = (bb_up - bb_lo).replace(0, np.nan)
    features[f"{p}_bb_pct_b"] = (c - bb_lo) / bb_width_raw
    features[f"{p}_bb_width"] = bb_width_raw / bb_mid.replace(0, np.nan)

    # ── EMAs ─────────────────────────────────────────────────────────────────
    ema9 = tat.EMAIndicator(close=c, window=9, fillna=False).ema_indicator()
    ema21 = tat.EMAIndicator(close=c, window=21, fillna=False).ema_indicator()
    ema50 = tat.EMAIndicator(close=c, window=50, fillna=False).ema_indicator()
    ema200 = tat.EMAIndicator(close=c, window=200, fillna=False).ema_indicator()

    features[f"{p}_ema_9"] = ema9
    features[f"{p}_ema_21"] = ema21
    features[f"{p}_ema_50"] = ema50
    features[f"{p}_ema_200"] = ema200

    features[f"{p}_ema_9_above_21"] = (ema9 > ema21).astype(float)
    features[f"{p}_ema_21_above_50"] = (ema21 > ema50).astype(float)
    features[f"{p}_ema_50_above_200"] = (ema50 > ema200).astype(float)

    features[f"{p}_dist_close_to_ema_21_pips"] = (c - ema21) / PIP_VALUE
    features[f"{p}_dist_close_to_ema_50_pips"] = (c - ema50) / PIP_VALUE
    features[f"{p}_dist_close_to_ema_200_pips"] = (c - ema200) / PIP_VALUE

    # ── ADX(14) ──────────────────────────────────────────────────────────────
    adx_ind = tat.ADXIndicator(high=h, low=l, close=c, window=14, fillna=False)
    adx = adx_ind.adx()
    features[f"{p}_adx_14"] = adx
    features[f"{p}_adx_14_strong"] = (adx >= 25).astype(float)

    # ── Stochastic(14, 3) ────────────────────────────────────────────────────
    stoch_ind = tam.StochasticOscillator(
        high=h, low=l, close=c, window=14, smooth_window=3, fillna=False
    )
    stoch_k = stoch_ind.stoch()
    stoch_d = stoch_ind.stoch_signal()
    features[f"{p}_stoch_k"] = stoch_k
    features[f"{p}_stoch_d"] = stoch_d
    features[f"{p}_stoch_overbought"] = (stoch_k >= 80).astype(float)
    features[f"{p}_stoch_oversold"] = (stoch_k <= 20).astype(float)

    # ── Construir y shiftar ───────────────────────────────────────────────────
    result = pd.DataFrame(features, index=ohlc_df.index)
    result = result.shift(1)

    logger.debug(
        "Indicadores técnicos %s: %d filas × %d features",
        prefix,
        len(result),
        len(result.columns),
    )
    return result
