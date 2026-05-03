"""Technical indicator features computed on OHLCV bars.

All features use only information available at bar-close time to prevent
look-ahead bias. Indicators are appended as new columns to the input
DataFrame without modifying the originals.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

# Features produced by this module (subset of the 45-feature spec)
FEATURE_NAMES: list[str] = [
    "rsi_14",
    "macd_line", "macd_signal", "macd_hist",
    "ema_20", "ema_50", "ema_200",
    "ema_cross_20_50",   # +1 bullish cross, -1 bearish, 0 no cross
    "atr_14",
    "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "stoch_k", "stoch_d",
    "adx_14", "di_plus", "di_minus",
    "cci_20",
    "donchian_upper", "donchian_lower", "donchian_mid",
    "vwap_dev",          # (close - VWAP) / ATR — deviation in ATR units
]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicator features and append them to *df*.

    Parameters
    ----------
    df:
        OHLCV DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``tick_volume``, and ``time`` (UTC, tz-aware index or column).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns defined in
        :data:`FEATURE_NAMES`. NaN rows (warmup period) are retained.
    """
    df = df.copy()

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd_line"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    # ── EMAs + directional cross ──────────────────────────────────────────────
    df["ema_20"] = ta.ema(df["close"], length=20)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)

    above = (df["ema_20"] > df["ema_50"]).astype(int)
    prev_above = above.shift(1)
    df["ema_cross_20_50"] = np.where(
        above - prev_above == 1, 1,
        np.where(above - prev_above == -1, -1, 0),
    )

    # ── ATR ───────────────────────────────────────────────────────────────────
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]
    df["bb_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / bb["BBM_20_2.0"]
    df["bb_pct"] = bb["BBP_20_2.0"]

    # ── Stochastic ────────────────────────────────────────────────────────────
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]

    # ── ADX / DI ─────────────────────────────────────────────────────────────
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx_14"] = adx["ADX_14"]
    df["di_plus"] = adx["DMP_14"]
    df["di_minus"] = adx["DMN_14"]

    # ── CCI ───────────────────────────────────────────────────────────────────
    df["cci_20"] = ta.cci(df["high"], df["low"], df["close"], length=20)

    # ── Donchian Channel ─────────────────────────────────────────────────────
    don = ta.donchian(df["high"], df["low"], lower_length=20, upper_length=20)
    df["donchian_upper"] = don["DCU_20_20"]
    df["donchian_lower"] = don["DCL_20_20"]
    df["donchian_mid"] = don["DCM_20_20"]

    # ── VWAP deviation (daily rolling VWAP) ──────────────────────────────────
    df = _add_vwap_deviation(df)

    logger.debug("Technical features added: %d rows, %d columns.", len(df), len(df.columns))
    return df


def _add_vwap_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily VWAP and express close distance in ATR units."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["tick_volume"].astype(float)

    # Use calendar date as grouping key for daily VWAP reset
    if pd.api.types.is_datetime64_any_dtype(df["time"]):
        date_key = df["time"].dt.date
    else:
        date_key = pd.to_datetime(df["time"]).dt.date

    df["_tp_vol"] = typical * vol
    df["_cum_tp_vol"] = df.groupby(date_key)["_tp_vol"].cumsum()
    df["_cum_vol"] = df.groupby(date_key)[vol.name if hasattr(vol, "name") else "tick_volume"].cumsum()
    df["_vwap"] = df["_cum_tp_vol"] / df["_cum_vol"]
    df["vwap_dev"] = (df["close"] - df["_vwap"]) / df["atr_14"].replace(0, np.nan)

    return df.drop(columns=["_tp_vol", "_cum_tp_vol", "_cum_vol", "_vwap"])
