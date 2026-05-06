"""Technical indicators using the `ta` library (Python 3.11 compatible).

All functions take a standard OHLCV DataFrame and return it with additional
indicator columns. An optional `prefix` argument disambiguates multi-timeframe
features (e.g. prefix="h1_" → "h1_rsi_14").
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
from ta.volatility import AverageTrueRange, BollingerBands

logger = logging.getLogger(__name__)


def calculate_all_indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Compute all technical indicators and append them to *df*.

    Parameters
    ----------
    df:
        OHLCV DataFrame. Must have columns ``open``, ``high``, ``low``,
        ``close``. Index should be tz-aware DatetimeIndex (UTC).
    prefix:
        String prepended to every added column. Use ``"h1_"`` etc. to
        distinguish features from different timeframes.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional indicator columns.
    """
    df = df.copy()
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    p = prefix

    # ── RSI ──────────────────────────────────────────────────────────────────
    df[f"{p}rsi_14"] = RSIIndicator(c, window=14).rsi()

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd = MACD(c, window_fast=12, window_slow=26, window_sign=9)
    df[f"{p}macd_line"]   = macd.macd()
    df[f"{p}macd_signal"] = macd.macd_signal()
    df[f"{p}macd_hist"]   = macd.macd_diff()

    # ── EMAs ─────────────────────────────────────────────────────────────────
    df[f"{p}ema_20"]  = EMAIndicator(c, window=20).ema_indicator()
    df[f"{p}ema_50"]  = EMAIndicator(c, window=50).ema_indicator()
    df[f"{p}ema_200"] = EMAIndicator(c, window=200).ema_indicator()

    df[f"{p}ema_bull_20_50"]  = (df[f"{p}ema_20"] > df[f"{p}ema_50"]).astype(int)
    df[f"{p}ema_full_bull"] = (
        (df[f"{p}ema_20"] > df[f"{p}ema_50"]) & (df[f"{p}ema_50"] > df[f"{p}ema_200"])
    ).astype(int)
    df[f"{p}ema_full_bear"] = (
        (df[f"{p}ema_20"] < df[f"{p}ema_50"]) & (df[f"{p}ema_50"] < df[f"{p}ema_200"])
    ).astype(int)

    # ── ATR ──────────────────────────────────────────────────────────────────
    df[f"{p}atr_14"] = AverageTrueRange(h, l, c, window=14).average_true_range()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = BollingerBands(c, window=20, window_dev=2)
    df[f"{p}bb_upper"]  = bb.bollinger_hband()
    df[f"{p}bb_lower"]  = bb.bollinger_lband()
    df[f"{p}bb_middle"] = bb.bollinger_mavg()
    df[f"{p}bb_width"]  = (df[f"{p}bb_upper"] - df[f"{p}bb_lower"]) / df[f"{p}bb_middle"]
    df[f"{p}bb_pct"]    = bb.bollinger_pband()  # 0 = at lower band, 1 = at upper

    # ── Stochastic ───────────────────────────────────────────────────────────
    stoch = StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df[f"{p}stoch_k"] = stoch.stoch()
    df[f"{p}stoch_d"] = stoch.stoch_signal()

    # ── Supertrend (manual — not in `ta` library) ────────────────────────────
    st_val, st_dir = _supertrend(df, period=10, multiplier=3.0)
    df[f"{p}supertrend"]     = st_val
    df[f"{p}supertrend_dir"] = st_dir  # +1 bullish, -1 bearish

    # ── Ichimoku ─────────────────────────────────────────────────────────────
    ichi = IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
    df[f"{p}ichi_tenkan"] = ichi.ichimoku_conversion_line()
    df[f"{p}ichi_kijun"]  = ichi.ichimoku_base_line()
    df[f"{p}ichi_a"]      = ichi.ichimoku_a()
    df[f"{p}ichi_b"]      = ichi.ichimoku_b()
    cloud_top    = df[[f"{p}ichi_a", f"{p}ichi_b"]].max(axis=1)
    cloud_bottom = df[[f"{p}ichi_a", f"{p}ichi_b"]].min(axis=1)
    df[f"{p}ichi_above_cloud"] = (c > cloud_top).astype(int)
    df[f"{p}ichi_below_cloud"] = (c < cloud_bottom).astype(int)

    # ── Derived ──────────────────────────────────────────────────────────────
    atr = df[f"{p}atr_14"].replace(0, np.nan)
    df[f"{p}close_vs_ema50_atr"] = (c - df[f"{p}ema_50"]) / atr
    df[f"{p}bar_direction"]      = np.sign(c - o).astype(int)

    logger.debug("Indicators computed: %d bars, prefix='%s'", len(df), prefix)
    return df


def _supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Compute Supertrend; returns (supertrend_line, direction) as Series.

    direction = +1 when price is above Supertrend (bullish),
    direction = -1 when price is below Supertrend (bearish).
    """
    atr_vals = AverageTrueRange(
        df["high"], df["low"], df["close"], window=period
    ).average_true_range().values

    hl2   = ((df["high"] + df["low"]) / 2).values
    close = df["close"].values
    n     = len(close)

    upper = hl2 + multiplier * atr_vals
    lower = hl2 - multiplier * atr_vals

    final_upper = upper.copy()
    final_lower = lower.copy()
    st    = np.full(n, np.nan)
    direc = np.zeros(n, dtype=int)

    for i in range(1, n):
        if np.isnan(atr_vals[i]):
            continue

        final_lower[i] = (
            lower[i]
            if lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]
            else final_lower[i - 1]
        )
        final_upper[i] = (
            upper[i]
            if upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]
            else final_upper[i - 1]
        )

        if np.isnan(st[i - 1]):
            direc[i] = 1
            st[i]    = final_lower[i]
        elif st[i - 1] == final_upper[i - 1]:
            if close[i] <= final_upper[i]:
                direc[i] = -1
                st[i]    = final_upper[i]
            else:
                direc[i] = 1
                st[i]    = final_lower[i]
        else:
            if close[i] >= final_lower[i]:
                direc[i] = 1
                st[i]    = final_lower[i]
            else:
                direc[i] = -1
                st[i]    = final_upper[i]

    return pd.Series(st, index=df.index), pd.Series(direc, index=df.index)
