"""Microstructure features derived from OHLCV bar geometry.

These features capture intra-bar price action patterns that may signal
order flow imbalances, momentum, or institutional activity.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    "bar_range",           # high - low (absolute)
    "bar_range_pct",       # bar_range as percentile vs. rolling 100 bars
    "body_ratio",          # |close - open| / bar_range
    "upper_wick_ratio",    # upper wick / bar_range
    "lower_wick_ratio",    # lower wick / bar_range
    "bar_direction",       # +1 bullish, -1 bearish, 0 doji
    "volume_ma_ratio",     # tick_volume / rolling 20-bar mean volume
    "range_ma_ratio",      # bar_range / rolling 20-bar mean range
    "spread_proxy",        # (high - low - |close - open|) / close — approx spread
    "successive_direction",# count of bars in same direction (streak)
]


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute microstructure features and append them to *df*.

    Parameters
    ----------
    df:
        OHLCV DataFrame with ``open``, ``high``, ``low``, ``close``,
        ``tick_volume`` columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional microstructure feature columns.
    """
    df = df.copy()

    # ── Bar geometry ──────────────────────────────────────────────────────────
    df["bar_range"] = df["high"] - df["low"]
    df["bar_body"] = (df["close"] - df["open"]).abs()

    _safe_range = df["bar_range"].replace(0, np.nan)
    df["body_ratio"] = df["bar_body"] / _safe_range

    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["upper_wick_ratio"] = upper_wick / _safe_range
    df["lower_wick_ratio"] = lower_wick / _safe_range

    # ── Bar direction ─────────────────────────────────────────────────────────
    df["bar_direction"] = np.sign(df["close"] - df["open"]).astype(int)

    # ── Range percentile (rolling 100-bar window) ─────────────────────────────
    df["bar_range_pct"] = (
        df["bar_range"]
        .rolling(100, min_periods=10)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    # ── Volume ratios ─────────────────────────────────────────────────────────
    vol_ma = df["tick_volume"].rolling(20, min_periods=1).mean()
    df["volume_ma_ratio"] = df["tick_volume"] / vol_ma.replace(0, np.nan)

    range_ma = df["bar_range"].rolling(20, min_periods=1).mean()
    df["range_ma_ratio"] = df["bar_range"] / range_ma.replace(0, np.nan)

    # ── Spread proxy ──────────────────────────────────────────────────────────
    df["spread_proxy"] = (df["bar_range"] - df["bar_body"]) / df["close"]

    # ── Directional streak ────────────────────────────────────────────────────
    df["successive_direction"] = _directional_streak(df["bar_direction"])

    # Drop helper column
    df = df.drop(columns=["bar_body"])

    logger.debug("Microstructure features added.")
    return df


# ── private helpers ────────────────────────────────────────────────────────────

def _directional_streak(direction: pd.Series) -> pd.Series:
    """Count consecutive bars in the same direction (signed streak).

    A bullish streak is positive; bearish streak is negative; doji resets to 0.
    """
    streak = pd.Series(0, index=direction.index, dtype=int)
    for i in range(1, len(direction)):
        d = direction.iloc[i]
        if d == 0:
            streak.iloc[i] = 0
        elif d == direction.iloc[i - 1]:
            streak.iloc[i] = streak.iloc[i - 1] + d
        else:
            streak.iloc[i] = d
    return streak
