"""Smart Money Concepts (SMC) feature builder.

Wraps the ``smartmoneyconcepts`` library to extract structure-based
features: BOS, CHoCH, Fair Value Gaps, and Order Blocks.

All features are computed on the bar that *closed*; no future information
is used. Features are appended as new columns to the input DataFrame.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import smartmoneyconcepts as smc  # type: ignore[import]
    _SMC_AVAILABLE = True
except ImportError:
    _SMC_AVAILABLE = False
    logger.warning(
        "smartmoneyconcepts not installed — SMC features will be NaN. "
        "Run: pip install smartmoneyconcepts"
    )

# Features produced by this module
FEATURE_NAMES: list[str] = [
    "bos_bullish",       # 1 if a bullish BOS occurred on this bar
    "bos_bearish",       # 1 if a bearish BOS occurred on this bar
    "choch_bullish",     # 1 if a bullish CHoCH occurred on this bar
    "choch_bearish",     # 1 if a bearish CHoCH occurred on this bar
    "fvg_bullish",       # 1 if bar is inside a bullish FVG
    "fvg_bearish",       # 1 if bar is inside a bearish FVG
    "ob_bullish",        # 1 if bar touches a bullish Order Block zone
    "ob_bearish",        # 1 if bar touches a bearish Order Block zone
    "premium_discount",  # +1 premium (above EQ), -1 discount (below EQ)
    "liquidity_sweep",   # 1 if a recent swing high/low was swept this bar
]


def add_smc_features(df: pd.DataFrame, swing_length: int = 10) -> pd.DataFrame:
    """Compute SMC features and append them to *df*.

    Parameters
    ----------
    df:
        OHLCV DataFrame (must have ``open``, ``high``, ``low``, ``close``).
    swing_length:
        Look-back for swing high/low detection (default 10 bars).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional SMC feature columns.
    """
    df = df.copy()

    if not _SMC_AVAILABLE:
        for col in FEATURE_NAMES:
            df[col] = np.nan
        return df

    ohlc = df[["open", "high", "low", "close"]].copy()

    # ── Structure (BOS / CHoCH) ───────────────────────────────────────────────
    structure = smc.swing_highs_lows(ohlc, swing_length=swing_length)
    bos_df = smc.bos_choch(ohlc, structure, close_break=True)

    df["bos_bullish"] = _flag_series(bos_df, "BOS", "Bullish", len(df))
    df["bos_bearish"] = _flag_series(bos_df, "BOS", "Bearish", len(df))
    df["choch_bullish"] = _flag_series(bos_df, "CHOCH", "Bullish", len(df))
    df["choch_bearish"] = _flag_series(bos_df, "CHOCH", "Bearish", len(df))

    # ── Fair Value Gaps ───────────────────────────────────────────────────────
    fvg_df = smc.fvg(ohlc)
    df["fvg_bullish"] = _binary_col(fvg_df, "FVG", "Bullish", len(df))
    df["fvg_bearish"] = _binary_col(fvg_df, "FVG", "Bearish", len(df))

    # ── Order Blocks ──────────────────────────────────────────────────────────
    ob_df = smc.ob(ohlc, structure)
    df["ob_bullish"] = _binary_col(ob_df, "OB", "Bullish", len(df))
    df["ob_bearish"] = _binary_col(ob_df, "OB", "Bearish", len(df))

    # ── Premium / Discount (relative to recent swing range) ──────────────────
    df["premium_discount"] = _premium_discount(ohlc, structure)

    # ── Liquidity sweep (high/low taken out) ──────────────────────────────────
    df["liquidity_sweep"] = _liquidity_sweep(ohlc, structure)

    logger.debug("SMC features added.")
    return df


# ── private helpers ────────────────────────────────────────────────────────────

def _flag_series(
    df_struct: pd.DataFrame,
    col_type: str,
    direction: str,
    n: int,
) -> pd.Series:
    """Return a binary Series of length *n* for a structure event."""
    out = pd.Series(0, index=range(n))
    if df_struct is None or df_struct.empty:
        return out
    mask = (df_struct.get("Type", pd.Series()) == col_type) & (
        df_struct.get("Direction", pd.Series()) == direction
    )
    idx = df_struct[mask].index
    out.iloc[idx[idx < n]] = 1
    return out


def _binary_col(
    df_feat: pd.DataFrame,
    col: str,
    direction: str,
    n: int,
) -> pd.Series:
    """Return a binary Series where the feature column matches *direction*."""
    out = pd.Series(0, index=range(n))
    if df_feat is None or df_feat.empty or col not in df_feat.columns:
        return out
    mask = df_feat[col] == direction
    idx = df_feat[mask].index
    out.iloc[idx[idx < n]] = 1
    return out


def _premium_discount(
    ohlc: pd.DataFrame,
    structure: pd.DataFrame,
    lookback: int = 50,
) -> pd.Series:
    """Classify each bar as premium (+1) or discount (-1) vs. mid of recent range."""
    mid = (ohlc["high"].rolling(lookback).max() + ohlc["low"].rolling(lookback).min()) / 2
    return np.sign(ohlc["close"] - mid).fillna(0).astype(int)


def _liquidity_sweep(ohlc: pd.DataFrame, structure: pd.DataFrame) -> pd.Series:
    """Flag bars where the current high/low takes out a recent swing level."""
    highs = ohlc["high"]
    lows = ohlc["low"]
    prev_high = highs.shift(1).rolling(5).max()
    prev_low = lows.shift(1).rolling(5).min()
    sweep = ((highs > prev_high) | (lows < prev_low)).astype(int)
    return sweep.fillna(0)
