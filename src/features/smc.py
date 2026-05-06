"""Smart Money Concepts feature extraction via the smartmoneyconcepts library.

The library exposes a class-based API: ``from smartmoneyconcepts import smc``,
then call ``smc.swing_highs_lows(...)``, ``smc.bos_choch(...)``, etc.

``ob()`` requires a ``volume`` column; we rename ``tick_volume`` automatically.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from smartmoneyconcepts import smc as _smc
    _SMC_AVAILABLE = True
except ImportError:
    _SMC_AVAILABLE = False
    logger.warning("smartmoneyconcepts not installed — SMC features will be NaN.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _prep_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV DataFrame acceptable by smartmoneyconcepts.

    Resets the integer index (required by the library) and ensures a
    ``volume`` column exists (``ob()`` needs it).
    """
    cols = ["open", "high", "low", "close"]
    out = df[cols].copy().reset_index(drop=True)
    if "tick_volume" in df.columns:
        out["volume"] = df["tick_volume"].values
    elif "volume" not in df.columns:
        out["volume"] = 0.0
    return out


# ── public API ────────────────────────────────────────────────────────────────

def precompute_smc(
    df: pd.DataFrame,
    swing_length: int = 10,
) -> dict[str, Optional[pd.DataFrame]]:
    """Pre-compute all SMC structures on a full OHLC DataFrame.

    Parameters
    ----------
    df:
        OHLCV DataFrame (DatetimeIndex, any standard OHLCV columns).
    swing_length:
        Look-back for swing high/low detection. Smaller values → more swings.

    Returns
    -------
    dict with keys:
        ``swings``  — DataFrame(HighLow, Level)
        ``bos``     — DataFrame(BOS, CHOCH, Level, BrokenIndex)
        ``fvg``     — DataFrame(FVG, Top, Bottom, MitigatedIndex)
        ``ob``      — DataFrame(OB, Top, Bottom, OBVolume, MitigatedIndex) or None

    All DataFrames share integer index aligned to *df*.
    """
    result: dict[str, Optional[pd.DataFrame]] = {
        "swings": None, "bos": None, "fvg": None, "ob": None
    }

    if not _SMC_AVAILABLE:
        return result

    ohlc = _prep_ohlc(df)

    try:
        result["swings"] = _smc.swing_highs_lows(ohlc, swing_length=swing_length)
    except Exception as exc:
        logger.warning("swing_highs_lows failed: %s", exc)
        return result

    try:
        result["bos"] = _smc.bos_choch(ohlc, result["swings"], close_break=True)
    except Exception as exc:
        logger.warning("bos_choch failed: %s", exc)

    try:
        result["fvg"] = _smc.fvg(ohlc, join_consecutive=False)
    except Exception as exc:
        logger.warning("fvg failed: %s", exc)

    try:
        result["ob"] = _smc.ob(ohlc, result["swings"], close_mitigation=False)
    except Exception as exc:
        logger.warning("ob failed: %s", exc)

    return result


def get_smc_snapshot(
    smc_data: dict,
    bar_idx: int,
    price: float,
    atr: float,
    lookback: int = 20,
    fvg_atr_threshold: float = 2.0,
) -> dict[str, bool | int]:
    """Extract SMC state at a specific bar index.

    Produces binary features (True/False) suitable for ML or rule evaluation.

    Parameters
    ----------
    smc_data:
        Output of :func:`precompute_smc`.
    bar_idx:
        Integer position of the current bar (last completed bar before entry).
    price:
        Current close price (for proximity checks).
    atr:
        Current ATR value (for proximity checks).
    lookback:
        How many bars back to look for BOS/CHoCH events.
    fvg_atr_threshold:
        FVG Top or Bottom must be within this many ATRs of *price*.

    Returns
    -------
    dict
        Keys: bos_bull_recent, bos_bear_recent, choch_bull_recent,
        choch_bear_recent, fvg_bull_near, fvg_bear_near,
        ob_bull_near, ob_bear_near.
    """
    out: dict[str, bool | int] = {
        "bos_bull_recent":  False,
        "bos_bear_recent":  False,
        "choch_bull_recent": False,
        "choch_bear_recent": False,
        "fvg_bull_near":    False,
        "fvg_bear_near":    False,
        "ob_bull_near":     False,
        "ob_bear_near":     False,
    }

    # ── BOS / CHoCH ───────────────────────────────────────────────────────────
    bos_df = smc_data.get("bos")
    if bos_df is not None and not bos_df.empty and "BrokenIndex" in bos_df.columns:
        bi = bos_df["BrokenIndex"].dropna()
        window_mask = (bi <= bar_idx) & (bi >= bar_idx - lookback)
        window = bos_df.loc[bi[window_mask].index]

        out["bos_bull_recent"]   = bool((window.get("BOS",   pd.Series()) == 1.0).any())
        out["bos_bear_recent"]   = bool((window.get("BOS",   pd.Series()) == -1.0).any())
        out["choch_bull_recent"] = bool((window.get("CHOCH", pd.Series()) == 1.0).any())
        out["choch_bear_recent"] = bool((window.get("CHOCH", pd.Series()) == -1.0).any())

    # ── FVG (unmitigated, near price) ─────────────────────────────────────────
    fvg_df = smc_data.get("fvg")
    if fvg_df is not None and not fvg_df.empty and atr > 0:
        # Keep only FVGs that formed before current bar
        active = fvg_df[fvg_df.index <= bar_idx].copy()
        # Unmitigated: MitigatedIndex is NaN OR mitigated after current bar
        unmit_mask = active["MitigatedIndex"].isna() | (active["MitigatedIndex"] > bar_idx)
        active = active[unmit_mask]
        # Near price
        prox = fvg_atr_threshold * atr
        near = active[
            ((active["Top"] - price).abs() < prox) |
            ((active["Bottom"] - price).abs() < prox) |
            ((price >= active["Bottom"]) & (price <= active["Top"]))
        ]
        out["fvg_bull_near"] = bool((near.get("FVG", pd.Series()) == 1.0).any())
        out["fvg_bear_near"] = bool((near.get("FVG", pd.Series()) == -1.0).any())

    # ── Order Blocks (unmitigated, near price) ────────────────────────────────
    ob_df = smc_data.get("ob")
    if ob_df is not None and not ob_df.empty and atr > 0:
        active = ob_df[ob_df.index <= bar_idx].copy()
        unmit_mask = active["MitigatedIndex"].isna() | (active["MitigatedIndex"] > bar_idx)
        active = active[unmit_mask]
        prox = fvg_atr_threshold * atr
        near = active[
            ((active["Top"] - price).abs() < prox) |
            ((active["Bottom"] - price).abs() < prox) |
            ((price >= active["Bottom"]) & (price <= active["Top"]))
        ]
        out["ob_bull_near"] = bool((near.get("OB", pd.Series()) == 1.0).any())
        out["ob_bear_near"] = bool((near.get("OB", pd.Series()) == -1.0).any())

    return out
