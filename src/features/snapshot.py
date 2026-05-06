"""Build per-trade feature snapshots (no look-ahead bias).

For each trade, all indicators are evaluated at the *last completed bar*
strictly before ``open_time``. This is enforced by slicing the index:
    last_bar = df[df.index < open_time].iloc[-1]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .microstructure import (
    get_distance_to_round_levels,
    get_killzone,
    get_session_active,
    get_session_levels,
)
from .smc import get_smc_snapshot, precompute_smc
from .technical import calculate_all_indicators

logger = logging.getLogger(__name__)

# ── Timeframe suffixes used as column prefixes ────────────────────────────────
_TF_PREFIX = {"M5": "m5_", "M15": "m15_", "H1": "h1_", "H4": "h4_"}


def _last_bar_before(df: pd.DataFrame, ts: pd.Timestamp) -> Optional[pd.Series]:
    """Return the last row whose index is strictly less than *ts*."""
    mask = df.index < ts
    if not mask.any():
        return None
    return df[mask].iloc[-1]


def _position_before(df: pd.DataFrame, ts: pd.Timestamp) -> int:
    """Return the integer position of the last bar before *ts*."""
    mask = df.index < ts
    if not mask.any():
        return 0
    return int(np.where(mask)[0][-1])


# ── Main public functions ─────────────────────────────────────────────────────

def build_features_per_trade(
    trades_df: pd.DataFrame,
    ohlc_dict: dict[str, pd.DataFrame],
    swing_length: int = 10,
) -> pd.DataFrame:
    """Build a feature DataFrame with one row per trade.

    Parameters
    ----------
    trades_df:
        DataFrame of trades (from ``trades_XAUUSD_target.parquet``).
        Must have columns ``ticket``, ``open_time``, ``open_price``, ``side``.
    ohlc_dict:
        Mapping timeframe → OHLCV DataFrame (DatetimeIndex, UTC).
        Expected keys: ``M5``, ``M15``, ``H1``, ``H4``.
    swing_length:
        Passed to SMC swing detection.

    Returns
    -------
    pd.DataFrame
        Shape (n_trades, ~50). Index = trade ticket.
    """
    logger.info("Computing indicators on all timeframes …")
    ind: dict[str, pd.DataFrame] = {}
    for tf, df_raw in ohlc_dict.items():
        prefix = _TF_PREFIX.get(tf, f"{tf.lower()}_")
        ind[tf] = calculate_all_indicators(df_raw, prefix=prefix)
        logger.info("  %s → %d bars, %d indicator columns", tf, len(df_raw), len(ind[tf].columns))

    logger.info("Pre-computing SMC on M15 and H1 …")
    smc_m15 = precompute_smc(ohlc_dict.get("M15", pd.DataFrame()), swing_length=swing_length)
    smc_h1  = precompute_smc(ohlc_dict.get("H1",  pd.DataFrame()), swing_length=swing_length)

    df_m15 = ohlc_dict.get("M15", pd.DataFrame())

    rows: list[dict] = []
    for _, trade in trades_df.iterrows():
        row = _build_single(trade, ind, smc_m15, smc_h1, df_m15)
        rows.append(row)
        logger.debug("  trade %s → %d features", trade["ticket"], len(row))

    df_features = pd.DataFrame(rows).set_index("ticket")
    logger.info("Feature matrix: %d trades × %d features", *df_features.shape)
    return df_features


def _build_single(
    trade: pd.Series,
    ind: dict[str, pd.DataFrame],
    smc_m15: dict,
    smc_h1: dict,
    df_m15_raw: pd.DataFrame,
) -> dict:
    """Build the feature dict for a single trade."""
    ts: pd.Timestamp = trade["open_time"]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    price = float(trade["open_price"])
    side  = trade["side"]          # "buy" or "sell"
    direction = 1 if side == "buy" else -1

    row: dict = {
        "ticket":       trade["ticket"],
        "side":         side,
        "direction":    direction,
        "open_time":    ts,
        "open_price":   price,
    }

    # ── Multi-timeframe indicator values ──────────────────────────────────────
    for tf, df_ind in ind.items():
        prefix  = _TF_PREFIX.get(tf, f"{tf.lower()}_")
        bar     = _last_bar_before(df_ind, ts)
        if bar is None:
            logger.warning("No bars before %s for %s — filling NaN", ts, tf)
            for col in df_ind.columns:
                if col not in ("open", "high", "low", "close", "tick_volume",
                               "real_volume", "spread"):
                    row[col] = float("nan")
            continue

        # Extract only indicator columns (not raw OHLCV)
        raw_cols = {"open", "high", "low", "close", "tick_volume", "real_volume", "spread"}
        for col, val in bar.items():
            if col not in raw_cols:
                row[col] = float(val) if pd.notna(val) else float("nan")

    # ── SMC features on M15 ───────────────────────────────────────────────────
    if "M15" in ind:
        bar_idx_m15 = _position_before(ind["M15"], ts)
        atr_m15     = float(row.get("m15_atr_14", 1.0) or 1.0)
        smc_snap    = get_smc_snapshot(smc_m15, bar_idx_m15, price, atr_m15)
        for k, v in smc_snap.items():
            row[f"m15_{k}"] = int(v)

    # ── SMC features on H1 ────────────────────────────────────────────────────
    if "H1" in ind:
        bar_idx_h1 = _position_before(ind["H1"], ts)
        atr_h1     = float(row.get("h1_atr_14", 1.0) or 1.0)
        smc_snap_h1 = get_smc_snapshot(smc_h1, bar_idx_h1, price, atr_h1)
        for k, v in smc_snap_h1.items():
            row[f"h1_{k}"] = int(v)

    # ── Session / microstructure features ────────────────────────────────────
    row["hour_utc"]      = ts.hour
    row["session"]       = get_session_active(ts)
    kz = get_killzone(ts)
    row["killzone"]      = kz if kz else "none"
    row["is_killzone"]   = int(kz is not None)

    rnd = get_distance_to_round_levels(price)
    row["dist_50"]  = rnd["dist_50"]
    row["dist_100"] = rnd["dist_100"]

    # ATR-normalised distances to round levels
    atr_ref = float(row.get("m15_atr_14", 1.0) or 1.0)
    row["dist_50_atr"]  = rnd["dist_50"]  / atr_ref
    row["dist_100_atr"] = rnd["dist_100"] / atr_ref

    # Session H/L
    if not df_m15_raw.empty:
        sess_levels = get_session_levels(df_m15_raw, ts)
        for k, v in sess_levels.items():
            row[k] = v
        # Distance from price to session levels (normalised by ATR)
        for k in ("asia_high", "asia_low", "london_high", "london_low"):
            lvl = row.get(k, float("nan"))
            row[f"{k}_dist_atr"] = abs(price - lvl) / atr_ref if pd.notna(lvl) else float("nan")

    # ── SMC alignment: does BOS direction match trade direction? ──────────────
    row["m15_bos_aligned"] = int(
        (direction == 1  and bool(row.get("m15_bos_bull_recent"))) or
        (direction == -1 and bool(row.get("m15_bos_bear_recent")))
    )
    row["m15_fvg_aligned"] = int(
        (direction == 1  and bool(row.get("m15_fvg_bull_near"))) or
        (direction == -1 and bool(row.get("m15_fvg_bear_near")))
    )
    row["h1_ema_aligned"] = int(
        (direction == 1  and bool(row.get("h1_ema_full_bull"))) or
        (direction == -1 and bool(row.get("h1_ema_full_bear")))
    )
    row["m15_st_aligned"] = int(
        float(row.get("m15_supertrend_dir", 0)) == direction
    )

    return row
