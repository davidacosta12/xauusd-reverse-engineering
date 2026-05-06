"""Microstructure features: sessions, killzones, round levels, session H/L.

All times are UTC. Session definitions follow ICT / retail convention:
  Asian   00:00 – 07:00 UTC
  London  07:00 – 16:00 UTC
  NY      12:00 – 21:00 UTC
  Overlap 12:00 – 16:00 UTC (London + NY)

Killzones (ICT):
  London KZ   07:00 – 10:00 UTC
  NY KZ       12:00 – 15:00 UTC
  Asian KZ    20:00 – 23:00 UTC
"""

from __future__ import annotations

import logging
from datetime import time

import pandas as pd

logger = logging.getLogger(__name__)

# ── Session time boundaries (UTC, half-open intervals [start, end)) ───────────
_SESSIONS: dict[str, tuple[int, int]] = {
    "london":  (7,  16),
    "ny":      (12, 21),
    "asian":   (0,   7),   # simplified: also covers 21-24 from previous day
}

_KILLZONES: dict[str, tuple[int, int]] = {
    "london": (7,  10),
    "ny":     (12, 15),
    "asian":  (20, 23),
}


def get_session_active(ts: pd.Timestamp) -> str:
    """Return the trading session active at *ts* (UTC).

    Returns one of: ``'london'``, ``'ny'``, ``'overlap'``, ``'asian'``, ``'off'``.
    """
    h = ts.hour + ts.minute / 60.0
    in_london = 7.0 <= h < 16.0
    in_ny     = 12.0 <= h < 21.0
    in_asian  = h < 7.0 or h >= 21.0

    if in_london and in_ny:
        return "overlap"
    if in_london:
        return "london"
    if in_ny:
        return "ny"
    if in_asian:
        return "asian"
    return "off"


def get_killzone(ts: pd.Timestamp) -> str | None:
    """Return the active killzone at *ts* (UTC), or ``None``."""
    h = ts.hour + ts.minute / 60.0
    for kz, (start, end) in _KILLZONES.items():
        if start <= h < end:
            return kz
    return None


def dist_to_round_level(price: float, step: float) -> float:
    """Absolute distance from *price* to the nearest multiple of *step*."""
    nearest = round(price / step) * step
    return abs(price - nearest)


def get_distance_to_round_levels(price: float) -> dict[str, float]:
    """Return distances (in price units) to nearest $50 and $100 round levels.

    Parameters
    ----------
    price:
        Current price (XAUUSD, e.g. 3250.40).

    Returns
    -------
    dict
        Keys: ``dist_50``, ``dist_100``.
    """
    return {
        "dist_50":  dist_to_round_level(price, 50.0),
        "dist_100": dist_to_round_level(price, 100.0),
    }


def get_session_levels(df_m15: pd.DataFrame, ts: pd.Timestamp) -> dict[str, float]:
    """Compute session high/low for Asian and London sessions visible at *ts*.

    Looks back at bars before *ts* to find:
    - Previous Asian session (00:00-07:00 UTC, yesterday)
    - Current or previous London session (07:00-16:00 UTC)

    Parameters
    ----------
    df_m15:
        M15 OHLCV DataFrame with tz-aware DatetimeIndex.
    ts:
        Timestamp of the trade entry (UTC-aware).

    Returns
    -------
    dict
        Keys: asia_high, asia_low, london_high, london_low.
        Returns NaN for any session with no data.
    """
    bars_before = df_m15[df_m15.index < ts]
    if bars_before.empty:
        return {k: float("nan") for k in ("asia_high", "asia_low", "london_high", "london_low")}

    today = ts.normalize()  # midnight UTC of ts's day

    # ── Asian session high/low: 00:00-07:00 on today OR yesterday ────────────
    _asia_high = _asia_low = float("nan")
    for day_offset in (0, -1):
        day = today + pd.Timedelta(days=day_offset)
        asia_bars = bars_before[
            (bars_before.index >= day) & (bars_before.index < day + pd.Timedelta(hours=7))
        ]
        if not asia_bars.empty:
            _asia_high = asia_bars["high"].max()
            _asia_low  = asia_bars["low"].min()
            break

    # ── London session high/low: 07:00-16:00 on today OR yesterday ───────────
    _lon_high = _lon_low = float("nan")
    for day_offset in (0, -1):
        day = today + pd.Timedelta(days=day_offset)
        lon_bars = bars_before[
            (bars_before.index >= day + pd.Timedelta(hours=7)) &
            (bars_before.index <  day + pd.Timedelta(hours=16))
        ]
        if not lon_bars.empty:
            _lon_high = lon_bars["high"].max()
            _lon_low  = lon_bars["low"].min()
            break

    return {
        "asia_high":   _asia_high,
        "asia_low":    _asia_low,
        "london_high": _lon_high,
        "london_low":  _lon_low,
    }
