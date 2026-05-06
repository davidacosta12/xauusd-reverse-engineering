"""Download OHLCV bars and tick data from MT5.

All functions assume ``mt5.initialize()`` + ``mt5.login()`` have already
been called (i.e. a :class:`MT5Client` is connected).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

logger = logging.getLogger(__name__)

TIMEFRAME_MAP: dict[str, int] = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def get_bars(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Download OHLCV bars for *symbol* / *timeframe* from MT5.

    Parameters
    ----------
    symbol:
        Instrument ticker (e.g. ``"XAUUSD"``).
    timeframe:
        Bar period string — one of ``TIMEFRAME_MAP`` keys.
    start:
        Window start (UTC-aware datetime).
    end:
        Window end (UTC-aware). Defaults to ``datetime.now(UTC)``.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC), columns: open, high, low, close,
        tick_volume, real_volume, spread.

    Raises
    ------
    ValueError
        If *timeframe* is not in :data:`TIMEFRAME_MAP`.
    RuntimeError
        If MT5 returns no bars (includes broker error code).
    """
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. Valid options: {list(TIMEFRAME_MAP)}"
        )
    if end is None:
        end = datetime.now(timezone.utc)

    # Make symbol visible in Market Watch (required before copy_rates_range)
    if not mt5.symbol_select(symbol, True):
        code, msg = mt5.last_error()
        raise RuntimeError(f"symbol_select({symbol!r}) failed [{code}]: {msg}")

    bars = mt5.copy_rates_range(symbol, TIMEFRAME_MAP[timeframe], start, end)

    if bars is None or len(bars) == 0:
        code, msg = mt5.last_error()
        raise RuntimeError(
            f"copy_rates_range({symbol!r}, {timeframe}) returned no data [{code}]: {msg}\n"
            f"Window: {start} → {end}"
        )

    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()

    logger.info(
        "%s %s: %d bars  [%s → %s]",
        symbol, timeframe, len(df), df.index[0].date(), df.index[-1].date(),
    )
    return df


def get_ticks(
    symbol: str,
    start: datetime,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Download all ticks for *symbol* in the given window.

    Parameters
    ----------
    symbol:
        Instrument ticker.
    start:
        Window start (UTC-aware).
    end:
        Window end (UTC-aware). Defaults to now.

    Returns
    -------
    pd.DataFrame
        Index: ``time_msc`` (UTC, ms precision).
        Columns: time, bid, ask, last, volume, flags, volume_real.

    Raises
    ------
    RuntimeError
        If MT5 returns no ticks.
    """
    if end is None:
        end = datetime.now(timezone.utc)

    if not mt5.symbol_select(symbol, True):
        code, msg = mt5.last_error()
        raise RuntimeError(f"symbol_select({symbol!r}) failed [{code}]: {msg}")

    ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)

    if ticks is None or len(ticks) == 0:
        code, msg = mt5.last_error()
        raise RuntimeError(
            f"copy_ticks_range({symbol!r}) returned no data [{code}]: {msg}"
        )

    df = pd.DataFrame(ticks)
    df["time"]     = pd.to_datetime(df["time"],     unit="s",  utc=True)
    df["time_msc"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    df = df.set_index("time_msc").sort_index()

    logger.info("%s ticks: %d  [%s → %s]", symbol, len(df), df.index[0], df.index[-1])
    return df


def download_all_timeframes(
    symbol: str,
    start: datetime,
    end: Optional[datetime] = None,
    output_dir: Path = Path("data/raw"),
) -> dict[str, Path]:
    """Download bars for M1/M5/M15/M30/H1/H4/D1 and save as Parquet.

    Parameters
    ----------
    symbol:
        Instrument ticker.
    start:
        Window start (UTC-aware).
    end:
        Window end (UTC-aware). Defaults to now.
    output_dir:
        Directory for Parquet output. Created if absent.

    Returns
    -------
    dict[str, Path]
        Mapping timeframe → saved Parquet path for each successful download.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    for tf in TIMEFRAME_MAP:
        try:
            df = get_bars(symbol, tf, start, end)
            path = output_dir / f"bars_{symbol}_{tf}.parquet"
            df.to_parquet(path)
            saved[tf] = path
        except RuntimeError as exc:
            logger.warning("Skipping %s %s: %s", symbol, tf, exc)

    return saved
