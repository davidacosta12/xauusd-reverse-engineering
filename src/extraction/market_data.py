"""Download OHLCV bar data from MT5 for a given symbol and timeframe."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

from .mt5_client import MT5Client, MT5ConnectionError

logger = logging.getLogger(__name__)

# Mapping of human-readable timeframe strings to MT5 constants
TIMEFRAME_MAP: dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class MarketDataExtractor:
    """Download OHLCV bars from MT5 for one symbol and timeframe.

    Parameters
    ----------
    client:
        Authenticated :class:`MT5Client` instance.
    symbol:
        Instrument ticker (e.g. ``"XAUUSD"``).
    timeframe:
        Bar timeframe string, one of ``TIMEFRAME_MAP`` keys (default ``"M15"``).
    """

    def __init__(
        self,
        client: MT5Client,
        symbol: str = "XAUUSD",
        timeframe: str = "M15",
    ) -> None:
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. Valid: {list(TIMEFRAME_MAP)}"
            )
        self._client = client
        self._symbol = symbol
        self._tf_str = timeframe
        self._tf_mt5 = TIMEFRAME_MAP[timeframe]

    # ── public interface ───────────────────────────────────────────────────────

    def fetch(
        self,
        date_from: datetime,
        date_to: Optional[datetime] = None,
        extra_bars: int = 200,
    ) -> pd.DataFrame:
        """Download OHLCV bars for the requested window.

        Parameters
        ----------
        date_from:
            Start of the window (UTC). A buffer of ``extra_bars`` is
            prepended automatically so indicators have warmup data.
        date_to:
            End of the window (UTC). Defaults to now.
        extra_bars:
            Number of bars to prepend before ``date_from`` for indicator
            warmup (default 200).

        Returns
        -------
        pd.DataFrame
            Columns: time (UTC, tz-aware), open, high, low, close, tick_volume,
            real_volume, spread.

        Raises
        ------
        MT5ConnectionError
            If MT5 returns no bars.
        """
        if date_to is None:
            date_to = datetime.now(timezone.utc)

        self._client._require_connected()

        # Fetch with buffer using copy_rates_range
        bars = mt5.copy_rates_range(self._symbol, self._tf_mt5, date_from, date_to)

        if bars is None or len(bars) == 0:
            code, msg = mt5.last_error()
            raise MT5ConnectionError(
                f"copy_rates_range() returned no data [{code}]: {msg}"
            )

        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "tick_volume", "real_volume": "real_volume"})
        df = df.sort_values("time").reset_index(drop=True)

        logger.info(
            "Downloaded %d %s bars for %s  [%s → %s]",
            len(df),
            self._tf_str,
            self._symbol,
            df["time"].iloc[0],
            df["time"].iloc[-1],
        )
        return df

    def fetch_for_trades(
        self,
        trades: pd.DataFrame,
        extra_bars: int = 200,
    ) -> pd.DataFrame:
        """Convenience method: fetch bars spanning the full trade history.

        Parameters
        ----------
        trades:
            DataFrame with ``open_time`` and ``close_time`` columns (UTC).
        extra_bars:
            Warmup buffer bars prepended before the first trade.

        Returns
        -------
        pd.DataFrame
            Full OHLCV bar set covering all trades.
        """
        date_from: datetime = trades["open_time"].min().to_pydatetime()
        date_to: datetime = trades["close_time"].max().to_pydatetime()
        return self.fetch(date_from=date_from, date_to=date_to, extra_bars=extra_bars)

    def save(self, df: pd.DataFrame, path: Path) -> None:
        """Save OHLCV DataFrame as Parquet.

        Parameters
        ----------
        df:
            DataFrame returned by :meth:`fetch`.
        path:
            Destination file (should end in ``.parquet``).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("OHLCV data saved → %s (%d rows)", path, len(df))
