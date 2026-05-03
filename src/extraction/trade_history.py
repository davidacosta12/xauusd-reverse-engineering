"""Extract and clean closed-trade history from an MT5 account."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

from .mt5_client import MT5Client, MT5ConnectionError

logger = logging.getLogger(__name__)

# MT5 deal entry types
_ENTRY_IN = mt5.DEAL_ENTRY_IN
_ENTRY_OUT = mt5.DEAL_ENTRY_OUT
_ENTRY_INOUT = mt5.DEAL_ENTRY_INOUT


class TradeHistoryExtractor:
    """Download and normalise the closed-deal history from MT5.

    Only *completed* round-trip trades (entry + exit deals) are returned.
    Pending orders and deposits/withdrawals are filtered out.

    Parameters
    ----------
    client:
        An authenticated :class:`MT5Client` instance.
    symbol:
        Instrument filter (e.g. ``"XAUUSD"``). Pass ``None`` to fetch all.
    """

    def __init__(self, client: MT5Client, symbol: Optional[str] = "XAUUSD") -> None:
        self._client = client
        self._symbol = symbol

    # ── public interface ───────────────────────────────────────────────────────

    def fetch(
        self,
        date_from: datetime,
        date_to: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Download deals and reconstruct round-trip trades.

        Parameters
        ----------
        date_from:
            Start of the history window (UTC).
        date_to:
            End of the history window (UTC). Defaults to ``datetime.now(UTC)``.

        Returns
        -------
        pd.DataFrame
            One row per completed trade with columns:
            ticket, symbol, direction, open_time, close_time,
            open_price, close_price, volume, sl, tp,
            profit, commission, swap, duration_min.

        Raises
        ------
        MT5ConnectionError
            If MT5 returns no deals unexpectedly.
        """
        if date_to is None:
            date_to = datetime.now(timezone.utc)

        self._client._require_connected()

        deals = mt5.history_deals_get(date_from, date_to)
        if deals is None:
            code, msg = mt5.last_error()
            raise MT5ConnectionError(f"history_deals_get() failed [{code}]: {msg}")

        df_deals = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        logger.info("Fetched %d raw deals from MT5.", len(df_deals))

        df_trades = self._reconstruct_trades(df_deals)

        if self._symbol:
            df_trades = df_trades[df_trades["symbol"] == self._symbol].copy()

        logger.info(
            "Reconstructed %d completed trades for %s.", len(df_trades), self._symbol
        )
        return df_trades.reset_index(drop=True)

    def save(self, df: pd.DataFrame, path: Path) -> None:
        """Persist the trade DataFrame as a Parquet file.

        Parameters
        ----------
        df:
            DataFrame returned by :meth:`fetch`.
        path:
            Destination file path (should end in ``.parquet``).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Trade history saved → %s (%d rows)", path, len(df))

    # ── private helpers ────────────────────────────────────────────────────────

    def _reconstruct_trades(self, df_deals: pd.DataFrame) -> pd.DataFrame:
        """Match entry and exit deals into completed round-trip trades."""
        # Keep only real trades (exclude balance/credit operations)
        df_deals = df_deals[df_deals["entry"].isin([_ENTRY_IN, _ENTRY_OUT, _ENTRY_INOUT])].copy()
        df_deals["time_dt"] = pd.to_datetime(df_deals["time"], unit="s", utc=True)

        entries = df_deals[df_deals["entry"] == _ENTRY_IN].copy()
        exits = df_deals[df_deals["entry"].isin([_ENTRY_OUT, _ENTRY_INOUT])].copy()

        rows: list[dict] = []
        for _, entry in entries.iterrows():
            # Match exit deal by position_id
            matched = exits[exits["position_id"] == entry["position_id"]]
            if matched.empty:
                continue  # open trade — skip
            exit_deal = matched.iloc[0]

            direction = "buy" if entry["type"] == mt5.DEAL_TYPE_BUY else "sell"
            open_t: pd.Timestamp = entry["time_dt"]
            close_t: pd.Timestamp = exit_deal["time_dt"]

            rows.append(
                {
                    "ticket": int(entry["position_id"]),
                    "symbol": entry["symbol"],
                    "direction": direction,
                    "open_time": open_t,
                    "close_time": close_t,
                    "open_price": float(entry["price"]),
                    "close_price": float(exit_deal["price"]),
                    "volume": float(entry["volume"]),
                    "sl": float(entry.get("sl", 0.0)),
                    "tp": float(entry.get("tp", 0.0)),
                    "profit": float(exit_deal["profit"]),
                    "commission": float(entry["commission"]) + float(exit_deal["commission"]),
                    "swap": float(exit_deal["swap"]),
                    "duration_min": (close_t - open_t).total_seconds() / 60,
                }
            )

        return pd.DataFrame(rows)
