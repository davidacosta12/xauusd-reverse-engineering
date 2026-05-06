"""Extract and reconstruct closed round-trip trades from MT5 deal history.

MT5 stores trades as individual *deals* (one for entry, one for exit).
This module groups them by ``position_id`` into single round-trip rows,
preserving millisecond timestamps, magic numbers, and comments.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

logger = logging.getLogger(__name__)

# Column order for the output DataFrame
_COLUMNS = [
    "ticket", "symbol", "side", "volume",
    "open_time", "open_price",
    "close_time", "close_price",
    "sl", "tp",
    "commission", "swap", "profit",
    "magic", "comment_open", "comment_close",
    "duration_min",
]


def get_full_history(
    date_from: datetime,
    date_to: Optional[datetime] = None,
    symbol_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Extract and reconstruct all closed round-trip trades from MT5.

    Groups entry + exit deals by ``position_id``. Millisecond-precision
    timestamps come from ``time_msc``. SL/TP are sourced from the
    corresponding historical orders (last order per position wins).

    Parameters
    ----------
    date_from:
        Window start (UTC-aware datetime).
    date_to:
        Window end (UTC-aware datetime). Defaults to ``datetime.now(UTC)``.
    symbol_filter:
        If given, keep only rows where ``symbol.startswith(symbol_filter)``.
        Pass ``None`` to return all instruments.

    Returns
    -------
    pd.DataFrame
        Columns: ticket, symbol, side, volume, open_time, open_price,
        close_time, close_price, sl, tp, commission, swap, profit,
        magic, comment_open, comment_close, duration_min.

    Raises
    ------
    RuntimeError
        If MT5 returns no deals for the requested window.
    """
    if date_to is None:
        date_to = datetime.now(timezone.utc)

    # ── 1. Fetch raw deals ────────────────────────────────────────────────────
    raw_deals = mt5.history_deals_get(date_from, date_to)
    if raw_deals is None or len(raw_deals) == 0:
        code, msg = mt5.last_error()
        raise RuntimeError(
            f"history_deals_get() returned nothing [{code}]: {msg}\n"
            f"Window: {date_from} → {date_to}"
        )
    df_d = pd.DataFrame(list(raw_deals), columns=raw_deals[0]._asdict().keys())
    logger.info("Raw deals fetched: %d", len(df_d))

    # ── 2. Fetch historical orders for SL/TP ─────────────────────────────────
    raw_orders = mt5.history_orders_get(date_from, date_to)
    sl_tp: dict[int, tuple[float, float]] = {}
    if raw_orders and len(raw_orders) > 0:
        df_o = pd.DataFrame(list(raw_orders), columns=raw_orders[0]._asdict().keys())
        # Last order per position (SL/TP may be modified during the trade)
        for pos_id, grp in df_o.groupby("position_id"):
            last = grp.sort_values("time_done").iloc[-1]
            sl_tp[int(pos_id)] = (float(last["sl"]), float(last["tp"]))

    # ── 3. Keep only real trade deals (exclude balance/credit) ───────────────
    df_d = df_d[df_d["type"].isin([mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL])].copy()

    # ── 4. Separate entry and exit legs ──────────────────────────────────────
    entries = df_d[df_d["entry"] == mt5.DEAL_ENTRY_IN].copy()
    exits   = df_d[df_d["entry"].isin([mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT])].copy()

    if entries.empty:
        logger.warning("No entry deals found in window.")
        return pd.DataFrame(columns=_COLUMNS)

    # ── 5. Aggregate exit legs (handles partial closes) ──────────────────────
    def _agg_exits(grp: pd.DataFrame) -> pd.Series:
        total_vol = grp["volume"].sum()
        vwap = (grp["price"] * grp["volume"]).sum() / total_vol if total_vol > 0 else 0.0
        last = grp.sort_values("time_msc").iloc[-1]
        return pd.Series(
            {
                "close_time_msc": grp["time_msc"].max(),
                "close_price":    vwap,
                "profit":         grp["profit"].sum(),
                "swap":           grp["swap"].sum(),
                "commission_exit": grp["commission"].sum(),
                "comment_close":  last["comment"],
            }
        )

    exit_agg = exits.groupby("position_id", group_keys=False).apply(_agg_exits).reset_index()

    # ── 6. Merge entry + exit on position_id ─────────────────────────────────
    # Drop fields that either conflict with exit_agg or are not needed in output:
    #   - "ticket"  : deal-level ticket (different from position_id → our "ticket")
    #   - "profit"  : always 0.0 on entry deals; real P&L comes from exit_agg
    #   - "swap"    : always 0.0 on entry deals; real swap comes from exit_agg
    #   - "fee"     : rarely used, not in output schema
    #   - "external_id" : not in output schema
    entries = entries.rename(columns={
        "comment":    "comment_open",
        "commission": "commission_entry",
    }).drop(columns=["ticket", "profit", "swap", "fee", "external_id"], errors="ignore")
    merged = entries.merge(exit_agg, on="position_id", how="inner")

    if merged.empty:
        logger.warning("No completed trades found (no matching exit deals).")
        return pd.DataFrame(columns=_COLUMNS)

    # ── 7. Build output columns ───────────────────────────────────────────────
    merged["open_time"]  = pd.to_datetime(merged["time_msc"],       unit="ms", utc=True)
    merged["close_time"] = pd.to_datetime(merged["close_time_msc"], unit="ms", utc=True)
    merged["side"]       = merged["type"].map(
        {mt5.DEAL_TYPE_BUY: "buy", mt5.DEAL_TYPE_SELL: "sell"}
    )
    merged["commission"]   = merged["commission_entry"] + merged["commission_exit"]
    merged["duration_min"] = (
        (merged["close_time"] - merged["open_time"]).dt.total_seconds() / 60
    )
    merged["sl"] = merged["position_id"].map(lambda x: sl_tp.get(x, (0.0, 0.0))[0])
    merged["tp"] = merged["position_id"].map(lambda x: sl_tp.get(x, (0.0, 0.0))[1])

    result = (
        merged
        .rename(columns={"position_id": "ticket", "price": "open_price"})
        [_COLUMNS]
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    # ── 8. Optional symbol filter ─────────────────────────────────────────────
    if symbol_filter:
        result = result[result["symbol"].str.startswith(symbol_filter)].copy()
        result = result.reset_index(drop=True)

    logger.info(
        "Reconstructed %d trades (filter=%r, window=%s → %s)",
        len(result), symbol_filter, date_from.date(), date_to.date(),
    )
    return result


def save_trades(df: pd.DataFrame, path: Path) -> None:
    """Persist a trades DataFrame as Parquet.

    Parameters
    ----------
    df:
        DataFrame as returned by :func:`get_full_history`.
    path:
        Destination file (should end in ``.parquet``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    size_kb = path.stat().st_size / 1024
    logger.info("Trades saved → %s  (%d rows, %.1f KB)", path.name, len(df), size_kb)
