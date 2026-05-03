"""Trade-level matching between original and replicated trade lists."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def match_trades(
    original: pd.DataFrame,
    replicated: pd.DataFrame,
    time_tolerance_min: float = 15.0,
) -> pd.DataFrame:
    """Pair original trades with replicated trades by open time proximity.

    Parameters
    ----------
    original:
        DataFrame of real trades (must have ``open_time``, ``direction``).
    replicated:
        DataFrame of backtest trades (must have ``open_time``, ``direction``).
    time_tolerance_min:
        Maximum difference in minutes between open times to count as a match.

    Returns
    -------
    pd.DataFrame
        Matched pairs with columns:
        orig_open_time, orig_direction, rep_open_time, rep_direction,
        time_diff_min, direction_match.
    """
    rows = []
    rep_used: set[int] = set()

    for _, orig_row in original.iterrows():
        t_orig: pd.Timestamp = orig_row["open_time"]

        candidates = replicated[
            (replicated.index.isin(rep_used) == False)  # noqa: E712
            & (
                (replicated["open_time"] - t_orig).abs()
                <= pd.Timedelta(minutes=time_tolerance_min)
            )
        ]

        if candidates.empty:
            rows.append(
                {
                    "orig_open_time": t_orig,
                    "orig_direction": orig_row["direction"],
                    "rep_open_time": pd.NaT,
                    "rep_direction": None,
                    "time_diff_min": float("nan"),
                    "direction_match": False,
                }
            )
            continue

        # Pick the closest candidate
        best_idx = (candidates["open_time"] - t_orig).abs().idxmin()
        rep_row = replicated.loc[best_idx]
        rep_used.add(best_idx)

        time_diff = abs((rep_row["open_time"] - t_orig).total_seconds() / 60)
        dir_match = orig_row["direction"] == rep_row["direction"]

        rows.append(
            {
                "orig_open_time": t_orig,
                "orig_direction": orig_row["direction"],
                "rep_open_time": rep_row["open_time"],
                "rep_direction": rep_row["direction"],
                "time_diff_min": time_diff,
                "direction_match": dir_match,
            }
        )

    df_matched = pd.DataFrame(rows)
    match_rate = df_matched["direction_match"].mean()
    logger.info(
        "Matched %d / %d trades. Directional match rate: %.1f%%",
        df_matched["direction_match"].sum(),
        len(original),
        match_rate * 100,
    )
    return df_matched
