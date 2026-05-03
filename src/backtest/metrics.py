"""Performance metrics and comparison utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def directional_match(
    predicted: pd.Series,
    actual: pd.Series,
) -> float:
    """Compute the fraction of trades where direction matches.

    Parameters
    ----------
    predicted:
        Series of predicted directions (+1 long, -1 short).
    actual:
        Series of actual trade directions (+1 long, -1 short).

    Returns
    -------
    float
        Match rate in [0, 1].
    """
    aligned_pred, aligned_act = predicted.align(actual, join="inner")
    return float((aligned_pred == aligned_act).mean())


def equity_correlation(
    equity_a: pd.Series,
    equity_b: pd.Series,
) -> float:
    """Pearson correlation between two equity curves.

    Parameters
    ----------
    equity_a, equity_b:
        Equity time-series (must share a comparable index or same length).

    Returns
    -------
    float
        Pearson ρ in [-1, 1].
    """
    a, b = equity_a.reset_index(drop=True), equity_b.reset_index(drop=True)
    min_len = min(len(a), len(b))
    return float(np.corrcoef(a.iloc[:min_len], b.iloc[:min_len])[0, 1])


def profit_factor(returns: pd.Series) -> float:
    """Gross profit / gross loss ratio.

    Parameters
    ----------
    returns:
        Per-trade P&L series.

    Returns
    -------
    float
        Profit factor (inf if no losing trades).
    """
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    return float(gains / losses) if losses > 0 else float("inf")


def summary_metrics(returns: pd.Series, equity: pd.Series) -> dict:
    """Compute a standard set of performance metrics.

    Parameters
    ----------
    returns:
        Per-trade P&L.
    equity:
        Cumulative equity curve.

    Returns
    -------
    dict
        Keys: win_rate, profit_factor, max_drawdown_pct,
              total_return_pct, n_trades.
    """
    wins = (returns > 0).sum()
    n = len(returns)
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min() * 100

    return {
        "n_trades": int(n),
        "win_rate": float(wins / n) if n > 0 else 0.0,
        "profit_factor": profit_factor(returns),
        "max_drawdown_pct": float(max_dd),
        "total_return_pct": float(
            (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        ),
    }
