"""Backtesting engine wrappers for both backtesting.py and vectorbt."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Type

import pandas as pd
from backtesting import Backtest, Strategy

logger = logging.getLogger(__name__)


def run_backtest(
    ohlc: pd.DataFrame,
    strategy_class: Type[Strategy],
    cash: float = 10_000.0,
    commission: float = 0.0002,
    margin: float = 1.0,
    trade_on_close: bool = False,
    exclusive_orders: bool = False,
    **strategy_params: object,
) -> dict:
    """Run an event-driven backtest using ``backtesting.py``.

    Parameters
    ----------
    ohlc:
        DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``
        (title-case as required by backtesting.py).
    strategy_class:
        A :class:`backtesting.Strategy` subclass.
    cash:
        Starting equity in account currency.
    commission:
        Per-trade commission as a fraction of trade value.
    margin:
        Required margin fraction (1.0 = no leverage).
    trade_on_close:
        Execute signals on the close of the signal bar (vs. next open).
    exclusive_orders:
        Cancel existing orders when a new one is placed.
    **strategy_params:
        Parameters forwarded to the strategy (accessible via ``self.*``).

    Returns
    -------
    dict
        Backtest stats dictionary produced by ``bt.run()``.
    """
    bt = Backtest(
        ohlc,
        strategy_class,
        cash=cash,
        commission=commission,
        margin=margin,
        trade_on_close=trade_on_close,
        exclusive_orders=exclusive_orders,
    )
    stats = bt.run(**strategy_params)
    logger.info(
        "Backtest complete — Return: %.2f%%, Sharpe: %.2f, Max DD: %.2f%%",
        stats["Return [%]"],
        stats["Sharpe Ratio"],
        stats["Max. Drawdown [%]"],
    )
    return stats
