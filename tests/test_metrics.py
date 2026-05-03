"""Unit tests for backtest metrics (no MT5 connection required)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import directional_match, equity_correlation, profit_factor


def test_directional_match_perfect() -> None:
    pred = pd.Series([1, -1, 1, 1, -1])
    actual = pd.Series([1, -1, 1, 1, -1])
    assert directional_match(pred, actual) == 1.0


def test_directional_match_zero() -> None:
    pred = pd.Series([1, 1, 1])
    actual = pd.Series([-1, -1, -1])
    assert directional_match(pred, actual) == 0.0


def test_equity_correlation_perfect() -> None:
    eq = pd.Series([10_000, 10_100, 10_250, 10_400])
    assert abs(equity_correlation(eq, eq) - 1.0) < 1e-9


def test_profit_factor_no_losses() -> None:
    returns = pd.Series([100, 200, 50])
    assert profit_factor(returns) == float("inf")


def test_profit_factor_mixed() -> None:
    returns = pd.Series([200, -100])
    assert abs(profit_factor(returns) - 2.0) < 1e-9
