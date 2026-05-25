"""Tests de la estrategia v3 y sus filtros operacionales."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from v2.src.strategies.asian_momentum_v3 import (
    StrategyParamsV3,
    _is_in_core_session,
    _session_date,
)


def test_core_session_hours():
    """Filtro 1: 22:00-02:59 UTC inclusive, fuera de ese rango no."""
    p = StrategyParamsV3()

    # Dentro de la sesion core
    assert _is_in_core_session(pd.Timestamp("2026-03-19 22:00", tz="UTC"), p)
    assert _is_in_core_session(pd.Timestamp("2026-03-19 23:30", tz="UTC"), p)
    assert _is_in_core_session(pd.Timestamp("2026-03-20 00:15", tz="UTC"), p)
    assert _is_in_core_session(pd.Timestamp("2026-03-20 01:00", tz="UTC"), p)
    assert _is_in_core_session(pd.Timestamp("2026-03-20 02:59", tz="UTC"), p)

    # Fuera de la sesion core
    assert not _is_in_core_session(pd.Timestamp("2026-03-20 03:00", tz="UTC"), p)
    assert not _is_in_core_session(pd.Timestamp("2026-03-20 04:30", tz="UTC"), p)
    assert not _is_in_core_session(pd.Timestamp("2026-03-20 05:00", tz="UTC"), p)
    assert not _is_in_core_session(pd.Timestamp("2026-03-19 20:00", tz="UTC"), p)
    assert not _is_in_core_session(pd.Timestamp("2026-03-19 21:59", tz="UTC"), p)


def test_session_date_assignment():
    """22h+ pertenece a la sesion del dia siguiente; 0-2h al dia actual."""
    p = StrategyParamsV3()

    # 22h del 18 marzo → sesion del 19 marzo
    assert _session_date(pd.Timestamp("2026-03-18 22:30", tz="UTC"), p) == datetime(2026, 3, 19).date()
    # 23h del 18 marzo → sesion del 19 marzo
    assert _session_date(pd.Timestamp("2026-03-18 23:45", tz="UTC"), p) == datetime(2026, 3, 19).date()
    # 01h del 19 marzo → sesion del 19 marzo
    assert _session_date(pd.Timestamp("2026-03-19 01:30", tz="UTC"), p) == datetime(2026, 3, 19).date()
    # 02h del 19 marzo → sesion del 19 marzo
    assert _session_date(pd.Timestamp("2026-03-19 02:50", tz="UTC"), p) == datetime(2026, 3, 19).date()
    # 22h del 26 marzo → sesion del 27 marzo
    assert _session_date(pd.Timestamp("2026-03-26 22:00", tz="UTC"), p) == datetime(2026, 3, 27).date()
