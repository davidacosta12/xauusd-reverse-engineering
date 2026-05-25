"""Estrategia v3.1: v3 SIN cooldown.

Diferencia vs v3:
- cooldown_minutes_within_day = 0 (eliminado)

Base empirica: el operador real toma trades consecutivos en hasta 18 minutos en
algunos dias. El cooldown de 90 min carecia de base y bloqueaba trades validos.

Filtros que SE MANTIENEN:
- Ventana horaria core 22:00-02:59 UTC (base: 29/30 trades en esa ventana)
- Max 2 trades por dia (base: 29/30 dias con <= 2 trades)
"""
from dataclasses import dataclass

import pandas as pd

from v2.src.strategies.asian_momentum_v3 import (
    StrategyParamsV3,
    apply_strategy_v3_to_snapshots,
)


@dataclass
class StrategyParamsV3_1(StrategyParamsV3):
    """v3.1 = v3 con cooldown=0 (eliminado)."""
    cooldown_minutes_within_day: int = 0


def apply_strategy_v3_1_to_snapshots(
    snapshots_df: pd.DataFrame,
    params: StrategyParamsV3_1 | None = None,
) -> pd.DataFrame:
    """Aplicacion identica a v3 pero con params v3.1 (cooldown=0).

    Reusa la logica de apply_strategy_v3_to_snapshots porque solo cambia el parametro.
    """
    params = params or StrategyParamsV3_1()
    return apply_strategy_v3_to_snapshots(snapshots_df, params)
