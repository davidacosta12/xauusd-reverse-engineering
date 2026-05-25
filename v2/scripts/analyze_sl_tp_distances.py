"""Analiza como el operador define SL/TP en funcion del precio de entrada.

Outputs:
- Distancia media SL->entry en pips (por direccion)
- Distancia media entry->TP en pips
- Ratio R:R medio del operador
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from v2.config.settings import DATA_GROUND_TRUTH

PIP = 0.10  # XAUUSD


def main() -> dict:
    trades = pd.read_parquet(DATA_GROUND_TRUTH / "trades.parquet")
    trades = trades[trades["sample"] == "in_sample"].copy()

    def compute_distances(row: pd.Series) -> pd.Series:
        e  = row["price_open"]
        sl = row["sl_initial"]
        tp = row["tp_initial"]
        if row["type"] == "BUY":
            sl_pips = (e - sl) / PIP
            tp_pips = (tp - e) / PIP
        else:
            sl_pips = (sl - e) / PIP
            tp_pips = (e - tp) / PIP
        return pd.Series({"sl_distance_pips": sl_pips, "tp_distance_pips": tp_pips})

    distances = trades.apply(compute_distances, axis=1)
    trades = pd.concat([trades, distances], axis=1)
    trades["rr_ratio"] = trades["tp_distance_pips"] / trades["sl_distance_pips"]

    print("=" * 70)
    print("           ANALISIS SL/TP DEL OPERADOR (in-sample)")
    print("=" * 70)
    print()

    for direction in ["BUY", "SELL"]:
        subset = trades[trades["type"] == direction]
        print(f"--- {direction} ({len(subset)} trades) ---")
        print(f"  SL distance pips: media={subset['sl_distance_pips'].mean():.1f}  "
              f"mediana={subset['sl_distance_pips'].median():.1f}  "
              f"std={subset['sl_distance_pips'].std():.1f}")
        print(f"  TP distance pips: media={subset['tp_distance_pips'].mean():.1f}  "
              f"mediana={subset['tp_distance_pips'].median():.1f}  "
              f"std={subset['tp_distance_pips'].std():.1f}")
        print(f"  R:R ratio:        media={subset['rr_ratio'].mean():.2f}  "
              f"mediana={subset['rr_ratio'].median():.2f}")
        print()

    print("=== Distribucion SL en pips (todos) ===")
    print(trades["sl_distance_pips"].describe().to_string())
    print()
    print("=== Distribucion TP en pips (todos) ===")
    print(trades["tp_distance_pips"].describe().to_string())
    print()

    profile = {
        "sl_pips_buy_median":  float(trades[trades["type"] == "BUY"]["sl_distance_pips"].median()),
        "sl_pips_sell_median": float(trades[trades["type"] == "SELL"]["sl_distance_pips"].median()),
        "tp_pips_buy_median":  float(trades[trades["type"] == "BUY"]["tp_distance_pips"].median()),
        "tp_pips_sell_median": float(trades[trades["type"] == "SELL"]["tp_distance_pips"].median()),
        "rr_median":           float(trades["rr_ratio"].median()),
    }
    print("=== PERFIL ESTADISTICO (para uso en backtest) ===")
    for k, v in profile.items():
        print(f"  {k}: {v:.1f}")

    return profile


if __name__ == "__main__":
    main()
