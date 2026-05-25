"""Verifica la cobertura temporal de los parquets criticos del proyecto."""
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from v2.config.settings import DATA_FEATURES, DATA_GROUND_TRUTH

print("=" * 70)
print("    VERIFICACION DE COBERTURA TEMPORAL")
print("=" * 70)

trades = pd.read_parquet(DATA_GROUND_TRUTH / "trades.parquet")
trades["time_open_utc"] = pd.to_datetime(trades["time_open_utc"], utc=True)
print(f"\nTrades.parquet:")
print(f"  Total: {len(trades)} (in-sample={(trades['sample']=='in_sample').sum()}, oos={(trades['sample']=='out_of_sample').sum()})")
print(f"  Primer trade in-sample: {trades[trades['sample']=='in_sample']['time_open_utc'].min()}")
print(f"  Ultimo trade in-sample: {trades[trades['sample']=='in_sample']['time_open_utc'].max()}")
print(f"  Primer trade OOS:       {trades[trades['sample']=='out_of_sample']['time_open_utc'].min()}")
print(f"  Ultimo trade OOS:       {trades[trades['sample']=='out_of_sample']['time_open_utc'].max()}")

print(f"\nOHLC ground_truth:")
for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
    path = DATA_GROUND_TRUTH / f"ohlc_{tf}.parquet"
    if not path.exists():
        print(f"  {tf}: NO EXISTE")
        continue
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    print(f"  {tf}: {len(df):,} barras | {df.index.min()} -> {df.index.max()}")

print(f"\nSnapshots:")
snap_path = DATA_FEATURES / "snapshots_M1.parquet"
if snap_path.exists():
    snap = pd.read_parquet(snap_path)
    if snap.index.tz is None:
        snap.index = snap.index.tz_localize("UTC")
    print(f"  snapshots_M1: {len(snap):,} filas | {snap.index.min()} -> {snap.index.max()}")
else:
    print(f"  snapshots_M1: NO EXISTE")

print()

oos_last = trades[trades["sample"] == "out_of_sample"]["time_open_utc"].max()
m1_path  = DATA_GROUND_TRUTH / "ohlc_M1.parquet"
if m1_path.exists():
    m1 = pd.read_parquet(m1_path)
    if m1.index.tz is None:
        m1.index = m1.index.tz_localize("UTC")
    m1_last   = m1.index.max()
    gap_hours = (oos_last - m1_last).total_seconds() / 3600
    print(f"GAP entre ultimo OHLC M1 ({m1_last}) y ultimo trade OOS ({oos_last}):")
    print(f"  {gap_hours:.1f} horas faltantes ({gap_hours/24:.1f} dias)")
    if gap_hours > 1:
        next_bar = m1_last + pd.Timedelta(minutes=1)
        target   = oos_last + pd.Timedelta(days=1)
        print(f"  HAY QUE EXTENDER el OHLC hasta cubrir {oos_last}")
        print(f"  RANGO A EXTRAER: {next_bar} -> {target}")
    else:
        print(f"  OK — cobertura completa.")
