"""Analisis de operaciones duplicadas en demo MEX Atlantic.

NO MODIFICA NADA. Solo lee historial y reporta duplicados.
"""
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

# Conectar a MT5
if not mt5.initialize():
    print(f"Error: {mt5.last_error()}")
    exit()

# Obtener historial completo desde 29/05/2026
desde = datetime(2026, 5, 29)
hasta = datetime.now()

deals = mt5.history_deals_get(desde, hasta)
if deals is None:
    print(f"No hay deals: {mt5.last_error()}")
    exit()

# Convertir a DataFrame
deals_df = pd.DataFrame([d._asdict() for d in deals])
print(f"Total deals: {len(deals_df)}")

# Filtrar solo entradas (no cierres)
entries = deals_df[deals_df['entry'] == 0].copy()
entries['time'] = pd.to_datetime(entries['time'], unit='s')

print(f"\nTotal entradas: {len(entries)}")
print("\n=== TODAS LAS ENTRADAS ===")
print(entries[['ticket', 'time', 'symbol', 'type', 'volume', 'price']].to_string())

# Buscar duplicados (mismo minuto, mismo simbolo, mismo tipo)
entries['minute'] = entries['time'].dt.floor('min')
dupes = entries.groupby(['minute', 'symbol', 'type']).filter(lambda x: len(x) > 1)

print(f"\n=== DUPLICADOS DETECTADOS: {len(dupes)} ordenes en grupos duplicados ===")
if len(dupes) > 0:
    print(dupes[['ticket', 'time', 'symbol', 'type', 'volume', 'price']].to_string())

    # Estadisticas
    grupos = dupes.groupby(['minute', 'symbol', 'type']).size()
    print(f"\nGrupos de duplicados: {len(grupos)}")
    print(f"Maximo duplicados simultaneos: {grupos.max()}")

# Detectar trades por dia
entries['date'] = entries['time'].dt.date
por_dia = entries.groupby('date').size().reset_index(name='trades_dia')
print("\n=== TRADES POR DIA ===")
print(por_dia.to_string())

print(f"\nMaximo trades en un dia: {por_dia['trades_dia'].max()}")
print(f"Promedio: {por_dia['trades_dia'].mean():.2f}")
print(f"Dias con 3+ trades: {(por_dia['trades_dia'] >= 3).sum()}")

mt5.shutdown()
