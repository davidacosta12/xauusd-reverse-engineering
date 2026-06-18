"""Analisis de lectura: PnL diario del backtest 3 años ya ejecutado.

NO MODIFICA NADA. Solo lee trades.parquet y calcula estadisticas.
"""
import pandas as pd
from pathlib import Path

trades_path = Path('v2/data/backtest_3years_twelvedata/XAU_USD_2023-05-29_2026-05-28/trades.parquet')
trades = pd.read_parquet(trades_path)

print('=' * 70)
print('ANALISIS DIARIO DEL BACKTEST 3 AÑOS (SOLO LECTURA)')
print('=' * 70)
print(f'\nTotal trades: {len(trades)}')

# Convertir entry_time a fecha
trades['entry_time'] = pd.to_datetime(trades['entry_time'])
trades['date'] = trades['entry_time'].dt.date

# Agrupar por dia
daily = trades.groupby('date').agg(
    pnl_dia=('net_pnl_usd', 'sum'),
    trades_dia=('net_pnl_usd', 'count'),
).reset_index()

# Ordenar
daily_sorted = daily.sort_values('pnl_dia')

print('\n=== TOP 15 PEORES DIAS ===')
print(daily_sorted.head(15).to_string(index=False))

print('\n=== TOP 10 MEJORES DIAS ===')
print(daily_sorted.tail(10).sort_values('pnl_dia', ascending=False).to_string(index=False))

print('\n=== DISTRIBUCION TRADES POR DIA ===')
print(daily['trades_dia'].value_counts().sort_index().to_string())

print('\n=== ESTADISTICAS DIARIAS ===')
print(f"Total dias con trades: {len(daily)}")
print(f"Peor dia (PnL): ${daily['pnl_dia'].min():.2f}")
print(f"Mejor dia (PnL): ${daily['pnl_dia'].max():.2f}")
print(f"PnL diario promedio: ${daily['pnl_dia'].mean():.2f}")
print(f"Max trades en un dia: {daily['trades_dia'].max()}")
print(f"Promedio trades por dia: {daily['trades_dia'].mean():.2f}")

print('\n=== SIMULACION FTMO POR LOTAJE ===')
print(f"Daily Loss Limit FTMO = -$5,000")
print()
limit = -5000
for factor, name in [(10, '0.20'), (15, '0.30'), (20, '0.40'), (25, '0.50'), (37.5, '0.75')]:
    bad_days = (daily['pnl_dia'] * factor < limit).sum()
    worst = daily['pnl_dia'].min() * factor
    print(f"  Lotaje {name} (x{factor}): peor dia = ${worst:.2f} | {bad_days} dias eliminados ({bad_days/len(daily)*100:.1f}%)")

# Detalle peor dia
print('\n=== DETALLE DEL PEOR DIA UNICO ===')
peor_dia = daily_sorted.iloc[0]
print(f"Fecha: {peor_dia['date']}")
print(f"PnL: ${peor_dia['pnl_dia']:.2f}")
print(f"Trades ese dia: {peor_dia['trades_dia']}")

# Trades del peor dia
trades_peor = trades[trades['date'] == peor_dia['date']]
print(f"\nDetalle trades del peor dia:")
cols = ['entry_time', 'direction', 'entry_price', 'exit_price', 'net_pnl_usd', 'exit_reason']
print(trades_peor[cols].to_string(index=False))

# Rachas perdedoras
print('\n=== RACHAS PERDEDORAS CONSECUTIVAS ===')
daily['perdedor'] = daily['pnl_dia'] < 0
daily['cambio'] = (daily['perdedor'] != daily['perdedor'].shift()).cumsum()
rachas = daily[daily['perdedor']].groupby('cambio').agg(
    dias_seguidos=('pnl_dia', 'count'),
    pnl_acumulado=('pnl_dia', 'sum'),
    fecha_inicio=('date', 'first'),
    fecha_fin=('date', 'last'),
).reset_index(drop=True)

if len(rachas) > 0:
    print(f"\nRacha perdedora mas larga: {rachas['dias_seguidos'].max()} dias seguidos")
    print(f"Peor racha por PnL acumulado: ${rachas['pnl_acumulado'].min():.2f}")
    print("\nTop 5 peores rachas:")
    print(rachas.nsmallest(5, 'pnl_acumulado').to_string(index=False))
