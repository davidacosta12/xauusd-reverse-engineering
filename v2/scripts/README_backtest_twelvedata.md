# Backtest Twelve Data — Guia de uso

Valida la estrategia v3 sobre cualquier periodo usando datos de Twelve Data API
(alternativa a MT5 para periodos historicos o para brokers sin API).

## Setup

### 1. Obtener API key

- Registrate en https://twelvedata.com (plan gratuito: 800 req/dia, 8 req/min)
- Copia tu key en `.env`:

```
TWELVE_DATA_API_KEY=tu_key_aqui
```

### 2. Instalar dependencias

```powershell
cd C:\Users\lenovo\OneDrive\Escritorio\xauusd-reverse-engineering
.\.venv\Scripts\Activate.ps1
pip install requests
```

`requests` es la unica dependencia nueva — todo lo demas ya estaba instalado.

---

## Uso basico

```powershell
# Activar entorno
.\.venv\Scripts\Activate.ps1

# Backtest periodo abril 2026 (ejemplo)
python -m v2.scripts.backtest_twelvedata --start 2026-04-01 --end 2026-04-30

# Periodo custom con parametros
python -m v2.scripts.backtest_twelvedata `
  --start 2026-01-01 `
  --end   2026-03-31 `
  --sl-buy 260.7 --tp-buy 272.1 `
  --sl-sell 342.8 --tp-sell 289.8 `
  --volume 0.02 `
  --spread 3.0 `
  --balance 2000

# Sin graficos (mas rapido)
python -m v2.scripts.backtest_twelvedata --start 2026-04-01 --end 2026-04-30 --no-plots

# Forzar re-descarga ignorando cache
python -m v2.scripts.backtest_twelvedata --start 2026-04-01 --end 2026-04-30 --force-refresh
```

## Todos los argumentos

| Argumento | Default | Descripcion |
|-----------|---------|-------------|
| `--symbol` | `XAU/USD` | Simbolo Twelve Data |
| `--start` | `2026-04-01` | Fecha inicio (YYYY-MM-DD) |
| `--end` | `2026-04-30` | Fecha fin (YYYY-MM-DD) |
| `--sl-buy` | `260.7` | SL en pips para BUY |
| `--tp-buy` | `272.1` | TP en pips para BUY |
| `--sl-sell` | `342.8` | SL en pips para SELL |
| `--tp-sell` | `289.8` | TP en pips para SELL |
| `--volume` | `0.02` | Lotes por operacion |
| `--spread` | `3.0` | Spread asumido (pips) — Twelve Data no provee |
| `--slip-entry` | `0.5` | Slippage de entrada (pips) |
| `--slip-exit` | `0.5` | Slippage de salida (pips) |
| `--balance` | `2000.0` | Balance inicial USD |
| `--force-refresh` | False | Ignorar cache y re-descargar |
| `--no-plots` | False | No generar graficos |
| `--output-dir` | auto | Directorio de salida |

## Outputs generados

Por defecto en `v2/data/backtest_twelve/XAU_USD_YYYY-MM-DD_YYYY-MM-DD/`:

```
snapshots_M1.parquet       # Features M1 del periodo
backtest_results.parquet   # Trades del backtest con PnL
metrics.json               # Metricas completas
plots/
  equity_curve.png
  pnl_distribution.png
  exit_reasons.png
  monthly_pnl.png
```

## Cache de datos

El extractor guarda los OHLC descargados en `v2/data/twelve_data/`:

```
XAU_USD_1min.parquet    # ~1MB por mes de datos M1
XAU_USD_5min.parquet
XAU_USD_15min.parquet
...
```

Las proximas ejecuciones sobre el mismo rango son instantaneas (0 llamadas API).
Para periodos nuevos solo descarga los datos faltantes.

## Limitaciones del plan gratuito

- 800 requests/dia — suficiente para descargar ~6 meses de datos M1 en una sesion
- 8 requests/minuto — el extractor duerme 8 segundos entre requests automaticamente
- 5000 velas por request — el extractor pagina automaticamente para periodos largos

Para 1 mes de datos M1 (~18,000 velas) se necesitan ~4 requests por timeframe x 5 timeframes = ~20 requests totales.

## Diferencias vs backtest con datos MT5

| Aspecto | MT5 | Twelve Data |
|---------|-----|-------------|
| Tick volume | Real | 0 (no disponible) |
| Spread | Real | Fijo (input `--spread`) |
| Precio | Exacto del broker | Precio medio de mercado |
| Disponibilidad | Solo mientras MT5 conectado | API publica, siempre disponible |

El volumen cero afecta features como `volume_rel_20` pero **no afecta** las features criticas de v3 (asian_mid, BB%B, consecutive candles).
