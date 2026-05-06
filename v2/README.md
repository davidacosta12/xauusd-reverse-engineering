# XAUUSD M15 — Reverse Engineering v2

Segunda iteración del proyecto de reverse-engineering de estrategia de trading.
Todo el trabajo nuevo vive en esta carpeta `v2/`. El trabajo previo (raíz del repo) se preserva.

---

## Objetivo

Replicar una estrategia **100% manual y discrecional** operada sobre XAUUSD M15
a partir de 30 trades reales cerrados en una cuenta MT5 investor-only (MEX Atlantic).

### Metas de éxito

| Métrica | Target |
|---------|--------|
| Coincidencia direccional (BUY/SELL) | ≥ 80% |
| Correlación equity curve (ρ Pearson) | ≥ 0.85 |

Si se cumplen ambas → generar EA funcional en MQL5.

---

## Hallazgos clave del diagnóstico previo (v1)

- Estrategia **discrecional pura**: sin magic number, sin patrones sistemáticos obvios.
- **96.7% de entradas en sesión asiática** (22:00–06:00 UTC) — foco casi exclusivo.
- Servidor MT5 = GMT+3. Todo tiempo almacenado en v1 era hora server sin convertir → fuente de bugs.
- Símbolo en MEX Atlantic: `XAUUSD..` (dos puntos al final) — crítico para `copy_rates_range`.
- Win rate real: 73% | P&L: +$421.40 | 30 trades | 2026-03-19 → 2026-04-27.

---

## Notas críticas de zona horaria

```
Servidor MT5 (MEX Atlantic): GMT+3
Almacenamiento interno v2:   UTC explícito (tz-aware)

Conversión: v2/src/utils/timezone.py
  server_to_utc(ts)      — naive/tz-aware GMT+3 → UTC
  utc_to_server(ts)      — UTC → GMT+3 (para llamadas a MT5)
  df_server_to_utc(df)   — DataFrame completo
  get_session_utc(ts)    — clasifica sesión de mercado
```

**Regla:** ningún DataFrame sale de los extractores sin UTC explícito.

---

## Notas de símbolo

```python
SYMBOL_BROKER   = "XAUUSD.."   # MEX Atlantic — incluye los dos puntos
SYMBOL_STANDARD = "XAUUSD"     # Dukascopy, yfinance, referencias externas
```

Si usas el símbolo incorrecto en `copy_rates_range`, MT5 devuelve `None` silenciosamente.

---

## Estructura de carpetas

```
v2/
├── README.md                   # este archivo
├── config/
│   └── settings.py             # configuración central (paths, símbolos, periodos)
├── notebooks/
│   └── 00_ground_truth_extraction.ipynb   # Fase 0: extracción y validación
├── src/
│   ├── data/
│   │   ├── mt5_extractor.py    # extractor MT5 (trades + OHLCV)
│   │   ├── dukascopy_extractor.py  # extractor Dukascopy (referencia externa)
│   │   └── data_validator.py   # validación cross-fuente
│   ├── features/               # (Fase 2) ingeniería de features
│   ├── rules/                  # (Fase 3) reglas de estrategia
│   ├── backtest/               # (Fase 4) backtesting
│   └── utils/
│       └── timezone.py         # conversión de zonas horarias
├── data/
│   ├── raw_mt5/                # OHLCV + trades desde MT5 (gitignored)
│   ├── raw_dukascopy/          # OHLCV desde Dukascopy (gitignored)
│   ├── ground_truth/           # datos validados y elegidos como fuente de verdad
│   ├── features/               # datasets con features calculados
│   └── reports/                # reportes HTML de validación
├── ea/                         # EA MQL5 generados (Fase 5, condicional)
└── tests/
    └── __init__.py
```

---

## Pipeline de fases

| Fase | Notebook | Descripción | Estado |
|------|----------|-------------|--------|
| 0 | `00_ground_truth_extraction.ipynb` | Extracción MT5 + Dukascopy, cross-validación, ground truth UTC | **Implementada** |
| 1 | `01_eda.ipynb` | EDA: distribución temporal, sesiones, hold time, SL/TP | Pendiente |
| 2 | `02_features.ipynb` | Features técnicas + SMC + microestructura | Pendiente |
| 3 | `03_strategy_id.ipynb` | Hipótesis manuales + XGBoost + SHAP (≤5 features) | Pendiente |
| 4 | `04_backtest.ipynb` | Backtesting event-driven, param sweep, métricas | Pendiente |
| 5 | `05_ea_generation.ipynb` | Generación MQL5 EA (si criterios cumplidos) | Condicional |

---

## Cómo ejecutar

### Requisitos previos

- Windows 10/11 con MetaTrader5 terminal instalado y logueado en cuenta 921339
- Python 3.11+ con `.venv` activo (ver `setup.ps1` en raíz del proyecto)
- Archivo `.env` en la raíz con `MT5_PASSWORD` y `MT5_PATH`

### Instalar dependencia adicional (Dukascopy)

```powershell
.\.venv\Scripts\Activate.ps1
pip install dukascopy-python
```

### Ejecutar Fase 0

```powershell
# Desde la raíz del proyecto, con .venv activo:
.\.venv\Scripts\Activate.ps1
jupyter notebook v2/notebooks/00_ground_truth_extraction.ipynb
```

O con JupyterLab:

```powershell
jupyter lab v2/notebooks/00_ground_truth_extraction.ipynb
```

### Variables .env necesarias

```env
MT5_LOGIN=921339
MT5_PASSWORD=<tu password investor>
MT5_SERVER=MEXAtlantic-Real
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

---

## Checks post-ejecución del notebook 00

Tras correr el notebook, verifica:

- [ ] `data/ground_truth/trades.parquet` existe y tiene exactamente **30 filas**
- [ ] El periodo de trades es **2026-03-19 → 2026-04-27** (UTC)
- [ ] El profit total es aproximadamente **$421.40**
- [ ] `data/ground_truth/ohlc_M15.parquet` tiene índice UTC tz-aware
- [ ] La validación cross-fuente M15 muestra **≥ 95% de barras dentro de 1 pip**
- [ ] Los reportes HTML están en `data/reports/cross_validation_*.html`
- [ ] No hay warnings de "símbolo no encontrado" en el log de MT5

---

## Seguridad

- `.env` está en `.gitignore` — nunca se commitea.
- `data/raw_mt5/`, `data/raw_dukascopy/`, `data/ground_truth/` están en `.gitignore`.
- La cuenta MT5 es investor-only: no se pueden colocar órdenes.
