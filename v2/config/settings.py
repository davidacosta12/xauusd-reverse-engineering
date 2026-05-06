"""Configuración central del proyecto v2."""
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
V2_ROOT = PROJECT_ROOT / "v2"
DATA_RAW_MT5 = V2_ROOT / "data" / "raw_mt5"
DATA_RAW_DUKA = V2_ROOT / "data" / "raw_dukascopy"
DATA_GROUND_TRUTH = V2_ROOT / "data" / "ground_truth"
DATA_FEATURES = V2_ROOT / "data" / "features"
DATA_REPORTS = V2_ROOT / "data" / "reports"

# ---------------------------------------------------------------------------
# Símbolo
# CRÍTICO: el broker MEX Atlantic usa "XAUUSD.." (dos puntos al final)
# ---------------------------------------------------------------------------
SYMBOL_BROKER = "XAUUSD.."      # Como aparece en MT5 MEX Atlantic
SYMBOL_STANDARD = "XAUUSD"      # Como aparece en Dukascopy y fuentes externas

# ---------------------------------------------------------------------------
# Periodos de análisis (siempre en UTC)
# Buffer de 2 semanas antes para features históricas
# ---------------------------------------------------------------------------
PERIOD_START_UTC = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
PERIOD_END_UTC = datetime(2026, 4, 30, 23, 59, 59, tzinfo=timezone.utc)

# Periodo exacto de los 30 trades reales
TRADES_PERIOD_START_UTC = datetime(2026, 3, 19, 0, 0, 0, tzinfo=timezone.utc)
TRADES_PERIOD_END_UTC = datetime(2026, 4, 27, 23, 59, 59, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# MT5 — Cuenta inversora (solo lectura)
# ---------------------------------------------------------------------------
MT5_SERVER = "MEXAtlantic-Real"
MT5_LOGIN = 921339
# CONFIRMADO vía symbol_info_tick: servidor corre en UTC+3
MT5_GMT_OFFSET_HOURS = 3

# ---------------------------------------------------------------------------
# Timeframes a extraer
# ---------------------------------------------------------------------------
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

# ---------------------------------------------------------------------------
# Tolerancia de validación cross-fuente
# XAU pip = $0.10 → 1 pip = 0.10 puntos de precio
# ---------------------------------------------------------------------------
PRICE_TOLERANCE_PIPS = 1.0
PIP_VALUE = 0.10

# ---------------------------------------------------------------------------
# Sesiones en UTC (límites hora)
# Asia real: 22:00–06:00 UTC
# London: 07:00–12:00 UTC
# NY: 13:00–21:00 UTC
# ---------------------------------------------------------------------------
SESSION_BOUNDS_UTC = {
    "asia":             (22, 6),
    "london":           (7, 12),
    "ny":               (13, 21),
}

# ---------------------------------------------------------------------------
# Ground truth de trades reales
# ---------------------------------------------------------------------------
TOTAL_TRADES = 30
WIN_RATE_REAL = 0.73
TOTAL_PNL_REAL = 421.40

# ---------------------------------------------------------------------------
# Targets de replicación (criterios de éxito v2)
# ---------------------------------------------------------------------------
TARGET_DIRECTIONAL_MATCH = 0.80   # ≥ 80% coincidencia direccional
TARGET_EQUITY_CORRELATION = 0.85  # ρ ≥ 0.85 con curva de equity real
