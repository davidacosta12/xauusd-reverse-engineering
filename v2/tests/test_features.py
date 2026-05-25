"""Tests unitarios para los módulos de features v2.

Cubre point-in-time correctness y lógica de cálculo de niveles institucionales.
"""
import sys
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Asegurar que la raíz del proyecto está en el path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import ta.momentum as tam

from v2.src.features.levels import (
    compute_asian_levels,
    compute_distances_to_levels,
    compute_pdh_pdl,
)
from v2.src.features.microstructure import compute_microstructure_features
from v2.src.features.technical import compute_technical_indicators


# ─────────────────────────────────────────────────────────────────────────────
# Helpers para construir datos sintéticos
# ─────────────────────────────────────────────────────────────────────────────

def _make_m15_bars(
    start: str,
    periods: int,
    base_price: float = 3000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera un DataFrame OHLCV sintético de barras M15."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    close = base_price + np.cumsum(rng.normal(0, 1.0, periods))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.5, 3.0, periods)
    low = np.minimum(open_, close) - rng.uniform(0.5, 3.0, periods)
    vol = rng.integers(100, 1000, periods).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "tick_volume": vol},
        index=idx,
    )


def _make_ny_bars(date_str: str, high: float, low: float) -> pd.DataFrame:
    """Genera barras para una sesión NY completa con high/low controlados."""
    idx = pd.date_range(
        f"{date_str} 13:00", periods=36, freq="15min", tz="UTC"  # 13:00-21:45 UTC
    )
    n = len(idx)
    prices = np.linspace(low + 10, high - 10, n)
    # Asegurar que alguna barra toca exactamente el high y low esperados
    h_arr = prices.copy()
    l_arr = prices.copy()
    h_arr[len(h_arr) // 2] = high
    l_arr[len(l_arr) // 3] = low
    return pd.DataFrame(
        {
            "open": prices,
            "high": h_arr,
            "low": l_arr,
            "close": prices,
            "tick_volume": np.ones(n) * 100.0,
        },
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: PDH/PDL sin lookahead
# ─────────────────────────────────────────────────────────────────────────────

def test_pdh_pdl_no_lookahead():
    """El PDH del día actual NO puede ser el high del día actual."""
    # Día 1 (lunes 2026-03-16): sesión NY con high=5000, low=4900
    # Día 2 (martes 2026-03-17): sesión NY con high=6000, low=4800
    # Para timestamps del martes, PDH debe ser 5000 (del lunes), no 6000

    monday_bars = _make_ny_bars("2026-03-16", high=5000.0, low=4900.0)
    tuesday_bars = _make_ny_bars("2026-03-17", high=6000.0, low=4800.0)

    # Barras extra del martes fuera de NY (para tener timestamps de referencia)
    tuesday_extra = _make_m15_bars(
        "2026-03-17 09:00", periods=16, base_price=5200.0, seed=1
    )

    all_bars = pd.concat([monday_bars, tuesday_bars, tuesday_extra]).sort_index()
    all_bars = all_bars[~all_bars.index.duplicated(keep="first")]

    result = compute_pdh_pdl(all_bars)

    # Para cualquier timestamp del martes que tenga PDH (puede ser NaN en primeras horas)
    tuesday_with_pdh = result.loc["2026-03-17"].dropna(subset=["pdh"])

    assert len(tuesday_with_pdh) > 0, "Deberían existir filas del martes con PDH válido"

    # El PDH del martes debe ser el high del lunes (5000), NO el del martes (6000)
    assert ((tuesday_with_pdh["pdh"] - 5000.0).abs() < 0.01).all(), (
        f"PDH esperado ~5000, obtenido: {tuesday_with_pdh['pdh'].unique()}"
    )
    assert ((tuesday_with_pdh["pdl"] - 4900.0).abs() < 0.01).all(), (
        f"PDL esperado ~4900, obtenido: {tuesday_with_pdh['pdl'].unique()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Asian levels — ventana temporal correcta
# ─────────────────────────────────────────────────────────────────────────────

def test_asian_levels_window_correct():
    """Para T=03:00 UTC el asian_high es correcto; para T=01:00 UTC es NaN."""
    # Crear barras que cubren:
    # Domingo 22:00 - Lunes 06:00 UTC (ventana asian de "lunes")
    # Lunes: primeras horas y luego barras después de las 02:00

    # Barras en ventana asian del lunes (22:00 dom - 02:00 lun)
    asian_idx = pd.date_range(
        "2026-03-15 22:00", periods=16, freq="15min", tz="UTC"  # 22:00-25:45 (01:45 lun)
    )
    # Barras post-02:00 del lunes
    post_idx = pd.date_range(
        "2026-03-16 02:00", periods=20, freq="15min", tz="UTC"
    )

    known_high = 3100.0
    known_low = 3050.0

    asian_prices = np.linspace(3060, 3090, len(asian_idx))
    asian_high_arr = asian_prices.copy()
    asian_high_arr[5] = known_high   # poner el high conocido en alguna barra
    asian_low_arr = asian_prices.copy()
    asian_low_arr[10] = known_low    # poner el low conocido

    post_prices = np.linspace(3080, 3085, len(post_idx))

    asian_bars = pd.DataFrame(
        {
            "open": asian_prices,
            "high": asian_high_arr,
            "low": asian_low_arr,
            "close": asian_prices,
            "tick_volume": np.ones(len(asian_idx)) * 50.0,
        },
        index=asian_idx,
    )
    post_bars = pd.DataFrame(
        {
            "open": post_prices,
            "high": post_prices + 2,
            "low": post_prices - 2,
            "close": post_prices,
            "tick_volume": np.ones(len(post_idx)) * 50.0,
        },
        index=post_idx,
    )
    all_bars = pd.concat([asian_bars, post_bars]).sort_index()

    result = compute_asian_levels(all_bars)

    # Barra de las 01:00 UTC (DENTRO de la ventana) → rolling: tiene valor (no NaN)
    # Con rolling, 01:00 UTC puede ver barras desde 22:00 hasta 00:45.
    # El known_high=3100 está en el bar de las 23:15, que ya es visible a las 01:00.
    bar_01h = result.loc["2026-03-16 01:00":"2026-03-16 01:15"]
    assert not bar_01h["asian_high"].isna().all(), (
        "Con rolling: las barras dentro de la ventana deben tener asian_high corriente"
    )
    assert bar_01h["asian_high"].iloc[0] == pytest.approx(known_high, abs=0.01), (
        f"asian_high rolling en 01:00 debe ser {known_high}, "
        f"obtenido {bar_01h['asian_high'].iloc[0]}"
    )
    assert bar_01h["is_window_open"].all(), (
        "is_window_open debe ser True para barras dentro de la ventana asiática"
    )

    # Barra de las 03:00 UTC (tras cierre de ventana) → valor cerrado correcto
    bar_03h = result.loc["2026-03-16 03:00":"2026-03-16 03:15"]
    assert not bar_03h["asian_high"].isna().all(), (
        "Las barras después de las 02:00 UTC deben tener asian_high válido"
    )
    assert bar_03h["asian_high"].iloc[0] == pytest.approx(known_high, abs=0.01), (
        f"asian_high esperado {known_high}, obtenido {bar_03h['asian_high'].iloc[0]}"
    )
    assert bar_03h["asian_low"].iloc[0] == pytest.approx(known_low, abs=0.01), (
        f"asian_low esperado {known_low}, obtenido {bar_03h['asian_low'].iloc[0]}"
    )
    assert not bar_03h["is_window_open"].any(), (
        "is_window_open debe ser False para barras fuera de la ventana"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Sin NaN después del warmup en indicadores técnicos
# ─────────────────────────────────────────────────────────────────────────────

def test_no_nan_after_warmup():
    """Después de 201 filas no hay NaN en features técnicos (warmup EMA200)."""
    ohlc = _make_m15_bars("2026-01-01 00:00", periods=350, base_price=3000.0, seed=7)
    features = compute_technical_indicators(ohlc, prefix="m15")

    after_warmup = features.iloc[201:]
    float_cols = features.select_dtypes(include=[np.floating]).columns.tolist()

    nan_counts = after_warmup[float_cols].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]

    assert len(cols_with_nan) == 0, (
        f"Columnas con NaN después del warmup (row 201+):\n{cols_with_nan}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Shift aplicado correctamente (RSI)
# ─────────────────────────────────────────────────────────────────────────────

def test_shift_applied():
    """El RSI en row T=100 debe ser el RSI calculado SOLO sobre rows 0..99."""
    n = 110
    ohlc = _make_m15_bars("2026-01-01 00:00", periods=n, base_price=3000.0, seed=99)
    features = compute_technical_indicators(ohlc, prefix="m15")

    # Valor en row 100 tras shift(1) = RSI calculado en row 99
    # = RSI(close[0..99]) at index 99
    rsi_in_features_at_100 = features["m15_rsi_14"].iloc[100]

    # Calcular manualmente RSI sobre primeras 100 barras
    rsi_manual = (
        tam.RSIIndicator(close=ohlc["close"].iloc[:100], window=14, fillna=False)
        .rsi()
        .iloc[-1]
    )

    assert abs(rsi_in_features_at_100 - rsi_manual) < 1e-6, (
        f"Shift no aplicado correctamente: "
        f"features[100]={rsi_in_features_at_100:.6f} vs manual={rsi_manual:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Signo de distancias a niveles
# ─────────────────────────────────────────────────────────────────────────────

def test_distance_to_levels_signs():
    """dist_to_pdh_signed > 0 si close > pdh, < 0 si close < pdh."""
    idx = pd.date_range("2026-03-17 14:00", periods=4, freq="15min", tz="UTC")

    prices_above = pd.DataFrame(
        {"close": [5100.0, 5200.0, 5050.0, 5150.0]}, index=idx
    )
    prices_below = pd.DataFrame(
        {"close": [4800.0, 4700.0, 4900.0, 4750.0]}, index=idx
    )

    pdh_val = 5000.0
    pdl_val = 4900.0
    levels = pd.DataFrame(
        {
            "pdh": pdh_val,
            "pdl": pdl_val,
        },
        index=idx,
    )

    # Close encima del PDH → signed positivo
    dist_above = compute_distances_to_levels(prices_above, levels)
    assert (dist_above["dist_to_pdh_pips_signed"] > 0).all(), (
        "dist_to_pdh_signed debe ser positivo cuando close > pdh"
    )

    # Close debajo del PDH → signed negativo
    dist_below = compute_distances_to_levels(prices_below, levels)
    assert (dist_below["dist_to_pdh_pips_signed"] < 0).all(), (
        "dist_to_pdh_signed debe ser negativo cuando close < pdh"
    )

    # Magnitud: close=5100, pdh=5000 → diff=100, pips=100/0.10=1000
    expected_pips = (5100.0 - 5000.0) / 0.10
    assert dist_above["dist_to_pdh_pips_signed"].iloc[0] == pytest.approx(
        expected_pips, abs=0.01
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Asian levels — rolling running dentro de la ventana
# ─────────────────────────────────────────────────────────────────────────────

def test_asian_levels_rolling_running():
    """Verifica que dentro de la ventana 22:00-02:00 UTC, asian_high crece
    monótonamente con las barras disponibles (sin lookahead)."""
    # Tres barras M15 dentro de la ventana asiática:
    #   23:00 UTC  high=4800
    #   23:15 UTC  high=4810   ← nuevo máximo
    #   23:30 UTC  high=4805
    # El dataset comienza en 23:00 (no hay barras a las 22:xx en el test)
    idx_win = pd.date_range("2026-03-15 23:00", periods=3, freq="15min", tz="UTC")
    highs = [4800.0, 4810.0, 4805.0]
    lows  = [4780.0, 4790.0, 4785.0]

    bars_win = pd.DataFrame(
        {
            "open":        [4790.0, 4800.0, 4807.0],
            "high":        highs,
            "low":         lows,
            "close":       [4798.0, 4808.0, 4790.0],
            "tick_volume": [100.0, 120.0, 90.0],
        },
        index=idx_win,
    )

    # Añadir barras fuera de la ventana para tener un índice completo
    idx_post = pd.date_range("2026-03-16 02:00", periods=4, freq="15min", tz="UTC")
    bars_post = pd.DataFrame(
        {
            "open":  [4800.0] * 4,
            "high":  [4815.0] * 4,
            "low":   [4795.0] * 4,
            "close": [4808.0] * 4,
            "tick_volume": [80.0] * 4,
        },
        index=idx_post,
    )

    all_bars = pd.concat([bars_win, bars_post]).sort_index()
    result = compute_asian_levels(all_bars)

    # T=23:00 → primer bar en la ventana, no hay barras previas → NaN
    val_2300 = result.loc["2026-03-15 23:00", "asian_high"]
    assert pd.isna(val_2300), (
        f"T=23:00 (primer bar de la ventana): asian_high debe ser NaN, obtenido {val_2300}"
    )
    assert result.loc["2026-03-15 23:00", "is_window_open"], "23:00 debe estar en ventana"

    # T=23:15 → solo ve el bar de 23:00 (high=4800)
    val_2315 = result.loc["2026-03-15 23:15", "asian_high"]
    assert val_2315 == pytest.approx(4800.0, abs=0.01), (
        f"T=23:15: asian_high debe ser 4800 (solo bar 23:00 visible), obtenido {val_2315}"
    )

    # T=23:30 → ve bars de 23:00 (4800) y 23:15 (4810) → max=4810
    val_2330 = result.loc["2026-03-15 23:30", "asian_high"]
    assert val_2330 == pytest.approx(4810.0, abs=0.01), (
        f"T=23:30: asian_high debe ser 4810 (max de 23:00 y 23:15), obtenido {val_2330}"
    )

    # T=02:00 (fuera de la ventana) → valor cerrado = max de toda la ventana = 4810
    val_0200 = result.loc["2026-03-16 02:00", "asian_high"]
    assert val_0200 == pytest.approx(4810.0, abs=0.01), (
        f"T=02:00 (ventana cerrada): asian_high debe ser 4810 (máximo total), obtenido {val_0200}"
    )
    assert not result.loc["2026-03-16 02:00", "is_window_open"], "02:00 debe estar fuera de ventana"
