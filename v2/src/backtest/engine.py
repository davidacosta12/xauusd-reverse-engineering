"""Motor de backtest discreto para la estrategia Asian Momentum.

Para cada señal:
- Apertura: entry_price + slippage_entry pips (en contra)
- Salida por SL/TP: precio hit +/- slippage_exit pips
- Forced close al final del periodo si no hit SL ni TP

Convenciones:
- Pip XAUUSD = 0.10 USD por oz
- Tamaño de posicion: 0.01 lote (1 oz) -> $0.10 por pip
- contract_size = 100 (oz por lote); 0.01 lote = 1 oz
"""
from dataclasses import dataclass

import pandas as pd
import numpy as np

PIP = 0.10          # USD por pip por oz
CONTRACT_SIZE = 100  # oz por lote


@dataclass
class BacktestConfig:
    slippage_entry_pips: float = 0.5
    slippage_exit_pips:  float = 0.5
    spread_pips:         float = 3.0   # spread round-trip XAUUSD MEX Atlantic

    # SL/TP empíricos (se rellenan con medianas del operador real)
    sl_pips_buy:  float = 260.7
    sl_pips_sell: float = 342.8
    tp_pips_buy:  float = 272.1
    tp_pips_sell: float = 289.8

    volume_lots: float = 0.01   # 1 oz -> $0.10/pip
    max_hold_hours: int = 12


def execute_trade(
    signal_row: dict,
    ohlc_m1: pd.DataFrame,
    config: BacktestConfig,
) -> dict | None:
    """Ejecuta una señal desde su entrada hasta SL/TP/forced close.

    Returns:
        Dict con columnas de resultado, o None si no hay barras disponibles.
    """
    direction = signal_row["signal"]
    entry_ts  = pd.Timestamp(signal_row["timestamp_utc"])

    # Primera vela M1 DESPUÉS de la señal
    entry_idx = ohlc_m1.index.searchsorted(entry_ts, side="right")
    if entry_idx >= len(ohlc_m1):
        return None

    entry_bar = ohlc_m1.iloc[entry_idx]
    raw_entry = float(entry_bar["open"])

    if direction == "BUY":
        entry_price = raw_entry + config.slippage_entry_pips * PIP
        sl_price    = entry_price - config.sl_pips_buy  * PIP
        tp_price    = entry_price + config.tp_pips_buy  * PIP
    else:
        entry_price = raw_entry - config.slippage_entry_pips * PIP
        sl_price    = entry_price + config.sl_pips_sell * PIP
        tp_price    = entry_price - config.tp_pips_sell * PIP

    # Ventana de barras para simular la posición
    max_exit_ts = entry_ts + pd.Timedelta(hours=config.max_hold_hours)
    exit_end_idx = ohlc_m1.index.searchsorted(max_exit_ts, side="right")
    bars = ohlc_m1.iloc[entry_idx:exit_end_idx]

    if len(bars) == 0:
        return None

    exit_reason = "forced_close"
    exit_price: float | None = None
    exit_time:  pd.Timestamp | None = None

    for ts, bar in bars.iterrows():
        low  = float(bar["low"])
        high = float(bar["high"])

        if direction == "BUY":
            # Chequear SL primero (pesimista: asume que el low ocurre antes)
            if low <= sl_price:
                exit_price  = sl_price  - config.slippage_exit_pips * PIP
                exit_reason = "sl"
                exit_time   = ts
                break
            if high >= tp_price:
                exit_price  = tp_price  - config.slippage_exit_pips * PIP
                exit_reason = "tp"
                exit_time   = ts
                break
        else:  # SELL
            if high >= sl_price:
                exit_price  = sl_price  + config.slippage_exit_pips * PIP
                exit_reason = "sl"
                exit_time   = ts
                break
            if low <= tp_price:
                exit_price  = tp_price  + config.slippage_exit_pips * PIP
                exit_reason = "tp"
                exit_time   = ts
                break

    if exit_price is None:
        last_bar   = bars.iloc[-1]
        exit_price = float(last_bar["close"])
        exit_time  = last_bar.name

    # PnL
    if direction == "BUY":
        gross_pips = (exit_price - entry_price) / PIP
    else:
        gross_pips = (entry_price - exit_price) / PIP

    oz          = config.volume_lots * CONTRACT_SIZE   # 0.01 * 100 = 1 oz
    gross_pnl   = gross_pips * PIP * oz
    cost_usd    = config.spread_pips * PIP * oz        # spread round-trip
    net_pnl     = gross_pnl - cost_usd
    duration    = (exit_time - entry_ts).total_seconds() / 60

    return {
        "entry_time":    entry_ts,
        "direction":     direction,
        "entry_price":   round(entry_price, 2),
        "sl_price":      round(sl_price, 2),
        "tp_price":      round(tp_price, 2),
        "exit_time":     exit_time,
        "exit_price":    round(exit_price, 2),
        "exit_reason":   exit_reason,
        "duration_min":  round(duration, 1),
        "gross_pips":    round(gross_pips, 1),
        "gross_pnl_usd": round(gross_pnl, 2),
        "cost_usd":      round(cost_usd, 2),
        "net_pnl_usd":   round(net_pnl, 2),
    }


def run_backtest(
    signals_df: pd.DataFrame,
    ohlc_m1: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Aplica execute_trade a todas las señales y devuelve DataFrame de resultados."""
    results = []
    for _, row in signals_df.iterrows():
        result = execute_trade(row.to_dict(), ohlc_m1, config)
        if result is not None:
            results.append(result)
    return pd.DataFrame(results) if results else pd.DataFrame()
