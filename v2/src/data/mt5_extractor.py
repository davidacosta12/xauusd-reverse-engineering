"""Extractor MT5 para XAUUSD.. desde el broker MEX Atlantic.

Modo de operación:
- Trades: usa mt5.history_deals_get() y agrupa por position_id para
  reconstruir los trades reales (entrada + salida).
- OHLC: usa mt5.copy_rates_range() por timeframe.

Convención de tiempo:
- MT5 server = GMT+3. TODOS los timestamps se convierten a UTC antes de guardar.
"""
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from v2.config.settings import (
    DATA_RAW_MT5,
    MT5_GMT_OFFSET_HOURS,
    MT5_SERVER,
    MT5_LOGIN,
    SYMBOL_BROKER,
    PERIOD_START_UTC,
    PERIOD_END_UTC,
    TIMEFRAMES,
)

logger = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)

_TF_MAP: dict[str, int] = {}


def _get_tf_constant(tf_name: str) -> int:
    """Resuelve el nombre 'M15' a la constante MetaTrader5.TIMEFRAME_M15."""
    import MetaTrader5 as mt5
    mapping = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    if tf_name not in mapping:
        raise ValueError(f"Timeframe desconocido: {tf_name}. Opciones: {list(mapping)}")
    return mapping[tf_name]


def connect_mt5() -> bool:
    """Conecta a MT5 leyendo .env. Devuelve True si OK."""
    import MetaTrader5 as mt5

    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER", MT5_SERVER)
    login = int(os.getenv("MT5_LOGIN", str(MT5_LOGIN)))
    path = os.getenv("MT5_PATH") or None

    init_kwargs: dict = {}
    if path:
        init_kwargs["path"] = path

    if not mt5.initialize(**init_kwargs):
        error = mt5.last_error()
        raise RuntimeError(f"MT5 initialize() falló: {error}")

    if not mt5.login(login=login, password=password, server=server):
        error = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"MT5 login() falló para cuenta {login}: {error}")

    info = mt5.account_info()
    logger.info(
        "MT5 conectado | cuenta=%s | broker=%s | server=%s",
        info.login,
        info.company,
        info.server,
    )
    return True


def disconnect_mt5() -> None:
    """Cierra la conexión MT5."""
    import MetaTrader5 as mt5
    mt5.shutdown()


# Alias privado para compatibilidad interna
_connect_mt5 = connect_mt5
_shutdown_mt5 = disconnect_mt5


def _infer_close_type(comment: object) -> str:
    """Infiere el tipo de cierre a partir del comment del deal OUT."""
    if not isinstance(comment, str) or not comment.strip():
        return "manual"
    c = comment.lower()
    if "[sl" in c:
        return "sl"
    if "[tp" in c:
        return "tp"
    return "manual"



def extract_trade_history(
    date_from_utc: datetime | None = None,
    date_to_utc: datetime | None = None,
    symbol: str = SYMBOL_BROKER,
    save_parquet: bool = True,
) -> pd.DataFrame:
    """Extrae los trades del periodo. Devuelve DataFrame con UN row por posición real.

    Columnas:
        position_id, ticket_in, ticket_out,
        time_open_utc, time_close_utc, duration_minutes,
        type (str: 'BUY'/'SELL'), volume,
        price_open, price_close, sl, tp,
        swap, commission, profit,
        magic, comment_open, comment_close, comment_close_inferred_type

    comment_close_inferred_type es 'sl' / 'tp' / 'manual' inferido del comment del deal OUT.
    Tiempos en UTC tz-aware. Ordenado por time_open_utc ascendente.
    Guarda en v2/data/raw_mt5/trades.parquet si save_parquet=True.
    """
    import MetaTrader5 as mt5

    date_from_utc = date_from_utc or PERIOD_START_UTC
    date_to_utc = date_to_utc or PERIOD_END_UTC

    deals = mt5.history_deals_get(date_from_utc, date_to_utc)
    if deals is None or len(deals) == 0:
        raise RuntimeError(f"No deals retornados. last_error: {mt5.last_error()}")

    deals_df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    logger.info("Deals raw descargados: %d", len(deals_df))

    # Filtrar por símbolo exacto
    deals_df = deals_df[deals_df["symbol"] == symbol].copy()
    logger.info("Deals de %s: %d", symbol, len(deals_df))

    # Convertir time (Unix server GMT+3) → UTC datetime tz-aware
    deals_df["time_utc"] = (
        pd.to_datetime(deals_df["time"], unit="s", utc=True)
        - pd.Timedelta(hours=MT5_GMT_OFFSET_HOURS)
    )

    # Filtrar solo deals con entry IN (0) o OUT (1); descartar balance, credit, etc.
    deals_df = deals_df[deals_df["entry"].isin([mt5.DEAL_ENTRY_IN, mt5.DEAL_ENTRY_OUT])].copy()

    deals_in  = deals_df[deals_df["entry"] == mt5.DEAL_ENTRY_IN].copy()
    deals_out = deals_df[deals_df["entry"] == mt5.DEAL_ENTRY_OUT].copy()

    logger.info("Deals IN: %d | Deals OUT: %d", len(deals_in), len(deals_out))

    # Merge por position_id (inner = solo posiciones cerradas dentro del periodo)
    merged = pd.merge(
        deals_in.add_suffix("_in"),
        deals_out.add_suffix("_out"),
        left_on="position_id_in",
        right_on="position_id_out",
        how="inner",
    )

    result = pd.DataFrame({
        "position_id":    merged["position_id_in"].astype("int64"),
        "ticket_in":      merged["ticket_in"].astype("int64"),
        "ticket_out":     merged["ticket_out"].astype("int64"),
        "time_open_utc":  merged["time_utc_in"],
        "time_close_utc": merged["time_utc_out"],
        "duration_minutes": (
            (merged["time_utc_out"] - merged["time_utc_in"])
            .dt.total_seconds() / 60
        ).round(1),
        "type":           merged["type_in"].map({mt5.DEAL_TYPE_BUY: "BUY", mt5.DEAL_TYPE_SELL: "SELL"}),
        "volume":         merged["volume_in"],
        "price_open":     merged["price_in"],
        "price_close":    merged["price_out"],
        "sl":             np.nan,
        "tp":             np.nan,
        "swap":           merged["swap_out"],
        "commission":     merged["commission_in"].fillna(0) + merged["commission_out"].fillna(0),
        "profit":         merged["profit_out"],
        "magic":          merged["magic_in"],
        "comment_open":   merged["comment_in"],
        "comment_close":  merged["comment_out"],
    })

    result["comment_close_inferred_type"] = result["comment_close"].apply(_infer_close_type)

    # MEX Atlantic no almacena SL/TP en el historial de la API para órdenes de mercado.
    # Mejor aproximación disponible: cuando el broker confirma el cierre vía comment
    # ("[sl ...]" / "[tp ...]"), el price_close ES el nivel ejecutado de SL o TP.
    sl_mask = result["comment_close_inferred_type"] == "sl"
    tp_mask = result["comment_close_inferred_type"] == "tp"
    result.loc[sl_mask, "sl"] = result.loc[sl_mask, "price_close"]
    result.loc[tp_mask, "tp"] = result.loc[tp_mask, "price_close"]
    logger.info(
        "SL/TP recuperados via comment: sl=%d tp=%d manual(sin SL/TP)=%d",
        sl_mask.sum(), tp_mask.sum(), (~sl_mask & ~tp_mask).sum(),
    )

    result = result.sort_values("time_open_utc").reset_index(drop=True)

    logger.info("Total trades: %d", len(result))
    logger.info("  BUY:  %d", (result["type"] == "BUY").sum())
    logger.info("  SELL: %d", (result["type"] == "SELL").sum())
    logger.info("  Ganadores: %d", (result["profit"] > 0).sum())
    logger.info("  Win rate: %.1f%%", (result["profit"] > 0).mean() * 100)
    logger.info("  P&L total: $%.2f", result["profit"].sum())
    logger.info("  Periodo: %s → %s", result["time_open_utc"].min(), result["time_open_utc"].max())
    logger.info("  Cierres por tipo: %s", result["comment_close_inferred_type"].value_counts().to_dict())

    if save_parquet:
        out_path = DATA_RAW_MT5 / "trades.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path, index=False)
        logger.info("Trades guardados: %s (%d trades)", out_path, len(result))

    return result


def extract_ohlc(
    timeframe: str,
    date_from_utc: datetime | None = None,
    date_to_utc: datetime | None = None,
    symbol: str = SYMBOL_BROKER,
    save_parquet: bool = True,
) -> pd.DataFrame:
    """Extrae OHLCV de un timeframe.

    Devuelve DataFrame con índice DatetimeIndex UTC y columnas:
        open, high, low, close, tick_volume, real_volume, spread.
    Guarda en v2/data/raw_mt5/ohlc_{TF}.parquet si save_parquet=True.
    """
    import MetaTrader5 as mt5
    from v2.src.utils.timezone import utc_to_server, df_server_to_utc

    start_utc = date_from_utc or PERIOD_START_UTC
    end_utc = date_to_utc or PERIOD_END_UTC

    start_server = utc_to_server(start_utc).to_pydatetime().replace(tzinfo=None)
    end_server = utc_to_server(end_utc).to_pydatetime().replace(tzinfo=None)

    tf_const = _get_tf_constant(timeframe)
    rates = mt5.copy_rates_range(symbol, tf_const, start_server, end_server)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        logger.warning("No se obtuvieron velas para %s %s: %s", symbol, timeframe, error)
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df_server_to_utc(df)
    df.index.name = "time_utc"

    for col in ["real_volume", "spread"]:
        if col not in df.columns:
            df[col] = 0

    df = df[["open", "high", "low", "close", "tick_volume", "real_volume", "spread"]]

    logger.info(
        "%s %s: %d velas | %s → %s",
        symbol, timeframe, len(df), df.index.min(), df.index.max(),
    )

    if save_parquet:
        out_path = DATA_RAW_MT5 / f"ohlc_{timeframe}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        logger.info("OHLC %s guardado en %s (%.1f KB)", timeframe, out_path, out_path.stat().st_size / 1024)

    return df


def extract_all_timeframes(
    timeframes: list[str] | None = None,
    date_from_utc: datetime | None = None,
    date_to_utc: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Extrae OHLCV para todos los timeframes especificados."""
    timeframes = timeframes or TIMEFRAMES
    results: dict[str, pd.DataFrame] = {}

    for tf in timeframes:
        logger.info("Extrayendo MT5 %s...", tf)
        try:
            df = extract_ohlc(tf, date_from_utc=date_from_utc, date_to_utc=date_to_utc)
            results[tf] = df
        except Exception as exc:
            logger.error("Error extrayendo %s: %s", tf, exc)

    return results
