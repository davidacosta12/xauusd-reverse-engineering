"""Extractor de datos desde MetaTrader5 para el proyecto v2.

Conecta al broker MEX Atlantic (investor read-only) y extrae:
- Historial completo de trades cerrados
- OHLCV en múltiples timeframes

Todo el tiempo se convierte a UTC antes de persistir.
"""
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from v2.src.utils.timezone import df_server_to_utc, server_to_utc
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

# Carga .env desde la raíz del proyecto (dos niveles arriba de v2/)
_ENV_PATH = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)

# Mapa de nombre de timeframe a constante MT5
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


def _connect_mt5() -> bool:
    """Inicializa y autentica la conexión con MetaTrader5.

    Lee credenciales de .env: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH.

    Returns:
        True si la conexión fue exitosa.

    Raises:
        RuntimeError: si la inicialización o login falla.
    """
    import MetaTrader5 as mt5

    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER", MT5_SERVER)
    login = int(os.getenv("MT5_LOGIN", MT5_LOGIN))
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


def _shutdown_mt5() -> None:
    import MetaTrader5 as mt5
    mt5.shutdown()


def extract_trade_history(
    save_parquet: bool = True,
) -> pd.DataFrame:
    """Extrae el historial completo de deals (trades cerrados) de la cuenta.

    Filtra únicamente deals de tipo DEAL_TYPE_BUY y DEAL_TYPE_SELL
    en la entrada/salida de posiciones. Empareja entrada + salida por position_id.

    Returns:
        DataFrame con columnas:
            ticket, position_id,
            time_open_utc, time_close_utc, duration_minutes,
            type (BUY/SELL),
            volume, price_open, price_close,
            sl, tp, swap, commission, profit,
            magic, comment
    """
    import MetaTrader5 as mt5

    _connect_mt5()
    try:
        # Descarga TODOS los deals del historial completo de la cuenta
        deals = mt5.history_deals_get(
            datetime(2000, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        if deals is None or len(deals) == 0:
            raise RuntimeError(f"No se obtuvieron deals: {mt5.last_error()}")

        df_deals = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        logger.info("Total deals raw: %d", len(df_deals))

    finally:
        _shutdown_mt5()

    # Filtrar solo deals de XAUUSD (el símbolo puede tener los "..")
    df_deals = df_deals[df_deals["symbol"].str.startswith("XAUUSD")]

    # Separar entradas y salidas
    # DEAL_ENTRY_IN = 0, DEAL_ENTRY_OUT = 1
    df_in = df_deals[df_deals["entry"] == 0].copy()
    df_out = df_deals[df_deals["entry"] == 1].copy()

    # Convertir tiempos server → UTC
    df_in["time_utc"] = pd.to_datetime(df_in["time"], unit="s", utc=True) - pd.Timedelta(hours=MT5_GMT_OFFSET_HOURS)
    df_out["time_utc"] = pd.to_datetime(df_out["time"], unit="s", utc=True) - pd.Timedelta(hours=MT5_GMT_OFFSET_HOURS)

    # Emparejar por position_id
    df_in = df_in.set_index("position_id")
    df_out = df_out.set_index("position_id")

    merged = df_in.join(df_out, how="inner", lsuffix="_open", rsuffix="_close")

    trades = pd.DataFrame()
    trades["ticket"] = merged["deal_open"] if "deal_open" in merged.columns else merged.index
    trades["position_id"] = merged.index
    trades["time_open_utc"] = merged["time_utc_open"]
    trades["time_close_utc"] = merged["time_utc_close"]
    trades["duration_minutes"] = (
        (trades["time_close_utc"] - trades["time_open_utc"])
        .dt.total_seconds() / 60
    ).round(1)

    # Tipo de operación
    # DEAL_TYPE_BUY = 0, DEAL_TYPE_SELL = 1
    trades["type"] = merged["type_open"].map({0: "BUY", 1: "SELL"})

    trades["volume"] = merged["volume_open"]
    trades["price_open"] = merged["price_open"]
    trades["price_close"] = merged["price_close"]
    trades["sl"] = merged.get("sl_open", 0.0)
    trades["tp"] = merged.get("tp_open", 0.0)
    trades["swap"] = merged["swap_close"] if "swap_close" in merged.columns else 0.0
    trades["commission"] = (
        merged.get("commission_open", 0.0).fillna(0)
        + merged.get("commission_close", 0.0).fillna(0)
    )
    trades["profit"] = merged["profit_close"] if "profit_close" in merged else merged.get("profit_open", 0.0)
    trades["magic"] = merged.get("magic_open", 0)
    trades["comment"] = merged.get("comment_open", "")

    trades = trades.reset_index(drop=True).sort_values("time_open_utc")

    logger.info(
        "Trades extraídos: %d | periodo: %s → %s",
        len(trades),
        trades["time_open_utc"].min(),
        trades["time_open_utc"].max(),
    )
    logger.info(
        "Profit total: $%.2f | BUY: %d | SELL: %d",
        trades["profit"].sum(),
        (trades["type"] == "BUY").sum(),
        (trades["type"] == "SELL").sum(),
    )

    if save_parquet:
        DATA_RAW_MT5.mkdir(parents=True, exist_ok=True)
        out_path = DATA_RAW_MT5 / "trades.parquet"
        trades.to_parquet(out_path, index=False)
        logger.info("Trades guardados en %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)

    return trades


def extract_ohlc(
    timeframe: str,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    save_parquet: bool = True,
) -> pd.DataFrame:
    """Extrae velas OHLCV del broker MT5 para el símbolo XAUUSD.

    Los tiempos en MT5 son hora server (GMT+3). Esta función los convierte a UTC
    antes de devolver y guardar el DataFrame.

    Args:
        timeframe: nombre del timeframe ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1').
        start_utc: inicio del periodo en UTC. Default: PERIOD_START_UTC.
        end_utc: fin del periodo en UTC. Default: PERIOD_END_UTC.
        save_parquet: si True, guarda el resultado en data/raw_mt5/.

    Returns:
        DataFrame con índice DatetimeIndex UTC y columnas:
            open, high, low, close, tick_volume, real_volume, spread
    """
    import MetaTrader5 as mt5

    start_utc = start_utc or PERIOD_START_UTC
    end_utc = end_utc or PERIOD_END_UTC

    # MT5 copy_rates_range espera datetimes en hora local/server
    # Usamos UTC directamente — la librería lo maneja correctamente cuando
    # el sistema está configurado. Para seguridad, convertimos a hora server.
    from v2.src.utils.timezone import utc_to_server
    start_server = utc_to_server(start_utc).to_pydatetime().replace(tzinfo=None)
    end_server = utc_to_server(end_utc).to_pydatetime().replace(tzinfo=None)

    _connect_mt5()
    try:
        tf_const = _get_tf_constant(timeframe)
        rates = mt5.copy_rates_range(SYMBOL_BROKER, tf_const, start_server, end_server)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.warning("No se obtuvieron velas para %s %s: %s", SYMBOL_BROKER, timeframe, error)
            return pd.DataFrame()

    finally:
        _shutdown_mt5()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")  # naive, hora server

    # Convertir hora server → UTC
    df = df.set_index("time")
    df = df_server_to_utc(df)
    df.index.name = "time_utc"

    # Renombrar columnas al estándar del proyecto
    df = df.rename(columns={
        "tick_volume": "tick_volume",
        "real_volume": "real_volume",
        "spread": "spread",
    })

    # Asegurar columnas presentes
    for col in ["real_volume", "spread"]:
        if col not in df.columns:
            df[col] = 0

    df = df[["open", "high", "low", "close", "tick_volume", "real_volume", "spread"]]

    logger.info(
        "%s %s: %d velas | %s → %s",
        SYMBOL_BROKER,
        timeframe,
        len(df),
        df.index.min(),
        df.index.max(),
    )

    if save_parquet:
        DATA_RAW_MT5.mkdir(parents=True, exist_ok=True)
        out_path = DATA_RAW_MT5 / f"ohlc_{timeframe}.parquet"
        df.to_parquet(out_path)
        logger.info(
            "OHLC %s guardado en %s (%.1f KB)",
            timeframe,
            out_path,
            out_path.stat().st_size / 1024,
        )

    return df


def extract_all_timeframes(
    timeframes: list[str] | None = None,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Extrae OHLCV para todos los timeframes especificados.

    Args:
        timeframes: lista de nombres de timeframe. Default: TIMEFRAMES de settings.
        start_utc: inicio del periodo en UTC.
        end_utc: fin del periodo en UTC.

    Returns:
        Dict mapeando nombre de timeframe → DataFrame OHLCV.
    """
    timeframes = timeframes or TIMEFRAMES
    results: dict[str, pd.DataFrame] = {}

    for tf in timeframes:
        logger.info("Extrayendo MT5 %s...", tf)
        try:
            df = extract_ohlc(tf, start_utc=start_utc, end_utc=end_utc)
            results[tf] = df
        except Exception as exc:
            logger.error("Error extrayendo %s: %s", tf, exc)

    return results
