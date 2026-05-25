"""Extiende los OHLC parquets para cubrir hasta el ultimo trade OOS + 2 dias."""
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from v2.config.settings import DATA_GROUND_TRUTH, SYMBOL_BROKER

import MetaTrader5 as mt5

TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

MT5_OFFSET_HOURS = 3   # server = GMT+3


def connect() -> None:
    ok = mt5.initialize(
        path=os.getenv("MT5_PATH"),
        login=int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
    )
    if not ok:
        raise RuntimeError(f"No se pudo conectar a MT5: {mt5.last_error()}")
    info = mt5.account_info()
    print(f"MT5 conectado: cuenta={info.login} broker={info.company}")


def main() -> None:
    connect()
    try:
        trades = pd.read_parquet(DATA_GROUND_TRUTH / "trades.parquet")
        trades["time_open_utc"] = pd.to_datetime(trades["time_open_utc"], utc=True)
        target_end_utc = trades["time_open_utc"].max() + pd.Timedelta(days=2)
        print(f"Extendiendo hasta: {target_end_utc}\n")

        for tf_name, tf_const in TF_MAP.items():
            path = DATA_GROUND_TRUTH / f"ohlc_{tf_name}.parquet"
            if not path.exists():
                print(f"{tf_name}: skip (no existe)")
                continue

            existing = pd.read_parquet(path)
            if existing.index.tz is None:
                existing.index = existing.index.tz_localize("UTC")
            last_existing = existing.index.max()

            start_utc = last_existing + pd.Timedelta(minutes=1)
            if start_utc >= target_end_utc:
                print(f"{tf_name}: ya cubre el periodo OOS — skip")
                continue

            # Convertir UTC -> server (GMT+3) para la peticion MT5
            start_srv = (start_utc + pd.Timedelta(hours=MT5_OFFSET_HOURS)).to_pydatetime().replace(tzinfo=None)
            end_srv   = (target_end_utc + pd.Timedelta(hours=MT5_OFFSET_HOURS)).to_pydatetime().replace(tzinfo=None)

            print(f"{tf_name}: extrayendo {start_utc} -> {target_end_utc} ...")
            rates = mt5.copy_rates_range(SYMBOL_BROKER, tf_const, start_srv, end_srv)

            if rates is None or len(rates) == 0:
                print(f"  sin datos ({mt5.last_error()})")
                continue

            new_df = pd.DataFrame(rates)
            new_df["time_utc"] = (
                pd.to_datetime(new_df["time"], unit="s", utc=True)
                - pd.Timedelta(hours=MT5_OFFSET_HOURS)
            )
            new_df = new_df.set_index("time_utc")

            keep_cols = [c for c in ["open", "high", "low", "close", "tick_volume", "real_volume", "spread"]
                         if c in new_df.columns]
            new_df = new_df[keep_cols]

            # Alinear columnas con existing (puede tener nombres distintos)
            existing_cols = existing.columns.tolist()
            for col in keep_cols:
                if col not in existing_cols:
                    existing[col] = 0

            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined = combined[existing_cols]

            combined.to_parquet(path)
            print(f"  +{len(new_df)} barras -> total {len(combined):,} | ultimo: {combined.index.max()}")

    finally:
        mt5.shutdown()
        print("\nMT5 desconectado.")


if __name__ == "__main__":
    main()
