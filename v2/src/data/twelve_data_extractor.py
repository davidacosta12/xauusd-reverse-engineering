"""Extractor de datos OHLC desde Twelve Data API.

Limitaciones plan gratuito:
- 800 requests/dia
- 8 requests/minuto
- Max 5000 velas por request
- Necesita paginacion para periodos largos

Politica implementada:
- Sleep entre requests (respetar rate limit)
- Cache local en parquet para no re-descargar lo ya bajado
- Resume si se interrumpe
- Retry con backoff exponencial en errores 429/5xx
"""
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)

logger = logging.getLogger(__name__)

TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"
RATE_LIMIT_SLEEP_SEC = 8  # 8 req/min = 1 cada 7.5s, con buffer


class TwelveDataExtractor:
    """Descarga OHLC de Twelve Data con paginacion y cache local."""

    SUPPORTED_INTERVALS = {
        "1min", "5min", "15min", "30min", "1h", "4h", "1day",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        rate_limit_sleep_sec: float = RATE_LIMIT_SLEEP_SEC,
    ) -> None:
        self.api_key = api_key or os.getenv("TWELVE_DATA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TWELVE_DATA_API_KEY no esta en .env ni se paso al constructor. "
                "Consigue una key gratuita en https://twelvedata.com"
            )
        self.cache_dir = Path(cache_dir) if cache_dir else Path("v2/data/twelve_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_sleep_sec = rate_limit_sleep_sec

    def _cache_path(self, symbol: str, interval: str) -> Path:
        safe = symbol.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe}_{interval}.parquet"

    def _request_chunk(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Una llamada a la API. Devuelve DataFrame con OHLCV indexado en UTC."""
        params = {
            "symbol":     symbol,
            "interval":   interval,
            "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date":   end.strftime("%Y-%m-%d %H:%M:%S"),
            "format":     "JSON",
            "outputsize": 5000,
            "timezone":   "UTC",
            "apikey":     self.api_key,
        }
        for attempt in range(3):
            try:
                resp = requests.get(
                    f"{TWELVE_DATA_BASE_URL}/time_series",
                    params=params,
                    timeout=30,
                )
                data = resp.json()

                if data.get("status") == "error":
                    msg = data.get("message", "unknown error")
                    if "rate limit" in msg.lower() or resp.status_code == 429:
                        wait = 30 * (attempt + 1)
                        logger.warning("Rate limit. Durmiendo %ds...", wait)
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"Twelve Data API error: {msg}")

                values = data.get("values", [])
                if not values:
                    return pd.DataFrame()

                df = pd.DataFrame(values)
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                df = df.set_index("datetime").sort_index()

                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col].astype(float)
                if "volume" in df.columns:
                    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
                else:
                    df["volume"] = 0.0

                return df[["open", "high", "low", "close", "volume"]]

            except requests.RequestException as exc:
                if attempt == 2:
                    raise
                wait = 5 * (attempt + 1)
                logger.warning("Request error (%s). Reintentando en %ds...", exc, wait)
                time.sleep(wait)

        return pd.DataFrame()

    def _fetch_range_paginated(
        self,
        symbol: str,
        interval: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Descarga [start, end] paginando hacia atras cuando el API tope en 5000 velas.

        El API de Twelve Data limita cada respuesta a 5000 velas (outputsize). Si el
        rango pedido produce mas, hay que seguir pidiendo hacia atras desde la vela
        mas antigua recibida hasta cubrir todo el rango.
        """
        chunks: list[pd.DataFrame] = []
        current_end = end
        chunk_num = 0

        while True:
            chunk_num += 1
            df_chunk = self._request_chunk(
                symbol, interval,
                start.to_pydatetime(),
                current_end.to_pydatetime(),
            )
            if df_chunk.empty:
                break

            chunks.append(df_chunk)
            logger.info(
                "    Chunk %d: %s a %s -> %d velas",
                chunk_num,
                df_chunk.index.min().strftime("%Y-%m-%d %H:%M"),
                df_chunk.index.max().strftime("%Y-%m-%d %H:%M"),
                len(df_chunk),
            )

            if len(df_chunk) < 5000 or df_chunk.index.min() <= start:
                break

            current_end = df_chunk.index.min() - pd.Timedelta(seconds=1)
            time.sleep(self.rate_limit_sleep_sec)

        if not chunks:
            return pd.DataFrame()

        combined = pd.concat(chunks)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined = combined.loc[start:end]

        if not combined.empty:
            logger.info(
                "  Total: %d velas, cobertura: %s a %s",
                len(combined),
                combined.index.min().strftime("%Y-%m-%d %H:%M"),
                combined.index.max().strftime("%Y-%m-%d %H:%M"),
            )
        return combined

    def _compute_missing_ranges(
        self,
        cached: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Identifica que rangos faltan dentro de [start, end] dado el cache."""
        if cached.empty:
            return [(start, end)]

        cached_in_range = cached.loc[start:end]
        if cached_in_range.empty:
            return [(start, end)]

        ranges = []
        if start < cached_in_range.index.min() - pd.Timedelta(minutes=1):
            ranges.append((start, cached_in_range.index.min() - pd.Timedelta(seconds=1)))
        if end > cached_in_range.index.max() + pd.Timedelta(minutes=1):
            ranges.append((cached_in_range.index.max() + pd.Timedelta(seconds=1), end))
        return ranges

    def fetch(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Descarga el rango [start, end] paginando si es necesario.

        Usa cache local en parquet para evitar redescargar lo ya bajado.

        Args:
            symbol: ej. "XAU/USD"
            interval: ej. "1min", "15min", "1h"
            start, end: datetime UTC (con o sin timezone)
            force_refresh: si True, ignora el cache y re-descarga todo

        Returns:
            DataFrame con columnas open, high, low, close, volume.
            Indice: DatetimeIndex UTC tz-aware.
        """
        if interval not in self.SUPPORTED_INTERVALS:
            raise ValueError(f"Intervalo no soportado: {interval}. Opciones: {self.SUPPORTED_INTERVALS}")

        start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start).tz_convert("UTC")
        end_ts   = pd.Timestamp(end,   tz="UTC") if pd.Timestamp(end).tz is None   else pd.Timestamp(end).tz_convert("UTC")

        cache_path = self._cache_path(symbol, interval)
        cached = pd.DataFrame()

        if cache_path.exists() and not force_refresh:
            cached = pd.read_parquet(cache_path)
            if cached.index.tz is None:
                cached.index = cached.index.tz_localize("UTC")
            logger.info(
                "Cache %s/%s: %d velas (%s -> %s)",
                symbol, interval, len(cached), cached.index.min(), cached.index.max(),
            )
            ranges_to_fetch = self._compute_missing_ranges(cached, start_ts, end_ts)
        else:
            ranges_to_fetch = [(start_ts, end_ts)]

        if not ranges_to_fetch:
            logger.info("Cache cubre el rango completo.")
            return cached.loc[start_ts:end_ts].copy()

        logger.info("Descargando %d chunk(s) desde Twelve Data...", len(ranges_to_fetch))
        all_new: list[pd.DataFrame] = []

        for i, (chunk_start, chunk_end) in enumerate(ranges_to_fetch, 1):
            logger.info(
                "  Chunk %d/%d: %s -> %s",
                i, len(ranges_to_fetch), chunk_start, chunk_end,
            )
            df_chunk = self._fetch_range_paginated(
                symbol, interval,
                chunk_start, chunk_end,
            )
            if not df_chunk.empty:
                all_new.append(df_chunk)
            else:
                logger.warning("    -> 0 velas retornadas para este chunk")

            if i < len(ranges_to_fetch):
                time.sleep(self.rate_limit_sleep_sec)

        if all_new:
            new_df = pd.concat(all_new)
            combined = pd.concat([cached, new_df]) if not cached.empty else new_df
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined.to_parquet(cache_path)
            logger.info("Cache actualizado: %d velas totales -> %s", len(combined), cache_path)
            return combined.loc[start_ts:end_ts].copy()

        return cached.loc[start_ts:end_ts].copy() if not cached.empty else pd.DataFrame()
