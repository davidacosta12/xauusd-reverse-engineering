"""Pipeline de feature engineering reutilizable.

Extrae la logica del notebook 01 / regenerate_features_and_test_v2.py
como funcion standalone que acepta DataFrames en lugar de leer de disco.

Usa EXCLUSIVAMENTE los modulos ya validados:
  - levels.py (compute_asian_levels, etc.)
  - microstructure.py (compute_microstructure_features)
  - technical.py (compute_technical_indicators)
"""
import logging

import pandas as pd

from v2.src.features.levels import (
    compute_asian_levels,
    compute_distances_to_levels,
    compute_monthly_levels,
    compute_pdh_pdl,
    compute_weekly_levels,
)
from v2.src.features.microstructure import compute_microstructure_features
from v2.src.features.technical import compute_technical_indicators

logger = logging.getLogger(__name__)


def _align_to_base(base: pd.DataFrame, feat: pd.DataFrame, name: str) -> pd.DataFrame:
    """Alinea feat al indice de base usando merge_asof backward."""
    if feat.empty:
        logger.warning("Feature block '%s' esta vacio — omitiendo", name)
        return pd.DataFrame(index=base.index)

    left = pd.DataFrame({"_ts": base.index}).sort_values("_ts")
    right = feat.reset_index()
    ts_col = right.columns[0]
    right = right.rename(columns={ts_col: "_ts_feat"}).sort_values("_ts_feat")
    right = right.drop_duplicates("_ts_feat")

    merged = pd.merge_asof(
        left, right,
        left_on="_ts", right_on="_ts_feat",
        direction="backward",
    ).drop(columns=["_ts_feat"])

    merged.index = base.index
    result = merged.drop(columns=["_ts"])
    logger.debug("Alineado '%s': %d columnas", name, result.shape[1])
    return result


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas OHLC a la convencion interna del proyecto.

    Twelve Data devuelve 'volume'; los modulos internos esperan 'tick_volume'.
    Tambien asegura que el indice sea DatetimeIndex UTC tz-aware.
    """
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    # Renombrar volume -> tick_volume si hace falta
    if "volume" in df.columns and "tick_volume" not in df.columns:
        df = df.rename(columns={"volume": "tick_volume"})
    # Asegurar tick_volume existe
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0.0
    # Asegurar columnas basicas presentes
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Columna requerida faltante en OHLC: '{col}'")
    return df


def build_snapshots(
    m1:  pd.DataFrame,
    m5:  pd.DataFrame,
    m15: pd.DataFrame,
    h1:  pd.DataFrame,
    h4:  pd.DataFrame,
) -> pd.DataFrame:
    """Construye el DataFrame de snapshots con todas las features.

    Replica exactamente el pipeline del notebook 01 y de
    regenerate_features_and_test_v2.py, aceptando DataFrames directamente
    en lugar de leer de disco.

    Args:
        m1, m5, m15, h1, h4: DataFrames OHLC con indice DatetimeIndex UTC.
            Columnas esperadas: open, high, low, close, tick_volume.
            (Si 'volume' existe en lugar de 'tick_volume', se renombra automaticamente.)

    Returns:
        DataFrame con indice igual al de m1 y todas las features alineadas.
    """
    # Normalizar columnas de todos los timeframes
    m1  = normalize_ohlc(m1)
    m5  = normalize_ohlc(m5)
    m15 = normalize_ohlc(m15)
    h1  = normalize_ohlc(h1)
    h4  = normalize_ohlc(h4)

    logger.info("Calculando niveles institucionales (PDH/PDL, weekly, monthly, asian)...")
    pdh_pdl  = compute_pdh_pdl(m15)
    weekly   = compute_weekly_levels(m15)
    monthly  = compute_monthly_levels(m15)
    asian    = compute_asian_levels(m15, m5_df=m5 if not m5.empty else None)
    all_levels = pd.concat([pdh_pdl, weekly, monthly, asian], axis=1)

    logger.info("Calculando distancias a niveles...")
    distances = compute_distances_to_levels(m15, all_levels)

    logger.info("Calculando microestructura (m5, m15, h1)...")
    micro: dict[str, pd.DataFrame] = {}
    for tf_name, df in [("m5", m5), ("m15", m15), ("h1", h1)]:
        if not df.empty:
            micro[tf_name] = compute_microstructure_features(df, prefix=tf_name)

    logger.info("Calculando indicadores tecnicos (m15, h1, h4)...")
    tech: dict[str, pd.DataFrame] = {}
    for tf_name, df in [("m15", m15), ("h1", h1), ("h4", h4)]:
        if not df.empty:
            tech[tf_name] = compute_technical_indicators(df, prefix=tf_name)

    logger.info("Alineando features al indice M1...")
    base = m1
    blocks: list[pd.DataFrame] = [
        _align_to_base(base, all_levels, "levels"),
        _align_to_base(base, distances, "distances"),
    ]
    for name, df in micro.items():
        blocks.append(_align_to_base(base, df, f"micro_{name}"))
    for name, df in tech.items():
        blocks.append(_align_to_base(base, df, f"tech_{name}"))

    snapshots = pd.concat([b for b in blocks if not b.empty], axis=1)
    snapshots = snapshots.loc[:, ~snapshots.columns.duplicated()]

    logger.info(
        "Snapshots construidos: %d filas x %d columnas",
        len(snapshots), snapshots.shape[1],
    )
    return snapshots
