"""Niveles institucionales para XAUUSD M15 — v2.

Todos los outputs son point-in-time correctos:
  para timestamp T se usan SOLO datos con timestamp < T.

Convención: para cada nivel, los stats del período D se hacen disponibles
a partir del comienzo del período D+1 y se mapean usando pd.merge_asof.
"""
import logging

import numpy as np
import pandas as pd

from v2.config.settings import PIP_VALUE

logger = logging.getLogger(__name__)

# NY session: 13:00–21:59 UTC
_NY_START = 13
_NY_END = 22  # exclusive

# Asian session que cubre "day D": [D-1 22:00 UTC, D 02:00 UTC)
_ASIAN_START = 22  # hora del día anterior
_ASIAN_END = 2    # hora del día actual (exclusive)

# Nombres de nivel para compute_distances_to_levels
_LEVEL_COLS = [
    "pdh", "pdl",
    "weekly_high", "weekly_low",
    "monthly_high", "monthly_low",
    "asian_high", "asian_low", "asian_mid", "asian_vwap",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_asof_levels(
    m15_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    avail_col: str,
) -> pd.DataFrame:
    """merge_asof preservando el orden original del índice de m15_df.

    Args:
        m15_df: DataFrame source (índice usado para resultado final).
        stats_df: DataFrame con columna `avail_col` indicando desde cuándo está disponible.
        avail_col: nombre de la columna de disponibilidad en stats_df.

    Returns:
        DataFrame con mismo índice que m15_df, columnas de stats_df (sin avail_col).
    """
    left = pd.DataFrame(
        {"_ts": m15_df.index, "_pos": np.arange(len(m15_df))}
    ).sort_values("_ts")

    right = stats_df.rename(columns={avail_col: "_avail"}).sort_values("_avail")

    merged = pd.merge_asof(left, right, left_on="_ts", right_on="_avail", direction="backward")
    merged = merged.sort_values("_pos")
    merged.index = m15_df.index

    drop_cols = ["_ts", "_pos", "_avail"]
    return merged.drop(columns=[c for c in drop_cols if c in merged.columns])


# ─────────────────────────────────────────────────────────────────────────────
# Niveles NY (PDH / PDL)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pdh_pdl(m15_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula Previous Day High/Low de sesión NY (13:00–21:59 UTC).

    Para cada timestamp T, devuelve el PDH/PDL de la última sesión NY
    COMPLETADA ANTES de T. Si T cae en lunes, usa la sesión del viernes.

    Returns:
        DataFrame con columnas: pdh, pdl, pdh_pdl_source_date.
        Indexado por m15_df.index.
    """
    hours = m15_df.index.hour
    ny_mask = (hours >= _NY_START) & (hours < _NY_END)
    ny_bars = m15_df.loc[ny_mask, ["high", "low"]].copy()

    if ny_bars.empty:
        logger.warning("compute_pdh_pdl: no hay barras en sesión NY (13-22 UTC).")
        return pd.DataFrame(
            {"pdh": np.nan, "pdl": np.nan, "pdh_pdl_source_date": pd.NaT},
            index=m15_df.index,
        )

    ny_bars["ny_date"] = ny_bars.index.normalize()
    daily_ny = (
        ny_bars.groupby("ny_date")
        .agg(pdh=("high", "max"), pdl=("low", "min"))
        .rename_axis("ny_date")
        .reset_index()
    )
    daily_ny["pdh_pdl_source_date"] = daily_ny["ny_date"]

    # Disponible a partir de la medianoche siguiente (ny_date + 1 día)
    daily_ny["_avail"] = daily_ny["ny_date"] + pd.Timedelta(days=1)

    stats = daily_ny[["_avail", "pdh", "pdl", "pdh_pdl_source_date"]].sort_values("_avail")
    result = _merge_asof_levels(m15_df, stats, "_avail")

    return result[["pdh", "pdl", "pdh_pdl_source_date"]]


# ─────────────────────────────────────────────────────────────────────────────
# Niveles Semanales
# ─────────────────────────────────────────────────────────────────────────────

def compute_weekly_levels(m15_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula weekly high/low de la semana cerrada anterior (lun–dom).

    Para timestamp T en la semana con inicio Monday W:
        weekly_high = max(high) de Monday W-7 a Sunday W-1.

    Returns:
        DataFrame con: weekly_high, weekly_low, weekly_range_size.
    """
    df = m15_df.copy()
    # Día de inicio de semana (lunes) para cada barra
    weekday = df.index.weekday  # 0=Lunes
    week_start = df.index.normalize() - pd.to_timedelta(weekday, unit="D")
    df["_week_start"] = week_start

    weekly = (
        df.groupby("_week_start")
        .agg(weekly_high=("high", "max"), weekly_low=("low", "min"))
        .rename_axis("week_start")
        .reset_index()
    )
    weekly["weekly_range_size"] = (weekly["weekly_high"] - weekly["weekly_low"]) / PIP_VALUE

    # Disponible a partir del lunes siguiente (semana + 7 días)
    weekly["_avail"] = weekly["week_start"] + pd.Timedelta(weeks=1)

    stats = weekly[["_avail", "weekly_high", "weekly_low", "weekly_range_size"]].sort_values("_avail")
    result = _merge_asof_levels(m15_df, stats, "_avail")

    return result[["weekly_high", "weekly_low", "weekly_range_size"]]


# ─────────────────────────────────────────────────────────────────────────────
# Niveles Mensuales
# ─────────────────────────────────────────────────────────────────────────────

def compute_monthly_levels(m15_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula monthly high/low del mes calendario anterior.

    Returns:
        DataFrame con: monthly_high, monthly_low.
    """
    df = m15_df.copy()
    # Inicio del mes para cada barra
    month_start = df.index.to_series().dt.to_period("M").dt.start_time
    if month_start.dt.tz is None:
        month_start = month_start.dt.tz_localize("UTC")
    else:
        month_start = month_start.dt.tz_convert("UTC")
    df["_month_start"] = month_start.values

    monthly = (
        df.groupby("_month_start")
        .agg(monthly_high=("high", "max"), monthly_low=("low", "min"))
        .rename_axis("month_start")
        .reset_index()
    )

    # Disponible a partir del primer día del mes siguiente
    monthly["_avail"] = monthly["month_start"] + pd.offsets.MonthBegin(1)
    # Asegurar UTC tz-aware
    monthly["_avail"] = pd.to_datetime(monthly["_avail"]).dt.tz_localize(
        "UTC", ambiguous="NaT", nonexistent="NaT"
    )

    stats = monthly[["_avail", "monthly_high", "monthly_low"]].sort_values("_avail")
    result = _merge_asof_levels(m15_df, stats, "_avail")

    return result[["monthly_high", "monthly_low"]]


# ─────────────────────────────────────────────────────────────────────────────
# Asian Range
# ─────────────────────────────────────────────────────────────────────────────

def compute_asian_levels(
    m15_df: pd.DataFrame,
    m5_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcula Asian Range en modo ROLLING durante la formación de la ventana.

    Para cada timestamp T en la ventana 22:00-02:00 UTC (formación):
        asian_high(T) = max(high) de barras desde 22:00 UTC hasta T-1 (sin lookahead)
        asian_low(T)  = min(low) desde 22:00 UTC hasta T-1
        asian_mid(T)  = (asian_high + asian_low) / 2
        asian_vwap(T) = VWAP de barras M5/M15 desde 22:00 UTC hasta T-1

    Para T fuera de la ventana (02:00-22:00 UTC):
        Se devuelve el valor CERRADO del rango más reciente (behavior original).

    El valor en T=22:00 (primer bar de la ventana) es NaN porque no hay barras previas.

    Args:
        m15_df: DataFrame con índice UTC. Usado para el índice de salida.
        m5_df: DataFrame M5 opcional. Si se provee, da mayor precisión al VWAP.

    Returns:
        DataFrame con: asian_high, asian_low, asian_mid, asian_vwap,
        asian_range_pips, asian_window_start_utc, asian_window_end_utc,
        is_window_open (bool).
        Indexado por m15_df.index.
    """
    src = (m5_df if m5_df is not None else m15_df).copy()

    _float_cols = ["asian_high", "asian_low", "asian_mid", "asian_vwap", "asian_range_pips"]
    _dt_cols = ["asian_window_start_utc", "asian_window_end_utc"]

    # ── 1. Etiquetar barras fuente con su "asian_day" y flag in-window ────────
    src_hrs = src.index.hour
    in_window_src = (src_hrs >= _ASIAN_START) | (src_hrs < _ASIAN_END)

    win_bars = src[in_window_src][["high", "low", "close", "tick_volume"]].copy()
    if win_bars.empty:
        result = pd.DataFrame(index=m15_df.index)
        for c in _float_cols:
            result[c] = np.nan
        for c in _dt_cols:
            result[c] = pd.NaT
        result["is_window_open"] = False
        return result

    # asian_day: día al que pertenece la ventana (hora >= 22 → día siguiente)
    _dates = win_bars.index.normalize()
    _late = win_bars.index.hour >= _ASIAN_START
    asian_day_arr = np.where(
        _late,
        _dates + pd.Timedelta(days=1),
        _dates,
    )
    win_bars["_asian_day"] = pd.DatetimeIndex(asian_day_arr)
    win_bars = win_bars.sort_index()

    # VWAP
    win_bars["_tv"] = win_bars["tick_volume"].fillna(0)
    win_bars["_tp"] = (win_bars["high"] + win_bars["low"] + win_bars["close"]) / 3
    win_bars["_tp_vol"] = win_bars["_tp"] * win_bars["_tv"]

    # ── 2. Running (cumulative shifted) values para barras dentro de la ventana ──
    # shift(1) dentro de cada grupo → valor en T = stats de barras hasta T-1
    def _shift_max(s):  return s.shift(1).expanding().max()
    def _shift_min(s):  return s.shift(1).expanding().min()
    def _shift_sum(s):  return s.shift(1).expanding().sum()

    win_bars["_run_high"]    = win_bars.groupby("_asian_day")["high"].transform(_shift_max)
    win_bars["_run_low"]     = win_bars.groupby("_asian_day")["low"].transform(_shift_min)
    win_bars["_run_tp_vol"]  = win_bars.groupby("_asian_day")["_tp_vol"].transform(_shift_sum)
    win_bars["_run_vol"]     = win_bars.groupby("_asian_day")["_tv"].transform(_shift_sum)

    win_bars["_run_mid"] = (win_bars["_run_high"] + win_bars["_run_low"]) / 2
    win_bars["_run_vwap"] = np.where(
        win_bars["_run_vol"] > 0,
        win_bars["_run_tp_vol"] / win_bars["_run_vol"],
        win_bars["_run_mid"],
    )
    win_bars["_run_range_pips"] = (win_bars["_run_high"] - win_bars["_run_low"]) / PIP_VALUE
    win_bars["_run_win_start"] = win_bars["_asian_day"] - pd.Timedelta(hours=2)
    win_bars["_run_win_end"]   = win_bars["_asian_day"] + pd.Timedelta(hours=2)

    # ── 3. Valores cerrados por asian_day (para timestamps fuera de la ventana) ─
    closed = win_bars.groupby("_asian_day").agg(
        asian_high=("high", "max"),
        asian_low=("low", "min"),
        _tp_vol_sum=("_tp_vol", "sum"),
        _vol_sum=("_tv", "sum"),
    ).reset_index()
    closed["asian_mid"] = (closed["asian_high"] + closed["asian_low"]) / 2
    closed["asian_range_pips"] = (closed["asian_high"] - closed["asian_low"]) / PIP_VALUE
    closed["asian_vwap"] = np.where(
        closed["_vol_sum"] > 0,
        closed["_tp_vol_sum"] / closed["_vol_sum"],
        closed["asian_mid"],
    )
    closed["asian_window_start_utc"] = closed["_asian_day"] - pd.Timedelta(hours=2)
    closed["asian_window_end_utc"]   = closed["_asian_day"] + pd.Timedelta(hours=2)
    # Disponible a partir del cierre de la ventana (02:00 UTC)
    closed["_avail"] = closed["asian_window_end_utc"]

    # ── 4. Construir resultado para m15_df ────────────────────────────────────
    result = pd.DataFrame(index=m15_df.index)
    for c in _float_cols:
        result[c] = np.nan
    for c in _dt_cols:
        result[c] = pd.NaT
    result["is_window_open"] = False

    m15_hrs = m15_df.index.hour
    m15_in_window = (m15_hrs >= _ASIAN_START) | (m15_hrs < _ASIAN_END)

    # 4a. Timestamps dentro de la ventana → running values (merge_asof backward)
    if m15_in_window.any():
        _run_cols = ["_run_high", "_run_low", "_run_mid", "_run_vwap",
                     "_run_range_pips", "_run_win_start", "_run_win_end"]
        win_src = win_bars[_run_cols].reset_index(names="_ts_src").sort_values("_ts_src")

        m15_win_ts = pd.DataFrame(
            {"_ts": m15_df.index[m15_in_window]}
        ).sort_values("_ts")

        merged_win = pd.merge_asof(
            m15_win_ts, win_src,
            left_on="_ts", right_on="_ts_src",
            direction="backward",
        ).set_index("_ts").reindex(m15_df.index[m15_in_window])

        result.loc[m15_in_window, "asian_high"]            = merged_win["_run_high"].values
        result.loc[m15_in_window, "asian_low"]             = merged_win["_run_low"].values
        result.loc[m15_in_window, "asian_mid"]             = merged_win["_run_mid"].values
        result.loc[m15_in_window, "asian_vwap"]            = merged_win["_run_vwap"].values
        result.loc[m15_in_window, "asian_range_pips"]      = merged_win["_run_range_pips"].values
        result.loc[m15_in_window, "asian_window_start_utc"] = merged_win["_run_win_start"].values
        result.loc[m15_in_window, "asian_window_end_utc"]   = merged_win["_run_win_end"].values
        result.loc[m15_in_window, "is_window_open"]         = True

    # 4b. Timestamps fuera de la ventana → valores cerrados (merge_asof backward)
    m15_outside = ~m15_in_window
    if m15_outside.any():
        closed_src = closed[[
            "_avail", "asian_high", "asian_low", "asian_mid", "asian_vwap",
            "asian_range_pips", "asian_window_start_utc", "asian_window_end_utc",
        ]].sort_values("_avail")

        m15_out_ts = pd.DataFrame(
            {"_ts": m15_df.index[m15_outside]}
        ).sort_values("_ts")

        merged_out = pd.merge_asof(
            m15_out_ts, closed_src,
            left_on="_ts", right_on="_avail",
            direction="backward",
        ).set_index("_ts").reindex(m15_df.index[m15_outside])

        for c in _float_cols + _dt_cols:
            if c in merged_out.columns:
                result.loc[m15_outside, c] = merged_out[c].values

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Distancias a niveles
# ─────────────────────────────────────────────────────────────────────────────

def compute_distances_to_levels(
    prices_df: pd.DataFrame,
    levels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula distancias en pips desde el close a cada nivel institucional.

    Args:
        prices_df: DataFrame con columna 'close', indexado por timestamp UTC.
        levels_df: Output combinado de compute_pdh_pdl, compute_weekly_levels, etc.

    Returns:
        DataFrame con:
            dist_to_<nivel>_pips_signed  (positivo = close ENCIMA del nivel)
            dist_to_<nivel>_pips_abs
            nearest_level                (nombre del nivel más cercano)
            dist_to_nearest_pips         (distancia absoluta mínima)
        Indexado por prices_df.index.
    """
    close = prices_df["close"]
    result = pd.DataFrame(index=prices_df.index)

    available_levels = [c for c in _LEVEL_COLS if c in levels_df.columns]
    if not available_levels:
        logger.warning("compute_distances_to_levels: levels_df no tiene columnas reconocidas.")
        return result

    abs_dist_dict: dict[str, pd.Series] = {}
    for lvl in available_levels:
        level = levels_df[lvl]
        signed = (close - level) / PIP_VALUE
        abs_d = signed.abs()
        result[f"dist_to_{lvl}_pips_signed"] = signed
        result[f"dist_to_{lvl}_pips_abs"] = abs_d
        abs_dist_dict[lvl] = abs_d

    abs_dist_df = pd.concat(abs_dist_dict, axis=1)
    # Nearest level (ignora NaN en la comparación)
    result["nearest_level"] = abs_dist_df.idxmin(axis=1)
    result["dist_to_nearest_pips"] = abs_dist_df.min(axis=1, skipna=True)

    return result
