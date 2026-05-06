"""Validador cross-fuente: compara OHLCV de MT5 broker vs Dukascopy.

El propósito no es encontrar la fuente "correcta" — el broker MT5 es la fuente
de verdad porque es lo que vio el operador. El propósito es cuantificar el
nivel de desacuerdo entre fuentes para decidir si la diferencia importa.

Tolerancia aceptable: ≤ 1.0 pip (0.10 puntos de precio en XAUUSD).
"""
import logging
from pathlib import Path

import pandas as pd

from v2.config.settings import (
    DATA_RAW_MT5,
    DATA_RAW_DUKA,
    DATA_REPORTS,
    PRICE_TOLERANCE_PIPS,
    PIP_VALUE,
    TIMEFRAMES,
)

logger = logging.getLogger(__name__)


def load_ohlc(source: str, timeframe: str) -> pd.DataFrame:
    """Carga un OHLCV guardado en parquet desde la fuente especificada.

    Args:
        source: 'mt5' o 'dukascopy'.
        timeframe: nombre del timeframe ('M1', 'M5', etc.).

    Returns:
        DataFrame con índice DatetimeIndex UTC.

    Raises:
        FileNotFoundError: si el archivo parquet no existe aún.
        ValueError: si source no es válido.
    """
    paths = {
        "mt5": DATA_RAW_MT5,
        "dukascopy": DATA_RAW_DUKA,
    }
    if source not in paths:
        raise ValueError(f"Fuente '{source}' no válida. Opciones: {list(paths)}")

    path = paths[source] / f"ohlc_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {path}. Ejecuta primero el extractor de {source}."
        )

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    return df


def validate_ohlc_sources(
    timeframe: str,
    save_report: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Valida la consistencia entre MT5 y Dukascopy para un timeframe dado.

    Alinea ambas fuentes por timestamp UTC (inner join), calcula diferencias
    en pips para OHLC y genera estadísticas de desacuerdo.

    Args:
        timeframe: nombre del timeframe a validar.
        save_report: si True, guarda un HTML en data/reports/.

    Returns:
        Tuple de:
            - DataFrame de comparación barra a barra
            - Dict con estadísticas de validación
    """
    logger.info("Validando %s MT5 vs Dukascopy...", timeframe)

    try:
        df_mt5 = load_ohlc("mt5", timeframe)
    except FileNotFoundError as exc:
        logger.error("MT5 %s no disponible: %s", timeframe, exc)
        return pd.DataFrame(), {"error": str(exc)}

    try:
        df_duka = load_ohlc("dukascopy", timeframe)
    except FileNotFoundError as exc:
        logger.error("Dukascopy %s no disponible: %s", timeframe, exc)
        return pd.DataFrame(), {"error": str(exc)}

    # Inner join por timestamp UTC
    df_mt5_aligned = df_mt5[["open", "high", "low", "close"]].add_suffix("_mt5")
    df_duka_aligned = df_duka[["open", "high", "low", "close"]].add_suffix("_duka")

    comparison = df_mt5_aligned.join(df_duka_aligned, how="inner")
    n_aligned = len(comparison)

    if n_aligned == 0:
        logger.error(
            "No hay barras coincidentes entre MT5 y Dukascopy para %s. "
            "Verifica que los periodos y zonas horarias sean correctos.",
            timeframe,
        )
        return pd.DataFrame(), {"n_aligned": 0, "error": "No coincidencias"}

    n_mt5 = len(df_mt5)
    n_duka = len(df_duka)
    logger.info(
        "%s | MT5: %d barras | Dukascopy: %d barras | Coincidentes: %d (%.1f%%)",
        timeframe,
        n_mt5,
        n_duka,
        n_aligned,
        100 * n_aligned / max(n_mt5, n_duka),
    )

    # Calcular diferencias en pips
    for price in ["open", "high", "low", "close"]:
        comparison[f"diff_{price}_pips"] = (
            (comparison[f"{price}_mt5"] - comparison[f"{price}_duka"]).abs() / PIP_VALUE
        )

    comparison["max_diff_pips"] = comparison[
        [f"diff_{p}_pips" for p in ["open", "high", "low", "close"]]
    ].max(axis=1)

    comparison["within_tolerance"] = comparison["max_diff_pips"] <= PRICE_TOLERANCE_PIPS

    # Barras con desacuerdo significativo
    outliers = comparison[~comparison["within_tolerance"]]
    if len(outliers) > 0:
        logger.warning(
            "%s: %d barras (%.1f%%) con diferencia > %.1f pip",
            timeframe,
            len(outliers),
            100 * len(outliers) / n_aligned,
            PRICE_TOLERANCE_PIPS,
        )
        for ts, row in outliers.head(10).iterrows():
            logger.warning(
                "  %s | open_mt5=%.3f open_duka=%.3f | max_diff=%.2f pips",
                ts,
                row["open_mt5"],
                row["open_duka"],
                row["max_diff_pips"],
            )

    # Estadísticas
    stats = {
        "timeframe": timeframe,
        "n_mt5_bars": n_mt5,
        "n_duka_bars": n_duka,
        "n_aligned": n_aligned,
        "pct_aligned": round(100 * n_aligned / max(n_mt5, n_duka), 2),
        "n_within_tolerance": int(comparison["within_tolerance"].sum()),
        "pct_within_tolerance": round(
            100 * comparison["within_tolerance"].mean(), 2
        ),
        "n_outliers": len(outliers),
        "diff_open_mean_pips": round(comparison["diff_open_pips"].mean(), 4),
        "diff_open_p95_pips": round(comparison["diff_open_pips"].quantile(0.95), 4),
        "diff_open_p99_pips": round(comparison["diff_open_pips"].quantile(0.99), 4),
        "diff_open_max_pips": round(comparison["diff_open_pips"].max(), 4),
        "diff_close_mean_pips": round(comparison["diff_close_pips"].mean(), 4),
        "diff_close_p95_pips": round(comparison["diff_close_pips"].quantile(0.95), 4),
        "diff_close_p99_pips": round(comparison["diff_close_pips"].quantile(0.99), 4),
        "diff_close_max_pips": round(comparison["diff_close_pips"].max(), 4),
        "max_diff_mean_pips": round(comparison["max_diff_pips"].mean(), 4),
        "max_diff_median_pips": round(comparison["max_diff_pips"].median(), 4),
        "max_diff_p95_pips": round(comparison["max_diff_pips"].quantile(0.95), 4),
        "max_diff_max_pips": round(comparison["max_diff_pips"].max(), 4),
        "tolerance_pips": PRICE_TOLERANCE_PIPS,
    }

    logger.info(
        "%s | %.1f%% de barras dentro de tolerancia (%.1f pip) | "
        "Max diff promedio: %.4f pips",
        timeframe,
        stats["pct_within_tolerance"],
        PRICE_TOLERANCE_PIPS,
        stats["max_diff_mean_pips"],
    )

    if save_report:
        _save_html_report(comparison, stats, timeframe)

    return comparison, stats


def _save_html_report(
    comparison: pd.DataFrame,
    stats: dict,
    timeframe: str,
) -> None:
    """Genera un reporte HTML simple con los resultados de validación."""
    DATA_REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = DATA_REPORTS / f"cross_validation_{timeframe}.html"

    stats_df = pd.DataFrame([stats]).T.rename(columns={0: "Valor"})

    # Muestra máximo 200 barras en el reporte para no generar HTMLs enormes
    sample = comparison.head(200)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cross-Validation {timeframe} — MT5 vs Dukascopy</title>
<style>
  body {{ font-family: monospace; padding: 2rem; }}
  h2 {{ color: #333; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
  th {{ background: #eee; }}
  .ok {{ background: #d4edda; }}
  .warn {{ background: #fff3cd; }}
  .bad {{ background: #f8d7da; }}
</style>
</head>
<body>
<h2>Cross-Validation: {timeframe} | MT5 vs Dukascopy</h2>
<h3>Resumen estadístico</h3>
{stats_df.to_html(classes="stats")}
<h3>Muestra de barras alineadas (primeras 200)</h3>
{sample.to_html(classes="comparison")}
<p><small>Generado por v2/src/data/data_validator.py</small></p>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    logger.info("Reporte HTML guardado en %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)


def report_validation_summary(
    timeframes: list[str] | None = None,
) -> pd.DataFrame:
    """Ejecuta la validación cross-fuente para múltiples timeframes y resume.

    Args:
        timeframes: lista de timeframes a validar.
            Default: ['M1', 'M5', 'M15', 'H1'].

    Returns:
        DataFrame con una fila por timeframe y las estadísticas clave.
    """
    timeframes = timeframes or ["M1", "M5", "M15", "H1"]

    rows: list[dict] = []
    for tf in timeframes:
        logger.info("--- Validando %s ---", tf)
        _, stats = validate_ohlc_sources(tf)
        if stats:
            rows.append(stats)

    if not rows:
        logger.warning("No se pudo validar ningún timeframe.")
        return pd.DataFrame()

    summary = pd.DataFrame(rows).set_index("timeframe")
    logger.info("\n=== RESUMEN DE VALIDACIÓN ===\n%s", summary.to_string())
    return summary
