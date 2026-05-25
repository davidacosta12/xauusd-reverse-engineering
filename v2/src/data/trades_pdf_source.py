"""Fuente de verdad de trades de la cuenta 921339, broker MEX Atlantic.

Origen: Informe oficial del broker generado el 2026-05-12 13:49.
Archivo PDF: historial_estrategia.pdf
Total trades XAUUSD..: 43
Trades excluidos: 1 (position_id 427944598, sin SL/TP definidos)
Trades válidos para análisis: 42

Convención temporal:
- El PDF reporta en hora del servidor MT5 = GMT+3.
- TODOS los timestamps en este módulo se convierten a UTC antes de exportar.

Split:
- IN-SAMPLE: hasta 2026-04-27 inclusive (30 trades) — para descubrir la regla
- OUT-OF-SAMPLE: 2026-04-28 al 2026-05-12 (12 trades) — para validar
"""
from datetime import datetime, timezone, timedelta

import pandas as pd

MT5_GMT_OFFSET_HOURS = 3
SYMBOL = "XAUUSD.."
SPLIT_CUTOFF_UTC = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
EXCLUDED_POSITION_IDS = {427944598}  # Sin SL/TP definidos

# Tabla de trades del PDF (43 filas, hora server GMT+3)
# Columnas: time_open_server, position_id, type, volume, price_open, sl, tp,
#           time_close_server, price_close, commission, swap, profit, close_comment
TRADES_RAW = [
    # IN-SAMPLE (30 trades, hasta 27 abril)
    ("2026-03-19 03:45:58", 413086299, "BUY",  0.01, 4838.09, 4815.13, 4858.20, "2026-03-19 04:37:28", 4857.95, 0.00,  0.00,  19.86, ""),
    ("2026-03-20 02:01:23", 413589012, "SELL", 0.01, 4647.54, 4672.74, 4627.17, "2026-03-20 04:06:15", 4673.35, 0.00,  0.00, -25.81, "[sl 4672.74]"),
    ("2026-03-23 02:50:29", 414044287, "SELL", 0.01, 4465.86, 4527.82, 4413.73, "2026-03-23 03:15:30", 4413.71, 0.00,  0.00,  52.15, "[tp 4413.73]"),
    ("2026-03-24 02:37:41", 414647846, "BUY",  0.01, 4429.37, 4376.64, 4469.82, "2026-03-24 03:28:19", 4375.18, 0.00,  0.00, -54.19, "[sl 4376.64]"),
    ("2026-03-24 03:30:01", 414661794, "SELL", 0.01, 4378.17, 4423.29, 4308.07, "2026-03-24 05:12:18", 4307.99, 0.00,  0.00,  70.18, "[tp 4308.07]"),
    ("2026-03-25 02:00:07", 415090558, "BUY",  0.02, 4484.35, 4461.07, 4506.15, "2026-03-25 02:25:15", 4502.02, 0.00,  0.00,  35.34, ""),
    ("2026-03-26 01:35:20", 415549986, "BUY",  0.02, 4520.76, 4507.49, 4535.80, "2026-03-26 01:44:18", 4533.48, 0.00,  0.00,  25.44, ""),
    ("2026-03-30 02:15:04", 416447591, "SELL", 0.01, 4478.49, 4513.62, 4450.16, "2026-03-30 02:35:36", 4452.42, 0.00,  0.00,  26.07, ""),
    ("2026-03-30 08:20:21", 416557098, "BUY",  0.01, 4512.57, 4479.75, 4544.18, "2026-03-30 08:58:22", 4525.31, 0.00,  0.00,  12.74, ""),
    ("2026-03-31 02:44:30", 416898251, "BUY",  0.01, 4518.70, 4492.32, 4542.96, "2026-03-31 03:20:17", 4484.11, 0.00,  0.00, -34.59, "[sl 4492.32]"),
    ("2026-04-01 01:41:58", 417330273, "BUY",  0.01, 4689.85, 4663.71, 4720.52, "2026-04-01 01:47:56", 4663.61, 0.00,  0.00, -26.24, "[sl 4663.71]"),
    ("2026-04-01 01:59:29", 417336677, "BUY",  0.01, 4680.90, 4658.95, 4722.63, "2026-04-01 03:25:09", 4716.79, 0.00,  0.00,  35.89, ""),
    ("2026-04-02 01:31:17", 417750955, "BUY",  0.01, 4771.87, 4752.24, 4791.91, "2026-04-02 02:10:23", 4790.40, 0.00,  0.00,  18.53, ""),
    ("2026-04-06 03:29:35", 418218005, "SELL", 0.01, 4604.33, 4651.94, 4564.24, "2026-04-06 05:17:48", 4652.03, 0.00,  0.00, -47.70, "[sl 4651.94]"),
    ("2026-04-06 05:32:37", 418249118, "BUY",  0.01, 4660.08, 4659.67, 4735.74, "2026-04-06 12:23:17", 4699.42, 0.00,  0.00,  39.34, ""),
    ("2026-04-07 04:50:45", 418619632, "SELL", 0.01, 4634.39, 4667.83, 4604.76, "2026-04-07 04:53:21", 4622.82, 0.00,  0.00,  11.57, ""),
    ("2026-04-07 05:32:51", 418634723, "SELL", 0.01, 4638.02, 4667.98, 4616.64, "2026-04-07 05:37:49", 4638.98, 0.00,  0.00,  -0.96, ""),
    ("2026-04-08 01:39:23", 419009914, "BUY",  0.01, 4769.80, 4720.27, 4817.11, "2026-04-08 01:56:00", 4807.21, 0.00,  0.00,  37.41, ""),
    ("2026-04-09 04:59:26", 419502130, "BUY",  0.01, 4728.35, 4691.15, 4759.04, "2026-04-09 15:01:17", 4759.43, 0.00,  0.00,  31.08, "[tp 4759.04]"),
    ("2026-04-13 03:35:05", 420215087, "BUY",  0.01, 4679.81, 4644.17, 4709.93, "2026-04-13 04:01:55", 4707.93, 0.00,  0.00,  28.12, ""),
    ("2026-04-14 01:55:03", 420630607, "BUY",  0.03, 4755.86, 4744.26, 4765.63, "2026-04-14 02:04:31", 4764.86, 0.00,  0.00,  27.00, ""),
    ("2026-04-15 04:14:09", 421070456, "BUY",  0.02, 4849.67, 4823.50, 4876.02, "2026-04-15 04:17:50", 4870.95, 0.00,  0.00,  42.56, ""),
    ("2026-04-16 01:55:08", 421443190, "BUY",  0.02, 4810.38, 4789.58, 4826.44, "2026-04-16 02:48:46", 4826.49, 0.00,  0.00,  32.22, "[tp 4826.44]"),
    ("2026-04-20 02:59:04", 422212533, "SELL", 0.02, 4754.01, 4791.18, 4723.86, "2026-04-20 04:11:20", 4791.31, 0.00,  0.00, -74.60, "[sl 4791.18]"),
    ("2026-04-20 05:47:51", 422270792, "BUY",  0.02, 4794.41, 4759.82, 4844.65, "2026-04-21 01:17:55", 4832.19, 0.00, -1.33,  75.56, ""),
    ("2026-04-21 04:55:01", 422646835, "SELL", 0.01, 4809.77, 4831.96, 4789.58, "2026-04-21 07:50:07", 4789.52, 0.00,  0.00,  20.25, "[tp 4789.58]"),
    ("2026-04-22 02:19:56", 423047007, "BUY",  0.01, 4732.12, 4706.12, 4751.64, "2026-04-22 04:30:31", 4748.45, 0.00,  0.00,  16.33, ""),
    ("2026-04-23 02:05:03", 423389702, "SELL", 0.01, 4724.39, 4747.00, 4703.44, "2026-04-23 03:13:26", 4706.33, 0.00,  0.00,  18.06, ""),
    ("2026-04-24 03:32:04", 423823679, "BUY",  0.01, 4703.51, 4684.96, 4717.21, "2026-04-24 05:24:45", 4684.93, 0.00,  0.00, -18.58, "[sl 4684.96]"),
    ("2026-04-27 04:08:17", 424146108, "BUY",  0.01, 4699.27, 4668.78, 4727.34, "2026-04-27 06:41:49", 4727.64, 0.00,  0.00,  28.37, "[tp 4727.34]"),
    # OUT-OF-SAMPLE (12 trades + 1 excluido, del 28 abril en adelante)
    ("2026-04-28 03:09:20", 424497463, "BUY",  0.02, 4700.45, 4681.82, 4716.00, "2026-04-28 04:50:49", 4681.65, 0.00,  0.00, -37.60, "[sl 4681.82]"),
    ("2026-04-29 03:19:49", 424904483, "BUY",  0.01, 4602.23, 4586.27, 4615.33, "2026-04-29 04:03:15", 4585.96, 0.00,  0.00, -16.27, "[sl 4586.27]"),
    ("2026-04-29 04:05:41", 424917471, "SELL", 0.01, 4584.55, 4601.45, 4557.14, "2026-04-29 05:07:47", 4601.49, 0.00,  0.00, -16.94, "[sl 4601.45]"),
    ("2026-04-30 02:59:39", 425308244, "BUY",  0.01, 4561.23, 4539.03, 4581.23, "2026-04-30 04:31:20", 4581.27, 0.00,  0.00,  20.04, "[tp 4581.23]"),
    ("2026-05-01 03:01:34", 425705525, "SELL", 0.01, 4626.21, 4640.34, 4613.15, "2026-05-01 04:20:52", 4614.95, 0.00,  0.00,  11.26, ""),
    ("2026-05-04 04:06:40", 426066239, "SELL", 0.01, 4602.48, 4633.47, 4575.27, "2026-05-04 11:05:01", 4574.80, 0.00,  0.00,  27.68, "[tp 4575.27]"),
    ("2026-05-05 03:09:12", 426476644, "BUY",  0.01, 4525.42, 4525.17, 4537.47, "2026-05-05 04:13:24", 4537.60, 0.00,  0.00,  12.18, "[tp 4537.47]"),
    ("2026-05-06 02:52:14", 426812628, "BUY",  0.01, 4593.15, 4572.38, 4608.17, "2026-05-06 03:32:33", 4606.12, 0.00,  0.00,  12.97, ""),
    ("2026-05-07 03:11:54", 427209160, "BUY",  0.02, 4699.67, 4689.01, 4709.81, "2026-05-07 03:31:31", 4689.06, 0.00,  0.00, -21.22, "[sl 4689.01]"),
    ("2026-05-07 04:20:14", 427223395, "BUY",  0.01, 4712.63, 4688.59, 4747.67, "2026-05-07 09:43:46", 4747.71, 0.00,  0.00,  35.08, "[tp 4747.67]"),
    ("2026-05-11 04:58:35", 427944598, "SELL", 0.01, 4682.41, None,    None,    "2026-05-11 04:59:49", 4685.47, 0.00,  0.00,  -3.06, ""),  # EXCLUIDO
    ("2026-05-11 05:58:34", 427961630, "SELL", 0.01, 4680.55, 4711.62, 4653.53, "2026-05-11 08:52:15", 4653.38, 0.00,  0.00,  27.17, "[tp 4653.53]"),
    ("2026-05-12 01:50:02", 428259471, "BUY",  0.02, 4752.26, 4734.63, 4766.62, "2026-05-12 02:52:23", 4763.61, 0.00,  0.00,  22.70, ""),
]

_TZ_SERVER = timezone(timedelta(hours=MT5_GMT_OFFSET_HOURS))


def _server_to_utc(server_str: str) -> pd.Timestamp:
    """Convierte string 'YYYY-MM-DD HH:MM:SS' en hora server GMT+3 a Timestamp UTC."""
    ts = pd.Timestamp(server_str)
    return ts.tz_localize(_TZ_SERVER).tz_convert(timezone.utc)


def _infer_close_type(close_comment: str, has_sl_tp: bool) -> str:
    """Infiere el tipo de cierre desde el comentario del deal OUT."""
    if not has_sl_tp:
        return "no_sl_tp"
    c = (close_comment or "").lower()
    if c.startswith("[sl"):
        return "sl"
    if c.startswith("[tp"):
        return "tp"
    return "manual"


def build_trades_dataframe(include_excluded: bool = False) -> pd.DataFrame:
    """Construye el DataFrame canónico de trades desde la tabla hardcoded.

    Args:
        include_excluded: Si True, incluye el trade 427944598 (sin SL/TP).
                         Default False.

    Returns:
        DataFrame con todos los trades válidos, ordenado por time_open_utc.
    """
    rows = []
    for tup in TRADES_RAW:
        (t_open_srv, pos_id, ttype, vol, p_open, sl, tp,
         t_close_srv, p_close, comm, swap, profit, close_comment) = tup

        excluded = pos_id in EXCLUDED_POSITION_IDS
        if excluded and not include_excluded:
            continue

        has_sl_tp = sl is not None and tp is not None
        time_open_utc = _server_to_utc(t_open_srv)
        time_close_utc = _server_to_utc(t_close_srv)

        sample = "in_sample" if time_open_utc < SPLIT_CUTOFF_UTC else "out_of_sample"

        rows.append({
            "position_id":      pos_id,
            "time_open_utc":    time_open_utc,
            "time_close_utc":   time_close_utc,
            "duration_minutes": (time_close_utc - time_open_utc).total_seconds() / 60,
            "type":             ttype,
            "volume":           vol,
            "symbol":           SYMBOL,
            "price_open":       p_open,
            "sl_initial":       sl,
            "tp_initial":       tp,
            "price_close":      p_close,
            "profit":           profit,
            "commission":       comm,
            "swap":             swap,
            "close_comment":    close_comment,
            "close_type":       _infer_close_type(close_comment, has_sl_tp),
            "sample":           sample,
            "excluded":         excluded,
        })

    df = pd.DataFrame(rows).sort_values("time_open_utc").reset_index(drop=True)
    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Resumen para verificación rápida."""
    in_s = df[df["sample"] == "in_sample"]
    out_s = df[df["sample"] == "out_of_sample"]
    return {
        "total_trades":              len(df),
        "in_sample":                 int((df["sample"] == "in_sample").sum()),
        "out_of_sample":             int((df["sample"] == "out_of_sample").sum()),
        "buy":                       int((df["type"] == "BUY").sum()),
        "sell":                      int((df["type"] == "SELL").sum()),
        "win_rate_global":           float((df["profit"] > 0).mean() * 100),
        "win_rate_in_sample":        float((in_s["profit"] > 0).mean() * 100),
        "win_rate_out_of_sample":    float((out_s["profit"] > 0).mean() * 100),
        "profit_total":              float(df["profit"].sum()),
        "profit_in_sample":          float(in_s["profit"].sum()),
        "profit_out_of_sample":      float(out_s["profit"].sum()),
        "close_type_counts":         df["close_type"].value_counts().to_dict(),
    }
