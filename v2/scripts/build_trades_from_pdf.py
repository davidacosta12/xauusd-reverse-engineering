"""Construye trades.parquet desde la tabla extraída del PDF oficial del broker.

Reemplaza el archivo actual en v2/data/ground_truth/trades.parquet
(que tenía SL/TP en NaN por limitación de la API MT5).
"""
import shutil

from v2.src.data.trades_pdf_source import build_trades_dataframe, get_summary_stats
from v2.config.settings import DATA_GROUND_TRUTH, DATA_RAW_MT5


def main():
    # Backup del archivo viejo (extraído de MT5 sin SL/TP)
    old_path = DATA_GROUND_TRUTH / "trades.parquet"
    if old_path.exists():
        backup_path = DATA_RAW_MT5 / "trades_mt5_extracted_DEPRECATED.parquet"
        shutil.copy(old_path, backup_path)
        print(f"Backup del archivo viejo: {backup_path}")

    # Construir el nuevo trades.parquet desde el PDF
    print("\nConstruyendo trades.parquet desde la tabla del PDF...")
    df = build_trades_dataframe(include_excluded=False)

    # Guardar
    out_path = DATA_GROUND_TRUTH / "trades.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"Guardado: {out_path} ({len(df)} trades)")

    # Summary
    print("\n=== RESUMEN ===")
    stats = get_summary_stats(df)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Validación de cordura
    print("\n=== VALIDACIONES ===")
    assert stats["total_trades"] == 42,   f"Esperados 42 trades válidos, obtenidos {stats['total_trades']}"
    assert stats["in_sample"] == 30,      f"Esperados 30 in-sample, obtenidos {stats['in_sample']}"
    assert stats["out_of_sample"] == 12,  f"Esperados 12 out-of-sample, obtenidos {stats['out_of_sample']}"
    assert df["sl_initial"].notna().all(), "Hay NaN en sl_initial!"
    assert df["tp_initial"].notna().all(), "Hay NaN en tp_initial!"
    print("  OK Conteos correctos")
    print("  OK Todos los SL/TP poblados")

    # Mostrar primeros y últimos trades
    cols = ["time_open_utc", "type", "price_open", "sl_initial", "tp_initial", "price_close", "profit", "close_type"]
    print("\n=== PRIMEROS 3 TRADES (in-sample) ===")
    print(df[df["sample"] == "in_sample"].head(3)[cols].to_string())

    print("\n=== ÚLTIMOS 3 TRADES (out-of-sample) ===")
    print(df[df["sample"] == "out_of_sample"].tail(3)[cols].to_string())

    return df


if __name__ == "__main__":
    main()
