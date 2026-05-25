"""Re-extrae trades históricos del periodo correcto y valida contra valores esperados."""
import logging
from datetime import datetime, timezone

from v2.src.data.mt5_extractor import connect_mt5, disconnect_mt5, extract_trade_history
from v2.config.settings import (
    DATA_GROUND_TRUTH,
    TRADES_PERIOD_START_UTC,
    TRADES_PERIOD_END_UTC,
    TOTAL_TRADES,
    TOTAL_PNL_REAL,
    WIN_RATE_REAL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def main() -> None:
    if not connect_mt5():
        raise SystemExit("No se pudo conectar a MT5. Verifica el .env y que MT5 esté abierto.")

    try:
        # Periodo con margen de 1 día a cada lado
        d_from = datetime(2026, 3, 18, 0, 0, 0, tzinfo=timezone.utc)
        d_to   = datetime(2026, 4, 28, 23, 59, 59, tzinfo=timezone.utc)

        print(f"\nExtrayendo trades de {d_from} a {d_to}...")
        trades = extract_trade_history(d_from, d_to, save_parquet=True)

        print("\n=== TRADES EXTRAÍDOS (todos en el margen) ===")
        print(trades[[
            "time_open_utc", "type", "price_open", "sl", "tp",
            "price_close", "profit", "comment_close_inferred_type"
        ]].to_string())

        # Filtrar al periodo exacto de los 30 trades reales
        trades_filtered = trades[
            (trades["time_open_utc"] >= TRADES_PERIOD_START_UTC) &
            (trades["time_open_utc"] <= TRADES_PERIOD_END_UTC)
        ].reset_index(drop=True)

        print(f"\n=== DESPUÉS DE FILTRAR AL PERIODO EXACTO ===")
        print(f"Trades: {len(trades_filtered)} (esperados: {TOTAL_TRADES})")
        print(f"P&L total:  ${trades_filtered['profit'].sum():.2f}  (esperado: +${TOTAL_PNL_REAL:.2f})")
        wr = (trades_filtered["profit"] > 0).mean() * 100
        print(f"Win rate:   {wr:.1f}%  (esperado: {WIN_RATE_REAL * 100:.1f}%)")

        print("\n=== PRIMEROS 3 TRADES (verificación manual) ===")
        cols = ["time_open_utc", "type", "price_open", "sl", "tp", "price_close", "profit", "comment_close_inferred_type"]
        print(trades_filtered[cols].head(3).to_string())

        if len(trades_filtered) > 0:
            out_path = DATA_GROUND_TRUTH / "trades.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            trades_filtered.to_parquet(out_path, index=False)
            print(f"\nGuardado en {out_path}")

        print("\n=== CIERRE POR TIPO ===")
        print(trades_filtered["comment_close_inferred_type"].value_counts().to_string())

    finally:
        disconnect_mt5()


if __name__ == "__main__":
    main()
