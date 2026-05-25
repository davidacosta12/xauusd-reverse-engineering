"""Analisis discriminatorio: encontrar 1 feature que distinga MATCHED de EXTRA.

Compara distribuciones de features en momentos de señal MATCHED vs EXTRA.
Reporta candidatos a filtro con metricas de impacto:
  - Cuantas extras eliminaria
  - Cuantas matched conservaria
  - Score de utilidad = matched_preserved% x extras_eliminated%
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from v2.config.settings import DATA_FEATURES, DATA_GROUND_TRUTH


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Cargando datos...")

    bt_path = DATA_GROUND_TRUTH.parent / "backtest_v3_split.parquet"
    bt = pd.read_parquet(bt_path)
    bt["matched"] = bt["matched"].astype(bool)
    n_matched = bt["matched"].sum()
    n_extra   = (~bt["matched"]).sum()
    print(f"  Backtest trades: {len(bt)}  (matched={n_matched}, extra={n_extra})")

    snapshots = pd.read_parquet(DATA_FEATURES / "snapshots_M1.parquet")
    if snapshots.index.tz is None:
        snapshots.index = snapshots.index.tz_localize("UTC")

    print("Recuperando snapshots al momento de cada entrada...")
    bt = bt.sort_values("entry_time").reset_index(drop=True)

    snap_rows = []
    for _, row in bt.iterrows():
        entry_ts = pd.Timestamp(row["entry_time"]).tz_convert("UTC")
        idx = snapshots.index.searchsorted(entry_ts, side="right") - 1
        if 0 <= idx < len(snapshots):
            snap_rows.append(snapshots.iloc[idx])
        else:
            snap_rows.append(pd.Series(dtype=float))

    feat = pd.DataFrame(snap_rows).reset_index(drop=True)
    feat["matched"]   = bt["matched"].values
    feat["direction"] = bt["direction"].values

    matched_df = feat[feat["matched"]].copy()
    extra_df   = feat[~feat["matched"]].copy()
    print(f"  Matched con features: {len(matched_df)}")
    print(f"  Extra con features  : {len(extra_df)}")

    # ── Analisis discriminatorio ─────────────────────────────────────────────
    num_cols = feat.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != "matched"]

    results = []
    for col in num_cols:
        m_vals = matched_df[col].dropna()
        e_vals = extra_df[col].dropna()
        if len(m_vals) < 5 or len(e_vals) < 5:
            continue
        m_med  = float(m_vals.median())
        e_med  = float(e_vals.median())
        pooled = float(feat[col].std())
        if pooled == 0 or np.isnan(pooled):
            continue
        z = (m_med - e_med) / pooled
        results.append({
            "feature":         col,
            "matched_median":  m_med,
            "extra_median":    e_med,
            "abs_diff":        abs(m_med - e_med),
            "z_diff":          z,
            "abs_z_diff":      abs(z),
        })

    res_df = pd.DataFrame(results).sort_values("abs_z_diff", ascending=False).reset_index(drop=True)

    print()
    print("=" * 100)
    print("    TOP 15 FEATURES DISCRIMINADORAS MATCHED vs EXTRA")
    print("=" * 100)
    print(res_df.head(15).to_string(index=False, float_format=lambda x: f"{x:+8.3f}"))

    # ── Simulacion de filtros ────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("    SIMULACION DE FILTROS (top 10 features)")
    print("=" * 100)
    hdr = (f"{'Feature':<45}{'Threshold':>12}{'M_keep':>12}"
           f"{'E_drop':>12}{'Score':>10}")
    print(hdr)
    print("-" * 100)

    for _, row in res_df.head(10).iterrows():
        col = row["feature"]
        m_med = row["matched_median"]
        e_med = row["extra_median"]
        thr   = (m_med + e_med) / 2

        if row["z_diff"] > 0:
            kept_m  = int((matched_df[col].dropna() >= thr).sum())
            drop_e  = int((extra_df[col].dropna() < thr).sum())
            dirstr  = ">="
        else:
            kept_m  = int((matched_df[col].dropna() <= thr).sum())
            drop_e  = int((extra_df[col].dropna() > thr).sum())
            dirstr  = "<="

        m_pct  = kept_m / len(matched_df) * 100
        e_pct  = drop_e / len(extra_df)   * 100
        score  = m_pct * e_pct / 100

        col_display = col[:43]
        print(f"{col_display:<45}{dirstr}{thr:>10.2f}"
              f"  {kept_m}/{len(matched_df)}({m_pct:.0f}%)"
              f"  {drop_e}/{len(extra_df)}({e_pct:.0f}%)"
              f"  {score:>8.1f}")

    # ── Candidatos viables ────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("    CANDIDATOS VIABLES (conserva >=18 matched, elimina >=15 extras)")
    print("=" * 100)

    viable = []
    for _, row in res_df.head(30).iterrows():
        col = row["feature"]
        thr = (row["matched_median"] + row["extra_median"]) / 2

        m_col = matched_df[col].dropna()
        e_col = extra_df[col].dropna()

        if row["z_diff"] > 0:
            kept_m = int((m_col >= thr).sum())
            drop_e = int((e_col < thr).sum())
            dirstr = ">="
        else:
            kept_m = int((m_col <= thr).sum())
            drop_e = int((e_col > thr).sum())
            dirstr = "<="

        if kept_m >= 18 and drop_e >= 15:
            score = (kept_m / len(matched_df)) * (drop_e / len(extra_df)) * 100
            viable.append({
                "feature":       col,
                "z_diff":        row["z_diff"],
                "direction":     dirstr,
                "threshold":     thr,
                "matched_kept":  kept_m,
                "extra_dropped": drop_e,
                "score":         score,
            })

    if not viable:
        print("  NO se encontraron candidatos que cumplan el criterio.")
        print("  Esto sugiere que no hay un filtro simple que distinga matched de extra.")
        print("  Recomendacion: pasar a OOS sin filtro adicional.")
        viable_df = pd.DataFrame()
    else:
        viable_df = pd.DataFrame(viable).sort_values("score", ascending=False).reset_index(drop=True)
        print(viable_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print()
        top = viable_df.iloc[0]
        print("=== RECOMENDACION ===")
        print(f"  Feature recomendada : {top['feature']}")
        print(f"  Regla               : señal aceptada solo si feature {top['direction']} {top['threshold']:.2f}")
        print(f"  Impacto esperado    : conserva {top['matched_kept']}/21 matched, "
              f"elimina {top['extra_dropped']}/33 extras")

    return res_df, viable_df


if __name__ == "__main__":
    main()
