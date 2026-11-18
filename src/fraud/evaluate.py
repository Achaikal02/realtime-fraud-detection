"""Nilai keputusan pipeline dengan label asli PaySim, termasuk simulasi antrean REVIEW.

Di produksi, label datang berminggu-minggu kemudian. Di sini kita "mempercepat waktu":
label diambil dari file PaySim lewat tx_id (= nomor baris di file asli).

Keputusan DIHITUNG ULANG dari skor yang tersimpan di Postgres, jadi aturan baru bisa
dinilai dari replay lama tanpa menjalankan ulang pipeline.

    python -m fraud.evaluate                       # kapasitas 60/jam, batas tahan 1, 2, 4, 8 jam
    python -m fraud.evaluate --capacity 30 --holds 2 4
"""

import argparse
import pathlib

import numpy as np
import pandas as pd

from fraud.decision import FALSE_POSITIVE_COST, REVIEW_HOLD_COST, decide
from fraud.features import MODEL_TYPES
from fraud.review import review_cost_summary, simulate_review

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_scored(data_path) -> pd.DataFrame:
    import psycopg

    from fraud.db import POSTGRES_DSN

    cols = ["tx_id", "step", "type", "amount", "score", "degraded"]
    with psycopg.connect(POSTGRES_DSN) as conn:
        rows = conn.execute(f"SELECT {', '.join(cols)} FROM transactions "
                            "WHERE decision <> 'WARMUP'").fetchall()
    d = pd.DataFrame(rows, columns=cols)
    if d.empty:
        raise SystemExit("Belum ada transaksi yang dinilai di Postgres.")
    labels = pd.read_parquet(data_path, columns=["isFraud"]).isFraud.to_numpy()
    d["isFraud"] = labels[d.tx_id.str.slice(3).astype(int).to_numpy()]
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data/processed/paysim.parquet"))
    p.add_argument("--capacity", type=int, default=60, help="kasus REVIEW yang sanggup diperiksa per jam")
    p.add_argument("--holds", type=int, nargs="+", default=[1, 2, 4, 8], help="batas tahan dana (jam)")
    args = p.parse_args()

    d = load_scored(args.data)
    s = d[d.type.isin(MODEL_TYPES)].copy()
    print(f"Transaksi dinilai: {len(d):,} (step {d.step.min()}–{d.step.max()}), "
          f"TRANSFER/CASH_OUT: {len(s):,}, degraded: {int(d.degraded.sum()):,}\n")

    # 1. Tanpa REVIEW (aturan lama): BLOCK atau ALLOW
    s["decision"] = np.where(s.score * s.amount > (1 - s.score) * FALSE_POSITIVE_COST, "BLOCK", "ALLOW")
    base = review_cost_summary(s, pd.Series(dtype=object), REVIEW_HOLD_COST)
    nothing = float(s.amount[s.isFraud == 1].sum())

    # 2. Dengan tangga BLOCK / REVIEW / ALLOW
    s["decision"] = [decide(sc, a) for sc, a in zip(s.score, s.amount)]
    rev = s[s.decision == "REVIEW"].assign(priority=lambda x: x.score * x.amount)
    per_step = rev.groupby("step").size().reindex(range(s.step.min(), s.step.max() + 1), fill_value=0)
    print(f"Kasus REVIEW: {len(rev):,} | per jam: rata-rata {per_step.mean():.1f}, "
          f"maksimum {per_step.max()} (kapasitas {args.capacity})")
    print(f"Fraud total: {int(s.isFraud.sum()):,} | tanpa sistem: Rp {nothing:,.0f}\n")

    rows = {"tanpa REVIEW": base}
    for h in args.holds:
        outcome = simulate_review(rev[["step", "priority"]], capacity=args.capacity, hold_steps=h)
        rows[f"REVIEW, tahan {h} jam"] = review_cost_summary(s, outcome, REVIEW_HOLD_COST)
    table = pd.DataFrame(rows).T
    with pd.option_context("display.float_format", "{:,.0f}".format, "display.width", 200,
                           "display.max_columns", None):
        print(table[["biaya_total", "tertangkap_block", "tertangkap_review", "fraud_lolos",
                     "kasus_review", "kedaluwarsa", "asli_ditahan", "salah_blokir"]])


if __name__ == "__main__":
    main()
