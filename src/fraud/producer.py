"""Memutar ulang transaksi PaySim ke topic Kafka `transactions`, sesuai urutan aslinya.

Pemakaian (dari folder proyek, Docker menyala):
    python -m fraud.producer --limit 1000           # coba 1000 transaksi dulu
    python -m fraud.producer                        # step 377-743, 1000 transaksi/detik
    python -m fraud.producer --rate 0               # secepat mungkin
    python -m fraud.producer --dry-run --limit 5    # lihat isi pesan tanpa Kafka

Default mulai step 377: step 377-400 hanya untuk PEMANASAN riwayat Redis (24 step, sama
dengan jendela fitur terpanjang), lalu consumer menilai step 401-743, periode yang TIDAK
dipakai melatih model (model final dilatih dengan step <= 400). Ini meniru "produksi".
"""

import argparse
import pathlib
import time
from collections import Counter

import numpy as np
import pandas as pd

from fraud.stream import (BROKERS, MESSAGE_FIELDS, TOPIC_TRANSACTIONS, amount_key, encode,
                          ensure_topic)

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_COLUMNS = [c for c in MESSAGE_FIELDS if c != "tx_id"]


def load_range(path, start_step: int, end_step: int, limit: int | None = None) -> pd.DataFrame:
    """Ambil transaksi step [start_step, end_step] dengan tx_id = nomor baris di file asli.

    Nomor baris global dipakai sebagai id karena sama dengan urutan yang dipakai
    `build_features` saat latihan -> memudahkan uji paritas online vs batch nanti.
    File PaySim urut per step, jadi baris untuk rentang step selalu bersambung.
    """
    steps = pd.read_parquet(path, columns=["step"]).step.to_numpy()
    if not np.all(steps[1:] >= steps[:-1]):
        raise ValueError("File harus urut berdasarkan step.")
    lo = int(np.searchsorted(steps, start_step, side="left"))
    hi = int(np.searchsorted(steps, end_step, side="right"))

    df = pd.read_parquet(path, columns=SOURCE_COLUMNS,
                         filters=[("step", ">=", start_step), ("step", "<=", end_step)])
    if len(df) != hi - lo:
        raise RuntimeError("Jumlah baris tidak cocok dengan rentang step; urutan file berubah?")
    df.insert(0, "tx_id", [f"tx_{i}" for i in range(lo, hi)])
    df["type"] = df["type"].astype(str)
    return df.head(limit) if limit else df


def to_message(row) -> dict:
    return {
        "tx_id": row.tx_id,
        "step": int(row.step),
        "type": row.type,
        "amount": float(row.amount),
        "nameOrig": str(row.nameOrig),
        "oldbalanceOrg": float(row.oldbalanceOrg),
        "nameDest": str(row.nameDest),
        "oldbalanceDest": float(row.oldbalanceDest),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data/processed/paysim.parquet"))
    p.add_argument("--start-step", type=int, default=377)
    p.add_argument("--end-step", type=int, default=743)
    p.add_argument("--limit", type=int, default=None, help="kirim N transaksi pertama saja")
    p.add_argument("--rate", type=float, default=1000, help="transaksi per detik; 0 = secepatnya")
    p.add_argument("--dry-run", action="store_true", help="cetak pesan, tidak mengirim ke Kafka")
    args = p.parse_args()

    df = load_range(args.data, args.start_step, args.end_step, args.limit)
    print(f"{len(df):,} transaksi (step {df.step.min()}–{df.step.max()}), "
          f"tx_id {df.tx_id.iloc[0]} … {df.tx_id.iloc[-1]}")

    if args.dry_run:
        for row in df.head(5).itertuples(index=False):
            m = to_message(row)
            print(f"  key={amount_key(m['amount'])!r:>14}  {m}")
        return

    from confluent_kafka import Producer
    from confluent_kafka.admin import AdminClient

    if ensure_topic(AdminClient({"bootstrap.servers": BROKERS})):
        print(f"Topic '{TOPIC_TRANSACTIONS}' dibuat.")

    producer = Producer({
        "bootstrap.servers": BROKERS,
        # Idempoten: kalau terjadi kirim ulang (retry), pesan tidak terduplikasi dan
        # urutan per partisi tetap terjaga. Penting karena fitur kita bergantung urutan.
        "enable.idempotence": True,
        "linger.ms": 5,  # kumpulkan pesan beberapa milidetik agar dikirim berkelompok
    })

    per_partition, errors = Counter(), Counter()

    def on_delivery(err, msg):
        if err is not None:
            errors[str(err)] += 1
        else:
            per_partition[msg.partition()] += 1

    t0 = time.perf_counter()
    for n, row in enumerate(df.itertuples(index=False), start=1):
        m = to_message(row)
        while True:
            try:
                producer.produce(TOPIC_TRANSACTIONS, key=amount_key(m["amount"]), value=encode(m),
                                 on_delivery=on_delivery)
                break
            except BufferError:          # antrean lokal penuh: tunggu sebagian terkirim
                producer.poll(0.1)
        producer.poll(0)
        if args.rate > 0:                # atur kecepatan agar mirip arus transaksi
            ahead = t0 + n / args.rate - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)
        if n % 50_000 == 0:
            print(f"  {n:,} terkirim ({n / (time.perf_counter() - t0):,.0f}/detik), step {m['step']}")

    producer.flush(30)
    dt = time.perf_counter() - t0
    print(f"\nSelesai: {sum(per_partition.values()):,} terkirim dalam {dt:,.1f} detik "
          f"({sum(per_partition.values()) / dt:,.0f}/detik)")
    print("Per partisi:", dict(sorted(per_partition.items())))
    if errors:
        print("GAGAL:", dict(errors))


if __name__ == "__main__":
    main()
