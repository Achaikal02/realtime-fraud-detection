"""Consumer: membaca transaksi dari Kafka, menghitung fitur, memberi skor, memutuskan, menyimpan.

Pemakaian (dari folder proyek, Docker menyala, producer sudah/sedang mengirim):
    python -m fraud.consumer                      # jalan terus sampai Ctrl+C
    python -m fraud.consumer --idle-exit 30       # berhenti sendiri kalau 30 detik tanpa pesan
Jalankan sampai 3 kali di terminal berbeda: mereka berbagi 3 partisi (consumer group).

Urutan per batch (penting untuk keandalan):
  1. ambil sekumpulan pesan
  2. hitung fitur satu per satu SESUAI URUTAN (Redis: hitung -> tambah)
  3. beri skor sekaligus untuk seluruh batch
  4. simpan ke Postgres dan COMMIT
  5. baru setelah itu catat offset Kafka
Kalau consumer mati di antara 4 dan 5, batch itu diproses ulang setelah restart
("minimal sekali"). Aman, karena Redis dan Postgres sama-sama menolak duplikat.
"""

import argparse
import math
import time
from collections import Counter

import numpy as np

from fraud.db import connect, features_json, insert_rows
from fraud.features import MODEL_TYPES
from fraud.online_features import OnlineFeatureStore, static_features
from fraud.scoring import Scorer
from fraud.stream import BROKERS, TOPIC_TRANSACTIONS, decode

GROUP_ID = "fraud-scorer"


def process_batch(txs: list[dict], store: OnlineFeatureStore, scorer: Scorer,
                  score_from_step: int) -> list[dict]:
    """Inti consumer, terpisah dari Kafka dan Postgres supaya bisa diuji sendiri."""
    feats, degraded = [], []
    for tx in txs:                                    # HARUS berurutan: riwayat bergantung urutan
        hist, d = store.update_and_get(tx)
        feats.append({**static_features(tx), **hist})
        degraded.append(d)

    to_score = [i for i, tx in enumerate(txs)
                if tx["type"] in MODEL_TYPES and tx["step"] >= score_from_step]
    scores = np.full(len(txs), np.nan)
    if to_score:
        scores[to_score] = scorer.score([feats[i] for i in to_score])

    rows = []
    for i, tx in enumerate(txs):
        s = float(scores[i])
        if tx["step"] < score_from_step:
            decision = "WARMUP"                       # hanya mengisi riwayat Redis
        else:
            decision = scorer.decide(tx["type"], tx["amount"], s)
        rows.append({
            "tx_id": tx["tx_id"], "step": tx["step"], "type": tx["type"], "amount": tx["amount"],
            "name_orig": tx["nameOrig"], "oldbalance_org": tx["oldbalanceOrg"],
            "name_dest": tx["nameDest"], "oldbalance_dest": tx["oldbalanceDest"],
            "score": None if math.isnan(s) else s, "decision": decision,
            # urutan antrean REVIEW = perkiraan kerugian (skor x nominal)
            "review_priority": s * tx["amount"] if decision == "REVIEW" else None,
            "degraded": degraded[i], "features": features_json(feats[i]),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score-from-step", type=int, default=401,
                   help="step sebelum ini hanya mengisi riwayat Redis (pemanasan)")
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--idle-exit", type=float, default=0, help="berhenti setelah N detik tanpa pesan; 0 = tidak")
    p.add_argument("--redis-host", default="localhost")
    args = p.parse_args()

    from confluent_kafka import Consumer
    from confluent_kafka.admin import AdminClient

    from fraud.stream import ensure_topic

    # Pastikan topic ada sebelum berlangganan. Tanpa ini, consumer yang start lebih dulu
    # daripada producer (misalnya setelah `fraud.reset`) tidak mendapat partisi.
    if ensure_topic(AdminClient({"bootstrap.servers": BROKERS})):
        print(f"Topic '{TOPIC_TRANSACTIONS}' dibuat.")

    scorer = Scorer.from_registry()
    print(f"Model dimuat: {len(scorer.features)} fitur ({', '.join(scorer.features)})")
    store = OnlineFeatureStore.connect(host=args.redis_host)
    conn = connect()

    consumer = Consumer({
        "bootstrap.servers": BROKERS,
        "group.id": GROUP_ID,
        "enable.auto.commit": False,      # offset dicatat manual, SETELAH Postgres commit
        "auto.offset.reset": "earliest",  # grup baru mulai dari pesan pertama di topic
    })
    consumer.subscribe([TOPIC_TRANSACTIONS],
                       on_assign=lambda c, parts: print("Mendapat partisi:", sorted(p.partition for p in parts)))

    totals, t0, last_msg, last_report = Counter(), time.perf_counter(), time.monotonic(), 0
    try:
        while True:
            msgs = consumer.consume(num_messages=args.batch, timeout=1.0)
            good = [m for m in msgs if m.error() is None]
            for m in msgs:
                if m.error() is not None:
                    print("Error Kafka:", m.error())
            if not good:
                if args.idle_exit and time.monotonic() - last_msg > args.idle_exit:
                    print(f"Tidak ada pesan selama {args.idle_exit:.0f} detik, berhenti.")
                    break
                continue
            last_msg = time.monotonic()

            rows = process_batch([decode(m.value()) for m in good], store, scorer, args.score_from_step)
            for r, m in zip(rows, good):
                r["kafka_partition"], r["kafka_offset"] = m.partition(), m.offset()
            insert_rows(conn, rows)                    # 4. simpan dan commit Postgres
            consumer.commit(asynchronous=False)        # 5. baru catat offset

            totals["diproses"] += len(rows)
            totals.update(r["decision"] for r in rows)
            totals["degraded"] += sum(r["degraded"] for r in rows)
            if totals["diproses"] - last_report >= 20_000:
                last_report = totals["diproses"]
                rate = totals["diproses"] / (time.perf_counter() - t0)
                print(f"  {totals['diproses']:,} diproses ({rate:,.0f}/detik) | step {rows[-1]['step']} | "
                      f"BLOCK {totals['BLOCK']:,} | degraded {totals['degraded']:,}")
    except KeyboardInterrupt:
        print("\nDihentikan (Ctrl+C).")
    finally:
        consumer.close()                               # keluar dari grup dengan rapi
        conn.close()
        print("Ringkasan:", dict(totals))


if __name__ == "__main__":
    main()
